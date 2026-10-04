import json
from types import SimpleNamespace
from unittest.mock import patch

from graph_aware_classifier.llm_analyzer import (
    LLMAnalyzer,
    LLMConfig,
    _token_usage_from_response,
    _validate_analysis_response,
    build_analysis_prompt,
    build_evidence_payload,
)


def test_token_usage_normalizes_ollama_and_claude_counts():
    ollama_usage = _token_usage_from_response(
        {
            "prompt_eval_count": 123,
            "eval_count": 45,
        },
        input_field="prompt_eval_count",
        output_field="eval_count",
        source="ollama",
    )
    claude_usage = _token_usage_from_response(
        type("Usage", (), {"input_tokens": 234, "output_tokens": 56})(),
        input_field="input_tokens",
        output_field="output_tokens",
        source="claude",
    )
    missing_usage = _token_usage_from_response(
        {},
        input_field="prompt_eval_count",
        output_field="eval_count",
        source="ollama",
    )

    assert ollama_usage == {
        "input_tokens": 123,
        "output_tokens": 45,
        "source": "ollama",
    }
    assert claude_usage == {
        "input_tokens": 234,
        "output_tokens": 56,
        "source": "claude",
    }
    assert missing_usage == {
        "input_tokens": None,
        "output_tokens": None,
        "source": "ollama",
    }


def test_analysis_response_uses_authoritative_shap_value():
    evidence = [{"ngram": "CALL PUSH", "shap_value": 0.25}]
    parsed = {
        "tier": "low-confidence",
        "explanation": "A claim tied to the provided feature.",
        "evidence_analysis": [
            {
                "ngram": "CALL PUSH",
                "shap_contribution": 0.0,
                "assessment": "The feature occurs in the supplied evidence.",
            },
        ],
        "confidence_score": 0.5,
        "key_indicators": [],
        "recommended_action": "Review the sample.",
    }

    validated = _validate_analysis_response(parsed, evidence)

    assert validated["evidence_analysis"][0]["shap_contribution"] == 0.25


def test_analysis_response_rejects_unknown_shap_ngram():
    parsed = {
        "tier": "benign",
        "explanation": "No evidence supports the claim.",
        "evidence_analysis": [
            {
                "ngram": "MOV RAX",
                "shap_contribution": 0.0,
                "assessment": "Unsupported feature.",
            },
        ],
        "confidence_score": 0.5,
        "key_indicators": [],
        "recommended_action": "Review the sample.",
    }

    try:
        _validate_analysis_response(parsed, [{"ngram": "CALL PUSH", "shap_value": 0.2}])
    except ValueError as error:
        assert "unknown SHAP n-gram" in str(error)
    else:
        raise AssertionError("Unknown SHAP n-gram should be rejected")


def test_default_generation_token_budget_is_8192(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)

    assert LLMConfig.from_env(env_path="missing-test-env").max_tokens == 8192
    assert LLMConfig(max_tokens=2048).max_tokens == 2048


def test_ollama_generation_records_finish_reason_and_token_budget():
    client = SimpleNamespace(
        generate=lambda **kwargs: {
            "response": '{"tier":',
            "done_reason": "length",
            "done": True,
            "eval_count": 8192,
            "prompt_eval_count": 10,
        },
    )
    analyzer = LLMAnalyzer(
        config=LLMConfig(max_tokens=8192, max_retries=1),
    )

    with patch(
        "graph_aware_classifier.llm_analyzer._ollama_pkg",
        SimpleNamespace(Client=lambda host: client),
    ), patch.object(client, "generate", wraps=client.generate) as generate:
        output = analyzer._call_ollama("prompt")

    assert output == '{"tier":'
    assert generate.call_args.kwargs["options"]["num_predict"] == 8192
    assert analyzer._last_response_metadata == {
        "finish_reason": "length",
        "truncated": True,
    }


def test_unparseable_response_is_not_reported_as_benign(tmp_path):
    analyzer = LLMAnalyzer(
        config=LLMConfig(max_retries=1),
        results_dir=tmp_path / "llm_results",
    )

    with patch.object(
        analyzer,
        "_call_llm",
        return_value="Thinking Process: answer in JSON at the end...",
    ):
        result = analyzer.analyze_sample(
            binary_name="sample.exe",
            rf_prediction=1,
            rf_probability=0.9,
            top_features=[
                {"ngram": "CALL PUSH", "value": 0.5, "shap_value": 0.2},
            ],
            occurrences_path=tmp_path / "missing-occurrences.json",
        )

    assert result["status"] == "failed"
    assert result["analysis"]["tier"] is None
    assert result["analysis"]["confidence_score"] is None
    assert "parse_error" in result["analysis"]


def test_truncated_generation_is_failed_even_if_json_is_incomplete(tmp_path):
    analyzer = LLMAnalyzer(
        config=LLMConfig(max_tokens=8192, max_retries=1),
        results_dir=tmp_path / "llm_results",
    )

    def truncated_response(prompt):
        analyzer._last_response_metadata = {
            "finish_reason": "length",
            "truncated": True,
        }
        return '{"tier":"high-confidence"'

    with patch.object(analyzer, "_call_llm", side_effect=truncated_response):
        result = analyzer.analyze_sample(
            binary_name="truncated.exe",
            rf_prediction=1,
            rf_probability=0.9,
            top_features=[
                {"ngram": "CALL PUSH", "value": 0.5, "shap_value": 0.2},
            ],
            occurrences_path=tmp_path / "missing-occurrences.json",
        )

    assert result["status"] == "failed"
    assert result["response_metadata"]["truncated"] is True
    assert result["analysis"]["tier"] is None
    assert "reached its output limit" in result["failure_reason"]


def test_prompt_includes_full_disassembly_for_shap_functions(tmp_path):
    occurrence_path = tmp_path / "sample_vocabulary.json"
    occurrence_path.write_text(
        json.dumps({
            "features": [
                {
                    "ngram": ["CALL", "PUSH"],
                    "occurrences": [
                        {
                            "function": "hash_worker",
                            "addresses": ["0x1000", "0x1005"],
                        },
                    ],
                },
                {
                    "ngram": ["MOV", "SUB"],
                    "occurrences": [
                        {
                            "function": "hash_worker",
                            "addresses": ["0x1010", "0x1014"],
                        },
                    ],
                },
            ],
        }),
        encoding="utf-8",
    )
    disassembly = {
        "hash_worker": [
            "0x1000 CALL 0x2000",
            "0x1005 PUSH RAX",
            "0x1010 MOV RAX, RBX",
            "0x1014 SUB RAX, 1",
        ],
    }

    evidence = build_evidence_payload(
        [
            {
                "ngram": "CALL PUSH",
                "value": 0.5,
                "shap_value": 0.2,
            },
            {
                "ngram": "MOV SUB",
                "value": 0.25,
                "shap_value": 0.1,
            },
        ],
        occurrence_path,
    )
    prompt = build_analysis_prompt(
        binary_name="sample.exe",
        rf_prediction=1,
        rf_probability=0.9,
        evidence=evidence,
        function_disassembly=disassembly,
    )

    assert all(
        feature["functions"][0]["function"] == "hash_worker"
        for feature in evidence
    )
    assert "SPOTTED FUNCTION DISASSEMBLY" in prompt
    assert prompt.count("Function: hash_worker") == 1
    for instruction in disassembly["hash_worker"]:
        assert instruction in prompt

    analyzer = LLMAnalyzer(
        config=LLMConfig(max_retries=1),
        results_dir=tmp_path / "llm_results",
    )
    response = json.dumps({
        "tier": "low-confidence",
        "explanation": "Assessment uses the supplied function disassembly.",
        "evidence_analysis": [],
        "confidence_score": 0.5,
        "key_indicators": [],
        "recommended_action": "Review the function.",
    })
    with patch.object(analyzer, "_call_llm", return_value=response) as llm_call:
        result = analyzer.analyze_sample(
            binary_name="sample.exe",
            rf_prediction=1,
            rf_probability=0.9,
            top_features=[
                {
                    "ngram": "CALL PUSH",
                    "value": 0.5,
                    "shap_value": 0.2,
                },
                {
                    "ngram": "MOV SUB",
                    "value": 0.25,
                    "shap_value": 0.1,
                },
            ],
            occurrences_path=occurrence_path,
            function_disassembly=disassembly,
        )

    assert "0x1014 SUB RAX, 1" in llm_call.call_args.args[0]
    assert result["function_disassembly"] == disassembly
    assert result["status"] == "completed"
    assert result["analysis"]["evidence_analysis"] == []
    assert result["token_usage"] == {
        "input_tokens": None,
        "output_tokens": None,
        "source": "unavailable",
    }
    assert "disassembly" not in result["evidence"][0]["functions"][0]
    saved_result = json.loads(
        (tmp_path / "llm_results" / "sample.exe_llm_result.json").read_text(
            encoding="utf-8",
        )
    )
    assert saved_result["function_disassembly"] == disassembly
