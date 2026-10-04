import json
from unittest.mock import patch

from graph_aware_classifier.llm_analyzer import (
    LLMAnalyzer,
    LLMConfig,
    _token_usage_from_response,
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
