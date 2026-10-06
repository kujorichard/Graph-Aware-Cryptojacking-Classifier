"""Focused tests for LLM evidence privacy and serialization."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import graph_aware_classifier.llm_analyzer as llm_analyzer
from graph_aware_classifier.llm_analyzer import LLMAnalyzer, LLMConfig


def _write_occurrences(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "ngram": ["MOV", "SUB"],
                        "occurrences": [
                            {
                                "function": "FUN_14000d6e0",
                                "addresses": ["0x14000d6e0", "0x14000d6e5"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_analyze_sample_omits_addresses_from_prompt_and_saved_evidence(tmp_path):
    occurrences_path = tmp_path / "occurrences.json"
    _write_occurrences(occurrences_path)
    top_features = [
        {
            "feature_id": 1,
            "ngram": "MOV SUB",
            "value": 0.75,
            "shap_value": 0.12,
        }
    ]
    analyzer = LLMAnalyzer(
        config=LLMConfig(backend="ollama"),
        results_dir=tmp_path / "results",
    )
    response = json.dumps(
        {
            "tier": "low-confidence",
            "explanation": "Evidence is limited.",
            "evidence_analysis": [],
            "confidence_score": 0.5,
            "key_indicators": [],
            "recommended_action": "Review manually.",
        }
    )

    with patch.object(analyzer, "_call_llm", return_value=response) as call_llm:
        result = analyzer.analyze_sample(
            binary_name="sample.exe",
            rf_prediction=1,
            rf_probability=0.87,
            top_features=top_features,
            occurrences_path=occurrences_path,
            function_disassembly={
                "FUN_14000d6e0": [
                    "0x14000d6e0 MOV qword ptr [RSP + 0x8], RBX"
                ]
            },
        )

    prompt = call_llm.call_args.args[0]
    assert "MOV SUB" in prompt
    assert "FUN_14000d6e0" not in prompt
    assert "Functions :" not in prompt
    assert "0x14000d6e0" not in prompt
    assert "0x14000d6e5" not in prompt
    assert "MOV qword ptr" not in prompt

    saved_path = tmp_path / "results" / "sample.exe_llm_result.json"
    saved_result = json.loads(saved_path.read_text(encoding="utf-8"))
    assert result["evidence"] == saved_result["evidence"]
    assert saved_result["evidence"][0]["functions"] == [
        {"function": "FUN_14000d6e0"}
    ]
    assert "function_disassembly" not in saved_result
    assert "address_groups" not in json.dumps(saved_result["evidence"])
    assert "0x14000d6e0" not in json.dumps(saved_result["evidence"])

    occurrence_data = json.loads(occurrences_path.read_text(encoding="utf-8"))
    assert occurrence_data["features"][0]["occurrences"][0]["addresses"] == [
        "0x14000d6e0",
        "0x14000d6e5",
    ]


def test_claude_request_omits_unsupported_temperature_argument():
    analyzer = LLMAnalyzer(
        config=LLMConfig(backend="claude", claude_api_key="test-key")
    )
    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=12, output_tokens=4),
        content=[SimpleNamespace(text="response")],
    )
    client = Mock()
    client.messages.create.return_value = response
    anthropic_mock = Mock()
    anthropic_mock.Anthropic.return_value = client

    with patch.object(llm_analyzer, "anthropic", anthropic_mock):
        result = analyzer._call_claude("test prompt")

    assert result == "response"
    request = client.messages.create.call_args.kwargs
    assert "temperature" not in request
    assert request["messages"] == [
        {"role": "user", "content": "test prompt"}
    ]
