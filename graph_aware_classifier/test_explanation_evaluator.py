import json
from types import SimpleNamespace
from unittest.mock import patch

from graph_aware_classifier.explanation_evaluator import (
    CRITERIA,
    EvaluatorConfig,
    ExplanationEvaluator,
)


def _valid_eval_response(evidence_ids=None):
    reference_ids = evidence_ids or ["SHAP:1", "DISASM:worker:0"]
    return {
        "criteria": [
            {
                "name": name,
                "score": 4,
                "rationale": "The assessment is grounded in supplied evidence.",
                "evidence_ids": reference_ids if name != "shap_alignment" else ["SHAP:1"],
            }
            for name in CRITERIA
        ],
        "claims": [
            {
                "claim": "The explanation references the supplied feature.",
                "support": "supported",
                "evidence_ids": ["SHAP:1"],
                "rationale": "The n-gram is present in the evidence.",
            },
            {
                "claim": "The code contacts a mining pool.",
                "support": "unsupported",
                "evidence_ids": [],
                "rationale": "No networking behavior is in the supplied region.",
            },
            {
                "claim": "A hidden behavior is present.",
                "support": "not_verifiable",
                "evidence_ids": [],
                "rationale": "The input does not provide ground-truth annotations.",
            },
        ],
        "overall_rationale": "One claim exceeds the supplied evidence.",
    }


def test_evaluator_has_independent_2048_token_budget(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
    monkeypatch.delenv("GEVAL_MAX_TOKENS", raising=False)

    config = EvaluatorConfig.from_env()

    assert config.max_tokens == 2048
    assert EvaluatorConfig(max_tokens=512).max_tokens == 512


def test_evaluation_validates_refs_and_calculates_unsupported_claim_rate():
    evaluator = ExplanationEvaluator(
        config=EvaluatorConfig(),
        generate_fn=lambda prompt: json.dumps(_valid_eval_response()),
    )

    result = evaluator.evaluate(
        explanation={"explanation": "A claim tied to SHAP."},
        evidence=[{
            "rank": 1,
            "ngram": "CALL PUSH",
            "frequency": 0.5,
            "shap_value": 0.2,
            "functions": [],
        }],
        function_disassembly={"worker": ["0x1000 CALL 0x2000"]},
    )

    assert result["status"] == "completed"
    assert result["result"]["claim_summary"] == {
        "claim_count": 3,
        "unsupported_or_contradicted_count": 1,
        "unsupported_claim_rate": 1 / 3,
        "rate_definition": (
            "Unsupported and contradicted claims divided by all listed "
            "claims, including not_verifiable claims."
        ),
    }


def test_evaluation_rejects_invalid_evidence_reference():
    response = _valid_eval_response(evidence_ids=["SHAP:999"])
    evaluator = ExplanationEvaluator(
        config=EvaluatorConfig(),
        generate_fn=lambda prompt: json.dumps(response),
    )

    result = evaluator.evaluate(
        explanation={"explanation": "A claim."},
        evidence=[{
            "rank": 1,
            "ngram": "CALL PUSH",
            "frequency": 0.5,
            "shap_value": 0.2,
            "functions": [],
        }],
        function_disassembly={"worker": ["0x1000 CALL 0x2000"]},
    )

    assert result["status"] == "failed"
    assert "invalid evidence reference" in result["failure_reason"]
    assert "result" not in result


def test_ollama_evaluator_records_truncation():
    client = SimpleNamespace(
        generate=lambda **kwargs: {
            "response": '{"criteria":',
            "done_reason": "length",
            "done": True,
            "eval_count": 2048,
            "prompt_eval_count": 100,
        },
    )
    evaluator = ExplanationEvaluator(config=EvaluatorConfig(max_tokens=2048))
    with patch(
        "graph_aware_classifier.explanation_evaluator.ollama",
        SimpleNamespace(Client=lambda host: client),
    ):
        result = evaluator.evaluate(
            explanation={"explanation": "Some explanation."},
            evidence=[{
                "rank": 1,
                "ngram": "CALL PUSH",
                "frequency": 0.5,
                "shap_value": 0.2,
                "functions": [],
            }],
            function_disassembly={},
        )

    assert result["status"] == "failed"
    assert result["response_metadata"] == {
        "finish_reason": "length",
        "truncated": True,
    }
    assert "reached its output limit" in result["failure_reason"]
