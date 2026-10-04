import json
from types import SimpleNamespace
from unittest.mock import patch

from graph_aware_classifier.batch_evaluation import evaluate_batch_reports
from graph_aware_classifier.explanation_evaluator import (
    CRITERIA,
    EvaluatorConfig,
)


def test_batch_persists_failures_and_summarizes_only_scored_reports(tmp_path):
    evaluator_config = EvaluatorConfig()
    eligible = {
        "sample": {"binary_name": "eligible.exe"},
        "llm": {
            "status": "completed",
            "result": {
                "analysis": {"explanation": "A generated explanation."},
                "function_disassembly": {},
            },
        },
        "shap": {"top_features": []},
    }
    failed = {
        "sample": {"binary_name": "failed.exe"},
        "llm": {"status": "failed"},
        "shap": {"top_features": []},
    }
    skipped = {
        "sample": {"binary_name": "skipped.exe"},
        "llm": {"status": "skipped"},
        "shap": {"top_features": []},
    }
    reports = [eligible, failed, skipped]
    generator = SimpleNamespace(config=SimpleNamespace(
        backend="ollama",
        ollama_model="local-generator",
        temperature=0.2,
        max_tokens=8192,
    ))

    completed_evaluation = {
        "status": "completed",
        "result": {
            "criteria": [
                {"name": name, "score": 4}
                for name in CRITERIA
            ],
            "claim_summary": {
                "claim_count": 1,
                "unsupported_or_contradicted_count": 1,
                "unsupported_claim_rate": 1.0,
            },
        },
    }
    with patch(
        "graph_aware_classifier.batch_evaluation.ExplanationEvaluator",
    ) as evaluator_class:
        evaluator_class.return_value.evaluate.return_value = completed_evaluation
        summary = evaluate_batch_reports(
            batch_reports=reports,
            batch_output_dir=tmp_path,
            generator=generator,
            evaluator_config=evaluator_config,
        )
    assert reports[0]["evaluation"]["status"] == "completed"
    assert reports[1]["evaluation"]["status"] == "skipped"
    assert summary["counts"] == {
        "test_samples": 3,
        "rf_positive_generation_eligible": 2,
        "generation_completed": 1,
        "generation_failed": 1,
        "evaluation_completed": 1,
        "evaluation_failed": 0,
        "evaluation_skipped": 2,
    }
    assert summary["claim_support"]["unsupported_claim_rate"] == 1.0

    saved_report = json.loads(
        (tmp_path / "eligible.exe_analysis.json").read_text(encoding="utf-8")
    )
    assert saved_report["evaluation"]["status"] == "completed"
    assert json.loads(
        (tmp_path / "batch_evaluation_summary.json").read_text(encoding="utf-8")
    ) == summary
