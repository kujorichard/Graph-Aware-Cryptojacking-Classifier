"""Batch orchestration and summary reporting for the explanation G-Eval stage."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from graph_aware_classifier.explanation_evaluator import (
    CRITERIA,
    RUBRIC_VERSION,
    EvaluatorConfig,
    ExplanationEvaluator,
)
from graph_aware_classifier.llm_analyzer import ANALYSIS_PROMPT_VERSION


def evaluate_batch_reports(
    batch_reports: list[dict[str, Any]],
    batch_output_dir: str | Path,
    generator,
    evaluator_config: EvaluatorConfig | None = None,
) -> dict[str, Any]:
    """Evaluate successful generations, persist per-sample results and summarize."""
    output_dir = Path(batch_output_dir)
    evaluator_config = evaluator_config or EvaluatorConfig.from_env()
    evaluator = ExplanationEvaluator(config=evaluator_config)

    for report in batch_reports:
        llm_report = report["llm"]
        if llm_report["status"] != "completed":
            report["evaluation"] = {
                "status": "skipped",
                "reason": f"Generation status was {llm_report['status']!r}.",
            }
        else:
            llm_result = llm_report["result"]
            report["evaluation"] = evaluator.evaluate(
                explanation=llm_result["analysis"],
                evidence=report["shap"]["top_features"],
                function_disassembly=llm_result.get("function_disassembly", {}),
            )

        safe_name = re.sub(r"[^\w.-]", "_", report["sample"]["binary_name"])
        report_path = output_dir / f"{safe_name}_analysis.json"
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(
                report,
                report_file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

    eligible_reports = [
        report for report in batch_reports
        if report["llm"]["status"] != "skipped"
    ]
    generation_completed = [
        report for report in eligible_reports
        if report["llm"]["status"] == "completed"
    ]
    evaluation_completed = [
        report for report in batch_reports
        if report.get("evaluation", {}).get("status") == "completed"
    ]
    evaluation_failed = [
        report for report in batch_reports
        if report.get("evaluation", {}).get("status") == "failed"
    ]
    criterion_scores = {
        name: [
            criterion["score"]
            for report in evaluation_completed
            for criterion in report["evaluation"]["result"]["criteria"]
            if criterion["name"] == name
        ]
        for name in CRITERIA
    }
    claim_summaries = [
        report["evaluation"]["result"]["claim_summary"]
        for report in evaluation_completed
    ]
    total_claims = sum(item["claim_count"] for item in claim_summaries)
    total_unsupported = sum(
        item["unsupported_or_contradicted_count"] for item in claim_summaries
    )

    generator_backend = getattr(getattr(generator, "config", None), "backend", None)
    generator_config = getattr(generator, "config", None)
    generator_model = None
    generator_max_tokens = None
    if generator_config is not None:
        generator_model = (
            generator_config.claude_model
            if generator_backend == "claude"
            else generator_config.ollama_model
        )
        generator_max_tokens = generator_config.max_tokens

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_scope": "labeled test split; RF-positive samples only",
        "explanations_per_sample": 1,
        "generator": {
            "prompt_version": ANALYSIS_PROMPT_VERSION,
            "backend": generator_backend,
            "model": generator_model,
            "temperature": (
                generator_config.temperature if generator_config is not None else None
            ),
            "max_tokens": generator_max_tokens,
        },
        "evaluator": {
            "rubric_version": RUBRIC_VERSION,
            "backend": evaluator_config.backend,
            "model": evaluator_config.model,
            "temperature": evaluator_config.temperature,
            "max_tokens": evaluator_config.max_tokens,
        },
        "counts": {
            "test_samples": len(batch_reports),
            "rf_positive_generation_eligible": len(eligible_reports),
            "generation_completed": len(generation_completed),
            "generation_failed": len(eligible_reports) - len(generation_completed),
            "evaluation_completed": len(evaluation_completed),
            "evaluation_failed": len(evaluation_failed),
            "evaluation_skipped": (
                len(batch_reports)
                - len(evaluation_completed)
                - len(evaluation_failed)
            ),
        },
        "mean_criterion_scores_1_to_5": {
            name: mean(scores) if scores else None
            for name, scores in criterion_scores.items()
        },
        "claim_support": {
            "claim_count": total_claims,
            "unsupported_or_contradicted_count": total_unsupported,
            "unsupported_claim_rate": (
                total_unsupported / total_claims if total_claims else None
            ),
            "rate_definition": (
                "Unsupported and contradicted claims divided by all listed claims, "
                "including not_verifiable claims. This is an evaluator estimate, "
                "not ground-truth behavior verification."
            ),
        },
    }
    summary_path = output_dir / "batch_evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False, allow_nan=False)
    print(
        "G-Eval complete: "
        f"{len(evaluation_completed)} scored, {len(evaluation_failed)} failed."
    )
    print(f"Auditable batch summary saved to {summary_path}")
    return summary
