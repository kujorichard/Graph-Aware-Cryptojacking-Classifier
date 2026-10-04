"""G-Eval-style judging of explanation support against recorded evidence."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph_aware_classifier.llm_analyzer import (
    _extract_json,
    _response_value,
    _token_usage_from_response,
)

logger = logging.getLogger(__name__)

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs) -> None:
        return None


RUBRIC_VERSION = "shap-faithfulness-v1"
CRITERIA = (
    "evidence_faithfulness",
    "shap_alignment",
    "code_region_grounding",
)
SUPPORT_LABELS = {
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "not_verifiable",
}
_EVALUATOR_SYSTEM_PROMPT = (
    "You are an independent evaluator of model-generated malware-analysis "
    "explanations. Judge claims only against the evidence explicitly supplied "
    "to you; do not perform a new malware classification or infer ground truth."
)


@dataclass
class EvaluatorConfig:
    """Configuration independent from explanation-generation settings."""

    backend: str = "ollama"
    ollama_model: str = "qwen3:14b-q4_K_M"
    ollama_host: str = "http://localhost:11434"
    claude_model: str = "claude-sonnet-4-20250514"
    claude_api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048

    @classmethod
    def from_env(cls) -> EvaluatorConfig:
        load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
        backend = os.getenv(
            "GEVAL_BACKEND",
            os.getenv("LLM_BACKEND", "ollama"),
        ).lower()
        return cls(
            backend=backend,
            ollama_model=os.getenv(
                "GEVAL_OLLAMA_MODEL",
                os.getenv("OLLAMA_MODEL", "qwen3:14b-q4_K_M"),
            ),
            ollama_host=os.getenv(
                "GEVAL_OLLAMA_HOST",
                os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            ),
            claude_model=os.getenv(
                "GEVAL_CLAUDE_MODEL",
                os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            ),
            claude_api_key=os.getenv(
                "GEVAL_ANTHROPIC_API_KEY",
                os.getenv("ANTHROPIC_API_KEY", ""),
            ),
            temperature=float(os.getenv("GEVAL_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("GEVAL_MAX_TOKENS", "2048")),
        )

    @property
    def model(self) -> str:
        return self.claude_model if self.backend == "claude" else self.ollama_model


def _evidence_catalog(
    evidence: list[dict[str, Any]],
    function_disassembly: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], set[str]]:
    catalog: list[dict[str, Any]] = []
    valid_ids: set[str] = set()
    for item in evidence:
        evidence_id = f"SHAP:{item['rank']}"
        valid_ids.add(evidence_id)
        catalog.append({
            "id": evidence_id,
            "type": "shap_ngram",
            "ngram": item["ngram"],
            "frequency": item["frequency"],
            "shap_value": item["shap_value"],
            "functions": item.get("functions", []),
        })

    for function, instructions in function_disassembly.items():
        for index, instruction in enumerate(instructions):
            evidence_id = f"DISASM:{function}:{index}"
            valid_ids.add(evidence_id)
            catalog.append({
                "id": evidence_id,
                "type": "disassembly_instruction",
                "function": function,
                "instruction": instruction,
            })
    return catalog, valid_ids


def _build_evaluation_prompt(
    explanation: dict[str, Any],
    evidence_catalog: list[dict[str, Any]],
) -> str:
    rubric = {
        "evidence_faithfulness": (
            "Are substantive claims supported by the supplied SHAP evidence "
            "and disassembly, without adding absent APIs, behaviors, or facts?"
        ),
        "shap_alignment": (
            "Does the explanation accurately refer to supplied n-grams, SHAP "
            "values, and their direction without inventing attributions?"
        ),
        "code_region_grounding": (
            "Are code-level interpretations grounded in the supplied function "
            "disassembly and cited regions?"
        ),
    }
    return (
        "You are an evaluator, not a malware classifier. Evaluate the supplied "
        "explanation only for faithfulness to the evidence catalog. Do not infer "
        "that cryptographic-looking instructions prove mining. The catalog does "
        "not provide ground truth for mining behavior. Do not assess claims as "
        "factually true beyond this evidence.\n\n"
        "Use this 1-5 rubric (1=unsupported, 2=mostly unsupported, "
        "3=partially supported, 4=mostly supported, 5=fully supported). "
        "Give a concise rationale for each score. Extract the explanation's "
        "substantive factual claims and classify each as supported, "
        "partially_supported, unsupported, contradicted, or not_verifiable. "
        "Every cited evidence ID must exactly match an ID in the catalog. "
        "An unsupported or contradicted claim counts toward the unsupported "
        "claim rate; report every claim, including not_verifiable claims.\n\n"
        "Return ONLY JSON matching this schema:\n"
        '{"criteria":[{"name":"evidence_faithfulness","score":1,'
        '"rationale":"...","evidence_ids":["SHAP:1"]},'
        '{"name":"shap_alignment","score":1,"rationale":"...",'
        '"evidence_ids":[]},{"name":"code_region_grounding","score":1,'
        '"rationale":"...","evidence_ids":[]}],'
        '"claims":[{"claim":"...","support":"supported",'
        '"evidence_ids":["SHAP:1"],"rationale":"..."}],'
        '"overall_rationale":"..."}\n\n'
        f"RUBRIC: {json.dumps(rubric, ensure_ascii=False)}\n"
        f"EXPLANATION: {json.dumps(explanation, ensure_ascii=False)}\n"
        f"EVIDENCE_CATALOG: {json.dumps(evidence_catalog, ensure_ascii=False)}"
    )


class ExplanationEvaluator:
    """Judge generated explanations against SHAP and disassembly evidence."""

    def __init__(
        self,
        config: EvaluatorConfig | None = None,
        generate_fn=None,
    ) -> None:
        self.config = config or EvaluatorConfig.from_env()
        self.generate_fn = generate_fn
        self.token_usage: dict[str, int | str | None] = {
            "input_tokens": None,
            "output_tokens": None,
            "source": "unavailable",
        }
        self.response_metadata: dict[str, Any] = {
            "finish_reason": None,
            "truncated": False,
        }

    def _call_ollama(self, prompt: str) -> str:
        if ollama is None:
            raise ImportError(
                "The 'ollama' package is required for Ollama G-Eval."
            )
        response = ollama.Client(host=self.config.ollama_host).generate(
            model=self.config.ollama_model,
            prompt=prompt,
            format="json",
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )
        self.token_usage = _token_usage_from_response(
            response,
            input_field="prompt_eval_count",
            output_field="eval_count",
            source="ollama",
        )
        finish_reason = _response_value(response, "done_reason")
        done = _response_value(response, "done")
        output_tokens = self.token_usage["output_tokens"]
        self.response_metadata = {
            "finish_reason": finish_reason,
            "truncated": (
                str(finish_reason).lower() in {"length", "max_tokens", "length_limit"}
                or done is False
                or (
                    output_tokens is not None
                    and output_tokens >= self.config.max_tokens
                )
            ),
        }
        return str(_response_value(response, "response") or "").strip()

    def _call_claude(self, prompt: str) -> str:
        if anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for Claude G-Eval."
            )
        api_key = self.config.claude_api_key
        if not api_key:
            raise ValueError("GEVAL_ANTHROPIC_API_KEY is required for Claude G-Eval.")
        response = anthropic.Anthropic(api_key=api_key).messages.create(
            model=self.config.claude_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=_EVALUATOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        self.token_usage = _token_usage_from_response(
            response.usage,
            input_field="input_tokens",
            output_field="output_tokens",
            source="claude",
        )
        finish_reason = getattr(response, "stop_reason", None)
        self.response_metadata = {
            "finish_reason": finish_reason,
            "truncated": finish_reason == "max_tokens",
        }
        return response.content[0].text

    def _call_model(self, prompt: str) -> str:
        if self.generate_fn is not None:
            return self.generate_fn(prompt)
        if self.config.backend == "ollama":
            return self._call_ollama(prompt)
        if self.config.backend == "claude":
            return self._call_claude(prompt)
        raise ValueError(f"Unsupported G-Eval backend: {self.config.backend!r}")

    def evaluate(
        self,
        explanation: dict[str, Any],
        evidence: list[dict[str, Any]],
        function_disassembly: dict[str, list[str]],
    ) -> dict[str, Any]:
        catalog, valid_ids = _evidence_catalog(evidence, function_disassembly)
        self.token_usage = {
            "input_tokens": None,
            "output_tokens": None,
            "source": "unavailable",
        }
        self.response_metadata = {"finish_reason": None, "truncated": False}
        raw_response = ""
        prompt = ""
        try:
            prompt = _build_evaluation_prompt(explanation, catalog)
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            raw_response = self._call_model(prompt)
            if self.response_metadata["truncated"]:
                raise ValueError(
                    "G-Eval response reached its output limit "
                    f"(finish_reason={self.response_metadata['finish_reason']!r})"
                )
            validated = self._validate(_extract_json(raw_response), valid_ids)
            claims = validated["claims"]
            unsupported_count = sum(
                claim["support"] in {"unsupported", "contradicted"}
                for claim in claims
            )
            validated["claim_summary"] = {
                "claim_count": len(claims),
                "unsupported_or_contradicted_count": unsupported_count,
                "unsupported_claim_rate": (
                    unsupported_count / len(claims) if claims else None
                ),
                "rate_definition": (
                    "Unsupported and contradicted claims divided by all listed "
                    "claims, including not_verifiable claims."
                ),
            }
            return {
                "status": "completed",
                "rubric_version": RUBRIC_VERSION,
                "prompt_sha256": prompt_sha256,
                "backend": self.config.backend,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "token_usage": dict(self.token_usage),
                "response_metadata": dict(self.response_metadata),
                "result": validated,
                "raw_response": raw_response,
            }
        except Exception as exc:
            logger.exception("G-Eval failed for explanation")
            return {
                "status": "failed",
                "rubric_version": RUBRIC_VERSION,
                "prompt_sha256": (
                    hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    if prompt
                    else None
                ),
                "backend": self.config.backend,
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "token_usage": dict(self.token_usage),
                "response_metadata": dict(self.response_metadata),
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "raw_response": raw_response,
            }

    @staticmethod
    def _validate(
        parsed: dict[str, Any],
        valid_ids: set[str],
    ) -> dict[str, Any]:
        if not isinstance(parsed, dict):
            raise ValueError("G-Eval response must be a JSON object")
        criteria = parsed.get("criteria")
        if not isinstance(criteria, list):
            raise ValueError("criteria must be a list")
        criteria_by_name = {}
        for item in criteria:
            if not isinstance(item, dict):
                raise ValueError("Each criterion must be an object")
            name = item.get("name")
            score = item.get("score")
            if not isinstance(name, str) or name not in CRITERIA or name in criteria_by_name:
                raise ValueError(f"Invalid or duplicate criterion: {name!r}")
            if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
                raise ValueError(f"Criterion {name!r} score must be an integer from 1 to 5")
            if not isinstance(item.get("rationale"), str):
                raise ValueError(f"Criterion {name!r} rationale must be a string")
            ExplanationEvaluator._validate_refs(item.get("evidence_ids"), valid_ids)
            criteria_by_name[name] = item
        if set(criteria_by_name) != set(CRITERIA):
            raise ValueError(f"criteria must include exactly {list(CRITERIA)}")

        claims = parsed.get("claims")
        if not isinstance(claims, list):
            raise ValueError("claims must be a list")
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict) or not isinstance(claim.get("claim"), str):
                raise ValueError(f"claims[{index}] must contain a claim string")
            support = claim.get("support")
            if not isinstance(support, str) or support not in SUPPORT_LABELS:
                raise ValueError(f"claims[{index}] has an invalid support label")
            if not isinstance(claim.get("rationale"), str):
                raise ValueError(f"claims[{index}] rationale must be a string")
            ExplanationEvaluator._validate_refs(claim.get("evidence_ids"), valid_ids)
        if not isinstance(parsed.get("overall_rationale"), str):
            raise ValueError("overall_rationale must be a string")
        return parsed

    @staticmethod
    def _validate_refs(references: Any, valid_ids: set[str]) -> None:
        if not isinstance(references, list) or not all(
            isinstance(reference, str) and reference in valid_ids
            for reference in references
        ):
            raise ValueError("evidence_ids contains an invalid evidence reference")
