"""Stage 5 — LLM-based evidence-grounded code analysis and explanation.

After the Random Forest classifier flags a binary as malware, this module:
  1. Builds a structured evidence payload from the top-k SHAP features and
     the per-sample occurrence metadata.
  2. Constructs a prompt asking the LLM to perform a three-tier classification
     (high-confidence cryptojacking, low-confidence cryptojacking, or benign)
     with an interpretable, evidence-grounded explanation.
  3. Sends the prompt to either a local Ollama model or the Claude API.
  4. Parses the structured JSON response, with fallback for malformed output.
  5. Saves results to ``graph_aware_classifier/llm_results/``.

Configuration is loaded from a ``.env`` file located alongside this module.
See ``.env.example`` for the template.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Optional third-party imports
# ---------------------------------------------------------------------------
try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

try:
    import ollama as _ollama_pkg
except ImportError:  # pragma: no cover
    _ollama_pkg = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_a: Any, **_kw: Any) -> None:  # type: ignore[misc]
        pass

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_RESULTS_DIR = _MODULE_DIR / "llm_results"
_DEFAULT_ENV_PATH = _MODULE_DIR / ".env"
ANALYSIS_PROMPT_VERSION = "shap-grounded-analysis-v1"

__all__ = [
    "LLMConfig",
    "LLMAnalyzer",
    "build_evidence_payload",
    "build_analysis_prompt",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _reported_token_count(value: Any) -> int | None:
    """Return a non-negative integer token count when the backend reports one."""
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _token_usage_from_response(
    response: Any,
    input_field: str,
    output_field: str,
    source: str,
) -> dict[str, int | str | None]:
    """Normalize provider token usage from dict or SDK response objects."""
    if isinstance(response, dict):
        input_count = response.get(input_field)
        output_count = response.get(output_field)
    else:
        input_count = getattr(response, input_field, None)
        output_count = getattr(response, output_field, None)

    return {
        "input_tokens": _reported_token_count(input_count),
        "output_tokens": _reported_token_count(output_count),
        "source": source,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    """Configuration for the LLM analysis backend.

    Use :meth:`from_env` to load from a ``.env`` file and/or environment
    variables, or construct manually for programmatic use.
    """

    backend: str = "ollama"
    """``"ollama"`` for local inference, ``"claude"`` for Anthropic API."""

    ollama_model: str = "qwen3:14b-q4_K_M"
    """Ollama model tag (e.g. ``qwen3:14b-q4_K_M``)."""

    ollama_host: str = "http://localhost:11434"
    """Ollama server URL."""

    claude_model: str = "claude-sonnet-4-20250514"
    """Anthropic model identifier."""

    claude_api_key: str = ""
    """Anthropic API key (loaded from env)."""

    temperature: float = 0.2
    """Sampling temperature — low for consistent, structured output."""

    max_tokens: int = 8192
    """Maximum tokens in the LLM response."""

    max_retries: int = 3
    """Number of retry attempts on transient failures."""

    retry_delay: float = 1.0
    """Initial delay in seconds between retries (doubles each time)."""

    timeout: float = 120.0
    """Request timeout in seconds."""

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> LLMConfig:
        """Load configuration from a ``.env`` file and environment variables.

        Parameters
        ----------
        env_path:
            Path to the ``.env`` file.  Defaults to the ``.env`` file next
            to this module.
        """
        env_file = Path(env_path) if env_path else _DEFAULT_ENV_PATH
        load_dotenv(dotenv_path=env_file)

        return cls(
            backend=os.getenv("LLM_BACKEND", "ollama").lower(),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:14b-q4_K_M"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            claude_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "8192")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("LLM_RETRY_DELAY", "1.0")),
            timeout=float(os.getenv("LLM_TIMEOUT", "120.0")),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Evidence payload construction
# ═══════════════════════════════════════════════════════════════════════════

def build_evidence_payload(
    top_features: list[dict[str, Any]],
    occurrences_path: str | Path,
) -> list[dict[str, Any]]:
    """Build the evidence payload from SHAP features and occurrence data.

    For each feature in *top_features* (output of
    ``ranked_feature_contributions``), look up the originating functions and
    memory addresses in the occurrence file written by
    ``sample_transformer.transform_sample``.

    Parameters
    ----------
    top_features:
        List of dicts with keys ``feature_id``, ``ngram`` (space-separated
        string), ``value`` (frequency), ``shap_value``.
    occurrences_path:
        Path to the ``<binary>_vocabulary.json`` file in ``occurences/``.

    Returns
    -------
    list[dict]
        One dict per feature with the original SHAP data plus a
        ``functions`` list of ``{function, graph_type, addresses}`` records.
    """
    occurrences_path = Path(occurrences_path)

    # ------------------------------------------------------------------
    # Load occurrence data and build a lookup keyed by ngram tuple
    # ------------------------------------------------------------------
    occurrence_by_ngram: dict[tuple[str, ...], list[dict[str, Any]]] = {}

    if occurrences_path.is_file():
        with occurrences_path.open("r", encoding="utf-8") as fh:
            occurrence_data = json.load(fh)

        for feat in occurrence_data.get("features", []):
            ngram_tuple = tuple(feat.get("ngram", []))
            occurrence_by_ngram[ngram_tuple] = feat.get("occurrences", [])
    else:
        logger.warning("Occurrence file not found: %s", occurrences_path)

    # ------------------------------------------------------------------
    # Merge SHAP features with occurrence data
    # ------------------------------------------------------------------
    evidence: list[dict[str, Any]] = []

    for rank, feature in enumerate(top_features, start=1):
        ngram_str: str = feature.get("ngram", "")
        ngram_tuple = tuple(ngram_str.split())

        raw_occurrences = occurrence_by_ngram.get(ngram_tuple, [])

        # Deduplicate functions while preserving address detail
        seen_functions: dict[str, list[list[str]]] = {}
        for occ in raw_occurrences:
            func_name = occ.get("function", "UNKNOWN")
            addresses = occ.get("addresses", [])
            seen_functions.setdefault(func_name, []).append(addresses)

        functions = [
            {
                "function": func_name,
                "address_groups": addr_groups,
            }
            for func_name, addr_groups in seen_functions.items()
        ]

        evidence.append({
            "rank": rank,
            "ngram": ngram_str,
            "frequency": feature.get("value", 0.0),
            "shap_value": feature.get("shap_value", 0.0),
            "functions": functions,
        })

    return evidence


# ═══════════════════════════════════════════════════════════════════════════
# Prompt construction
# ═══════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are an expert malware analyst specializing in cryptojacking detection. "
    "You analyze assembly-level evidence from binary executables and produce "
    "structured JSON assessments. Output ONLY valid JSON — no markdown fences, "
    "no commentary outside the JSON object."
)

_JSON_SCHEMA_DESCRIPTION = """\
{
  "tier": "<one of: high-confidence, low-confidence, benign>",
  "explanation": "<detailed paragraph explaining your reasoning, referencing specific n-grams and functions from the evidence>",
  "evidence_analysis": [
    {
      "ngram": "<n-gram sequence>",
      "shap_contribution": <float>,
      "assessment": "<your interpretation of what this pattern indicates in the context of cryptojacking>"
    }
  ],
  "confidence_score": <float between 0.0 and 1.0>,
  "key_indicators": ["<indicator 1>", "<indicator 2>"],
  "recommended_action": "<action recommendation for a malware analyst>"
}"""


def build_analysis_prompt(
    binary_name: str,
    rf_prediction: int,
    rf_probability: float,
    evidence: list[dict[str, Any]],
    function_disassembly: dict[str, list[str]] | None = None,
) -> str:
    """Construct the LLM analysis prompt with the evidence payload.

    Parameters
    ----------
    binary_name:
        Name of the binary under analysis.
    rf_prediction:
        Class predicted by the Random Forest (1 = malware).
    rf_probability:
        ``predict_proba`` score for the predicted class.
    evidence:
        Output of :func:`build_evidence_payload`.
    function_disassembly:
        Complete disassembly lines keyed by functions referenced in *evidence*.

    Returns
    -------
    str
        The fully-formatted prompt string.
    """
    # ── Header ────────────────────────────────────────────────────────
    lines: list[str] = [
        f"Binary Name: {binary_name}",
        f"Random Forest Prediction: MALWARE (class {rf_prediction}) "
        f"with probability {rf_probability:.4f}.",
        "",
        f"The following are the Top-{len(evidence)} most influential "
        "features (assembly mnemonic n-gram sequences) that drove the "
        "Random Forest prediction, ranked by absolute SHAP contribution.  "
        "For each n-gram, the relative frequency in the binary, the SHAP "
        "contribution score, and every function where the n-gram was "
        "observed are listed.",
        "",
        "=" * 60,
        "SHAP EVIDENCE",
        "=" * 60,
    ]

    # ── Per-feature evidence ──────────────────────────────────────────
    for item in evidence:
        rank = item["rank"]
        ngram = item["ngram"]
        freq = item["frequency"]
        shap_val = item["shap_value"]
        direction = "toward cryptojacking" if shap_val > 0 else "toward benign"

        lines.append("")
        lines.append(f"Rank {rank}: {ngram}")
        lines.append(f"  Frequency : {freq:.6f}")
        lines.append(f"  SHAP      : {shap_val:+.6f} ({direction})")

        funcs = item.get("functions", [])
        if funcs:
            func_strs: list[str] = []
            # Limit functions shown in prompt to keep size manageable;
            # the full list is preserved in the saved evidence JSON.
            max_funcs_in_prompt = 10
            for f in funcs[:max_funcs_in_prompt]:
                addr_flat = [
                    a for group in f["address_groups"] for a in group
                ]
                unique_addrs = sorted(set(addr_flat))
                addr_display = ", ".join(unique_addrs[:6])
                if len(unique_addrs) > 6:
                    addr_display += f" … (+{len(unique_addrs) - 6} more)"
                func_strs.append(f"{f['function']} (at {addr_display})")
            if len(funcs) > max_funcs_in_prompt:
                func_strs.append(
                    f"… (+{len(funcs) - max_funcs_in_prompt} more functions)"
                )
            lines.append("  Functions : " + "; ".join(func_strs))
        else:
            lines.append("  Functions : (no function context available)")

    # ── Instructions ──────────────────────────────────────────────────
    if function_disassembly:
        lines.extend([
            "",
            "=" * 60,
            "SPOTTED FUNCTION DISASSEMBLY",
            "=" * 60,
        ])
        for function_name, instructions in function_disassembly.items():
            lines.append("")
            lines.append(f"Function: {function_name}")
            lines.extend(instructions)

    lines.append("")
    lines.append("=" * 60)
    lines.append("ANALYSIS INSTRUCTIONS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        "Based on the SHAP evidence above, classify this binary into "
        "one of three tiers:"
    )
    lines.append("  • high-confidence  — strong evidence of cryptojacking")
    lines.append("  • low-confidence   — some indicators present but "
                 "inconclusive")
    lines.append("  • benign           — evidence does not support "
                 "cryptojacking")
    lines.append("")
    lines.append(
        "IMPORTANT: Your analysis must be grounded ONLY in the evidence "
        "provided above. Do not reference functions, API calls, or "
        "behaviors not present in the evidence. Use the supplied function "
        "disassembly to explain what the spotted code does. Every claim must "
        "be traceable to specific n-grams, functions, or instructions listed "
        "above. In evidence_analysis, use only exact n-gram strings listed "
        "under SHAP EVIDENCE. Do not estimate or invent SHAP values; the "
        "pipeline will attach the authoritative contribution."
    )
    lines.append("")
    lines.append("Respond with a single JSON object matching this schema:")
    lines.append(_JSON_SCHEMA_DESCRIPTION)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Response parsing
# ═══════════════════════════════════════════════════════════════════════════

_REQUIRED_KEYS = [
    "tier",
    "explanation",
    "evidence_analysis",
    "confidence_score",
    "key_indicators",
    "recommended_action",
]


def _extract_json(raw: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from LLM output.

    Handles:
      - Direct JSON
      - Qwen/reasoning models placing JSON inside <think>...</think>
      - JSON following <think>...</think> blocks
      - Markdown code blocks (```json ... ```)
      - Embedded JSON objects within explanatory text
    """
    if not raw or not raw.strip():
        raise json.JSONDecodeError("Empty LLM response received", raw, 0)

    raw_clean = raw.strip()

    # 1. Try direct parse of raw response
    try:
        return json.loads(raw_clean)
    except json.JSONDecodeError:
        pass

    # 2. Try parsing after removing <think>...</think> tags
    cleaned = re.sub(r"<think>.*?</think>", "", raw_clean, flags=re.DOTALL).strip()
    if cleaned:
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

    # 3. If cleaned didn't yield valid JSON, look for JSON inside raw
    # (Reasoning models often place the complete valid JSON inside their thinking output)
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw_clean, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", raw_clean, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found in LLM response", raw, 0)


def _validate_analysis_response(
    parsed: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate LLM structure and replace model-supplied SHAP values."""
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")

    required_keys = {
        "tier",
        "explanation",
        "evidence_analysis",
        "confidence_score",
        "key_indicators",
        "recommended_action",
    }
    missing_keys = required_keys - parsed.keys()
    if missing_keys:
        raise ValueError(f"Missing required response keys: {sorted(missing_keys)}")

    if (
        not isinstance(parsed["tier"], str)
        or parsed["tier"] not in {"high-confidence", "low-confidence", "benign"}
    ):
        raise ValueError(f"Invalid tier value: {parsed['tier']!r}")
    if not isinstance(parsed["explanation"], str) or not parsed["explanation"].strip():
        raise ValueError("explanation must be a non-empty string")

    confidence = parsed["confidence_score"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("confidence_score must be a number between 0 and 1")
    if not isinstance(parsed["evidence_analysis"], list):
        raise ValueError("evidence_analysis must be a list")
    if not isinstance(parsed["key_indicators"], list) or not all(
        isinstance(indicator, str) for indicator in parsed["key_indicators"]
    ):
        raise ValueError("key_indicators must be a list of strings")
    if not isinstance(parsed["recommended_action"], str):
        raise ValueError("recommended_action must be a string")

    evidence_by_ngram = {item["ngram"]: item for item in evidence}
    validated_items = []
    for index, item in enumerate(parsed["evidence_analysis"]):
        if not isinstance(item, dict):
            raise ValueError(f"evidence_analysis[{index}] must be an object")
        ngram = item.get("ngram")
        if not isinstance(ngram, str) or ngram not in evidence_by_ngram:
            raise ValueError(
                f"evidence_analysis[{index}] references an unknown SHAP n-gram: "
                f"{ngram!r}"
            )
        if not isinstance(item.get("assessment"), str):
            raise ValueError(f"evidence_analysis[{index}].assessment must be a string")
        validated_items.append({
            **item,
            "shap_contribution": evidence_by_ngram[ngram]["shap_value"],
        })

    parsed["evidence_analysis"] = validated_items
    return parsed


def _response_value(response: Any, name: str) -> Any:
    if isinstance(response, dict):
        return response.get(name)
    return getattr(response, name, None)


# ═══════════════════════════════════════════════════════════════════════════
# Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class LLMAnalyzer:
    """Analyse RF-flagged samples using a configurable LLM backend.

    Usage
    -----
    >>> analyzer = LLMAnalyzer()                     # loads .env
    >>> result = analyzer.analyze_sample(...)         # single sample
    >>> results = analyzer.batch_analyze(samples)     # batch
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        results_dir: str | Path | None = None,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self.results_dir = Path(results_dir) if results_dir else _DEFAULT_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._last_token_usage: dict[str, int | str | None] = {
            "input_tokens": None,
            "output_tokens": None,
            "source": "unavailable",
        }
        self._last_response_metadata: dict[str, Any] = {
            "finish_reason": None,
            "truncated": False,
        }

        if self.config.backend == "claude" and not self.config.claude_api_key:
            logger.warning(
                "Claude backend selected but ANTHROPIC_API_KEY is not set."
            )

    # ------------------------------------------------------------------
    # Backend calls
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> str:
        """Send *prompt* to the local Ollama server."""
        if _ollama_pkg is None:
            raise ImportError(
                "The 'ollama' package is required for the Ollama backend.  "
                "Install it with:  pip install ollama"
            )

        client = _ollama_pkg.Client(host=self.config.ollama_host)
        response = client.generate(
            model=self.config.ollama_model,
            prompt=prompt,
            format="json",
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )
        self._last_token_usage = _token_usage_from_response(
            response,
            input_field="prompt_eval_count",
            output_field="eval_count",
            source="ollama",
        )
        finish_reason = _response_value(response, "done_reason")
        done = _response_value(response, "done")
        output_tokens = self._last_token_usage["output_tokens"]
        self._last_response_metadata = {
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

        # Extract text from response (supports both Pydantic GenerateResponse and dict)
        text = ""
        if hasattr(response, "response") and response.response:
            text = response.response.strip()
        elif isinstance(response, dict) and response.get("response"):
            text = str(response["response"]).strip()

        return text

    def _call_claude(self, prompt: str) -> str:
        """Send *prompt* to the Anthropic Messages API."""
        if anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for the Claude backend.  "
                "Install it with:  pip install anthropic"
            )

        client = anthropic.Anthropic(api_key=self.config.claude_api_key)
        response = client.messages.create(
            model=self.config.claude_model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        self._last_token_usage = _token_usage_from_response(
            response.usage,
            input_field="input_tokens",
            output_field="output_tokens",
            source="claude",
        )
        finish_reason = getattr(response, "stop_reason", None)
        self._last_response_metadata = {
            "finish_reason": finish_reason,
            "truncated": finish_reason == "max_tokens",
        }
        return response.content[0].text

    def _call_llm(self, prompt: str) -> str:
        """Route to the configured backend with exponential-backoff retry."""
        backend = self.config.backend
        delay = self.config.retry_delay

        for attempt in range(1, self.config.max_retries + 1):
            try:
                if backend == "claude":
                    return self._call_claude(prompt)
                if backend == "ollama":
                    return self._call_ollama(prompt)
                raise ValueError(f"Unknown LLM backend: '{backend}'")
            except Exception as exc:
                logger.error(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    self.config.max_retries,
                    exc,
                )
                if attempt == self.config.max_retries:
                    raise
                logger.info("Retrying in %.1f s …", delay)
                time.sleep(delay)
                delay *= 2  # exponential backoff

        # Unreachable in practice, but satisfies the type checker.
        return ""  # pragma: no cover

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_sample(
        self,
        binary_name: str,
        rf_prediction: int,
        rf_probability: float,
        top_features: list[dict[str, Any]],
        occurrences_path: str | Path,
        function_disassembly: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Run the full LLM analysis pipeline on a single sample.

        Parameters
        ----------
        binary_name:
            Filename of the binary (e.g. ``"example1.exe"``).
        rf_prediction:
            Class label predicted by the RF (``1`` = malware).
        rf_probability:
            ``predict_proba`` score for the malware class.
        top_features:
            Output of ``ranked_feature_contributions()`` from
            :mod:`sample_explainer`.
        occurrences_path:
            Path to the ``<binary>_vocabulary.json`` written by
            ``sample_transformer.transform_sample()``.
        function_disassembly:
            Full address/mnemonic/operand lines for functions associated with
            the top SHAP n-grams.

        Returns
        -------
        dict
            Full result including metadata, evidence, and the LLM analysis.
            Returns an empty dict if *rf_prediction* is not ``1``.
        """
        if rf_prediction != 1:
            logger.info(
                "Skipping LLM analysis for %s — RF predicted class %d (not malware).",
                binary_name,
                rf_prediction,
            )
            return {}

        logger.info("Starting LLM analysis for %s …", binary_name)

        # 1. Build evidence
        evidence = build_evidence_payload(
            top_features,
            occurrences_path,
        )

        # 2. Build prompt
        prompt = build_analysis_prompt(
            binary_name, rf_prediction, rf_probability, evidence,
            function_disassembly=function_disassembly,
        )
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        # 3. Call LLM
        self._last_token_usage = {
            "input_tokens": None,
            "output_tokens": None,
            "source": "unavailable",
        }
        self._last_response_metadata = {
            "finish_reason": None,
            "truncated": False,
        }
        try:
            raw_response = self._call_llm(prompt)
            call_error = None
        except Exception as exc:
            logger.exception("LLM generation failed for %s", binary_name)
            raw_response = ""
            call_error = f"{type(exc).__name__}: {exc}"
        token_usage = dict(self._last_token_usage)
        response_metadata = dict(self._last_response_metadata)

        # 4. Parse response
        try:
            if call_error:
                raise ValueError(f"LLM generation call failed: {call_error}")
            if response_metadata["truncated"]:
                raise ValueError(
                    "LLM response reached its output limit "
                    f"(finish_reason={response_metadata['finish_reason']!r})"
                )
            parsed = _validate_analysis_response(
                _extract_json(raw_response),
                evidence,
            )
            analysis_status = "completed"
            failure_reason = None
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error("LLM analysis response is unusable: %s", exc)
            parsed = {
                "tier": None,
                "explanation": "LLM response is unusable and requires manual review.",
                "evidence_analysis": [],
                "confidence_score": None,
                "key_indicators": [],
                "recommended_action": "Manual review required",
                "parse_error": str(exc),
            }
            analysis_status = "failed"
            failure_reason = str(exc)

        # 5. Assemble final result
        result: dict[str, Any] = {
            "status": analysis_status,
            "failure_reason": failure_reason,
            "binary_name": binary_name,
            "rf_prediction": rf_prediction,
            "rf_probability": rf_probability,
            "llm_backend": self.config.backend,
            "llm_model": (
                self.config.claude_model
                if self.config.backend == "claude"
                else self.config.ollama_model
            ),
            "prompt_version": ANALYSIS_PROMPT_VERSION,
            "prompt_sha256": prompt_sha256,
            "generation_config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            "evidence": evidence,
            "function_disassembly": function_disassembly or {},
            "token_usage": token_usage,
            "response_metadata": response_metadata,
            "analysis": parsed,
            "raw_response": raw_response,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        # 6. Persist
        safe_name = re.sub(r"[^\w\-.]", "_", binary_name)
        output_path = self.results_dir / f"{safe_name}_llm_result.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)

        logger.info("LLM result saved → %s", output_path)
        return result

    def batch_analyze(
        self,
        samples: list[dict[str, Any]],
        output_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Run LLM analysis on multiple samples sequentially.

        Parameters
        ----------
        samples:
            Each dict must contain keys: ``binary_name``, ``rf_prediction``,
            ``rf_probability``, ``top_features``, ``occurrences_path``.
        output_dir:
            Override the results directory for this batch.  If *None*, uses
            the instance default.

        Returns
        -------
        list[dict]
            Results for every sample where RF predicted malware.
        """
        # Use a local results dir for this batch without mutating self
        original_results_dir = self.results_dir
        if output_dir is not None:
            self.results_dir = Path(output_dir)
            self.results_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        total = len(samples)
        malware_count = sum(
            1 for s in samples if s.get("rf_prediction") == 1
        )
        logger.info(
            "Batch analysis: %d samples total, %d flagged as malware.",
            total,
            malware_count,
        )

        processed = 0
        for idx, sample in enumerate(samples, start=1):
            binary_name = sample.get("binary_name", f"unknown_{idx}")

            if sample.get("rf_prediction") != 1:
                logger.info(
                    "[%d/%d] %s — benign, skipping.", idx, total, binary_name
                )
                continue

            processed += 1
            logger.info(
                "[%d/%d] Analysing %s (%d/%d malware) …",
                idx,
                total,
                binary_name,
                processed,
                malware_count,
            )

            try:
                result = self.analyze_sample(
                    binary_name=binary_name,
                    rf_prediction=sample["rf_prediction"],
                    rf_probability=sample.get("rf_probability", 0.0),
                    top_features=sample.get("top_features", []),
                    occurrences_path=sample.get("occurrences_path", ""),
                )
                if result:
                    results.append(result)
            except Exception as exc:
                logger.error("Failed to analyse %s: %s", binary_name, exc)

        # Restore original results dir
        self.results_dir = original_results_dir

        logger.info(
            "Batch complete: %d/%d malware samples analysed successfully.",
            len(results),
            malware_count,
        )
        return results
