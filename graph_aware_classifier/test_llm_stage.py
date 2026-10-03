"""Test script for Stage 5: Evidence-Grounded LLM Analysis.

Usage examples:
    # 1. Test single default sample:
    py test_llm_stage.py

    # 2. Test any specific sample file:
    py test_llm_stage.py --sample dataset/malware/00a16089397d26dec07ba75d5ac027fba40482e0af441942d8de1475aa133aab.exe.json

    # 3. Test an entire folder of samples in batch:
    py test_llm_stage.py --dir dataset/malware

    # 4. Test with mock LLM (instant, no GPU / API key needed):
    py test_llm_stage.py --sample dataset/malware/00a16089397d26dec07ba75d5ac027fba40482e0af441942d8de1475aa133aab.exe.json --mock

    # 5. Switch to Claude API:
    py test_llm_stage.py --sample dataset/malware/00a16089397d26dec07ba75d5ac027fba40482e0af441942d8de1475aa133aab.exe.json --backend claude
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graph_aware_classifier.sample_transformer import transform_sample
import graph_aware_classifier.sample_explainer as sample_explainer
from graph_aware_classifier.llm_analyzer import LLMConfig, LLMAnalyzer, build_evidence_payload

NGRAM_SIZE = 2
OCCURRENCES_DIR = project_root / "graph_aware_classifier" / "occurences"

MOCK_RESPONSE = """{
  "tier": "high-confidence",
  "explanation": "The analyzed binary demonstrates strong behavioral markers of host-based cryptojacking. The high prevalence of PUSH MOV and MOV SUB instructions within identified worker loops corresponds to the dense register staging and state permutations typical of unrolled cryptographic hash functions (such as CryptoNight or SHA-256). These instruction sequences dominate the SHAP attributions, driving the malware classification.",
  "evidence_analysis": [
    {
      "ngram": "PUSH MOV",
      "shap_contribution": 0.087,
      "assessment": "High-frequency function prologue and register setup in worker subroutines, characteristic of tightly unrolled hashing kernels."
    },
    {
      "ngram": "MOV SUB",
      "shap_contribution": 0.065,
      "assessment": "Iterative state modification consistent with loop-counter decrements and hashing round transformations."
    }
  ],
  "confidence_score": 0.95,
  "key_indicators": [
    "Prolific stack-frame staging across core subroutines",
    "Dominance of arithmetic and register-transfer n-grams in worker loops"
  ],
  "recommended_action": "Isolate system and terminate suspicious process threads executing tight hashing loops."
}"""


def process_single_sample(sample_path: Path, analyzer: LLMAnalyzer, mock: bool = False, force_malware: bool = False):
    """Run full RF -> SHAP -> LLM pipeline on a single sample."""
    sample_path = sample_path.resolve()
    print("-" * 65)
    print(f"[*] Processing: {sample_path.name}")

    if not sample_path.exists():
        print(f"[X] Error: File does not exist: {sample_path}")
        return None

    # 1. Transform sample
    transformed = transform_sample(sample_path, NGRAM_SIZE)

    # 2. RF Prediction & TreeSHAP
    explanation, prediction = sample_explainer.explain_sample(NGRAM_SIZE, transformed)
    rf_model, _ = sample_explainer._load_model_and_vocabulary(NGRAM_SIZE)
    rf_proba = float(rf_model.predict_proba(transformed)[0][1])
    raw_pred = int(prediction[0])

    print(f"    RF Prediction      : {raw_pred} ({'MALWARE' if raw_pred == 1 else 'BENIGN'})")
    print(f"    Malware Probability: {rf_proba:.4f}")

    rf_pred = 1 if force_malware else raw_pred
    if rf_pred != raw_pred:
        print("    [!] Overriding prediction to 1 (Malware) for Stage 5 LLM verification.")

    if rf_pred != 1:
        print("    [-] RF predicted BENIGN. Stage 5 LLM analysis skipped (as per thesis pipeline design).")
        return None

    # 3. Top SHAP features
    top_features = sample_explainer.ranked_feature_contributions(explanation, limit=20)
    print(f"    Extracted {len(top_features)} top SHAP features.")

    # 4. Occurrence lookup path
    # Look for matching vocabulary in occurences/
    occ_path = OCCURRENCES_DIR / f"{sample_path.stem}_vocabulary.json"
    if not occ_path.exists():
        # Fallback without double extension
        alt_name = sample_path.name.replace(".json", "")
        occ_path = OCCURRENCES_DIR / f"{alt_name}_vocabulary.json"

    # 5. Stage 5 LLM Analysis
    binary_label = sample_path.stem if sample_path.name.endswith(".json") else sample_path.name
    print(f"    [>] Invoking Stage 5 LLM Analysis on {analyzer.config.backend.upper()} ({analyzer.config.ollama_model if analyzer.config.backend == 'ollama' else analyzer.config.claude_model})...")

    if mock:
        with patch.object(analyzer, "_call_llm", return_value=MOCK_RESPONSE):
            result = analyzer.analyze_sample(
                binary_name=binary_label,
                rf_prediction=rf_pred,
                rf_probability=rf_proba if rf_pred == raw_pred else 0.985,
                top_features=top_features,
                occurrences_path=occ_path
            )
    else:
        try:
            result = analyzer.analyze_sample(
                binary_name=binary_label,
                rf_prediction=rf_pred,
                rf_probability=rf_proba if rf_pred == raw_pred else 0.985,
                top_features=top_features,
                occurrences_path=occ_path
            )
        except Exception as e:
            print(f"    [X] LLM call failed: {e}")
            return None

    # 6. Display Summary
    if result:
        analysis = result.get("analysis", {})
        tier = analysis.get("tier", "N/A").upper()
        conf = analysis.get("confidence_score", 0.0)
        print(f"\n    [+] STAGE 5 VERDICT   : {tier} (Confidence: {conf:.2f})")
        print(f"    [+] EXPLANATION       :\n        {analysis.get('explanation')}")
        indicators = analysis.get("key_indicators", [])
        if indicators:
            print("    [+] KEY INDICATORS    :")
            for ind in indicators:
                print(f"        - {ind}")
        print(f"    [+] RECOMMENDED ACTION: {analysis.get('recommended_action')}")
        out_file = analyzer.results_dir / f"{binary_label}_llm_result.json"
        print(f"    [OK] Saved structured record -> {out_file.name}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test Stage 5 LLM Cryptojacking Analysis")
    parser.add_argument("--sample", type=str, help="Path to a single sample .json file")
    parser.add_argument("--dir", type=str, help="Directory containing sample .json files to batch test")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM output (no server/API needed)")
    parser.add_argument("--backend", choices=["ollama", "claude"], help="Override backend in .env")
    parser.add_argument("--force-malware", action="store_true",
                        help="Force rf_prediction=1 even if RF predicted benign")
    args = parser.parse_args()

    print("=" * 65)
    print("STAGE 5 LLM EVIDENCE-GROUNDED ANALYZER")
    print("=" * 65)

    # Load configuration
    cfg = LLMConfig.from_env()
    if args.backend:
        cfg.backend = args.backend

    print(f"Backend configured : {cfg.backend.upper()}")
    if cfg.backend == "ollama":
        print(f"Ollama model       : {cfg.ollama_model}")
        print(f"Ollama host        : {cfg.ollama_host}")
    else:
        print(f"Claude model       : {cfg.claude_model}")
        key_status = "SET" if cfg.claude_api_key else "NOT SET (will fail unless --mock is used)"
        print(f"Anthropic API Key  : {key_status}")
    print()

    analyzer = LLMAnalyzer(config=cfg)

    # Case A: Batch directory test
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_absolute():
            # Check relative to cwd or project root
            if not dir_path.exists():
                dir_path = project_root / args.dir
        if not dir_path.is_dir():
            print(f"[X] Directory not found: {args.dir}")
            return

        json_files = sorted(list(dir_path.glob("*.json")))
        print(f"[*] Found {len(json_files)} sample files in {dir_path.name}/")
        print(f"[*] Beginning batch evaluation...\n")

        analyzed_count = 0
        for sample_file in json_files:
            res = process_single_sample(sample_file, analyzer, mock=args.mock, force_malware=args.force_malware)
            if res:
                analyzed_count += 1

        print("\n" + "=" * 65)
        print(f"BATCH RUN COMPLETE: {analyzed_count}/{len(json_files)} malware samples evaluated by LLM.")
        print(f"All records saved in: {analyzer.results_dir}")
        print("=" * 65)
        return

    # Case B: Single sample test
    if args.sample:
        sample_path = Path(args.sample)
        if not sample_path.is_absolute():
            if not sample_path.exists():
                sample_path = project_root / args.sample
    else:
        # Default fallback
        sample_path = project_root / "graph_aware_classifier" / "dataset" / "malware" / "00a16089397d26dec07ba75d5ac027fba40482e0af441942d8de1475aa133aab.exe.json"

    process_single_sample(sample_path, analyzer, mock=args.mock, force_malware=args.force_malware)
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
