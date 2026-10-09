"""Streamlit UI for the existing graph-aware cryptojacking detection pipeline.

Place this file in a demo/ directory inside your project. The existing
 graph_aware_classifier/ package can remain in the project root. Run from the
 project root with:
    streamlit run demo/streamlit_app.py

This app expects a preprocessed sample JSON, matching the current notebook. It does
not disassemble raw executables.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.tree import export_text, plot_tree


# ---------------------------------------------------------------------------
# Paths and page configuration
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
# Find the nearest ancestor that contains the existing project package. This
# supports both a root-level app and an app placed inside demo/ (or a nested
# subdirectory) without copying the model package into the demo folder.
PROJECT_ROOT = next(
    (
        candidate
        for candidate in (APP_DIR, *APP_DIR.parents)
        if (candidate / "graph_aware_classifier").is_dir()
    ),
    APP_DIR,
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PACKAGE_DIR = PROJECT_ROOT / "graph_aware_classifier"
OCCURRENCES_DIR = PACKAGE_DIR / "occurences"  # Keep the notebook's existing spelling.

NGRAM_SIZE = 2
SHAP_TOP_FEATURES = 20

st.set_page_config(
    page_title="Cryptojacking Detection",
    page_icon="",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Load the project's existing modules. The classifier itself is not redefined.
# ---------------------------------------------------------------------------
def load_pipeline_modules():
    """Import the same project functions used by the notebook."""
    transformer = importlib.import_module("graph_aware_classifier.sample_transformer")
    explainer = importlib.import_module("graph_aware_classifier.sample_explainer")
    llm_module = importlib.import_module("graph_aware_classifier.llm_analyzer")
    return transformer, explainer, llm_module


@st.cache_resource(show_spinner="Loading the trained Random Forest model...")
def load_model_and_vocabulary(ngram_size: int):
    """Cache the trained model and vocabulary between Streamlit reruns."""
    _, explainer, _ = load_pipeline_modules()
    return explainer._load_model_and_vocabulary(ngram_size)


def get_occurrences_for_features(
    top_features: list[dict[str, Any]],
    occurrences_path: Path,
    get_ngram_functions,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve SHAP-ranked n-grams to function names when the lookup exists."""
    occurrence_results: dict[str, Any] = {}
    lookup_errors: list[str] = []

    if not occurrences_path.exists():
        lookup_errors.append(
            f"Occurrence lookup file was not found: {occurrences_path}"
        )
        return occurrence_results, lookup_errors

    for item in top_features:
        ngram = str(item.get("ngram", ""))
        if not ngram:
            continue
        try:
            occurrence_results[ngram] = get_ngram_functions(
                ngram=ngram,
                occurrences_path=str(occurrences_path),
            )
        except Exception as exc:  # Keep other feature lookups visible if one fails.
            occurrence_results[ngram] = []
            lookup_errors.append(f"Could not look up '{ngram}': {exc}")

    return occurrence_results, lookup_errors


def run_detection(uploaded_name: str, uploaded_bytes: bytes) -> dict[str, Any]:
    """Run the notebook's stages and return everything needed by the UI."""
    transformer, explainer, llm_module = load_pipeline_modules()

    sample_stem = Path(uploaded_name).stem
    occurrences_path = OCCURRENCES_DIR / f"{sample_stem}_vocabulary.json"

    try:
        parsed_sample = json.loads(uploaded_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "The uploaded file is not valid UTF-8 JSON. Upload the preprocessed "
            "sample JSON used by the notebook, not a raw .exe file."
        ) from exc

    # Keep the upload temporary instead of permanently writing it into the project.
    with tempfile.TemporaryDirectory(prefix="cryptojacking_streamlit_") as temp_dir:
        sample_path = Path(temp_dir) / Path(uploaded_name).name
        sample_path.write_bytes(uploaded_bytes)

        with st.status("Stage 1/5 — Transforming the sample", expanded=True) as status:
            status.write("Validating the uploaded JSON sample.")
            status.write(f"Using n-gram size: {NGRAM_SIZE}.")
            transformed_sample = transformer.transform_sample(
                sample_path=str(sample_path),
                ngram_size=NGRAM_SIZE,
            )
            status.write("Sample transformation completed.")
            try:
                transformed_shape = tuple(transformed_sample.shape)
            except Exception:
                transformed_shape = None
            status.update(label="Stage 1/5 — Sample transformation complete", state="complete")

        with st.status("Stage 2/5 — Random Forest prediction and SHAP", expanded=True) as status:
            status.write("Running the same explain_sample() function used by the notebook.")
            explanation, prediction = explainer.explain_sample(
                ngram_size=NGRAM_SIZE,
                transformed_sample=transformed_sample,
            )
            top_features = explainer.ranked_feature_contributions(
                explanation,
                limit=SHAP_TOP_FEATURES,
            )
            predicted_class = int(prediction[0])
            class_name = {
                0: "Benign",
                1: "Cryptojacking",
            }.get(predicted_class, "Unknown")
            status.write(f"Random Forest prediction: {class_name} (class {predicted_class}).")
            status.write(f"Prepared {len(top_features)} ranked SHAP features.")
            status.update(label="Stage 2/5 — Prediction and SHAP complete", state="complete")

        with st.status("Stage 3/5 — Inspecting model output", expanded=True) as status:
            rf_model, vocabulary = load_model_and_vocabulary(NGRAM_SIZE)
            probabilities = rf_model.predict_proba(transformed_sample)[0]
            # Match the notebook's convention: class 1 is the cryptojacking class.
            try:
                malware_class_index = list(rf_model.classes_).index(1)
                rf_probability = float(probabilities[malware_class_index])
            except ValueError:
                rf_probability = float(probabilities[1])
            status.write(f"Cryptojacking-class probability: {rf_probability:.4f}.")
            status.write("The displayed decision tree is one tree within the Random Forest, not the entire ensemble.")
            status.update(label="Stage 3/5 — Model output ready", state="complete")

        with st.status("Stage 4/5 — Mapping influential n-grams to functions", expanded=True) as status:
            status.write(f"Looking for occurrence data at `{occurrences_path}`.")
            occurrence_results, occurrence_errors = get_occurrences_for_features(
                top_features=top_features,
                occurrences_path=occurrences_path,
                get_ngram_functions=explainer.get_ngram_functions,
            )
            if occurrence_errors:
                for message in occurrence_errors:
                    status.write(f"Warning: {message}")
            else:
                status.write(f"Resolved function occurrences for {len(occurrence_results)} n-grams.")
            status.update(label="Stage 4/5 — Function lookup finished", state="complete")

        llm_result: dict[str, Any] = {}
        llm_error: str | None = None
        if predicted_class == 1:
            with st.status("Stage 5/5 — Evidence-grounded LLM analysis", expanded=True) as status:
                status.write("Random Forest predicted class 1, so the notebook's LLM gate is satisfied.")
                status.write("Initializing LLMAnalyzer using the project's existing configuration.")
                try:
                    analyzer = llm_module.LLMAnalyzer()
                    llm_result = analyzer.analyze_sample(
                        binary_name=sample_stem,
                        rf_prediction=predicted_class,
                        rf_probability=rf_probability,
                        top_features=top_features,
                        occurrences_path=str(occurrences_path),
                    )
                    status.write("LLM analysis returned.")
                except Exception as exc:
                    llm_error = f"{type(exc).__name__}: {exc}"
                    status.write(f"LLM analysis could not complete: {llm_error}")
                    status.update(label="Stage 5/5 — LLM analysis failed", state="error")
                else:
                    status.update(label="Stage 5/5 — LLM analysis complete", state="complete")
        else:
            # Match the notebook: benign RF predictions do not call the LLM.
            with st.status("Stage 5/5 — LLM analysis skipped", expanded=True, state="complete") as status:
                status.write("Random Forest predicted benign, so no LLM call was made.")

        return {
            "uploaded_name": uploaded_name,
            "sample_stem": sample_stem,
            "transformed_sample": transformed_sample,
            "transformed_shape": transformed_shape,
            "explanation": explanation,
            "prediction": prediction,
            "predicted_class": predicted_class,
            "class_name": class_name,
            "rf_probability": rf_probability,
            "top_features": top_features,
            "rf_model": rf_model,
            "vocabulary": vocabulary,
            "occurrences_path": str(occurrences_path),
            "occurrence_results": occurrence_results,
            "occurrence_errors": occurrence_errors,
            "llm_result": llm_result,
            "llm_error": llm_error,
        }


def render_results(results: dict[str, Any]) -> None:
    """Render the intermediate and final results in expandable sections."""
    predicted_class = results["predicted_class"]
    probability = results["rf_probability"]
    top_features = results["top_features"]
    rf_model = results["rf_model"]
    vocabulary = results["vocabulary"]

    st.divider()
    st.header("Detection results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest prediction", results["class_name"])
    c2.metric("Cryptojacking probability", f"{probability:.1%}")
    c3.metric("SHAP features displayed", len(top_features))
    st.progress(min(max(probability, 0.0), 1.0), text="Random Forest probability for class 1 (cryptojacking)")

    with st.expander("Stage 1 — Sample transformation", expanded=False):
        st.write("The uploaded sample was passed to the existing `transform_sample()` function.")
        if results["transformed_shape"] is not None:
            st.write(f"Transformed representation shape: `{results['transformed_shape']}`")
        try:
            st.write(f"Non-zero feature count: `{int(results['transformed_sample'].nnz)}`")
        except Exception:
            st.caption("The transformer did not expose a sparse-matrix non-zero count.")

    with st.expander("Stage 2 — Random Forest decision-tree view", expanded=False):
        tree_index = 0
        feature_names = [
            " ".join(vocabulary.get_entry(feature_id).ngram)
            for feature_id in range(len(vocabulary))
        ]
        tree = rf_model.estimators_[tree_index]
        st.caption(
            "This is a depth-limited view of tree 0 from the forest. It is included to make "
            "some model splits inspectable; the prediction above comes from the full Random Forest."
        )
        rules = export_text(
            tree,
            feature_names=feature_names,
            max_depth=4,
            decimals=6,
            show_weights=True,
        )
        st.code(rules, language="text")

        fig, ax = plt.subplots(figsize=(24, 12))
        plot_tree(
            tree,
            feature_names=feature_names,
            class_names=[str(label) for label in rf_model.classes_],
            filled=True,
            rounded=True,
            max_depth=3,
            proportion=False,
            label="all",
            impurity=False,
            precision=6,
            fontsize=9,
            ax=ax,
        )
        ax.set_title("Random Forest — Decision Tree 0 (n-gram features)")
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with st.expander("Stage 3 — SHAP feature contributions", expanded=True):
        if top_features:
            contribution_df = pd.DataFrame(top_features)
            desired_columns = [column for column in ["ngram", "value", "shap_value"] if column in contribution_df]
            contribution_df = contribution_df[desired_columns].rename(
                columns={
                    "ngram": "N-Gram sequence",
                    "value": "Frequency",
                    "shap_value": "SHAP contribution",
                }
            )
            st.dataframe(contribution_df, hide_index=True, width="stretch")
            if "shap_value" in pd.DataFrame(top_features).columns:
                shap_df = pd.DataFrame(top_features).copy()
                shap_df["ngram"] = shap_df["ngram"].astype(str)
                shap_df = shap_df.sort_values("shap_value", key=lambda values: values.abs())
                fig, ax = plt.subplots(figsize=(10, max(4, min(9, 0.35 * len(shap_df)))))
                ax.barh(shap_df["ngram"], shap_df["shap_value"])
                ax.axvline(0, linewidth=0.8)
                ax.set_xlabel("SHAP contribution")
                ax.set_ylabel("N-gram")
                ax.set_title("Top SHAP-ranked n-grams")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
        else:
            st.info("No ranked SHAP features were returned for this sample.")

    with st.expander("Stage 4 — N-gram occurrences in disassembled functions", expanded=False):
        st.caption(f"Occurrence lookup: `{results['occurrences_path']}`")
        if results["occurrence_errors"]:
            for message in results["occurrence_errors"]:
                st.warning(message)
        if results["occurrence_results"]:
            for ngram, functions in results["occurrence_results"].items():
                function_list = functions if isinstance(functions, list) else [functions]
                st.markdown(f"**{ngram}** — {len(function_list)} occurrence result(s)")
                if function_list:
                    # Limit the initial text rendering for very common n-grams.
                    shown_functions = function_list[:100]
                    st.code("\n".join(str(name) for name in shown_functions), language="text")
                    if len(function_list) > len(shown_functions):
                        st.caption(f"Showing the first {len(shown_functions)} of {len(function_list)} function names.")
                else:
                    st.caption("No function names were returned for this n-gram.")
        else:
            st.info(
                "No occurrence data is available to display. Place the matching lookup file "
                "in `graph_aware_classifier/occurences/` using the notebook's naming convention."
            )

    with st.expander("Stage 5 — LLM analysis", expanded=True):
        if predicted_class != 1:
            st.info("LLM analysis was skipped because Random Forest predicted **Benign**. This preserves the notebook's conditional LLM behavior.")
        elif results["llm_error"]:
            st.error(f"The Random Forest flagged the sample, but the LLM stage failed: {results['llm_error']}")
            st.caption("Check the project's graph_aware_classifier/.env configuration, model/API access, and occurrence lookup file.")
        elif not results["llm_result"]:
            st.warning("The sample was flagged as cryptojacking, but no LLM result was returned.")
        else:
            llm_result = results["llm_result"]
            analysis = llm_result.get("analysis", {})
            tier = analysis.get("tier", "N/A")
            tier_display = {
                "high-confidence": "⚠️ HIGH-CONFIDENCE CRYPTOJACKING",
                "low-confidence": "❓ LOW-CONFIDENCE CRYPTOJACKING",
                "benign": "✅ BENIGN (LLM assessment)",
            }.get(tier, str(tier))
            try:
                confidence = float(analysis.get("confidence_score", 0.0))
                confidence_display = f"{confidence:.2f}"
            except (TypeError, ValueError):
                confidence_display = str(analysis.get("confidence_score", "N/A"))

            c1, c2, c3 = st.columns(3)
            c1.metric("LLM assessment tier", tier_display)
            c2.metric("LLM confidence score", confidence_display)
            c3.metric("Backend / model", f"{llm_result.get('llm_backend', 'N/A')} / {llm_result.get('llm_model', 'N/A')}")
            st.markdown("**Explanation**")
            st.write(analysis.get("explanation", "No explanation provided."))

            indicators = analysis.get("key_indicators", [])
            st.markdown("**Key indicators**")
            if indicators:
                for indicator in indicators:
                    st.markdown(f"- {indicator}")
            else:
                st.caption("No key indicators were provided.")

            st.markdown("**Recommended action**")
            st.write(analysis.get("recommended_action", "N/A"))

            evidence_analysis = analysis.get("evidence_analysis", [])
            if evidence_analysis:
                st.markdown("**Per-feature LLM assessments**")
                evidence_df = pd.DataFrame(evidence_analysis)
                display_cols = [
                    column
                    for column in ["ngram", "shap_contribution", "assessment"]
                    if column in evidence_df.columns
                ]
                if display_cols:
                    evidence_df = evidence_df[display_cols].rename(
                        columns={
                            "ngram": "N-gram",
                            "shap_contribution": "SHAP contribution",
                            "assessment": "LLM assessment",
                        }
                    )
                    st.dataframe(evidence_df, hide_index=True, width="stretch")
            else:
                st.caption("No per-feature LLM analysis was included in the response.")

            st.caption(
                "The LLM supplies an evidence-grounded explanation after the Random Forest flags a sample; "
                "it does not replace the Random Forest classifier."
            )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("Graph-Aware Cryptojacking Detection")
st.write(
    "Run the existing Random Forest → SHAP → LLM pipeline and inspect its intermediate outputs. "
    "The LLM is invoked only for samples classified as cryptojacking by the Random Forest."
)

with st.sidebar:
    st.header("Pipeline configuration")
    st.write(f"**N-gram size:** {NGRAM_SIZE}")
    st.write(f"**Top SHAP features:** {SHAP_TOP_FEATURES}")
    st.write(f"**Project root:** `{PROJECT_ROOT}`")
    st.write(f"**Occurrence directory:** `{OCCURRENCES_DIR}`")
    st.divider()
    st.caption("Keep the trained model, vocabulary, package modules, and LLM configuration in the existing project. This app does not train a new classifier.")

uploaded_file = st.file_uploader(
    "Upload a preprocessed sample JSON",
    type=["json"],
    help="Use the disassembled/graph-aware JSON sample expected by transform_sample(). A raw .exe is not accepted by this notebook pipeline.",
)

if uploaded_file is not None:
    st.caption(f"Selected sample: `{uploaded_file.name}` ({uploaded_file.size:,} bytes)")
    expected_occurrences = OCCURRENCES_DIR / f"{Path(uploaded_file.name).stem}_vocabulary.json"
    if expected_occurrences.exists():
        st.success(f"Matching occurrence lookup found: `{expected_occurrences.name}`")
    else:
        st.warning(
            f"Matching occurrence lookup not found: `{expected_occurrences}`. "
            "Classification and SHAP can still run, but function-level occurrence display and LLM analysis may be limited."
        )

run_clicked = st.button("Run detection pipeline", type="primary", disabled=uploaded_file is None)
if run_clicked and uploaded_file is not None:
    st.session_state.pop("detection_results", None)
    try:
        results = run_detection(uploaded_file.name, uploaded_file.getvalue())
        st.session_state["detection_results"] = results
    except Exception as exc:
        st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
        st.exception(exc)

results = st.session_state.get("detection_results")
if results:
    if uploaded_file is not None and results["uploaded_name"] != uploaded_file.name:
        st.info(
            f"The results below are for `{results['uploaded_name']}`. Click **Run detection pipeline** again to analyze `{uploaded_file.name}`."
        )
    render_results(results)
else:
    st.info(
        "Upload a preprocessed sample JSON and click **Run detection pipeline**. "
        "Each stage will report its progress, and the intermediate results will appear below."
    )
