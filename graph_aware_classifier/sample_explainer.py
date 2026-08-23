"""Explain Random Forest predictions with readable n-gram feature names."""

from pathlib import Path
import json

import joblib
import numpy as np
import shap

from graph_ngrams.vocabulary import Vocabulary


CLASSIFIER_DIR = Path(__file__).resolve().parent / "models_and_vocabularies"


def get_ngram_functions(
    ngram: str | list[str] | tuple[str, ...],
    occurrences_path: str | Path,
) -> list[str]:
    """Return the unique functions where an n-gram occurs in a report."""

    requested_ngram = (
        tuple(ngram.split())
        if isinstance(ngram, str)
        else tuple(ngram)
    )

    with Path(occurrences_path).open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    for feature in report.get("features", []):
        if tuple(feature.get("ngram", [])) != requested_ngram:
            continue

        return sorted({
            occurrence["function"]
            for occurrence in feature.get("occurrences", [])
            if "function" in occurrence
        })

    return []


def _load_model_and_vocabulary(
    ngram_size: int,
):
    model_dir = CLASSIFIER_DIR / f"{ngram_size}gram"
    model = joblib.load(model_dir / f"{ngram_size}gram_rf.joblib")
    vocabulary = Vocabulary.load(model_dir / "vocabulary.json")
    return model, vocabulary


def explain_sample(
    ngram_size: int,
    transformed_sample,
) -> tuple[shap.Explanation, np.ndarray]:
    """Return the SHAP explanation and Random Forest prediction."""

    model, vocabulary = _load_model_and_vocabulary(ngram_size)
    sample_shape = transformed_sample.shape or (0, 0)
    feature_count = sample_shape[1]
    expected_features = getattr(model, "n_features_in_", feature_count)
    if expected_features != feature_count:
        raise ValueError(
            f"Model expects {expected_features} features, "
            f"but transformed sample provides {feature_count}"
        )
    if len(vocabulary) != feature_count:
        raise ValueError(
            f"Vocabulary contains {len(vocabulary)} features, "
            f"but transformed sample provides {feature_count}"
        )

    feature_names = [
        " ".join(vocabulary.get_entry(feature_id).ngram)
        for feature_id in range(feature_count)
    ]
    dense_sample = transformed_sample.toarray().astype(np.float32, copy=False)
    raw_explanation = shap.TreeExplainer(model)(dense_sample)

    explanation = shap.Explanation(
        values=raw_explanation.values,
        base_values=raw_explanation.base_values,
        data=dense_sample,
        feature_names=feature_names,
    )
    prediction = model.predict(transformed_sample)

    return explanation, prediction


def ranked_feature_contributions(
    explanation: shap.Explanation,
    class_index: int = 1,
    limit: int = 20,
) -> list[dict[str, float | int | str]]:
    """Return the strongest SHAP contributions using n-grams instead of IDs."""

    values = np.asarray(explanation.values)
    if values.ndim == 3:
        shap_values = values[0, :, class_index]
    elif values.ndim == 2:
        shap_values = values[0]
    else:
        shap_values = values

    data = np.asarray(explanation.data)
    sample_values = data[0] if data.ndim == 2 else data
    order = np.argsort(np.abs(shap_values))[::-1][:limit]

    return [
        {
            "feature_id": int(feature_id),
            "ngram": str(explanation.feature_names[feature_id]),
            "value": float(sample_values[feature_id]),
            "shap_value": float(shap_values[feature_id]),
        }
        for feature_id in order
    ]