"""Run a stored Random Forest model on a transformed sample."""

from pathlib import Path

import joblib
import numpy as np


CLASSIFIER_DIR = Path(__file__).resolve().parent / "models_and_vocabularies"


def predict_sample(
    ngram_size: int,
    transformed_sample,
) -> np.ndarray:
    """Predict a transformed sample with its matching n-gram model."""

    model_path = (
        CLASSIFIER_DIR
        / f"{ngram_size}gram"
        / f"{ngram_size}gram_rf.joblib"
    )
    model = joblib.load(model_path)

    sample_shape = transformed_sample.shape or (0, 0)
    feature_count = sample_shape[1]
    expected_features = getattr(model, "n_features_in_", feature_count)
    if expected_features != feature_count:
        raise ValueError(
            f"Model expects {expected_features} features, "
            f"but transformed sample provides {feature_count}"
        )

    return model.predict(transformed_sample)