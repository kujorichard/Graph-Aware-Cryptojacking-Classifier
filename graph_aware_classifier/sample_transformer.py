"""Transform one graph JSON sample into a trained-model feature vector."""

from pathlib import Path
import json

import numpy as np
from scipy.sparse import csr_matrix

from graph_ngrams.extractors import (
    extract_binary_cfg_ngrams,
    extract_binary_dfg_ngrams,
)
from graph_ngrams.loader import GraphLoader
from graph_ngrams.vocabulary import Vocabulary

from graph_aware_classifier.extract_sample_graph_ngrams import (
    extract_sample_graph_ngrams,
)


NGRAM_SIZE = 2
SAMPLE_PATH = Path(r"D:\path\to\your\sample.exe.json")
CLASSIFIER_DIR = Path(__file__).resolve().parent / "models_and_vocabularies"
OCCURRENCE_OUTPUT_DIR = Path(__file__).resolve().parent / "occurences"


def _paths_for_ngram_size(n: int) -> tuple[Path, Path]:
    model_dir = CLASSIFIER_DIR / f"{n}gram"
    return (
        model_dir / "vocabulary.json",
        model_dir / f"{n}gram_rf.joblib",
    )


def _build_feature_vector(
    binary_sample: str | Path,
    vocabulary: Vocabulary,
    n: int,
) -> csr_matrix:
    binary = GraphLoader.load(binary_sample)
    shingles = (
        extract_binary_cfg_ngrams(binary, n)
        + extract_binary_dfg_ngrams(binary, n)
    )
    features = vocabulary.vectorize(shingles)

    return csr_matrix(
        (
            np.asarray(list(features.values()), dtype=np.float32),
            (np.zeros(len(features), dtype=np.int32), list(features.keys())),
        ),
        shape=(1, len(vocabulary)),
        dtype=np.float32,
    )


def transform_sample(
    sample_path: str | Path,
    ngram_size: int = NGRAM_SIZE,
) -> csr_matrix:
    """Write occurrence metadata and return one model-ready sparse vector."""

    vocabulary_path, _ = _paths_for_ngram_size(ngram_size)
    vocabulary = Vocabulary.load(vocabulary_path)

    occurrence_report = extract_sample_graph_ngrams(
        sample_path,
        n=ngram_size,
    )
    OCCURRENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    occurrence_path = (
        OCCURRENCE_OUTPUT_DIR
        / f"{occurrence_report['binary']}_vocabulary.json"
    )
    with occurrence_path.open("w", encoding="utf-8") as output_file:
        json.dump(occurrence_report, output_file, indent=2)

    return _build_feature_vector(sample_path, vocabulary, ngram_size)


def main() -> None:
    if not SAMPLE_PATH.is_file():
        raise FileNotFoundError(
            f"Set SAMPLE_PATH to an existing graph JSON file: {SAMPLE_PATH}"
        )

    vector = transform_sample(SAMPLE_PATH, NGRAM_SIZE)
    print(f"Feature vector shape: {vector.shape}")


if __name__ == "__main__":
    main()