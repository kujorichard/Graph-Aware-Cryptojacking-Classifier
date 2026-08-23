from dataclasses import dataclass
from importlib.resources import path
from .models import Sample
from .vocabulary import Vocabulary
from .extractors import (
    extract_binary_cfg_ngrams,
    extract_binary_dfg_ngrams,
)
from pathlib import Path

from .loader import GraphLoader

class DatasetBuilder:

    def __init__(
        self,
        vocabulary: Vocabulary,
        n: int = 3,
    ):

        self.vocabulary = vocabulary
        self.n = n

    def build_sample(
        self,
        path: Path,
    ) -> Sample:

        binary = GraphLoader.load(
            str(path)
        )

        dfg = extract_binary_dfg_ngrams(
            binary,
            self.n,
        )

        cfg = extract_binary_cfg_ngrams(
            binary,
            self.n,
        )

        shingles = dfg + cfg

        features = self.vocabulary.vectorize(
            shingles,
        )

        return Sample(
            binary_name=binary.binary_name,
            features=features,
            label=binary.label,
        )

    def build_dataset(
        self,
        paths: list[Path],
    ) -> list[Sample]:

        dataset = []

        for path in paths:

            dataset.append(
                self.build_sample(path)
            )

        return dataset

    def build_feature_matrix(
        self,
        paths: list[Path],
    ) -> tuple[list[list[float]], list[int]]:
        X = []
        y = []

        for path in paths:
            sample = self.build_sample(path)
            row = [0.0] * len(self.vocabulary.entries)

            for feature_id, frequency in sample.features.items():
                row[feature_id] = frequency

            X.append(row)
            y.append(sample.label)

        return X, y
