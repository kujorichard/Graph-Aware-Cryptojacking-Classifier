from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz

from graph_ngrams.loader import GraphLoader
from graph_ngrams.extractors import (
    extract_binary_cfg_ngrams,
    extract_binary_dfg_ngrams,
)
from graph_ngrams.vocabulary import Vocabulary


class FeatureMatrixBuilder:

    def __init__(
        self,
        vocabulary: Vocabulary,
        n: int,
        output_dir: Path,
        batch_size: int = 100,
    ):
        self.vocabulary = vocabulary
        self.n = n
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size

    def build_split(
        self,
        paths: list[Path],
        labels: list[int],
        split_name: str,
    ) -> None:

        split_dir = (
            self.output_dir
            / f"{self.n}gram"
            / split_name
        )

        split_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        batch_rows = []
        batch_labels = []

        batch_number = 0

        for index, (path, label) in enumerate(
            zip(paths, labels),
            start=1,
        ):

            binary = GraphLoader.load(
                str(path)
            )

            cfg = extract_binary_cfg_ngrams(
                binary,
                self.n,
            )

            dfg = extract_binary_dfg_ngrams(
                binary,
                self.n,
            )

            shingles = cfg + dfg

            features = self.vocabulary.vectorize(
                shingles,
            )

            batch_rows.append(features)
            batch_labels.append(label)

            if (
                len(batch_rows) >= self.batch_size
                or index == len(paths)
            ):

                self._write_batch(
                    batch_rows,
                    batch_labels,
                    split_dir,
                    batch_number,
                )

                print(
                    f"{split_name}: "
                    f"processed {index}/{len(paths)}"
                )

                batch_rows = []
                batch_labels = []
                batch_number += 1

    def _write_batch(
        self,
        rows: list[dict[int, float]],
        labels: list[int],
        split_dir: Path,
        batch_number: int,
    ) -> None:

        matrix = csr_matrix(
            (
                [
                    value
                    for row in rows
                    for value in row.values()
                ],
                (
                    [
                        row_index
                        for row_index, row in enumerate(rows)
                        for _ in row
                    ],
                    [
                        feature_id
                        for row in rows
                        for feature_id in row.keys()
                    ],
                ),
            ),
            shape=(
                len(rows),
                len(self.vocabulary),
            ),
            dtype=np.float32,
        )

        X_path = (
            split_dir
            / f"X_batch_{batch_number:03d}.npz"
        )

        y_path = (
            split_dir
            / f"y_batch_{batch_number:03d}.npy"
        )

        save_npz(
            X_path,
            matrix,
        )

        np.save(
            y_path,
            np.asarray(labels, dtype=np.int8),
        )