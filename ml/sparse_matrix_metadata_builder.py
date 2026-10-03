#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent),
)

from dataset_loader import DatasetLoader


class SparseMatrixMetadataBuilder:
    """Save row-to-file provenance for each sparse-matrix batch."""

    def __init__(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path,
        n: int,
        batch_size: int = 100,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.n = n
        self.batch_size = batch_size

    def build_split_metadata(
        self,
        paths: list[Path],
        labels: list[int],
        split_name: str,
    ) -> None:
        """Write metadata for each batch in the same order as the sparse matrix."""
        split_dir = self.output_dir / f"{self.n}gram" / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        batch_entries: list[dict] = []
        batch_number = 0
        processed_count = 0

        for row_index, (path, label) in enumerate(zip(paths, labels)):
            row_index_in_batch = len(batch_entries)
            batch_entries.append(
                {
                    "split": split_name,
                    "batch_number": batch_number,
                    "row_index_in_split": processed_count,
                    "row_index_in_batch": row_index_in_batch,
                    "label": int(label),
                    "class_name": "benign" if label == 0 else "cryptojacking",
                    "source_path": str(path.resolve()),
                    "source_name": path.name,
                }
            )
            processed_count += 1

            if len(batch_entries) >= self.batch_size or row_index == len(paths) - 1:
                self._write_batch_metadata(
                    batch_entries,
                    split_dir,
                    batch_number,
                )
                batch_entries = []
                batch_number += 1

    def _write_batch_metadata(
        self,
        entries: list[dict],
        split_dir: Path,
        batch_number: int,
    ) -> None:
        metadata_path = split_dir / f"row_metadata_batch_{batch_number:03d}.json"

        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2)

    def validate_existing_batches(self) -> None:
        """Validate that saved metadata lines up with the .npz/.npy rows."""
        for split_name in ("train", "test"):
            split_dir = self.output_dir / f"{self.n}gram" / split_name

            if not split_dir.exists():
                continue

            batch_files = sorted(split_dir.glob("X_batch_*.npz"))
            if not batch_files:
                print(f"No sparse batches found for {split_name}; skipping validation.")
                continue

            for matrix_path in batch_files:
                batch_number = int(matrix_path.stem.split("_")[-1])
                metadata_path = split_dir / f"row_metadata_batch_{batch_number:03d}.json"
                y_path = split_dir / f"y_batch_{batch_number:03d}.npy"

                if not metadata_path.exists():
                    raise FileNotFoundError(f"Metadata missing for {matrix_path}")
                if not y_path.exists():
                    raise FileNotFoundError(f"Labels missing for {matrix_path}")

                matrix = load_npz(matrix_path)
                labels = np.load(y_path)

                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)

                if matrix.shape[0] != len(labels):
                    raise ValueError(
                        f"Row count mismatch in {matrix_path}: "
                        f"matrix rows={matrix.shape[0]}, labels={len(labels)}"
                    )

                if len(metadata) != matrix.shape[0]:
                    raise ValueError(
                        f"Metadata length mismatch in {metadata_path}: "
                        f"metadata={len(metadata)}, matrix rows={matrix.shape[0]}"
                    )

                for row_index, entry in enumerate(metadata):
                    if int(entry["label"]) != int(labels[row_index]):
                        raise ValueError(
                            f"Label mismatch at {split_name} row {row_index} in batch {batch_number}: "
                            f"metadata label={entry['label']}, y label={int(labels[row_index])}"
                        )

                print(
                    f"Validated {split_name} batch {batch_number:03d}: "
                    f"{matrix.shape[0]} rows, {len(metadata)} metadata entries"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate row-to-file metadata for the sparse feature matrices built by "
            "feature_matrix_builder.py."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dataset",
        help="Directory containing benign/cryptojacking JSON samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "feature_datasets",
        help="Directory where the matrices are written.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="N-gram size used to build vocabulary and sparse features.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size used when building sparse matrices.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the saved metadata against the existing sparse-matrix batches.",
    )
    args = parser.parse_args()

    loader = DatasetLoader(args.dataset_dir)
    files, labels = loader.load_files()

    X_train, X_test, y_train, y_test = loader.train_test_split()

    builder = SparseMatrixMetadataBuilder(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        n=args.n,
        batch_size=args.batch_size,
    )

    builder.build_split_metadata(X_train, y_train, "train")
    builder.build_split_metadata(X_test, y_test, "test")

    print(f"Generated metadata for {len(X_train)} train rows and {len(X_test)} test rows.")
    print(f"Metadata written to: {args.output_dir / f'{args.n}gram'}")

    if args.validate:
        builder.validate_existing_batches()


if __name__ == "__main__":
    main()
