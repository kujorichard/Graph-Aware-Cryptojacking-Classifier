from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from sklearn.model_selection import train_test_split


NGRAM_SIZES = (2, 3, 4)
CLASS_DIRECTORIES = (("benign", 0), ("cryptojacking", 1))


def load_dataset_files(
    dataset_dir: str | Path,
) -> tuple[list[Path], list[int]]:
    """Return JSON samples and labels in stable class/name order."""
    dataset_dir = Path(dataset_dir)
    files: list[Path] = []
    labels: list[int] = []

    for class_name, label in CLASS_DIRECTORIES:
        class_dir = dataset_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset class directory does not exist: {class_dir}"
            )

        class_files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".json"
        )
        files.extend(class_files)
        labels.extend([label] * len(class_files))

    if not files:
        raise ValueError(f"No JSON samples found under {dataset_dir}")

    return files, labels


def get_train_test_split(
    dataset_dir: str | Path,
) -> tuple[list[Path], list[Path], list[int], list[int]]:
    files, labels = load_dataset_files(dataset_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        files,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    return X_train, X_test, y_train, y_test


def iter_opcode_ngrams(
    sample_path: str | Path,
    n: int,
) -> Iterable[tuple[str, ...]]:
    """Yield n-grams from each function's listed instruction order."""
    if n < 1:
        raise ValueError(f"n must be positive; got {n}")

    sample_path = Path(sample_path)
    with sample_path.open("r", encoding="utf-8") as handle:
        sample = json.load(handle)

    for graph in sample["graphs"]:
        instructions = graph["instructions"]
        mnemonics = [instruction["mnemonic"] for instruction in instructions]
        for start in range(len(mnemonics) - n + 1):
            yield tuple(mnemonics[start : start + n])


def build_vocabulary(
    paths: list[Path],
    n: int,
) -> dict[tuple[str, ...], int]:
    """Build a deterministic vocabulary from the supplied samples."""
    vocabulary = {
        ngram
        for path in paths
        for ngram in iter_opcode_ngrams(path, n)
    }
    return {
        ngram: feature_id
        for feature_id, ngram in enumerate(sorted(vocabulary))
    }


def save_vocabulary(
    vocabulary: dict[tuple[str, ...], int],
    path: str | Path,
    n: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "n": n,
                "vocabulary_size": len(vocabulary),
                "features": {
                    " ".join(ngram): feature_id
                    for ngram, feature_id in vocabulary.items()
                },
            },
            handle,
            indent=2,
        )


def load_vocabulary(path: str | Path) -> dict[tuple[str, ...], int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        tuple(ngram.split()): int(feature_id)
        for ngram, feature_id in data["features"].items()
    }


def _write_batch(
    rows: list[dict[int, float]],
    labels: list[int],
    feature_count: int,
    split_dir: Path,
    batch_number: int,
) -> None:
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []

    for row_index, row in enumerate(rows):
        for feature_id, value in row.items():
            row_indices.append(row_index)
            column_indices.append(feature_id)
            values.append(value)

    matrix = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), feature_count),
        dtype=np.float32,
    )
    save_npz(split_dir / f"X_batch_{batch_number:03d}.npz", matrix)
    np.save(
        split_dir / f"y_batch_{batch_number:03d}.npy",
        np.asarray(labels, dtype=np.int8),
    )


def build_split(
    paths: list[Path],
    labels: list[int],
    split_name: str,
    n: int,
    vocabulary: dict[tuple[str, ...], int],
    output_dir: str | Path,
    batch_size: int = 100,
) -> None:
    if len(paths) != len(labels):
        raise ValueError(
            f"{split_name} paths/labels differ in length: "
            f"{len(paths)} != {len(labels)}"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {batch_size}")

    split_dir = Path(output_dir) / f"{n}gram" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for old_batch in split_dir.glob("X_batch_*.npz"):
        old_batch.unlink()
    for old_batch in split_dir.glob("y_batch_*.npy"):
        old_batch.unlink()

    rows: list[dict[int, float]] = []
    batch_labels: list[int] = []
    batch_number = 0

    for index, (path, label) in enumerate(zip(paths, labels), start=1):
        counts = Counter(
            vocabulary[ngram]
            for ngram in iter_opcode_ngrams(path, n)
            if ngram in vocabulary
        )
        total = sum(counts.values())
        rows.append(
            {
                feature_id: count / total
                for feature_id, count in counts.items()
            }
            if total
            else {}
        )
        batch_labels.append(label)

        if len(rows) >= batch_size or index == len(paths):
            _write_batch(
                rows,
                batch_labels,
                len(vocabulary),
                split_dir,
                batch_number,
            )
            print(
                f"{n}gram {split_name}: processed {index}/{len(paths)}"
            )
            rows = []
            batch_labels = []
            batch_number += 1
