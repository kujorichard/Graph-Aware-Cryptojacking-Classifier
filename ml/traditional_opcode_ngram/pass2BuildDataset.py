from pathlib import Path

from feature_pipeline import (
    NGRAM_SIZES,
    build_split,
    get_train_test_split,
    load_vocabulary,
)


def main() -> None:
    baseline_dir = Path(__file__).resolve().parent
    dataset_dir = baseline_dir.parent / "dataset"
    X_train, X_test, y_train, y_test = get_train_test_split(dataset_dir)

    print(
        f"Pass 2: building train/test datasets for {len(NGRAM_SIZES)} n-gram sizes "
        f"| train={len(X_train)} samples | test={len(X_test)} samples"
    )
    for ngram_index, n in enumerate(NGRAM_SIZES, start=1):
        print(
            f"\n[{ngram_index}/{len(NGRAM_SIZES)}] Building {n}gram datasets",
            flush=True,
        )
        vocabulary_path = (
            baseline_dir / "feature_datasets" / f"{n}gram" / "vocabulary.json"
        )
        vocabulary = load_vocabulary(vocabulary_path)
        print(f"Loaded {n}gram vocabulary: {len(vocabulary)} features")

        build_split(
            X_train,
            y_train,
            "train",
            n,
            vocabulary,
            baseline_dir / "feature_datasets",
        )
        build_split(
            X_test,
            y_test,
            "test",
            n,
            vocabulary,
            baseline_dir / "feature_datasets",
        )

    print("\nPass 2 complete: all train/test datasets are ready.", flush=True)


if __name__ == "__main__":
    main()
