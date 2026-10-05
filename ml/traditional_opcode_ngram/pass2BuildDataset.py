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

    for n in NGRAM_SIZES:
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


if __name__ == "__main__":
    main()
