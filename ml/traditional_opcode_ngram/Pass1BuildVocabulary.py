from pathlib import Path

from feature_pipeline import (
    NGRAM_SIZES,
    build_vocabulary,
    get_train_test_split,
    save_vocabulary,
)


def main() -> None:
    baseline_dir = Path(__file__).resolve().parent
    dataset_dir = baseline_dir.parent / "dataset"
    X_train, _, _, _ = get_train_test_split(dataset_dir)

    for n in NGRAM_SIZES:
        vocabulary = build_vocabulary(X_train, n)
        vocabulary_path = (
            baseline_dir / "feature_datasets" / f"{n}gram" / "vocabulary.json"
        )
        save_vocabulary(vocabulary, vocabulary_path, n)
        print(f"{n}gram vocabulary size: {len(vocabulary)}")
        print(f"Vocabulary saved to: {vocabulary_path}")


if __name__ == "__main__":
    main()
