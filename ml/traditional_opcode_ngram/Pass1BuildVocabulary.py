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

    print(
        f"Pass 1: building {len(NGRAM_SIZES)} training-only vocabularies "
        f"from {len(X_train)} training samples"
    )
    for ngram_index, n in enumerate(NGRAM_SIZES, start=1):
        print(
            f"\n[{ngram_index}/{len(NGRAM_SIZES)}] Building {n}gram vocabulary",
            flush=True,
        )
        vocabulary = build_vocabulary(X_train, n)
        vocabulary_path = (
            baseline_dir / "feature_datasets" / f"{n}gram" / "vocabulary.json"
        )
        save_vocabulary(vocabulary, vocabulary_path, n)
        print(
            f"Completed {n}gram vocabulary: {len(vocabulary):,} features "
            f"| saved to {vocabulary_path}",
            flush=True,
        )

    print("\nPass 1 complete: all vocabularies are ready.", flush=True)


if __name__ == "__main__":
    main()
