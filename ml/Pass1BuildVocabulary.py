import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from dataset_loader import DatasetLoader
from graph_ngrams.vocabulary import Vocabulary


def main():

    ml_dir = Path(__file__).resolve().parent

    loader = DatasetLoader(
        ml_dir / "dataset"
    )

    X_train, X_test, y_train, y_test = (
        loader.train_test_split()
    )

    vocabulary = Vocabulary(
        store_occurrences=False
    )

    loader.build_vocabulary(
        X_train,
        vocabulary,
        n=2,
    )

    vocabulary_path = (
        ml_dir
        / "feature_datasets"
        / "2gram"
        / "vocabulary.json"
    )

    vocabulary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vocabulary.save(
        vocabulary_path,
        n=2,
    )

    print(
        f"\nFinal vocabulary size: {len(vocabulary)}"
    )

    print(
        f"Skipped files: {len(loader.skipped_files)}"
    )

    for path, error in loader.skipped_files:

        print(
            f"- {path}"
        )

    print(
        f"Vocabulary saved to: {vocabulary_path}"
    )


if __name__ == "__main__":
    main()