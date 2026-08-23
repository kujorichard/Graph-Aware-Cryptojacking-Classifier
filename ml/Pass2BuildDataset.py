import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from dataset_loader import DatasetLoader
from graph_ngrams.vocabulary import Vocabulary
from feature_matrix_builder import FeatureMatrixBuilder

def main():

    ml_dir = Path(__file__).resolve().parent

    loader = DatasetLoader(
        ml_dir / "dataset"
    )

    X_train, X_test, y_train, y_test = (
        loader.train_test_split()
    )

    vocabulary_path = (
        ml_dir
        / "feature_datasets"
        / "2gram"
        / "vocabulary.json"
    )

    vocabulary = Vocabulary.load(
        vocabulary_path
    )


    print(
        f"Loaded vocabulary: {len(vocabulary)} features"
    )

    builder = FeatureMatrixBuilder(
        vocabulary=vocabulary,
        n=2,
        output_dir=ml_dir / "feature_datasets",
        batch_size=100,
    )


    builder.build_split(
        X_train,
        y_train,
        "train",
    )

    builder.build_split(
        X_test,
        y_test,
        "test",
    )


if __name__ == "__main__":

    main()