from pathlib import Path
import time
import json
from sklearn.model_selection import train_test_split

from graph_ngrams.extractors import extract_binary_cfg_ngrams, extract_binary_dfg_ngrams
from graph_ngrams.loader import GraphLoader
from graph_ngrams.vocabulary import Vocabulary


class DatasetLoader:

    def __init__(self, dataset_dir: str | Path):

        self.dataset_dir = Path(dataset_dir)

        self.benign_dir = self.dataset_dir / "benign"
        self.cryptojacking_dir = self.dataset_dir / "cryptojacking"

        self.skipped_files: list[tuple[Path, str]] = []


    def load_files(self):

        benign_files = sorted([
            path
            for path in self.benign_dir.iterdir()
            if path.is_file()
        ])

        cryptojacking_files = sorted([
            path
            for path in self.cryptojacking_dir.iterdir()
            if path.is_file()
        ])

        files = benign_files + cryptojacking_files

        labels = (
            [0] * len(benign_files) +
            [1] * len(cryptojacking_files)
        )

        return files, labels


    def load_binary(
        self,
        path: Path,
    ):
        return GraphLoader.load(
            str(path)
        )


    def load_binaries(
        self,
        paths: list[Path],
    ):
        binaries = []

        for path in paths:
            binaries.append(
                self.load_binary(path)
            )

        return binaries


    def train_test_split(self, test_size: float = 0.20):

        files, labels = self.load_files()

        X_train, X_test, y_train, y_test = train_test_split(
            files,
            labels,
            test_size=test_size,
            random_state=42,
            stratify=labels,
        )

        return (
            X_train,
            X_test, 
            y_train,
            y_test,
        )


    def iter_binaries(
        self,
        paths: list[Path],
    ):
        for path in paths:

            try:
                yield GraphLoader.load(
                    str(path)
                )

            except json.JSONDecodeError as exc:

                print(
                    f"\n[SKIP] Invalid JSON: {path}"
                )

                print(
                    f"JSONDecodeError: {exc}"
                )

                self.skipped_files.append(
                    (
                        path,
                        str(exc),
                    )
                )


    def build_vocabulary(
        self,
        paths: list[Path],
        vocabulary: Vocabulary,
        n: int = 3,
    ) -> Vocabulary:

        total = len(paths)
        start_time = time.perf_counter()

        for index, binary in enumerate(
            self.iter_binaries(paths),
            start=1,
        ):

            cfg = extract_binary_cfg_ngrams(
                binary,
                n,
            )

            dfg = extract_binary_dfg_ngrams(
                binary,
                n,
            )

            shingles = cfg + dfg

            vocabulary.add_shingles(
                binary.binary_name,
                shingles,
            )

            if index % 50 == 0 or index == total:

                elapsed = time.perf_counter() - start_time

                rate = (
                    index / elapsed
                    if elapsed > 0
                    else 0
                )

                remaining = total - index

                eta = (
                    remaining / rate
                    if rate > 0
                    else 0
                )

                print(
                    f"Processed {index}/{total} "
                    f"| Vocabulary: {len(vocabulary)} "
                    f"| {rate:.2f} binaries/sec "
                    f"| ETA: {eta / 60:.1f} min"
                )

        return vocabulary