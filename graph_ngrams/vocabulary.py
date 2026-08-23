from .models import (
    VocabularyEntry,
    FeatureOccurrence,
    Shingle,
)

import json
from pathlib import Path

class Vocabulary:

    def __init__(
        self,
        store_occurrences: bool = True,
    ):

        self.entries: dict[
            tuple[str, ...],
            VocabularyEntry,
        ] = {}

        self.id_to_entry: dict[
            int,
            VocabularyEntry,
        ] = {}

        self.next_feature_id = 0

        self.store_occurrences = store_occurrences

    def _build_occurrence(
        self,
        binary: str,
        shingle: Shingle,
    ) -> FeatureOccurrence:

        return FeatureOccurrence(
            binary=binary,
            function=shingle.function,
            graph_type=shingle.type,
            path_id=shingle.path_id,
            graph_path=shingle.graph_path.copy(),
            chain_id=shingle.chain_id,
            variable=shingle.variable,
            dependency_kind=shingle.dependency_kind,
            instruction_indices=shingle.instruction_indices.copy(),
            addresses=shingle.addresses.copy(),
        )

    def add_shingle(
        self,
        binary: str,
        shingle: Shingle,
    ) -> None:
        
        key = tuple(shingle.ngram)

        if key not in self.entries:

            entry = VocabularyEntry(
                feature_id=self.next_feature_id,
                ngram=key,
            )

            self.entries[key] = entry

            self.id_to_entry[entry.feature_id] = entry

            self.next_feature_id += 1

        entry = self.entries[key]

        if self.store_occurrences:

            entry.occurrences.append(
                self._build_occurrence(
                    binary,
                    shingle,
                )
            )

    def add_shingles(
        self,
        binary: str,
        shingles: list[Shingle],
    ) -> None:

        for shingle in shingles:
            self.add_shingle(
                binary,
                shingle,
            )

    @property
    def size(self) -> int:
        return len(self.entries)

    def __len__(self):
        return len(self.entries)

    def get_entry(
        self,
        feature_id: int,
    ) -> VocabularyEntry:
        """
        Retrieve a vocabulary entry by its feature ID.
        """

        return self.id_to_entry[feature_id]

    def get_feature_id(
        self,
        shingle: Shingle,
    ) -> int:
        """
        Return the feature ID assigned to a shingle.
        """

        key = tuple(shingle.ngram)

        return self.entries[key].feature_id


    def count_features(
        self,
        shingles: list[Shingle],
    ) -> dict[int, int]:
        """
        Count how many times each feature appears.
        """

        counts: dict[int, int] = {}

        for shingle in shingles:

            key = tuple(shingle.ngram)

            if key not in self.entries:
                continue

            feature_id = self.entries[key].feature_id

            counts[feature_id] = (
                counts.get(feature_id, 0) + 1
            )

        return counts


    def vectorize(
        self,
        shingles: list[Shingle],
    ) -> dict[int, float]:

        counts = self.count_features(shingles)

        total = sum(counts.values())

        if total == 0:
            return {}

        return {
            feature_id: count / total
            for feature_id, count in counts.items()
        }


    def save(
        self,
        path: Path,
        n: int,
    ) -> None:

        data = {
            "n": n,
            "vocabulary_size": len(self.entries),
            "features": {
                " ".join(ngram): entry.feature_id
                for ngram, entry in self.entries.items()
            },
        }

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
            )


    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "Vocabulary":

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        vocabulary = cls(
            store_occurrences=False
        )

        features = data["features"]

        for ngram_string, feature_id in features.items():

            ngram = tuple(
                ngram_string.split()
            )

            entry = VocabularyEntry(
                feature_id=feature_id,
                ngram=ngram,
            )

            vocabulary.entries[ngram] = entry
            vocabulary.id_to_entry[feature_id] = entry

        vocabulary.next_feature_id = (
            data["vocabulary_size"]
        )

        return vocabulary