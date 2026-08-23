import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from graph_ngrams.loader import GraphLoader
from graph_ngrams.extractors import (
    extract_binary_cfg_ngrams,
    extract_binary_dfg_ngrams,
)
from graph_ngrams.vocabulary import Vocabulary


def extract_sample_graph_ngrams(
    binary_sample: str | Path,
    n: int,
) -> dict:
    """Extract a sample's n-gram occurrences and return the report."""

    binary = GraphLoader.load(binary_sample)

    cfg_shingles = extract_binary_cfg_ngrams(binary, n)
    dfg_shingles = extract_binary_dfg_ngrams(binary, n)

    vocabulary = Vocabulary(store_occurrences=True)
    vocabulary.add_shingles(
        binary.binary_name,
        cfg_shingles + dfg_shingles,
    )

    output = {
        "binary": binary.binary_name,
        "label": binary.label,
        "label_name": binary.label_name,
        "n": n,
        "total_ngrams": len(cfg_shingles) + len(dfg_shingles),
        "vocabulary_size": vocabulary.size,
        "features": [
            {
                "feature_id": entry.feature_id,
                "ngram": list(entry.ngram),
                "occurrences": [
                    {
                        "function": occurrence.function,
                        "graph_type": occurrence.graph_type,
                        "addresses": occurrence.addresses,
                    }
                    for occurrence in entry.occurrences
                ],
            }
            for entry in sorted(
                vocabulary.entries.values(),
                key=lambda entry: entry.feature_id,
            )
        ],
    }

    return output


def main():
    curr_dir = Path(__file__).resolve().parent
    report = extract_sample_graph_ngrams(
        curr_dir / "dataset" / "malware" / "example1.exe.json",
        n=3,
    )
    print(
        f"Extracted {report['total_ngrams']} n-grams "
        f"({report['vocabulary_size']} unique)"
    )


if __name__ == "__main__":
    main()