from .dfg import enumerate_dfg_paths
from .models import BinaryGraph, FunctionGraph, Shingle
from .cfg import enumerate_cfg_paths
from .shingler import shingle_path

# ---------------------------------------------------------------------
# CFG Extractors
# ---------------------------------------------------------------------
def extract_cfg_ngrams(
    function: FunctionGraph,
    n: int,
) -> list[Shingle]:

    shingles: list[Shingle] = []

    for path in enumerate_cfg_paths(function):
        shingles.extend(
            shingle_path(
                function,
                path,
                n,
            )
        )

    return shingles


def extract_binary_cfg_ngrams(
    binary: BinaryGraph,
    n: int,
) -> list[Shingle]:

    shingles: list[Shingle] = []

    for function in binary.functions:

        try:
            shingles.extend(
                extract_cfg_ngrams(
                    function,
                    n,
                )
            )

        except ValueError as exc:

            print(
                f"[CFG WARNING] "
                f"{binary.binary_name} | "
                f"Function: {function.name} | "
                f"{exc}"
            )

    return shingles


# ---------------------------------------------------------------------
# DFG Extractors
# ---------------------------------------------------------------------
def extract_dfg_ngrams(
    function: FunctionGraph,
    n: int,
) -> list[Shingle]:
    """
    Extract all DFG-guided opcode n-grams for a function.
    """

    shingles: list[Shingle] = []

    #print(f"Processing function: {function.name}")
    paths = enumerate_dfg_paths(function)

    for path in paths:
        shingles.extend(
            shingle_path(
                function,
                path,
                n,
            )
        )

    return shingles

def extract_binary_dfg_ngrams(
    binary: BinaryGraph,
    n: int,
) -> list[Shingle]:

    shingles: list[Shingle] = []

    for function in binary.functions:

        try:
            shingles.extend(
                extract_dfg_ngrams(
                    function,
                    n,
                )
            )

        except ValueError as exc:

            print(
                f"[DFG WARNING] "
                f"{binary.binary_name} | "
                f"Function: {function.name} | "
                f"{exc}"
            )

    return shingles