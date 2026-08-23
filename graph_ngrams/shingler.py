from .models import (
    FunctionGraph,
    Instruction,
    InstructionPath,
    Shingle,
)


def create_shingle(
    function: FunctionGraph,
    path: InstructionPath,
    instructions: list[Instruction],
) -> Shingle:
    """
    Create a graph-guided opcode n-gram from an instruction window.
    """

    return Shingle(
        function=function.name,

        type=path.graph_type,

        path_id=path.path_id,

        graph_path=path.graph_path.copy(),

        chain_id=path.chain_id,

        variable=path.variable,

        dependency_kind=path.dependency_kind,

        instruction_indices=[
            ins.idx
            for ins in instructions
        ],

        addresses=[
            ins.address
            for ins in instructions
        ],

        ngram=[
            ins.mnemonic
            for ins in instructions
        ],
    )


def shingle_path(
    function: FunctionGraph,
    path: InstructionPath,
    n: int,
) -> list[Shingle]:
    """
    Convert an instruction path into opcode n-grams.
    """

    instructions = path.instructions

    if len(instructions) < n:
        return []

    shingles = []

    for start in range(
        len(instructions) - n + 1
    ):

        window = instructions[start:start+n]

        shingles.append(
            create_shingle(
                function,
                path,
                window,
            )
        )

    return shingles