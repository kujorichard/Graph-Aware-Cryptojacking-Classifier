from collections import defaultdict
from .models import BinaryGraph, FunctionGraph, InstructionPath, Shingle


def build_cfg_adjacency(function: FunctionGraph) -> dict[int, list[int]]:
    """
    Build an adjacency list for a function's Control Flow Graph (CFG).

    Returns
    -------
    dict[int, list[int]]

        {
            block_id: [successor_block_ids]
        }
    """

    adjacency: dict[int, list[int]] = defaultdict(list)

    # Ensure every block appears in the dictionary
    for block in function.cfg_nodes:
        adjacency[block.block_id] = []

    # Add directed edges
    for edge in function.cfg_edges:
        adjacency[edge.from_block].append(edge.to_block)

    return dict(adjacency)


def find_entry_block(function: FunctionGraph) -> int:
    """
    Return the block that contains the function's entry address.
    """

    entry = function.entry.lower()

    for block in function.cfg_nodes:

        for idx in block.instruction_indices:

            instr = function.instruction_by_idx(idx)

            if instr.address.lower() == entry:
                return block.block_id

    raise ValueError(
        f"Could not locate entry block for {function.name}"
    )


def enumerate_cfg_paths(
    function: FunctionGraph,
) -> list[InstructionPath]:
    """
    Enumerate all acyclic execution paths in a function's CFG.
    """

    adjacency = build_cfg_adjacency(function)

    entry = find_entry_block(function)

    paths: list[InstructionPath] = []

    path_counter = 0

    def dfs(
        current: int,
        visited: set[int],
        graph_path: list[int],
    ):

        nonlocal path_counter

        visited.add(current)

        graph_path.append(current)

        successors = adjacency[current]

        #
        # Exit node
        #
        if len(successors) == 0:

            instructions = []

            for block_id in graph_path:

                block = function.cfg_nodes[block_id]

                for idx in block.instruction_indices:
                    instructions.append(
                        function.instruction_by_idx(idx)
                    )

            paths.append(
                InstructionPath(
                    function=function.name,
                    graph_type="CFG",
                    path_id=path_counter,
                    graph_path=graph_path.copy(),
                    instructions=instructions,
                    chain_id=None,
                    variable=None,
                    dependency_kind=None,
                )
            )

            path_counter += 1

        else:

            for nxt in successors:

                #
                # Prevent infinite loops
                #
                if nxt not in visited:

                    dfs(
                        nxt,
                        visited.copy(),
                        graph_path.copy(),
                    )

    dfs(
        entry,
        set(),
        [],
    )

    return paths
