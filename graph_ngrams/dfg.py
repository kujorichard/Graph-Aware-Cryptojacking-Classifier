from collections import defaultdict
from .models import DependencySegment, FunctionGraph, Shingle
from .models import BinaryGraph, DFGEdge, FunctionGraph, InstructionPath, Shingle


def group_edges_by_variable(
    edges: list[DFGEdge],
) -> dict[str, list[DFGEdge]]:
    """
    Group DFG edges according to the variable they propagate.
    """

    groups: dict[str, list[DFGEdge]] = defaultdict(list)

    for edge in edges:
        groups[edge.var].append(edge)

    return dict(groups)


def build_adjacency(
    edges: list[DFGEdge],
) -> dict[int, list[int]]:
    """
    Build an adjacency list from a set of DFG edges.
    """

    adjacency: dict[int, list[int]] = defaultdict(list)

    for edge in edges:
        adjacency[edge.from_idx].append(edge.to_idx)

    return dict(adjacency)


def find_roots(
    edges: list[DFGEdge],
) -> list[int]:
    """
    Return all nodes with no incoming edges.
    """

    sources = set()
    destinations = set()

    for edge in edges:

        sources.add(edge.from_idx)

        destinations.add(edge.to_idx)

    roots = sorted(
        sources - destinations
    )

    return roots


def _dfs_chains(
    adjacency: dict[int, list[int]],
    current: int,
    path: list[int],
    chains: list[list[int]],
) -> None:
    """
    Enumerate every root-to-leaf path in a DFG.
    """

    path.append(current)

    children = adjacency.get(current, [])

    if not children:
        chains.append(path.copy())
    else:
        for child in children:

            if child in path:
                continue

            _dfs_chains(
                adjacency,
                child,
                path.copy(),
                chains,
            )


def enumerate_variable_chains(
    edges: list[DFGEdge],
) -> list[list[int]]:
    """
    Enumerate all dependency chains for one variable.
    """

    adjacency = build_adjacency(edges)

    roots = find_roots(edges)

    chains: list[list[int]] = []

    for root in roots:

        _dfs_chains(
            adjacency,
            root,
            [],
            chains,
        )

    return chains


def build_instruction_path(
    function: FunctionGraph,
    segment: DependencySegment,
    path_id: int,
) -> InstructionPath:
    """
    Convert a DependencySegment into an InstructionPath.
    """

    instructions = [
        function.instruction_by_idx(idx)
        for idx in segment.instruction_indices
    ]

    return InstructionPath(
        function=function.name,

        graph_type="DFG",

        path_id=path_id,

        graph_path=segment.instruction_indices.copy(),

        instructions=instructions,

        chain_id=path_id,

        variable=segment.variable,

        dependency_kind=segment.kind,
    )


def enumerate_dfg_paths(
    function: FunctionGraph,
) -> list[InstructionPath]:
    """
    Enumerate every DFG dependency chain in a function.
    """

    paths: list[InstructionPath] = []

    path_counter = 0

    groups = group_edges_by_variable(
        function.dfg_edges
    )

    for variable, edges in groups.items():

        dependency_kind = edges[0].kind

        segments = extract_dependency_segments(
            variable=variable,
            kind=dependency_kind,
            edges=edges,
        )

        for segment in segments:

            paths.append(
                build_instruction_path(
                    function=function,
                    segment=segment,
                    path_id=path_counter,
                )
            )

            path_counter += 1

    return paths



def build_dependency_chains(
    edges: list[DFGEdge],
) -> list[list[int]]:
    """
    Build linear def-use chains for a single variable.

    The algorithm:
        1. Remove loop-carried edges.
        2. Build adjacency.
        3. Find roots.
        4. Iteratively walk each chain.
    """

    # Ignore loop-carried dependencies
    filtered_edges = [
        edge
        for edge in edges
        if not edge.loop_carried
    ]

    adjacency = build_adjacency(filtered_edges)

    roots = find_roots(filtered_edges)

    chains: list[list[int]] = []

    stack: list[tuple[int, list[int]]] = []

    for root in roots:
        stack.append(
            (root, [root])
        )

    while stack:

        current, chain = stack.pop()

        successors = adjacency.get(current, [])

        #
        # No outgoing edges.
        #
        if not successors:
            chains.append(chain)
            continue

        #
        # Follow every successor.
        #
        for successor in successors:

            #
            # Prevent cycles.
            #
            if successor in chain:
                continue

            stack.append(
                (
                    successor,
                    chain + [successor]
                )
            )

    return chains


def build_indegree(
    edges: list[DFGEdge],
) -> dict[int, int]:
    """
    Compute the indegree of every node.
    """

    indegree: dict[int, int] = defaultdict(int)

    for edge in edges:
        indegree[edge.to_idx] += 1

        #
        # Ensure every source exists.
        #
        indegree.setdefault(edge.from_idx, 0)

    return dict(indegree)


def build_outdegree(
    edges: list[DFGEdge],
) -> dict[int, int]:
    """
    Compute the outdegree of every node.
    """

    outdegree: dict[int, int] = defaultdict(int)

    for edge in edges:
        outdegree[edge.from_idx] += 1

        # Ensure every destination exists.
        outdegree.setdefault(edge.to_idx, 0)

    return dict(outdegree)


def extract_dependency_segments(
    variable: str,
    kind: str,
    edges: list[DFGEdge],
) -> list[DependencySegment]:
    """
    Extract maximal linear dependency segments.
    """

    filtered_edges = [
        edge
        for edge in edges
        if not edge.loop_carried
    ]

    adjacency = build_adjacency(filtered_edges)

    indegree = build_indegree(filtered_edges)

    outdegree = build_outdegree(filtered_edges)

    # -----------------------------------------
    # Find segment start nodes
    # -----------------------------------------

    starts: list[int] = []

    all_nodes = set(indegree.keys()) | set(outdegree.keys())

    for node in all_nodes:

        if (
            indegree[node] == 0
            or indegree[node] > 1
            or outdegree[node] > 1
        ):
            starts.append(node)

    segments: list[DependencySegment] = []

    for start in starts:

        #
        # Every outgoing edge becomes a segment.
        #
        for successor in adjacency.get(start, []):

            segment = [start]

            current = successor

            while True:

                segment.append(current)

                #
                # Leaf node.
                #
                if outdegree[current] == 0:
                    break

                #
                # Branch node.
                #
                if outdegree[current] > 1:
                    break

                #
                # Merge node.
                #
                if indegree[current] > 1:
                    break

                #
                # Continue along the only successor.
                #
                next_nodes = adjacency[current]

                if not next_nodes:
                    break

                current = next_nodes[0]

            segments.append(
                DependencySegment(
                    variable=variable,
                    kind=kind,
                    instruction_indices=segment,
                )
            )

    return segments