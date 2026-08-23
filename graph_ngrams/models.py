from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------
# Basic structures
# ---------------------------------------------------------------------

@dataclass(slots=True)
class Instruction:
    """Single assembly instruction."""

    idx: int
    address: str
    mnemonic: str
    operands: str
    defs: list[dict[str, Any]] = field(default_factory=list)
    uses: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class CFGNode:
    """Basic block."""

    block_id: int
    instruction_indices: list[int]
    start: str
    end: str


@dataclass(slots=True)
class CFGEdge:
    from_block: int
    to_block: int


@dataclass(slots=True)
class DFGEdge:
    from_idx: int
    to_idx: int
    var: str
    kind: str
    loop_carried: bool = False

# ---------------------------------------------------------------------
# Function graph
# ---------------------------------------------------------------------

@dataclass(slots=True)
class FunctionGraph:
    name: str
    entry: str

    instructions: list[Instruction]
    cfg_nodes: list[CFGNode]
    cfg_edges: list[CFGEdge]
    dfg_edges: list[DFGEdge]

    _instruction_lookup: dict[int, Instruction] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._instruction_lookup = {
            ins.idx: ins
            for ins in self.instructions
        }

    def instruction_by_idx(self, idx: int) -> Instruction:
        return self._instruction_lookup[idx]


# ---------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------

@dataclass(slots=True)
class BinaryGraph:
    binary_name: str
    label: int
    label_name: str

    functions: list[FunctionGraph]

# ---------------------------------------------------------------------
# Instruction path
# ---------------------------------------------------------------------

@dataclass(slots=True)
class InstructionPath:
    function: str

    graph_type: Literal["CFG", "DFG"]

    path_id: int

    graph_path: list[int]

    instructions: list[Instruction]

    chain_id: int | None

    variable: str | None

    dependency_kind: str | None


# ---------------------------------------------------------------------
# Shingle data class
# ---------------------------------------------------------------------

from typing import Literal

@dataclass(slots=True)
class Shingle:
    function: str

    type: Literal["CFG", "DFG"]

    path_id: int
    graph_path: list[int]

    chain_id: int | None

    variable: str | None
    dependency_kind: str | None

    instruction_indices: list[int]
    addresses: list[str]
    ngram: list[str]


# ---------------------------------------------------------------------
# DependencySegment data class
# ---------------------------------------------------------------------

@dataclass(slots=True)
class DependencySegment:
    """
    A maximal linear data-dependency segment for a single variable.
    """

    variable: str

    kind: str

    instruction_indices: list[int]


# ---------------------------------------------------------------------
# FeatureOccurence data class
# ---------------------------------------------------------------------

@dataclass(slots=True)
class FeatureOccurrence:
    """
    Records one occurrence of an opcode n-gram.
    """

    binary: str

    function: str

    graph_type: str

    path_id: int

    graph_path: list[int]

    chain_id: int | None

    variable: str | None

    dependency_kind: str | None

    instruction_indices: list[int]

    addresses: list[str]


# ---------------------------------------------------------------------
# VocabularyEntry data class
# ---------------------------------------------------------------------

@dataclass(slots=True)
class VocabularyEntry:
    """
    One unique opcode n-gram.
    """

    feature_id: int

    ngram: tuple[str, ...]

    occurrences: list[FeatureOccurrence] = field(
        default_factory=list
    )

# ---------------------------------------------------------------------
# (Binary) Sample data class
# ---------------------------------------------------------------------
@dataclass
class Sample:
    binary_name: str
    label: int
    features: dict[int, float]


