import json
from pathlib import Path

from .models import (
    BinaryGraph,
    CFGEdge,
    CFGNode,
    DFGEdge,
    FunctionGraph,
    Instruction,
)


class GraphLoader:
    """Loads a Ghidra-exported graph JSON."""

    @staticmethod
    def load(path: str | Path) -> BinaryGraph:

        path = Path(path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        functions: list[FunctionGraph] = []

        for graph in data["graphs"]:

            # --------------------------
            # Instructions
            # --------------------------

            instructions = [
                Instruction(
                    idx=i["idx"],
                    address=i["addr"],
                    mnemonic=i["mnemonic"],
                    operands=i.get("operands", ""),
                    defs=i.get("defs", []),
                    uses=i.get("uses", []),
                )
                for i in graph["instructions"]
            ]

            # --------------------------
            # CFG Nodes
            # --------------------------

            cfg_nodes = [
                CFGNode(
                    block_id=block_id,
                    instruction_indices=node["instr_indices"],
                    start=node["start"],
                    end=node["end"],
                )
                for block_id, node in enumerate(graph["cfg_nodes"])
            ]

            # --------------------------
            # CFG Edges
            # --------------------------

            cfg_edges = []

            for edge in graph.get("cfg_edges", []):

                src = edge.get("src", edge.get("from"))
                dst = edge.get("dst", edge.get("to"))

                if src is None or dst is None:
                    continue

                cfg_edges.append(
                    CFGEdge(
                        from_block=src,
                        to_block=dst,
                    )
                )

            # --------------------------
            # DFG Edges
            # --------------------------

            dfg_edges = [
                DFGEdge(
                    from_idx=edge["from"],
                    to_idx=edge["to"],
                    var=edge["var"],
                    kind=edge["kind"],
                    loop_carried=edge.get(
                        "loop_carried",
                        False,
                    ),
                )
                for edge in graph.get("dfg_edges", [])
            ]

            functions.append(
                FunctionGraph(
                    name=graph["function"],
                    entry=graph["entry"],
                    instructions=instructions,
                    cfg_nodes=cfg_nodes,
                    cfg_edges=cfg_edges,
                    dfg_edges=dfg_edges,
                )
            )

        return BinaryGraph(
            binary_name=data["binary"],
            label=data["label"],
            label_name=data["label_name"],
            functions=functions,
        )