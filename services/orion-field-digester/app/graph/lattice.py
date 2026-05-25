from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from orion.schemas.field_state import FieldEdgeV1


@dataclass(frozen=True)
class LatticeGraph:
    nodes: list[str]
    capabilities: list[str]
    edges: list[FieldEdgeV1]


def load_lattice(path: Path) -> LatticeGraph:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    nodes = [f"node:{n['node_id']}" for n in raw["nodes"]]
    capabilities = [f"capability:{c['capability_id']}" for c in raw["capabilities"]]
    edges = [FieldEdgeV1.model_validate(e) for e in raw["edges"]]
    return LatticeGraph(nodes=nodes, capabilities=capabilities, edges=edges)
