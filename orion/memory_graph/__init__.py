from __future__ import annotations

from typing import TYPE_CHECKING

from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.json_to_rdf import draft_to_graph
from orion.memory_graph.project import project_graph_to_cards

if TYPE_CHECKING:
    from orion.memory_graph.validate import validate_graph as validate_graph


def __getattr__(name: str):
    if name == "validate_graph":
        from orion.memory_graph.validate import validate_graph as _validate_graph

        return _validate_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SuggestDraftV1",
    "draft_to_graph",
    "project_graph_to_cards",
    "validate_graph",
]
