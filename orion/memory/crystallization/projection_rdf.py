from __future__ import annotations

from dataclasses import dataclass

from orion.memory.crystallization.schemas import MemoryCrystallizationV1


@dataclass
class RdfProjectionResult:
    named_graph: str | None = None
    skipped: bool = False
    reason: str | None = None


def build_rdf_projection_hint(crystallization: MemoryCrystallizationV1) -> RdfProjectionResult:
    """
    Conservative RDF memory_graph projection hint.

    Does not write to Fuseki/GraphDB — operator should use existing
    /api/memory/graph/approve flow. Crystallizations only record intent.
    """
    if crystallization.status != "active":
        return RdfProjectionResult(skipped=True, reason="status_not_active")

    scope = crystallization.scope[0] if crystallization.scope else "orion"
    graph_uri = f"https://orion.local/memory/crystallization/{crystallization.crystallization_id}"
    return RdfProjectionResult(named_graph=graph_uri, reason=f"scope:{scope}")
