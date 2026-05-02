from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from orion.core.storage.memory_cards import insert_cards_and_edges_batch
from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.graphdb import compensate_batch, insert_batch
from orion.memory_graph.json_to_rdf import draft_to_graph
from orion.memory_graph.project import project_graph_to_cards
from orion.memory_graph.validate import validate_graph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApproveOutcome:
    ok: bool
    card_ids: List[UUID]
    violations: List[str]
    preview: Optional[Dict[str, Any]] = None


async def approve_memory_graph_draft(
    draft: SuggestDraftV1,
    pool: asyncpg.Pool,
    *,
    graphdb_url: str,
    graphdb_repo: str,
    graphdb_user: str = "",
    graphdb_pass: str = "",
    named_graph_iri: str,
) -> ApproveOutcome:
    """validate → GraphDB (named graph + revision batch) → Postgres cards/edges; compensate RDF if PG fails."""
    batch_id = str(uuid4())
    g2 = draft_to_graph(draft, revision_batch=batch_id)
    violations = validate_graph(g2)
    if violations:
        return ApproveOutcome(ok=False, card_ids=[], violations=violations)

    insert_batch(
        g2,
        named_graph=named_graph_iri,
        graphdb_url=graphdb_url,
        repo=graphdb_repo,
        user=graphdb_user,
        password=graphdb_pass,
    )

    pack = project_graph_to_cards(g2, draft, named_graphs=[named_graph_iri])
    card_ids: List[UUID] = []
    try:
        card_ids = await insert_cards_and_edges_batch(
            pool,
            creates=pack.creates,
            edge_indices=pack.edge_indices,
            actor="operator",
        )
    except Exception:
        logger.exception("approve_memory_graph_draft postgres_failed batch=%s", batch_id)
        try:
            compensate_batch(
                batch_id,
                graphdb_url=graphdb_url,
                repo=graphdb_repo,
                user=graphdb_user,
                password=graphdb_pass,
            )
        except Exception:
            logger.exception("compensate_batch_failed batch=%s", batch_id)
        raise

    return ApproveOutcome(ok=True, card_ids=card_ids, violations=[])


def preview_validate_only(draft: SuggestDraftV1) -> tuple[bool, List[str], Dict[str, Any]]:
    """Validate RDF + SHACL without persistence; return preview card shells."""
    g = draft_to_graph(draft)
    violations = validate_graph(g)
    pack = project_graph_to_cards(g, draft)
    preview = {
        "card_count": len(pack.creates),
        "edge_count": len(pack.edge_indices),
        "titles": [c.title for c in pack.creates],
    }
    return (len(violations) == 0, violations, preview)
