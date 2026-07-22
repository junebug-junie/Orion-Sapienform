from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID, uuid4

from orion.core.storage.memory_cards import insert_cards_and_edges_batch
from orion.memory_graph.dto import CardProjectionDefaultsV1, SuggestDraftV1
from orion.memory_graph.draft_sanitize import sanitize_suggest_draft_dict
from orion.memory_graph.json_to_rdf import draft_to_graph
from orion.memory_graph.utterance_text import ensure_draft_utterance_text
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
    named_graph_iri: str,
    card_defaults: Optional[CardProjectionDefaultsV1] = None,
) -> ApproveOutcome:
    """validate → Postgres (memory_cards + edges).

    The RDF graph store write (Fuseki graph-store HTTP + SPARQL update, or
    legacy GraphDB) was removed 2026-07-22: a live audit found the only
    content that had ever been written through it (16 AffectiveDisposition
    records across 19 batches) traced back to fixture utterance ids
    (turn-id-1..5, matching services/orion-hub/tests/
    test_memory_consolidation_draft_routes.py) with zero matching Postgres
    cards despite this same function writing both atomically -- i.e. test
    pollution from exercising this endpoint directly against live Fuseki,
    not real approved memory. The RDF graph (``g2``) is still built in
    memory (``draft_to_graph``) purely as an intermediate representation for
    ``project_graph_to_cards`` -- it is never persisted externally anymore.
    ``named_graph_iri`` is kept as a parameter (used by ``project_graph_to_cards``
    for evidence/id namespacing) even though nothing writes to it as a real
    RDF named graph any longer.
    """
    batch_id = str(uuid4())
    draft = SuggestDraftV1.model_validate(
        sanitize_suggest_draft_dict(draft.model_dump(mode="json"))
    )
    draft = ensure_draft_utterance_text(draft)
    g2 = draft_to_graph(draft, revision_batch=batch_id)
    violations = validate_graph(g2)
    if violations:
        return ApproveOutcome(ok=False, card_ids=[], violations=violations)

    pack = project_graph_to_cards(g2, draft, named_graphs=[named_graph_iri], card_defaults=card_defaults)
    card_ids: List[UUID] = await insert_cards_and_edges_batch(
        pool,
        creates=pack.creates,
        edge_indices=pack.edge_indices,
        actor="operator",
    )

    return ApproveOutcome(ok=True, card_ids=card_ids, violations=[])


def preview_validate_only(
    draft: SuggestDraftV1,
    *,
    supplemental_utterance_text: Optional[Mapping[str, str]] = None,
) -> tuple[bool, List[str], Dict[str, Any]]:
    """Validate RDF + SHACL without persistence; return preview card shells."""
    from orion.memory_graph.suggest_validate import collect_topical_spine_warnings

    try:
        draft = SuggestDraftV1.model_validate(
            sanitize_suggest_draft_dict(draft.model_dump(mode="json"))
        )
        draft = ensure_draft_utterance_text(draft, supplemental=supplemental_utterance_text)
    except ValueError as exc:
        if str(exc).startswith("utterance_text_missing:"):
            return (False, [str(exc)], {"card_count": 0, "edge_count": 0, "titles": [], "warnings": []})
        raise
    try:
        g = draft_to_graph(draft)
    except ValueError as exc:
        return (False, [str(exc)], {"card_count": 0, "edge_count": 0, "titles": [], "warnings": []})
    violations = validate_graph(g)
    pack = project_graph_to_cards(g, draft)
    corpus_parts: List[str] = []
    if supplemental_utterance_text:
        corpus_parts.extend(str(v) for v in supplemental_utterance_text.values() if str(v).strip())
    if draft.utterance_text_by_id:
        corpus_parts.extend(str(v) for v in draft.utterance_text_by_id.values() if str(v).strip())
    warnings = collect_topical_spine_warnings(
        draft.model_dump(mode="json"),
        utterance_text=" ".join(corpus_parts),
    )
    preview = {
        "card_count": len(pack.creates),
        "edge_count": len(pack.edge_indices),
        "titles": [c.title for c in pack.creates],
        "warnings": warnings,
    }
    return (len(violations) == 0, violations, preview)
