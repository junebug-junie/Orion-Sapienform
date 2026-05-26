from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID, uuid4

import requests

from orion.core.storage.memory_cards import insert_cards_and_edges_batch
from orion.memory_graph.dto import SuggestDraftV1
from orion.memory_graph.graphdb import compensate_batch, insert_batch
from orion.memory_graph.json_to_rdf import draft_to_graph
from orion.memory_graph.utterance_text import ensure_draft_utterance_text
from orion.memory_graph.project import project_graph_to_cards
from orion.memory_graph.rdf_target import (
    MemoryGraphSparqlTarget,
    resolve_memory_graph_rdf_target,
)
from orion.memory_graph.validate import validate_graph

logger = logging.getLogger(__name__)


def _sparql_insert_named_graph(
    graph,
    *,
    named_graph: str,
    graph_store_url: str,
    user: str = "",
    password: str = "",
    session: Optional[requests.Session] = None,
) -> None:
    from orion.graph.sparql_client import GraphStoreClient

    ttl = graph.serialize(format="turtle")
    body = ttl if isinstance(ttl, str) else ttl.decode("utf-8")
    client = GraphStoreClient(
        graph_store_url,
        user=user or None,
        password=password or None,
        session=session,
    )
    client.post_graph(body, graph_uri=named_graph, content_type="text/turtle")


def _sparql_compensate_batch(
    batch_id: str,
    *,
    update_url: str,
    user: str = "",
    password: str = "",
    session: Optional[requests.Session] = None,
) -> None:
    from orion.graph.sparql_client import SparqlUpdateClient

    sparql = f"""PREFIX orionmem: <https://orion.local/ns/mem/v2026-05#>
DELETE {{
  ?s ?p ?o .
}}
WHERE {{
  ?s orionmem:revisionBatch "{batch_id}" .
  ?s ?p ?o .
}}
"""
    client = SparqlUpdateClient(
        update_url,
        user=user or None,
        password=password or None,
        session=session,
    )
    client.update(sparql)


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
    """validate → RDF graph store (Fuseki graph-store HTTP + SPARQL update, or legacy GraphDB) → Postgres.

    When ``RDF_STORE_GRAPH_STORE_URL`` and ``RDF_STORE_UPDATE_URL`` are set (``MEMORY_GRAPH_APPROVAL_BACKEND=auto``),
    writes go to Fuseki. Legacy GraphDB is used only when ``MEMORY_GRAPH_APPROVAL_BACKEND=graphdb`` and
    ``GRAPHDB_URL`` is set in the environment (see ``resolve_memory_graph_rdf_target``). Hub must not
    implicitly select GraphDB from stale ``GRAPHDB_URL`` while ``MEMORY_GRAPH_APPROVAL_BACKEND=auto``.
    """
    batch_id = str(uuid4())
    draft = ensure_draft_utterance_text(draft)
    g2 = draft_to_graph(draft, revision_batch=batch_id)
    violations = validate_graph(g2)
    if violations:
        return ApproveOutcome(ok=False, card_ids=[], violations=violations)

    target = resolve_memory_graph_rdf_target()
    if target is None:
        raise ValueError("graph_backend_unconfigured")

    sess = requests.Session()
    if isinstance(target, MemoryGraphSparqlTarget):
        _sparql_insert_named_graph(
            g2,
            named_graph=named_graph_iri,
            graph_store_url=target.graph_store_url,
            user=target.user,
            password=target.password,
            session=sess,
        )
    else:
        insert_batch(
            g2,
            named_graph=named_graph_iri,
            graphdb_url=target.graphdb_url,
            repo=target.repo,
            user=target.user,
            password=target.password,
            session=sess,
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
            if isinstance(target, MemoryGraphSparqlTarget):
                _sparql_compensate_batch(
                    batch_id,
                    update_url=target.update_url,
                    user=target.user,
                    password=target.password,
                    session=sess,
                )
            else:
                compensate_batch(
                    batch_id,
                    graphdb_url=target.graphdb_url,
                    repo=target.repo,
                    user=target.user,
                    password=target.password,
                    session=sess,
                )
        except Exception:
            logger.exception("compensate_batch_failed batch=%s", batch_id)
        raise

    return ApproveOutcome(ok=True, card_ids=card_ids, violations=[])


def preview_validate_only(
    draft: SuggestDraftV1,
    *,
    supplemental_utterance_text: Optional[Mapping[str, str]] = None,
) -> tuple[bool, List[str], Dict[str, Any]]:
    """Validate RDF + SHACL without persistence; return preview card shells."""
    try:
        draft = ensure_draft_utterance_text(draft, supplemental=supplemental_utterance_text)
    except ValueError as exc:
        if str(exc).startswith("utterance_text_missing:"):
            return (False, [str(exc)], {"card_count": 0, "edge_count": 0, "titles": []})
        raise
    g = draft_to_graph(draft)
    violations = validate_graph(g)
    pack = project_graph_to_cards(g, draft)
    preview = {
        "card_count": len(pack.creates),
        "edge_count": len(pack.edge_indices),
        "titles": [c.title for c in pack.creates],
    }
    return (len(violations) == 0, violations, preview)
