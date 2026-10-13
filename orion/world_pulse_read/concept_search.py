"""Reading claim objects by meaning: existing atlas concepts close to a read's own concepts.

The exact path (``assertions.object_candidates``) offers an existing concept only when its
label appears verbatim in the read. Atlas labels are chat-topic phrases ("Health Crisis
Burden") and earlier readings' concept names, so a news page almost never contains one:
on the six reads with retained text before this patch, exact matching offered objects for
two. This module adds the concepts nearest each subject by embedding, through the same
search-by-meaning plumbing the introspect tools use (orion/introspect/semantic_index.py):

- Index: every non-fenced atlas concept, embedded once as ``label`` or
  ``label: definition`` by vector-host (bge-large-en-v1.5) and upserted through
  orion-vector-writer into the Chroma collection ``orion_atlas_concepts``. A pass embeds
  only concepts the index lacks or holds stale text for.
- Query: each subject's own ``label: definition`` is embedded and the nearest concepts are
  read back. Chroma is an index, never the record: every hit is re-read from the store and
  re-gated (stored concept/entity, not a fenced memory node, not one of the read's own
  concepts, not already offered by label).

Offered is not accepted. ``plan_claims`` is unchanged: a link is accepted only when the
quote is verbatim in the retained text AND names the object by its stored label. An object
offered here has a label the text does not contain (otherwise the exact path offered it),
so a claim to it is journalled ``proposed`` (reason ``quote_not_about_endpoints``) with no
link: visible for review, never in the graph by itself.

``OBJECT_MIN_SIMILARITY`` was set from live data (2026-10-13 replay, PR report
docs/superpowers/pr-reports/2026-10-13-reading-claim-semantic-objects-pr.md): over 6,080
(subject, atlas concept) pairs from the six reads, cosine rests at a median of 0.531
(p90 0.605, p95 0.630, p99 0.693); bge never rests near 0. 0.65 sits between p95 and p99:
a read whose subjects have no real neighbours (a 404 page) gets 2-3 offers, a read in a
dense atlas region (GPU hardware) fills the cap.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import httpx

from orion.introspect.semantic_index import IndexPass, SearchConfig, embed, index_docs, nearest
from orion.substrate.reconcile import is_identity_fenced

from .assertions import _SEMANTIC_KINDS, MAX_OBJECTS, ClaimContextV1, EndpointV1

ATLAS_CONCEPT_COLLECTION = "orion_atlas_concepts"
OBJECT_MIN_SIMILARITY = 0.65
# Hits read back per subject before re-gating. The read's own concepts and exact-label
# objects are near their own subjects by construction, so ask for the cap plus room for them.
NEAREST_PER_SUBJECT = 2 * MAX_OBJECTS
# Index the whole concept atlas (1,016 concepts live on 2026-10-13), not the
# region-rank slice the exact path scans.
INDEX_SCAN_LIMIT = 5000
# Embeds per index pass (~70 ms each on vector-host): a first build of ~1,000 concepts
# takes five passes; after that a pass embeds only new or edited concepts.
INDEX_BATCH = 200
DOC_PREFIX = "atlas-concept"
HTTP_TIMEOUT_SEC = 5.0


def concept_doc(node: Any) -> str:
    """The text a concept is embedded as: its label, plus its definition when it has one."""
    label = str(getattr(node, "label", "") or "").strip()
    definition = str(getattr(node, "definition", None) or "").strip()
    return f"{label}: {definition}" if label and definition else label


def index_rows(store: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """(node_id, text, meta) for every concept the index should hold."""
    region = store.read_concept_region(limit_nodes=INDEX_SCAN_LIMIT, limit_edges=1)
    rows = []
    for node in getattr(region, "nodes", None) or []:
        if node.node_kind not in _SEMANTIC_KINDS or is_identity_fenced(node):
            # A memory referent never becomes a reading object; keep it out of the index too.
            continue
        text = concept_doc(node)
        if text:
            rows.append((node.node_id, text, {"node_id": node.node_id, "kind": node.node_kind}))
    return rows


async def index_atlas_concepts(
    store: Any, cfg: SearchConfig, *, client: httpx.AsyncClient, bus: Any, source: Any,
) -> IndexPass:
    rows = await asyncio.to_thread(index_rows, store)
    return await index_docs(rows, cfg, client=client, bus=bus, source=source, doc_prefix=DOC_PREFIX)


def _offerable(store: Any, node_id: str) -> EndpointV1 | None:
    node = store.get_node_by_id(node_id)
    if node is None or node.node_kind not in _SEMANTIC_KINDS or is_identity_fenced(node):
        return None
    return EndpointV1(node_id=node.node_id, kind=node.node_kind,
                      label=str(getattr(node, "label", "") or ""), via="meaning")


async def related_objects(
    store: Any, ctx: ClaimContextV1, cfg: SearchConfig, *, client: httpx.AsyncClient,
) -> list[tuple[EndpointV1, float]]:
    """Existing concepts nearest this read's subjects, best first, at or above the floor,
    up to the room left under MAX_OBJECTS. Raises SearchUnavailableError when the embedder
    or index cannot answer (the caller keeps the exact-label objects)."""
    room = MAX_OBJECTS - len(ctx.objects)
    if room <= 0 or not ctx.subjects:
        return []
    docs = []
    for s in ctx.subjects:
        node = store.get_node_by_id(s.node_id)
        doc = concept_doc(node) if node is not None else s.label
        if doc:
            docs.append(doc)

    async def hits(doc: str) -> list[tuple[str, float]]:
        vector, _model = await embed(client, cfg, doc, doc_prefix=DOC_PREFIX)
        return await nearest(client, cfg, vector, NEAREST_PER_SUBJECT)

    best: dict[str, float] = {}
    for scored in await asyncio.gather(*(hits(d) for d in docs)):
        for node_id, sim in scored:
            if sim >= cfg.min_similarity and sim > best.get(node_id, -1.0):
                best[node_id] = sim
    taken = {e.node_id for e in (*ctx.subjects, *ctx.objects)}
    out: list[tuple[EndpointV1, float]] = []
    for node_id, sim in sorted(best.items(), key=lambda kv: (-kv[1], kv[0])):
        if node_id in taken:
            continue
        endpoint = _offerable(store, node_id)
        if endpoint is None:
            continue
        out.append((endpoint, sim))
        taken.add(node_id)
        if len(out) >= room:
            break
    return out


def with_related_objects(ctx: ClaimContextV1, related: list[tuple[EndpointV1, float]]) -> ClaimContextV1:
    """The context with related objects appended after the exact-label ones (never reordered:
    a label hit always outranks a meaning hit), still capped at MAX_OBJECTS."""
    if not related:
        return ctx
    return replace(ctx, objects=(*ctx.objects, *(e for e, _ in related))[:MAX_OBJECTS])
