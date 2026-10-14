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
- Query: each subject's own ``label: definition`` is embedded, and every indexed concept is
  scored by its best mean-centered cosine to any subject (``rank_centered``; the mean is the
  index's own). Chroma's nearest-neighbour order is raw cosine, which centering does not
  follow, so the whole index (~1,000 vectors) is read with ``all_vectors``. Chroma is an
  index, never the record: every hit is re-read from the store and re-gated (stored
  concept, not a fenced memory node, label at least MIN_LABEL_CHARS, not one of the read's
  own concepts, not already offered by label).

Offered is not accepted. ``plan_claims`` is unchanged: a link is accepted only when the
quote is verbatim in the retained text AND names the object by its stored label. A meaning
object's label is almost always absent from the text (otherwise the exact path offered it),
so a claim to it is journalled ``proposed`` (reason ``quote_not_about_endpoints``) with no
link, visible for review. The one exception is by the same rule: a concept ranked below
the exact path's region scan (REGION_SCAN_LIMIT) whose label IS in the text arrives here,
and a quote naming it is accepted exactly as it would have been from the label path.
Labels shorter than MIN_LABEL_CHARS are never offered (the quote check is a substring
test; "ai" is inside "said").

Why centered (2026-10-14, PR report
docs/superpowers/pr-reports/2026-10-14-reading-link-targets-centered-pr.md): raw bge cosine
has no calm state (unrelated concept pairs rest at 0.53), so the old raw floor of 0.65
offered "gpu_section_hub_page" and "message matching / capitulation" to a NASA expo read.
Centered against the index mean, unrelated pairs rest at 0.05 and the floor
(OBJECT_MIN_CENTERED_SIMILARITY) cuts those while keeping the NASA and GPU neighbours.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any

import httpx

from orion.introspect.semantic_index import (
    IndexPass,
    SearchConfig,
    SearchUnavailableError,
    all_vectors,
    embed,
    index_docs,
)
from orion.substrate.reconcile import is_identity_fenced

from .assertions import _SEMANTIC_KINDS, MAX_OBJECTS, MIN_LABEL_CHARS, ClaimContextV1, EndpointV1

logger = logging.getLogger(__name__)

ATLAS_CONCEPT_COLLECTION = "orion_atlas_concepts"
# Centered-cosine floor for meaning objects (centered_matrix): a concept is offered when its
# best centered cosine to any of the read's subjects is at least this. Set from live data
# (2026-10-14 replay, services/orion-hub/evals/run_reading_link_targets_replay.py) on 625
# concept pairs over the 1,022 indexed atlas concepts, centered against the index mean:
# two concepts minted by the same read (related by construction) p10 0.18 / p50 0.37;
# a concept vs one from a different read p50 0.05 / p90 0.24 / p95 0.30 / p99 0.46.
# 0.35 sits between the unrelated p95 and p99, the same place the retired raw floor (0.65,
# between raw p95 0.63 and p99 0.69) was set, but it now rests near 0 instead of 0.53.
# The value is tied to the index mean it was measured against; re-run the replay after
# the atlas changes shape.
OBJECT_MIN_CENTERED_SIMILARITY = 0.35
# Index the whole concept atlas (1,016 concepts live on 2026-10-13), not the
# region-rank slice the exact path scans.
INDEX_SCAN_LIMIT = 5000
# Embeds per index pass (~70 ms each on vector-host): a first build of ~1,000 concepts
# takes five passes; after that a pass embeds only new or edited concepts.
INDEX_BATCH = 200
DOC_PREFIX = "atlas-concept"
HTTP_TIMEOUT_SEC = 5.0


def centered_matrix(vectors: Any) -> tuple[Any, Any]:
    """(mean, rows minus the mean, renormalized) for an (n, d) array of unit bge vectors.

    bge cosine has no calm state (unrelated text rests near 0.53-0.66); subtracting the
    pool mean (Mu & Viswanath 2018, as orion/world_pulse_read/resonance_similarity.py does
    for reads) moves the unrelated rest point near 0, so one floor means the same thing in
    a dense atlas region and a sparse one."""
    import numpy as np

    raw = np.asarray(vectors, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] == 0:
        raise ValueError("centered_matrix needs a non-empty (n, d) array")
    mu = raw.mean(axis=0)
    cen = raw - mu
    cen /= np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-12)
    return mu, cen


def concept_doc(node: Any) -> str:
    """The text a concept is embedded as: its label, plus its definition when it has one."""
    label = str(getattr(node, "label", "") or "").strip()
    definition = str(getattr(node, "definition", None) or "").strip()
    return f"{label}: {definition}" if label and definition else label


def index_rows(store: Any) -> list[tuple[str, str, dict[str, Any]]]:
    """(node_id, text, meta) for every concept the index should hold."""
    region = store.read_concept_region(limit_nodes=INDEX_SCAN_LIMIT, limit_edges=1)
    nodes = getattr(region, "nodes", None) or []
    if len(nodes) >= INDEX_SCAN_LIMIT:
        logger.warning("reading_concept_index_truncated limit=%d lowest-ranked concepts not indexed",
                       INDEX_SCAN_LIMIT)
    rows = []
    for node in nodes:
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
    label = str(getattr(node, "label", "") or "")
    if len(label.strip()) < MIN_LABEL_CHARS:
        return None
    return EndpointV1(node_id=node.node_id, kind=node.node_kind, label=label, via="meaning")


def _subject_docs(store: Any, ctx: ClaimContextV1) -> list[str]:
    docs = []
    for s in ctx.subjects:
        node = store.get_node_by_id(s.node_id)
        doc = concept_doc(node) if node is not None else s.label
        if doc:
            docs.append(doc)
    return docs


def _regate(store: Any, ranked: list[tuple[str, float]], taken: set[str], room: int) -> list[tuple[EndpointV1, float]]:
    out: list[tuple[EndpointV1, float]] = []
    for node_id, sim in ranked:
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


def rank_centered(
    index: list[tuple[str, list[float]]], subject_vecs: list[list[float]], floor: float,
) -> list[tuple[str, float]]:
    """(doc_id, best centered cosine to any subject) at or above ``floor``, best first.
    Pure: the mean is the index's own (every indexed concept), subjects are centered with it."""
    import numpy as np

    if not index or not subject_vecs:
        return []
    ids = [i for i, _ in index]
    raw = np.asarray([v for _, v in index], dtype=np.float64)
    raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-12)
    mu, cen = centered_matrix(raw)
    q = np.asarray(subject_vecs, dtype=np.float64)
    if q.shape[1] != raw.shape[1]:
        raise ValueError("subject and index vectors differ in dimension")
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    qc = q - mu
    qc /= np.maximum(np.linalg.norm(qc, axis=1, keepdims=True), 1e-12)
    best = (qc @ cen.T).max(axis=0)
    keep = [(ids[k], float(best[k])) for k in range(len(ids)) if best[k] >= floor]
    return sorted(keep, key=lambda kv: (-kv[1], kv[0]))


async def related_objects(
    store: Any, ctx: ClaimContextV1, cfg: SearchConfig, *, client: httpx.AsyncClient,
) -> list[tuple[EndpointV1, float]]:
    """Existing concepts nearest this read's subjects by centered cosine, best first, at or
    above ``cfg.min_similarity`` (OBJECT_MIN_CENTERED_SIMILARITY), up to the room left under
    MAX_OBJECTS. Raises SearchUnavailableError when the embedder or index cannot answer
    (the caller keeps the exact-label objects)."""
    room = MAX_OBJECTS - len(ctx.objects)
    if room <= 0 or not ctx.subjects:
        return []
    # Store reads can be Falkor round trips: keep them off the event loop, like
    # build_claim_context; only the embed and Chroma calls run on it.
    docs = await asyncio.to_thread(_subject_docs, store, ctx)
    if not docs:
        return []
    vectors = [v for v, _model in await asyncio.gather(*(embed(client, cfg, d, doc_prefix=DOC_PREFIX)
                                                          for d in docs))]
    # Centering needs the whole index (its mean, and an order Chroma's raw nearest does not
    # follow): ~1,000 x 1024 floats, about 2 s of /get against a Stage 2 turn of minutes.
    index = await all_vectors(client, cfg)
    try:
        ranked = await asyncio.to_thread(rank_centered, index, vectors, cfg.min_similarity)
    except ValueError as exc:
        raise SearchUnavailableError(f"index vectors unusable: {exc}") from exc
    taken = {e.node_id for e in (*ctx.subjects, *ctx.objects)}
    return await asyncio.to_thread(_regate, store, ranked, taken, room)


def with_related_objects(ctx: ClaimContextV1, related: list[tuple[EndpointV1, float]]) -> ClaimContextV1:
    """The context with related objects appended after the exact-label ones (never reordered:
    a label hit always outranks a meaning hit), still capped at MAX_OBJECTS."""
    if not related:
        return ctx
    return replace(ctx, objects=(*ctx.objects, *(e for e, _ in related))[:MAX_OBJECTS])
