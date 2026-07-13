from __future__ import annotations

import asyncio
import contextvars
import logging
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import asyncpg

from app.crystallization_ids import validate_crystallization_id
from app.store import neighborhood as pg_neighborhood
from app.store import upsert_episode as pg_upsert_episode

logger = logging.getLogger(__name__)

# Single shared group_id for every node/edge this adapter writes via graphiti-core.
# Graphiti.search()'s group_ids filter is optional/unset on the read side today, so this
# value only needs to be *consistent* across writers, not semantically meaningful yet.
# Do not build a multi-tenant group_id scheme on top of this without a concrete read-side
# need -- that would be an unrequested abstraction (see CLAUDE.md "no keyword cathedrals").
ORION_GROUP_ID = "orion"

# Process-local guard so index/constraint bootstrap only runs once per adapter process
# even if ensure_graphiti_indices() is invoked from more than one call site.
_indices_ready = False

# Process-local cache for the search() driver/client/Graphiti stack, keyed by the exact
# constructor inputs. In practice falkordb_uri/graph_name/embed_url all come straight from
# the process-wide `settings` singleton (see app/main.py's /v1/search route), so they are
# constant for the life of the process -- but keying by value (instead of a bare global)
# means a value change (e.g. across tests with different settings) rebuilds instead of
# silently returning a stale stack. Mirrors the _indices_ready lazy-init-once pattern above.
_search_stack_cache: dict[tuple[str, str, str], tuple[Any, Any, Any]] = {}

# Tracks whether _OrionEmbedderClient.create() produced a real vector *for the current
# search() call*, scoped via a ContextVar rather than instance state on the embedder object.
# The embedder instance is now cached and reused across /v1/search requests (see
# _get_search_stack) -- a plain `self.used` instance attribute would let one request's
# result leak into a concurrent request's trace (both share the same embedder object).
# asyncio gives each request-handling Task its own copy of the current context, so
# concurrent requests can't see each other's writes here even though they share one
# _OrionEmbedderClient instance.
_embed_used_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_embed_used_ctx", default=False
)


def _parse_falkordb_uri(uri: str) -> tuple[str, int]:
    parsed = urlparse(uri or "redis://localhost:6379")
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    return host, port


def _falkor_driver(falkordb_uri: str, graph_name: str):
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    host, port = _parse_falkordb_uri(falkordb_uri)
    return FalkorDriver(host=host, port=port, database=graph_name)


async def ensure_graphiti_indices(falkordb_uri: str, graph_name: str) -> None:
    """Idempotent bootstrap for graphiti-core's RELATES_TO fulltext + range indices.

    FalkorDB's `CREATE FULLTEXT INDEX` has no `IF NOT EXISTS` guard (unlike Neo4j), so a
    second call against an already-indexed graph raises "already indexed" --
    `FalkorDriver.execute_query` already catches that substring internally and returns
    None instead of raising, but we still wrap broadly here: a fresh deployment target
    won't have the indices yet, an already-bootstrapped host (like this one) will, and
    either way a bootstrap hiccup must not crash the adapter. Mirrors the
    GRAPHITI_AUTO_APPLY_SCHEMA / apply_graphiti_schema() pattern used for the Postgres
    projection schema in app/store.py + app/main.py's lifespan.
    """
    global _indices_ready
    if _indices_ready:
        return
    from graphiti_core.utils.maintenance.graph_data_operations import (
        build_indices_and_constraints,
    )

    driver = _falkor_driver(falkordb_uri, graph_name)
    try:
        await build_indices_and_constraints(driver)
        _indices_ready = True
        logger.info("graphiti_indices_bootstrapped graph=%s", graph_name)
    except Exception as exc:
        if "already indexed" in str(exc).lower() or "already exists" in str(exc).lower():
            logger.info("graphiti_indices_already_exist graph=%s error=%s", graph_name, exc)
            _indices_ready = True
        else:
            logger.warning("graphiti_indices_bootstrap_failed graph=%s error=%s", graph_name, exc)


def _extract_crystallization_ids(results: Any) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()

    def _maybe_add(value: Any) -> None:
        if not value:
            return
        cid = str(value).strip()
        if cid and cid not in seen:
            seen.add(cid)
            ids.append(cid)

    items = results if isinstance(results, list) else getattr(results, "edges", None) or []
    for item in items:
        if isinstance(item, dict):
            _maybe_add(item.get("crystallization_id"))
            node = item.get("node") or item.get("entity") or {}
            if isinstance(node, dict):
                _maybe_add(node.get("crystallization_id"))
                node_id = node.get("id") or node.get("uuid") or ""
                if isinstance(node_id, str):
                    if node_id.startswith("gent_"):
                        _maybe_add(node_id.removeprefix("gent_"))
                    elif node_id.startswith("gep_"):
                        _maybe_add(node_id.removeprefix("gep_"))
            continue
        # Real graphiti-core search() results are EntityEdge objects (graphiti_core/edges.py),
        # which carry source_node_uuid/target_node_uuid as plain strings -- there is no
        # nested source_node/target_node object with its own .crystallization_id attribute.
        # Both endpoints of a fact edge are entities we ingested, so both are candidates.
        for node_id in (
            getattr(item, "source_node_uuid", None),
            getattr(item, "target_node_uuid", None),
        ):
            if isinstance(node_id, str):
                if node_id.startswith("gent_"):
                    _maybe_add(node_id.removeprefix("gent_"))
                elif node_id.startswith("gep_"):
                    _maybe_add(node_id.removeprefix("gep_"))
    return ids


async def _embed_query(query: str, embed_url: str) -> list[float] | None:
    if not embed_url.strip():
        return None
    try:
        import httpx
        from orion.schemas.vector.schemas import EmbeddingGenerateV1, EmbeddingResultV1

        req = EmbeddingGenerateV1(
            doc_id="graphiti_search_query",
            text=query,
            embedding_profile="default",
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(embed_url.rstrip("/"), json=req.model_dump(mode="json"))
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return EmbeddingResultV1.model_validate(data).embedding
    except Exception as exc:
        logger.warning("graphiti_search_embed_failed error=%s", exc)
    return None


async def _filter_intimate_crystallization_ids(
    driver: Any,
    crystallization_ids: list[str],
) -> list[str]:
    if not crystallization_ids:
        return crystallization_ids
    try:
        rows, _header, _summary = await driver.execute_query(
            """
            UNWIND $ids AS cid
            MATCH (n)
            WHERE n.crystallization_id = cid AND n.sensitivity = 'intimate'
            RETURN DISTINCT n.crystallization_id AS cid
            """,
            ids=crystallization_ids,
        )
        intimate = {str(row.get("cid") if isinstance(row, dict) else row[0]) for row in (rows or [])}
        return [cid for cid in crystallization_ids if cid not in intimate]
    except Exception as exc:
        logger.warning("graphiti_search_intimate_filter_failed error=%s", exc)
        return crystallization_ids


async def _cast_embedding_to_vecf32(
    driver: Any, *, match_clause: str, property_name: str, uuid: str, embedding: list[float] | None
) -> None:
    """graphiti-core==0.19.0's single-entity EntityNode.save()/EntityEdge.save() queries for
    FalkorDB (get_entity_node_save_query / get_entity_edge_save_query) do `SET n/e =
    $entity_data`/`$edge_data` with a plain Python list for name_embedding/fact_embedding --
    FalkorDB stores that as a Cypher List, not its native Vectorf32 type. Only the separate
    *bulk* save query builders (get_entity_node_save_bulk_query /
    get_entity_edge_save_bulk_query) wrap embeddings in `vecf32(...)`; graphiti-core doesn't
    expose a bulk path off EntityNode/EntityEdge themselves. Without this cast,
    Graphiti.search()'s cosine-similarity queries (`vec.cosineDistance(..., vecf32($search_
    vector))`) raise `Type mismatch: expected Null or Vectorf32 but was List` for every
    embedded node/edge. Verified live against this host's FalkorDB (see
    docs/superpowers/specs/2026-07-13-graphiti-core-backend-activation-spec.md follow-up
    section). Known limitation: this SET runs as a second round-trip after .save(), so there
    is a narrow window where a concurrent search could still hit the List-typed value -- not
    eliminated in this pass because avoiding it means bypassing EntityNode/EntityEdge.save()
    entirely for the embedded-write path, a bigger change than this fix's scope."""
    if not embedding:
        return
    await driver.execute_query(
        f"MATCH {match_clause} SET {property_name} = vecf32($embedding)",
        uuid=uuid,
        embedding=embedding,
    )


async def _lookup_subject(pool: asyncpg.Pool | None, crystallization_id: str) -> str | None:
    """Look up the already-governed subject text for a crystallization from the Postgres
    dual-write projection (graphiti_episodes.subject), so link fact templates can reference
    a real subject instead of a raw id when the link target has already been ingested."""
    if pool is None:
        return None
    try:
        return await pool.fetchval(
            "SELECT subject FROM graphiti_episodes WHERE crystallization_id = $1",
            crystallization_id,
        )
    except Exception as exc:
        logger.warning("graphiti_target_subject_lookup_failed id=%s error=%s", crystallization_id, exc)
        return None


async def _ensure_target_entity_stub(
    driver: Any,
    *,
    target_entity_id: str,
    target_id: str,
    target_subject: str,
    target_sensitivity: str,
) -> None:
    """EntityEdge.save() on FalkorDB MATCHes both endpoints -- it does not create them. If
    the link target hasn't been ingested yet, the edge write would silently no-op without a
    stub node. Only create when missing: EntityNode.save() is a full property replace, and
    clobbering an already-real target's attributes/summary with stub data would destroy it."""
    from graphiti_core.nodes import EntityNode

    rows, _header, _summary = await driver.execute_query(
        "MATCH (n:Entity {uuid: $uuid}) RETURN n.uuid AS uuid",
        uuid=target_entity_id,
    )
    if rows:
        return

    stub = EntityNode(
        uuid=target_entity_id,
        name=target_subject,
        group_id=ORION_GROUP_ID,
        summary="",
        attributes={"crystallization_id": target_id, "sensitivity": target_sensitivity},
    )
    await stub.save(driver)


async def ingest_episode(
    pool: asyncpg.Pool | None,
    *,
    episode_id: str,
    crystallization_id: str,
    kind: str,
    subject: str,
    summary: str,
    status: str,
    metadata: dict[str, Any],
    links: list[dict[str, Any]] | None,
    falkordb_uri: str,
    graph_name: str,
    embed_url: str = "",
) -> dict[str, list[str]]:
    if metadata.get("sensitivity") == "intimate":
        return {"edge_ids": [], "skipped": True, "reason": "intimate_sensitivity"}

    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode

    crystallization_id = validate_crystallization_id(crystallization_id)
    sensitivity = str(metadata.get("sensitivity") or "public")
    driver = _falkor_driver(falkordb_uri, graph_name)
    entity_id = f"gent_{crystallization_id}"
    now = datetime.now(timezone.utc)
    edge_ids: list[str] = []

    # Entity identity: graphiti-core's own read/write queries (including Graphiti.search())
    # match nodes on `.uuid`, not a custom `.id` property, so uuid=entity_id keeps identity
    # stable across re-ingest. `attributes` keys get flattened onto the node as top-level
    # properties by EntityNode.save() on FalkorDB, which is what
    # _filter_intimate_crystallization_ids()'s `n.crystallization_id` / `n.sensitivity`
    # Cypher already expects -- do not change that function.
    #
    # Self-referential RELATES_TO edge for every ingested crystallization, even ones with
    # no links. Graphiti.search() only ever returns edges, never bare nodes -- without this,
    # crystallizations with no CrystallizationLinkV1 links (most of them, in practice) would
    # never be searchable at all. `fact` is a deterministic string template built from
    # already-governed subject/summary fields -- never LLM-generated (hard constraint, see
    # docs/superpowers/specs/2026-07-06-graphiti-rail-activation-design.md Non-goals).
    #
    # The node name embedding and self-edge fact embedding are independent embed-host calls
    # (self_fact doesn't depend on name_embedding), so run them concurrently instead of
    # serially -- halves embed-host round-trip latency on this hot path.
    self_fact = f"{subject}: {summary[:280]}"
    name_embedding, self_fact_embedding = await asyncio.gather(
        _embed_query(subject, embed_url),
        _embed_query(self_fact, embed_url),
    )
    entity_node = EntityNode(
        uuid=entity_id,
        name=subject,
        group_id=ORION_GROUP_ID,
        summary=summary[:500],
        name_embedding=name_embedding,
        attributes={"crystallization_id": crystallization_id, "sensitivity": sensitivity},
        created_at=now,
    )
    await entity_node.save(driver)
    await _cast_embedding_to_vecf32(
        driver,
        match_clause="(n:Entity {uuid: $uuid})",
        property_name="n.name_embedding",
        uuid=entity_id,
        embedding=name_embedding,
    )

    self_edge_id = f"ged_{crystallization_id}"
    self_edge = EntityEdge(
        uuid=self_edge_id,
        group_id=ORION_GROUP_ID,
        source_node_uuid=entity_id,
        target_node_uuid=entity_id,
        name="describes",
        fact=self_fact,
        fact_embedding=self_fact_embedding,
        episodes=[episode_id],
        created_at=now,
    )
    await self_edge.save(driver)
    await _cast_embedding_to_vecf32(
        driver,
        match_clause="()-[e:RELATES_TO {uuid: $uuid}]->()",
        property_name="e.fact_embedding",
        uuid=self_edge_id,
        embedding=self_fact_embedding,
    )
    edge_ids.append(self_edge_id)

    for link in links or []:
        target_id = validate_crystallization_id(str(link["target_crystallization_id"]))
        relation = str(link["relation"])
        confidence = float(link.get("confidence", 0.5))
        target_entity_id = f"gent_{target_id}"
        target_sensitivity = str(link.get("sensitivity") or "public")

        target_subject = (
            await _lookup_subject(pool, target_id) or f"related crystallization {target_id}"
        )
        await _ensure_target_entity_stub(
            driver,
            target_entity_id=target_entity_id,
            target_id=target_id,
            target_subject=target_subject,
            target_sensitivity=target_sensitivity,
        )

        link_fact = f"{subject} {relation} {target_subject}"
        link_fact_embedding = await _embed_query(link_fact, embed_url)
        link_edge_id = f"ged_{crystallization_id}_{target_id}_{relation}"
        link_edge = EntityEdge(
            uuid=link_edge_id,
            group_id=ORION_GROUP_ID,
            source_node_uuid=entity_id,
            target_node_uuid=target_entity_id,
            name=relation,
            fact=link_fact,
            fact_embedding=link_fact_embedding,
            episodes=[episode_id],
            created_at=now,
            # attributes flattens onto the edge the same way EntityNode.attributes flattens
            # onto a node (see entity_node above) -- preserves the confidence field the old
            # raw-Cypher RELATED edge stored as r.confidence.
            attributes={"confidence": confidence},
        )
        await link_edge.save(driver)
        await _cast_embedding_to_vecf32(
            driver,
            match_clause="()-[e:RELATES_TO {uuid: $uuid}]->()",
            property_name="e.fact_embedding",
            uuid=link_edge_id,
            embedding=link_fact_embedding,
        )
        edge_ids.append(link_edge_id)

    # Dual-write Postgres so neighborhood/search callers stay consistent with Phase B.
    if pool is not None:
        pg_edge_ids = await pg_upsert_episode(
            pool,
            episode_id=episode_id,
            crystallization_id=crystallization_id,
            kind=kind,
            subject=subject,
            summary=summary,
            status=status,
            metadata=metadata,
            links=links,
        )
        return {"edge_ids": pg_edge_ids or edge_ids}

    return {"edge_ids": edge_ids}


async def get_neighborhood(
    pool: asyncpg.Pool | None, crystallization_id: str, *, depth: int = 1
) -> dict[str, Any]:
    if pool is None:
        raise RuntimeError("store_unavailable")
    return await pg_neighborhood(pool, crystallization_id, depth=depth)


def _no_op_llm_client() -> Any:
    """graphiti-core's Graphiti() eagerly builds an OpenAIClient() for llm_client if one
    isn't supplied, which raises at import time without OPENAI_API_KEY. Orion never calls
    add_episode()/entity-extraction through graphiti-core (nodes/edges are written via
    explicit EntityNode/EntityEdge payloads in ingest_episode above, never LLM-derived), so
    search() never actually invokes the LLM client -- this stub only exists to satisfy
    Graphiti's constructor.
    """
    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.llm_client.config import LLMConfig

    class _NullLLMClient(LLMClient):
        def __init__(self) -> None:
            super().__init__(config=LLMConfig(), cache=False)

        async def _generate_response(self, messages, response_model=None, max_tokens=8192, model_size=None):  # type: ignore[override]
            return {}

    return _NullLLMClient()


def _no_op_cross_encoder() -> Any:
    """RRF (Orion's search config) doesn't invoke the cross-encoder reranker, but Graphiti()
    still eagerly builds an OpenAIRerankerClient() by default without one supplied. Identity
    stub only; never exercised by the reciprocal-rank-fusion search path used here.
    """
    from graphiti_core.cross_encoder.client import CrossEncoderClient

    class _NullCrossEncoder(CrossEncoderClient):
        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:  # type: ignore[override]
            return [(p, 0.0) for p in passages]

    return _NullCrossEncoder()


def _orion_embedder_client(embed_url: str) -> Any:
    """Bridges graphiti-core's cosine-similarity search rail to Orion's own embedding host
    (CRYSTALLIZER_EMBED_HOST_URL) instead of the library's default OpenAIEmbedder.

    Records whether a call actually produced a vector via `_embed_used_ctx` (a per-call
    ContextVar, not instance state) so callers can report an honest embed_used trace field
    without a second, redundant HTTP call to the embed host for the same query text. This
    instance is cached and reused across /v1/search requests (see `_get_search_stack`), so
    tracking success as instance state (`self.used`) would leak one request's result into a
    concurrent request's trace; the ContextVar is scoped per request-handling asyncio Task
    instead.
    """
    from graphiti_core.embedder.client import EmbedderClient

    class _OrionEmbedderClient(EmbedderClient):
        async def create(self, input_data) -> list[float]:  # type: ignore[override]
            text = input_data[0] if isinstance(input_data, list) and input_data else str(input_data)
            vector = await _embed_query(str(text), embed_url)
            if vector:
                _embed_used_ctx.set(True)
            return vector or []

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:  # type: ignore[override]
            return [await self.create(item) for item in input_data_list]

    return _OrionEmbedderClient()


def _get_search_stack(falkordb_uri: str, graph_name: str, embed_url: str) -> tuple[Any, Any, Any]:
    """Build (or reuse) the FalkorDriver + embedder + Graphiti instance used by search().

    Building this stack fresh on every /v1/search call was pure per-request overhead: a new
    FalkorDriver (redis-protocol client, graphiti_core/driver/falkordb_driver.py -- thin
    wrapper over falkordb.asyncio.FalkorDB, no eager connection-pool exhaustion or
    documented long-lived-reuse hazard) plus a new Graphiti() instance were constructed and
    thrown away after a single request. Reuse is the intended usage for FalkorDriver; this
    just stops rebuilding it (and the no-op llm/cross-encoder stubs) every call.
    """
    key = (falkordb_uri, graph_name, embed_url)
    cached = _search_stack_cache.get(key)
    if cached is not None:
        return cached

    from graphiti_core import Graphiti

    driver = _falkor_driver(falkordb_uri, graph_name)
    embedder = _orion_embedder_client(embed_url)
    graphiti = Graphiti(
        graph_driver=driver,
        llm_client=_no_op_llm_client(),
        embedder=embedder,
        cross_encoder=_no_op_cross_encoder(),
    )
    _search_stack_cache.clear()  # single-entry process cache; drop any stale key first
    _search_stack_cache[key] = (driver, embedder, graphiti)
    logger.info("graphiti_search_stack_built falkordb_uri=%s graph=%s", falkordb_uri, graph_name)
    return driver, embedder, graphiti


async def search(
    query: str,
    *,
    seed_crystallization_id: str,
    limit: int,
    embed_url: str,
    falkordb_uri: str,
    graph_name: str,
) -> dict[str, Any]:
    driver, embedder, graphiti = _get_search_stack(falkordb_uri, graph_name, embed_url)

    # embedder is cached/reused across requests (see _get_search_stack); _embed_used_ctx is
    # a per-Task ContextVar (see its definition above) so this reset+read pair can't race
    # with or leak into a concurrent request's trace even though they share one embedder.
    ctx_token = _embed_used_ctx.set(False)
    try:
        results = await graphiti.search(query=query, num_results=limit)
        embed_used = _embed_used_ctx.get()
    finally:
        _embed_used_ctx.reset(ctx_token)

    crystallization_ids = _extract_crystallization_ids(results)
    crystallization_ids = await _filter_intimate_crystallization_ids(driver, crystallization_ids)
    if seed_crystallization_id and seed_crystallization_id not in crystallization_ids:
        crystallization_ids.insert(0, seed_crystallization_id)
    return {
        "crystallization_ids": crystallization_ids[:limit],
        "trace": {
            "backend": "graphiti_core",
            "rails": ["vector", "graph"],
            "query": query,
            "embed_used": embed_used,
            "result_count": len(crystallization_ids),
            "raw_type": type(results).__name__,
        },
    }
