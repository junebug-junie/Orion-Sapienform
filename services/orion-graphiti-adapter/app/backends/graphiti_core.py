from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import asyncpg

from app.crystallization_ids import validate_crystallization_id
from app.store import neighborhood as pg_neighborhood
from app.store import upsert_episode as pg_upsert_episode

logger = logging.getLogger(__name__)


def _parse_falkordb_uri(uri: str) -> tuple[str, int]:
    parsed = urlparse(uri or "redis://localhost:6379")
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    return host, port


def _falkor_driver(falkordb_uri: str, graph_name: str):
    from graphiti_core.driver.falkordb_driver import FalkorDriver

    host, port = _parse_falkordb_uri(falkordb_uri)
    return FalkorDriver(host=host, port=port, database=graph_name)


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
        node = getattr(item, "source_node", None) or getattr(item, "target_node", None)
        if node is not None:
            _maybe_add(getattr(node, "crystallization_id", None))
            node_id = getattr(node, "uuid", None) or getattr(node, "id", None)
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
        rows, _ = await driver.execute_query(
            """
            UNWIND $ids AS cid
            MATCH (n)
            WHERE n.crystallization_id = cid AND n.sensitivity = 'intimate'
            RETURN DISTINCT n.crystallization_id AS cid
            """,
            {"ids": crystallization_ids},
        )
        intimate = {str(row.get("cid") if isinstance(row, dict) else row[0]) for row in (rows or [])}
        return [cid for cid in crystallization_ids if cid not in intimate]
    except Exception as exc:
        logger.warning("graphiti_search_intimate_filter_failed error=%s", exc)
        return crystallization_ids


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
) -> dict[str, list[str]]:
    if metadata.get("sensitivity") == "intimate":
        return {"edge_ids": [], "skipped": True, "reason": "intimate_sensitivity"}

    crystallization_id = validate_crystallization_id(crystallization_id)
    sensitivity = str(metadata.get("sensitivity") or "public")
    driver = _falkor_driver(falkordb_uri, graph_name)
    entity_id = f"gent_{crystallization_id}"
    edge_ids: list[str] = []

    await driver.execute_query(
        """
        MERGE (e:Entity {id: $entity_id, crystallization_id: $crystallization_id})
        SET e.name = $name, e.sensitivity = $sensitivity
        MERGE (ep:Episode {id: $episode_id, crystallization_id: $crystallization_id})
        SET ep.kind = $kind, ep.sensitivity = $sensitivity
        MERGE (e)-[:HAS_EPISODE]->(ep)
        """,
        {
            "entity_id": entity_id,
            "crystallization_id": crystallization_id,
            "episode_id": episode_id,
            "name": subject,
            "kind": kind,
            "sensitivity": sensitivity,
        },
    )
    edge_ids.append(f"ged_{crystallization_id}")

    for link in links or []:
        target_id = validate_crystallization_id(str(link["target_crystallization_id"]))
        relation = str(link["relation"])
        confidence = float(link.get("confidence", 0.5))
        target_entity_id = f"gent_{target_id}"
        await driver.execute_query(
            """
            MERGE (t:Entity {id: $target_entity_id, crystallization_id: $target_id})
            SET t.sensitivity = $target_sensitivity
            MERGE (e:Entity {id: $entity_id})
            MERGE (e)-[r:RELATED]->(t)
            SET r.relation = $relation, r.confidence = $confidence
            """,
            {
                "target_entity_id": target_entity_id,
                "target_id": target_id,
                "target_sensitivity": str(link.get("sensitivity") or "public"),
                "entity_id": entity_id,
                "relation": relation,
                "confidence": confidence,
            },
        )
        edge_ids.append(f"ged_{crystallization_id}_{target_id}_{relation}")

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


async def search(
    query: str,
    *,
    seed_crystallization_id: str,
    limit: int,
    embed_url: str,
    falkordb_uri: str,
    graph_name: str,
) -> dict[str, Any]:
    from graphiti_core import Graphiti

    driver = _falkor_driver(falkordb_uri, graph_name)
    graphiti = Graphiti(graph_driver=driver)
    embedding = await _embed_query(query, embed_url)

    search_kwargs: dict[str, Any] = {"query": query, "num_results": limit}
    if embedding is not None:
        search_kwargs["query_vector"] = embedding

    try:
        results = await graphiti.search(**search_kwargs)
    except TypeError:
        # Older graphiti-core builds may not accept query_vector.
        results = await graphiti.search(query=query, num_results=limit)

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
            "embed_used": embedding is not None,
            "result_count": len(crystallization_ids),
            "raw_type": type(results).__name__,
        },
    }
