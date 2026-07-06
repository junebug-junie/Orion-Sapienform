from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import asyncpg

from app.store import neighborhood as pg_neighborhood

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


async def ingest_episode(
    pool: asyncpg.Pool | None,
    *,
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
    del pool, summary, status, metadata  # prescribed MERGE only; no add_episode / LLM extraction
    driver = _falkor_driver(falkordb_uri, graph_name)
    entity_id = f"gent_{crystallization_id}"
    episode_id = f"gep_{crystallization_id}"
    edge_ids: list[str] = []

    await driver.execute_query(
        (
            f"MERGE (e:Entity {{id: '{entity_id}', crystallization_id: '{crystallization_id}'}}) "
            "SET e.name = $name "
            f"MERGE (ep:Episode {{id: '{episode_id}', crystallization_id: '{crystallization_id}'}}) "
            "SET ep.kind = $kind "
            "MERGE (e)-[:HAS_EPISODE]->(ep)"
        ),
        {"name": subject, "kind": kind},
    )
    edge_ids.append(f"ged_{crystallization_id}")

    for link in links or []:
        target_id = str(link["target_crystallization_id"])
        relation = str(link["relation"])
        confidence = float(link.get("confidence", 0.5))
        target_entity_id = f"gent_{target_id}"
        await driver.execute_query(
            (
                f"MERGE (t:Entity {{id: '{target_entity_id}', crystallization_id: '{target_id}'}}) "
                f"MERGE (e:Entity {{id: '{entity_id}'}}) "
                "MERGE (e)-[r:RELATED]->(t) "
                "SET r.relation = $relation, r.confidence = $confidence"
            ),
            {"relation": relation, "confidence": confidence},
        )
        edge_ids.append(f"ged_{crystallization_id}_{target_id}_{relation}")

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
    del embed_url  # reserved for embed host wiring; search uses graphiti hybrid retrieval only
    from graphiti_core import Graphiti

    driver = _falkor_driver(falkordb_uri, graph_name)
    graphiti = Graphiti(graph_driver=driver)
    results = await graphiti.search(query=query, num_results=limit)
    crystallization_ids = _extract_crystallization_ids(results)
    if seed_crystallization_id and seed_crystallization_id not in crystallization_ids:
        crystallization_ids.insert(0, seed_crystallization_id)
    return {
        "crystallization_ids": crystallization_ids[:limit],
        "trace": {
            "backend": "graphiti_core",
            "rails": ["vector", "graph"],
            "query": query,
            "result_count": len(crystallization_ids),
            "raw_type": type(results).__name__,
        },
    }
