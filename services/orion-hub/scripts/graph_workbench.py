"""Bounded, read-only snapshots for Gephi Lite. No graph mutations or layouts."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID
from xml.etree import ElementTree as ET

from orion.memory.crystallization.repository import normalize_crystallization_id

MAX_NODES = 1000
MAX_EDGES = 4000
SOURCE_LABELS = {
    "worldview": "Worldview",
    "substrate": "Substrate concepts",
    "crystallizations": "Memory crystallizations",
}
GEXF = "http://www.gexf.net/1.2draft"
ET.register_namespace("", GEXF)
_INVALID_XML = re.compile(r"[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def scalar(value: Any) -> str | bool | int | float:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return float(value) if isinstance(value, Decimal) else value
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, (datetime, date)):
        value = value.isoformat()
    return _INVALID_XML.sub("", str(value))


def caption(props: dict, fallback: str) -> str:
    return str(next((props[k] for k in ("label", "name", "subject", "title", "text", "node_id", "id") if props.get(k)), fallback))[:120]


@dataclass
class Snapshot:
    source: str
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: dict[str, dict] = field(default_factory=dict)
    truncated: bool = False
    seed: str = ""

    def node(self, key: str, properties: dict, limit: int) -> bool:
        if key not in self.nodes and len(self.nodes) >= limit:
            self.truncated = True
            return False
        self.nodes[key] = {k: scalar(v) for k, v in properties.items() if v is not None}
        return True

    def edge(self, key: str, source: str, target: str, properties: dict) -> None:
        if source not in self.nodes or target not in self.nodes:
            self.truncated = True
            return
        if key not in self.edges and len(self.edges) >= MAX_EDGES:
            self.truncated = True
            return
        self.edges[key] = {"source": source, "target": target, "properties": {
            k: scalar(v) for k, v in properties.items() if v is not None
        }}

    def gexf(self) -> bytes:
        root = ET.Element(f"{{{GEXF}}}gexf", version="1.2")
        meta = ET.SubElement(root, "meta", lastmodifieddate=date.today().isoformat())
        ET.SubElement(meta, "creator").text = "Orion Hub"
        ET.SubElement(meta, "description").text = json.dumps({
            "source": self.source, "seed": self.seed, "truncated": self.truncated,
            "exported_at": datetime.now().astimezone().isoformat(),
            "nodes": len(self.nodes), "edges": len(self.edges),
        })
        graph = ET.SubElement(root, "graph", mode="static", defaultedgetype="directed")
        for cls, records in (("node", list(self.nodes.values())), ("edge", [e["properties"] for e in self.edges.values()])):
            attrs = ET.SubElement(graph, "attributes", {"class": cls})
            types = {}
            for key in sorted({k for record in records for k in record}):
                values = [r[key] for r in records if key in r]
                kind = "boolean" if all(isinstance(v, bool) for v in values) else (
                    "double" if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values) else "string")
                types[key] = kind
                ET.SubElement(attrs, "attribute", id=key, title=key, type=kind)
            container = ET.SubElement(graph, "nodes" if cls == "node" else "edges")
            for key, record in (self.nodes.items() if cls == "node" else self.edges.items()):
                props = record if cls == "node" else record["properties"]
                label = caption(props, key) if cls == "node" else str(props.get("relation", key))
                kwargs = {"id": key, "label": _INVALID_XML.sub("", label)}
                if cls == "edge":
                    kwargs.update(source=record["source"], target=record["target"])
                element = ET.SubElement(container, cls, kwargs)
                values_element = ET.SubElement(element, "attvalues")
                for name, value in props.items():
                    text = str(value).lower() if types[name] == "boolean" else str(value)
                    ET.SubElement(values_element, "attvalue", {"for": name, "value": text})
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)


class FalkorReader:
    """Use RESP read-only queries with server timeout; preserve URI credentials."""
    def __init__(self, uri: str, graph_name: str):
        import redis
        from redis.commands.graph import Graph

        self.redis = redis.Redis.from_url(uri, decode_responses=True, socket_timeout=4, socket_connect_timeout=2)
        self.graph = Graph(self.redis, graph_name)

    def query(self, query: str, params: dict | None = None) -> list[dict]:
        from redis.commands.graph.query_result import QueryResult
        from orion.graph.falkor_client import _rows_from_query_result

        # Graph.query silently retries unknown RO_QUERY as writable GRAPH.QUERY.
        # Issue RO_QUERY explicitly so that fallback is impossible.
        prefixed = self.graph._build_params_header(params) + query
        response = self.redis.execute_command("GRAPH.RO_QUERY", self.graph.name, prefixed, "--compact", "timeout", 2000)
        result = QueryResult(self.graph, response)
        return _rows_from_query_result(result.header, result.result_set)

    def close(self):
        self.redis.close()


def graph_search(reader: FalkorReader, q: str) -> list[dict]:
    return reader.query("""
        MATCH (n)
        WHERE toLower(coalesce(n.label, n.name, n.title, n.node_id, '')) CONTAINS toLower($q)
        RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties
        ORDER BY id(n) LIMIT 20
    """, {"q": q})


def graph_snapshot(reader: FalkorReader, source: str, seed: str, depth: int, limit: int) -> Snapshot:
    result = Snapshot(source=source, seed=seed)
    if seed:
        rows = reader.query("MATCH (n) WHERE id(n) = $id RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties", {"id": int(seed)})
    elif source == "substrate":
        rows = reader.query("MATCH (n:Concept) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties ORDER BY n.activation DESC, id(n) LIMIT 1")
    else:
        rows = reader.query("MATCH (n) RETURN id(n) AS id, labels(n) AS labels, properties(n) AS properties ORDER BY id(n) LIMIT $limit", {"limit": limit + 1})
    if not rows:
        raise LookupError("No matching nodes in this source.")
    result.truncated = len(rows) > limit
    for row in rows[:limit]:
        result.node(str(row["id"]), {**row["properties"], "node_labels": row["labels"]}, limit)
    if seed or source == "substrate":
        result.seed = str(rows[0]["id"])
        frontier = list(map(int, result.nodes))
        for _ in range(depth):
            edges = reader.query("""
                MATCH (a)-[r]-(b) WHERE id(a) IN $ids
                RETURN id(r) AS id, id(startNode(r)) AS source, id(endNode(r)) AS target,
                       type(r) AS relation, properties(r) AS properties,
                       id(b) AS neighbor, labels(b) AS labels, properties(b) AS neighbor_properties
                ORDER BY id(r) LIMIT $limit
            """, {"ids": frontier, "limit": MAX_EDGES + 1})
            result.truncated |= len(edges) > MAX_EDGES
            new = []
            for row in edges[:MAX_EDGES]:
                key = str(row["neighbor"])
                if key not in result.nodes and result.node(key, {**row["neighbor_properties"], "node_labels": row["labels"]}, limit):
                    new.append(int(key))
                result.edge(str(row["id"]), str(row["source"]), str(row["target"]), {**row["properties"], "relation": row["relation"]})
            frontier = new
            if not frontier:
                break
    # Include every edge inside the selected node set (parallel edges preserved).
    edges = reader.query("""
        MATCH (a)-[r]->(b) WHERE id(a) IN $ids AND id(b) IN $ids
        RETURN id(r) AS id, id(a) AS source, id(b) AS target,
               type(r) AS relation, properties(r) AS properties
        ORDER BY id(r) LIMIT $limit
    """, {"ids": list(map(int, result.nodes)), "limit": MAX_EDGES + 1})
    result.truncated |= len(edges) > MAX_EDGES
    for row in edges[:MAX_EDGES]:
        result.edge(str(row["id"]), str(row["source"]), str(row["target"]), {**row["properties"], "relation": row["relation"]})
    return result


_VISIBLE = "COALESCE(governance->>'sensitivity', 'private') IN ('public', 'private')"


async def crystal_search(pool, q: str) -> list[dict]:
    async with pool.acquire() as conn, conn.transaction(readonly=True):
        return [dict(r) for r in await conn.fetch(f"""
            SELECT crystallization_id AS id, subject AS label, kind, status
            FROM memory_crystallizations WHERE {_VISIBLE} AND (subject ILIKE $1 OR summary ILIKE $1)
            ORDER BY updated_at DESC, crystallization_id LIMIT 20
        """, "%" + q + "%", timeout=3)]


async def crystal_snapshot(pool, seed: str, depth: int, limit: int, lineage: bool) -> Snapshot:
    result = Snapshot(source="crystallizations", seed=seed)
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        if seed:
            cid = UUID(normalize_crystallization_id(seed))
            rows = await conn.fetch(f"SELECT * FROM memory_crystallizations WHERE crystallization_id=$1 AND {_VISIBLE}", cid, timeout=3)
        else:
            rows = await conn.fetch(f"SELECT * FROM memory_crystallizations WHERE status='active' AND {_VISIBLE} ORDER BY updated_at DESC, crystallization_id LIMIT $1", min(50, limit) + 1, timeout=3)
            result.truncated = len(rows) > min(50, limit)
            rows = rows[:min(50, limit)]
        if not rows:
            raise LookupError("No matching public/private crystallizations.")
        ids = []
        for row in rows:
            ids.append(row["crystallization_id"])
            result.node(str(row["crystallization_id"]), {**dict(row), "node_labels": ["Crystallization"]}, limit)
        frontier = ids[:]
        for _ in range(depth if seed else 0):
            neighbors = await conn.fetch(f"""
                SELECT DISTINCT c.* FROM memory_crystallizations c
                JOIN memory_crystallization_links l ON c.crystallization_id IN (l.from_crystallization_id, l.to_crystallization_id)
                WHERE (l.from_crystallization_id=ANY($1::uuid[]) OR l.to_crystallization_id=ANY($1::uuid[]))
                  AND NOT c.crystallization_id=ANY($2::uuid[]) AND {_VISIBLE}
                ORDER BY c.crystallization_id LIMIT $3
            """, frontier, ids, limit - len(ids) + 1, timeout=3)
            frontier = []
            for row in neighbors:
                key = row["crystallization_id"]
                if result.node(str(key), {**dict(row), "node_labels": ["Crystallization"]}, limit):
                    ids.append(key)
                    frontier.append(key)
            if not frontier:
                break
        links = await conn.fetch("""SELECT * FROM memory_crystallization_links
            WHERE from_crystallization_id=ANY($1::uuid[]) AND to_crystallization_id=ANY($1::uuid[])
            ORDER BY link_id LIMIT $2""", ids, MAX_EDGES + 1, timeout=3)
        result.truncated |= len(links) > MAX_EDGES
        for row in links[:MAX_EDGES]:
            result.edge(str(row["link_id"]), str(row["from_crystallization_id"]), str(row["to_crystallization_id"]), dict(row))
        if lineage:
            # projector.py persists these references in the canonical row's JSONB.
            # The similarly named projection_refs TABLE has no runtime producer.
            for cid in ids:
                raw = result.nodes[str(cid)].get("projection_refs", "{}")
                refs = json.loads(raw) if isinstance(raw, str) else raw
                for kind, external_ids in refs.items():
                    if not isinstance(external_ids, list):
                        continue
                    for external_id in external_ids[:limit + 1]:
                        key = "projection:" + json.dumps([kind, str(external_id)])
                        result.node(key, {"external_id": external_id, "label": external_id,
                                         "projection_kind": kind, "node_labels": ["ProjectionRef", kind]}, limit)
                        result.edge(str(cid) + ":" + key, str(cid), key, {"relation": "has_projection_ref"})
                    result.truncated |= len(external_ids) > limit
            sources = await conn.fetch("""SELECT * FROM memory_crystallization_sources
                WHERE crystallization_id=ANY($1::uuid[]) ORDER BY source_ref_id LIMIT $2""", ids, limit + 1, timeout=3)
            result.truncated |= len(sources) > limit
            for row in sources[:limit]:
                key = "source:" + json.dumps([row["source_kind"], row["source_id"]])
                result.node(key, {"source_id": row["source_id"], "label": row["source_id"],
                                 "source_kind": row["source_kind"], "node_labels": ["Source", row["source_kind"]]}, limit)
                # Shared evidence has one node; each crystal's excerpt/strength
                # belongs on its own provenance edge and cannot overwrite another.
                result.edge("source:" + str(row["source_ref_id"]), str(row["crystallization_id"]), key, {**dict(row), "relation": "has_source"})
    return result
