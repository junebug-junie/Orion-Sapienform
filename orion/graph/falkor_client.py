"""Shared low-level FalkorDB Cypher client.

Extracted from ``orion.substrate.falkor_store`` (2026-07-18) -- this class has
zero substrate-specific coupling (no ``ConceptNodeV1``/``SubstrateEdgeV1``
knowledge), so it belongs in the shared ``orion/graph/`` home the FalkorDB
property-graph doctrine names for adapters
(``docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md``),
not duplicated per consumer. ``orion.substrate.falkor_store`` re-exports these
names so existing imports there continue to work unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import urlparse


class FalkorGraphClient(Protocol):
    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any: ...


class RecordingFalkorClient:
    """Test double that records Cypher and optionally returns scripted rows."""

    def __init__(
        self,
        *,
        hydrate_node_rows: list[dict[str, Any]] | None = None,
        hydrate_edge_rows: list[dict[str, Any]] | None = None,
        hydrate_legacy_node_rows: list[dict[str, Any]] | None = None,
        hydrate_legacy_edge_rows: list[dict[str, Any]] | None = None,
        hydrate_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        # hydrate_rows is a compatibility alias for hydrate_node_rows.
        if hydrate_rows is not None and hydrate_node_rows is None:
            hydrate_node_rows = hydrate_rows
        self.calls: list[tuple[str, dict[str, Any] | None]] = []
        self._hydrate_node_rows = list(hydrate_node_rows or [])
        self._hydrate_edge_rows = list(hydrate_edge_rows or [])
        self._hydrate_legacy_node_rows = list(hydrate_legacy_node_rows or [])
        self._hydrate_legacy_edge_rows = list(hydrate_legacy_edge_rows or [])

    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        self.calls.append((cypher, params))
        if "WHERE n.payload_json IS NOT NULL" in cypher:
            return self._hydrate_legacy_node_rows
        if "WHERE e.payload_json IS NOT NULL" in cypher:
            return self._hydrate_legacy_edge_rows
        if "RETURN n.node_id AS node_id" in cypher:
            return self._hydrate_node_rows
        if "RETURN e.edge_id AS edge_id" in cypher:
            return self._hydrate_edge_rows
        return []


class RedisGraphQueryClient:
    """Minimal sync Redis GRAPH.QUERY client for FalkorDB."""

    def __init__(self, *, uri: str, graph_name: str) -> None:
        import redis
        from redis.commands.graph import Graph

        parsed = urlparse(uri or "redis://localhost:6379")
        self._r = redis.Redis(
            host=parsed.hostname or "localhost",
            port=int(parsed.port or 6379),
            db=int((parsed.path or "/0").lstrip("/") or 0),
            decode_responses=True,
        )
        self._graph = Graph(self._r, graph_name)

    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        result = self._graph.query(cypher, params=params)
        # redis-py exposes list-shaped result_set rows and keeps column names on
        # QueryResult.header as [type, name] pairs. Zip to dicts so callers can
        # address fields by name (native multi-column and legacy 2-column alike).
        return _rows_from_query_result(getattr(result, "header", None), result.result_set)


def _header_field_names(header: Any) -> list[str]:
    names: list[str] = []
    if not header:
        return names
    for column in header:
        if isinstance(column, (list, tuple)) and len(column) >= 2:
            left, right = column[0], column[1]
            # redis-py QueryResult: [column_type:int, column_name:str]
            # some raw wire fixtures: [column_name:str, column_type:int]
            if isinstance(left, int) or (isinstance(left, str) and str(left).isdigit()):
                names.append(str(right).split(".")[-1])
            elif isinstance(right, int) or (isinstance(right, str) and str(right).isdigit()):
                names.append(str(left).split(".")[-1])
            else:
                names.append(str(right).split(".")[-1])
        elif isinstance(column, (list, tuple)) and column:
            names.append(str(column[0]).split(".")[-1])
        else:
            names.append(str(column).split(".")[-1])
    return names


def _rows_from_query_result(header: Any, result_set: Any) -> list[dict[str, Any]]:
    if result_set is None:
        return []
    names = _header_field_names(header)
    out: list[dict[str, Any]] = []
    for record in result_set:
        if isinstance(record, dict):
            out.append(record)
        elif isinstance(record, (list, tuple)):
            if names and len(names) == len(record):
                out.append(dict(zip(names, record)))
            else:
                out.append({"_positional": list(record)})
    return out
