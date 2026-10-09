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
            rows = self._hydrate_legacy_node_rows
        elif "WHERE e.payload_json IS NOT NULL" in cypher:
            rows = self._hydrate_legacy_edge_rows
        elif "RETURN n.node_id AS node_id" in cypher:
            rows = self._hydrate_node_rows
        elif "RETURN e.edge_id AS edge_id" in cypher:
            rows = self._hydrate_edge_rows
        else:
            return []
        if params is not None and "after_id" in params:
            numbered = [dict(row, object_id=row.get("object_id", i)) for i, row in enumerate(rows)]
            return [row for row in numbered if row["object_id"] > params["after_id"]][:params["page_size"]]
        return rows



class RedisGraphQueryClient:
    """Minimal sync Redis GRAPH.QUERY client for FalkorDB.

    ``read_only=True`` sends ``GRAPH.RO_QUERY``, so a bug in a reader cannot
    write -- the engine refuses a mutating clause outright rather than relying
    on the caller only having composed read queries. Verified live 2026-08-29:
    ``CREATE (:Tmp)`` through this path returns "graph.RO_QUERY is to be
    executed only on read-only queries". Same belt-and-braces
    ``orion/curiosity/worldview.py`` applies to Orion's own graph; defaults to
    False so every existing writer is unchanged.

    ONE CAVEAT, because the guarantee is not unconditional. redis-py's
    ``Graph.query`` catches ``ResponseError`` and, on ``"unknown command"``
    with ``read_only=True``, silently RE-ISSUES the query as a writable
    ``GRAPH.QUERY``. That path only triggers against a FalkorDB build with no
    ``GRAPH.RO_QUERY`` at all (this deployment has it -- see the refusal
    above), but on such a build the read-only promise degrades with no signal
    to the caller. Do not treat this flag as an authorization boundary; it is
    defence in depth behind a caller that already only composes reads. The
    real boundary for Orion's own graph is a FalkorDB ACL, not this flag.
    """

    # Class-level default so an instance built with __new__ -- which tests in
    # this repo do, to exercise graph_query without opening a real Redis
    # connection (orion/substrate/tests/test_falkor_store.py) -- still has a
    # defined mode instead of raising AttributeError inside graph_query.
    _read_only: bool = False

    def __init__(
        self,
        *,
        uri: str,
        graph_name: str,
        read_only: bool = False,
        socket_timeout: float | None = None,
        socket_connect_timeout: float | None = None,
    ) -> None:
        """``socket_timeout``/``socket_connect_timeout`` (seconds) are passed
        to ``redis.Redis`` only when set; the default (None) keeps redis-py's
        own default of no timeout, i.e. every existing caller is unchanged.
        A caller on a latency-bounded path (orion-recall's substrate store)
        opts in so a hung FalkorDB cannot pin its thread forever."""
        import redis
        from redis.commands.graph import Graph

        parsed = urlparse(uri or "redis://localhost:6379")
        timeout_kwargs: dict[str, float] = {}
        if socket_timeout is not None:
            timeout_kwargs["socket_timeout"] = float(socket_timeout)
        if socket_connect_timeout is not None:
            timeout_kwargs["socket_connect_timeout"] = float(socket_connect_timeout)
        self._r = redis.Redis(
            host=parsed.hostname or "localhost",
            port=int(parsed.port or 6379),
            db=int((parsed.path or "/0").lstrip("/") or 0),
            decode_responses=True,
            **timeout_kwargs,
        )
        self._graph = Graph(self._r, graph_name)
        self._graph_name = graph_name
        self._read_only = bool(read_only)

    @property
    def read_only(self) -> bool:
        return self._read_only

    def close(self) -> None:
        """Release this client's connection pool. Safe to call twice and on a
        client built with __new__ (no ``_r``)."""
        r = getattr(self, "_r", None)
        if r is not None:
            r.close()

    def graph_query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        """Run Cypher and return rows as name-keyed dicts.

        ``read_only`` routes to ``GRAPH.RO_QUERY`` via redis-py's own
        ``read_only=True`` keyword rather than a hand-issued
        ``execute_command``. That matters for more than tidiness: the library
        path asks for ``--compact`` and decodes the reply through
        ``QueryResult``, so a collection-valued column (``collect(...)``, a list
        comprehension over ``nodes(path)``) comes back as a real list. Issuing
        ``GRAPH.RO_QUERY`` by hand without ``--compact`` returns that same
        column as its *string repr*, which does not fail -- it silently yields
        a list of single characters when a caller iterates it. Caught live
        2026-08-29; both modes now share one parser and one behaviour.
        """
        # The read_only kwarg is passed ONLY when it is actually set. redis-py's
        # Graph.query accepts it, but this codebase also drives graph_query with
        # stand-in Graph objects whose query() takes just (cypher, params) --
        # unconditionally forwarding a third kwarg breaks them for a value that
        # changes nothing. The default path is therefore byte-identical to the
        # call this method made before read-only mode existed.
        if self._read_only:
            result = self._graph.query(cypher, params=params, read_only=True)
        else:
            result = self._graph.query(cypher, params=params)
        # redis-py exposes list-shaped result_set rows and keeps column names on
        # QueryResult.header as [type, name] pairs. Zip to dicts so callers can
        # address fields by name (native multi-column and legacy 2-column alike).
        return _rows_from_query_result(getattr(result, "header", None), result.result_set)


def set_assignments(alias: str, params: dict[str, Any], *, skip: set[str]) -> str:
    """Build a Cypher `SET alias.key = $key, ...` clause from a params dict,
    skipping keys (e.g. the MERGE identity key) that shouldn't be reassigned.
    Shared by orion.substrate.falkor_store and any other Cypher-native writer
    that needs a SET clause built from a partial params dict.
    """
    keys = sorted(k for k in params if k not in skip)
    return ", ".join(f"{alias}.{key} = ${key}" for key in keys)


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
    if not isinstance(result_set, (list, tuple)):
        raise ValueError("malformed Falkor result set")
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
        else:
            raise ValueError("malformed Falkor result row")
    return out
