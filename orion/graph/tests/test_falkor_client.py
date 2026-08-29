from __future__ import annotations

from orion.graph.falkor_client import (
    FalkorGraphClient,
    RecordingFalkorClient,
    RedisGraphQueryClient,
    _header_field_names,
    _rows_from_query_result,
)


def test_recording_falkor_client_records_calls() -> None:
    client: FalkorGraphClient = RecordingFalkorClient()
    client.graph_query("MERGE (n:Foo {id: $id})", {"id": "1"})
    assert client.calls == [("MERGE (n:Foo {id: $id})", {"id": "1"})]


def test_recording_falkor_client_hydrate_node_rows_dispatch() -> None:
    client = RecordingFalkorClient(hydrate_node_rows=[{"node_id": "n1"}])
    rows = client.graph_query("MATCH (n) RETURN n.node_id AS node_id")
    assert rows == [{"node_id": "n1"}]


def test_recording_falkor_client_hydrate_rows_alias() -> None:
    client = RecordingFalkorClient(hydrate_rows=[{"node_id": "aliased"}])
    rows = client.graph_query("MATCH (n) RETURN n.node_id AS node_id")
    assert rows == [{"node_id": "aliased"}]


def test_header_field_names_type_name_pairs() -> None:
    header = [[1, "n.turn_id"], [1, "n.session_id"]]
    assert _header_field_names(header) == ["turn_id", "session_id"]


def test_header_field_names_name_type_pairs() -> None:
    header = [["turn_id", 1], ["session_id", 1]]
    assert _header_field_names(header) == ["turn_id", "session_id"]


def test_header_field_names_empty() -> None:
    assert _header_field_names(None) == []
    assert _header_field_names([]) == []


def test_rows_from_query_result_zips_named_columns() -> None:
    header = [[1, "n.turn_id"], [1, "n.ts"]]
    result_set = [["turn-1", "2026-07-18T00:00:00Z"]]
    rows = _rows_from_query_result(header, result_set)
    assert rows == [{"turn_id": "turn-1", "ts": "2026-07-18T00:00:00Z"}]


def test_rows_from_query_result_falls_back_to_positional() -> None:
    result_set = [["turn-1", "extra"]]
    rows = _rows_from_query_result(None, result_set)
    assert rows == [{"_positional": ["turn-1", "extra"]}]


def test_rows_from_query_result_none_result_set() -> None:
    assert _rows_from_query_result([[1, "x"]], None) == []


def test_redis_graph_query_client_constructs_from_uri() -> None:
    # Real redis.Redis() construction is lazy (no connection attempt until a
    # command is issued), so this exercises the constructor path without
    # needing a live FalkorDB -- matches the existing RedisGraphQueryClient
    # test pattern in orion/substrate/tests/test_falkor_store.py.
    client = RedisGraphQueryClient(uri="redis://example.test:6380/2", graph_name="orion_recall")
    assert client._graph.name == "orion_recall"
    assert client._r.connection_pool.connection_kwargs["host"] == "example.test"
    assert client._r.connection_pool.connection_kwargs["port"] == 6380
    assert client._r.connection_pool.connection_kwargs["db"] == 2


# --- read-only mode must not break stand-in Graph objects -------------------
#
# Regression, 2026-08-29 (PR #1957): adding read_only mode made graph_query
# read self._read_only and forward a third kwarg unconditionally. Both broke
# orion/substrate/tests/test_falkor_store.py::
# test_redis_graph_client_returns_named_dicts_from_header, which builds the
# client with __new__ (no __init__, so no _read_only) and injects a fake Graph
# whose query() takes only (cypher, params). Main went red on merge because
# that consumer suite was never run against the shared-client change.


class _FakeResult:
    header = [[1, "node_id"]]
    result_set = [["concept-alpha"]]


class _TwoArgGraph:
    """A Graph stand-in whose query() predates the read_only kwarg."""

    def __init__(self):
        self.calls = []

    def query(self, cypher, params=None):
        self.calls.append((cypher, params))
        return _FakeResult()


class _ThreeArgGraph:
    def __init__(self):
        self.calls = []

    def query(self, cypher, params=None, read_only=False):
        self.calls.append((cypher, params, read_only))
        return _FakeResult()


def test_graph_query_works_on_an_instance_built_without_init():
    from orion.graph.falkor_client import RedisGraphQueryClient

    client = RedisGraphQueryClient.__new__(RedisGraphQueryClient)
    client._graph = _TwoArgGraph()

    assert client.graph_query("RETURN 1", {"a": 1}) == [{"node_id": "concept-alpha"}]
    assert client.read_only is False, "class-level default must exist"


def test_default_mode_does_not_forward_the_read_only_kwarg():
    """A two-argument query() must keep working: forwarding a kwarg that
    changes nothing would break it for no benefit."""
    from orion.graph.falkor_client import RedisGraphQueryClient

    client = RedisGraphQueryClient.__new__(RedisGraphQueryClient)
    graph = _TwoArgGraph()
    client._graph = graph
    client.graph_query("RETURN 1", {"a": 1})
    assert graph.calls == [("RETURN 1", {"a": 1})]


def test_read_only_mode_does_forward_the_kwarg():
    from orion.graph.falkor_client import RedisGraphQueryClient

    client = RedisGraphQueryClient.__new__(RedisGraphQueryClient)
    graph = _ThreeArgGraph()
    client._graph = graph
    client._read_only = True
    client.graph_query("RETURN 1", None)
    assert graph.calls == [("RETURN 1", None, True)]
