"""`written_at` is guaranteed by code on Orion's own run nodes, not by the prompt.

Regression for live 2026-10-01: 186/594 `:Hop` nodes in `orion_worldview` had
no `written_at` because the kickoff prompt's example used to omit it.
"""

from __future__ import annotations

import os

import pytest

from orion.curiosity.tests.fake_worldview_graph import FakeGraph
from orion.curiosity.write_stamp import (
    STAMP_SOURCE,
    STAMPED_LABELS,
    WriteStamper,
    completed_tool_use_ids,
    graph_write_tool_use_ids,
    stamp_cypher,
)


def test_new_unstamped_hop_is_stamped_and_legacy_hop_is_not() -> None:
    graph = FakeGraph()
    legacy = graph.create("Hop", run_id="old", n=1, note="legacy, real time unknown")
    stamper = WriteStamper(client=graph, graph_name="orion_worldview")
    assert stamper.take_baseline() is True

    # What Orion's model-written `CREATE (:Hop {run_id:..., n:1, note:...})` leaves.
    new = graph.create("Hop", run_id="new", n=1, note="forgot written_at")

    assert stamper.stamp() == 1
    assert graph.props(new)["written_at"] is not None
    assert graph.props(new)["written_at_source"] == STAMP_SOURCE
    assert graph.props(legacy).get("written_at") is None
    assert "written_at_source" not in graph.props(legacy)
    # Idempotent: a second stamp finds nothing and does not move the clock.
    first = graph.props(new)["written_at"]
    assert stamper.stamp() == 0
    assert graph.props(new)["written_at"] == first


def test_model_written_timestamps_are_left_exactly_as_written() -> None:
    graph = FakeGraph()
    stamper = WriteStamper(client=graph, graph_name="g")
    stamper.take_baseline()
    good = graph.create("Hop", run_id="r", n=1, written_at=42)
    iso = graph.create("Finding", run_id="r", written_at="2026-09-01T00:00:00Z")
    assert stamper.stamp() == 0
    assert graph.props(good) == {"run_id": "r", "n": 1, "written_at": 42}
    assert graph.props(iso)["written_at"] == "2026-09-01T00:00:00Z"


def test_sibling_run_labels_are_covered_but_other_labels_are_not() -> None:
    graph = FakeGraph()
    stamper = WriteStamper(client=graph, graph_name="g")
    stamper.take_baseline()
    finding = graph.create("Finding", run_id="r", text="t")
    outcome = graph.create("TurnOutcome", run_id="r")
    prior = graph.create("Prior", claim="x")  # updated in place, not a run node
    assert stamper.stamp() == 2
    assert graph.props(finding)["written_at"] is not None
    assert graph.props(outcome)["written_at"] is not None
    assert "written_at" not in graph.props(prior)
    assert "Hop" in STAMPED_LABELS and "Prior" not in STAMPED_LABELS


def test_no_baseline_means_nothing_is_stamped() -> None:
    """Fail closed: without the legacy set, a new node cannot be told apart."""
    graph = FakeGraph()
    graph.create("Hop", run_id="old", n=1)
    stamper = WriteStamper(client=graph, graph_name="g")
    graph.fail = True
    assert stamper.take_baseline() is False
    assert stamper.armed is False
    graph.fail = False
    graph.create("Hop", run_id="new", n=1)
    assert stamper.stamp() is None
    assert all(n["props"].get("written_at") is None for n in graph.nodes)


def test_stamp_error_is_swallowed() -> None:
    graph = FakeGraph()
    stamper = WriteStamper(client=graph, graph_name="g")
    stamper.take_baseline()
    graph.fail = True
    assert stamper.stamp() is None


def test_from_env_requires_every_graph_key() -> None:
    full = {
        "ORION_CURIOSITY_GRAPH_HOST": "h",
        "ORION_CURIOSITY_GRAPH_PORT": "6379",
        "ORION_CURIOSITY_GRAPH_USER": "orion_curiosity",
        "ORION_CURIOSITY_GRAPH_PASSWORD": "pw",
        "ORION_CURIOSITY_GRAPH_OWN": "orion_worldview",
    }
    assert WriteStamper.from_env(full) is not None
    for key in full:
        assert WriteStamper.from_env({**full, key: ""}) is None, key


def test_stamp_cypher_only_carries_integer_ids() -> None:
    q = stamp_cypher([3, 1, 3])
    assert q.startswith("CYPHER baseline=[1,3] ")
    with pytest.raises(ValueError):
        stamp_cypher(["1] MATCH (x) DETACH DELETE x //"])


def test_graph_write_detection_from_stream_events() -> None:
    assistant = {
        "type": "assistant",
        "message": {"content": [
            {"type": "tool_use", "id": "w1", "name": "Bash",
             "input": {"command": 'redis-cli -u "$U" GRAPH.QUERY orion_worldview "CREATE (:Hop {n:1})"'}},
            {"type": "tool_use", "id": "r1", "name": "Bash",
             "input": {"command": 'redis-cli GRAPH.RO_QUERY orion_worldview "MATCH (h:Hop) RETURN h"'}},
            {"type": "tool_use", "id": "x1", "name": "Read", "input": {"file_path": "GRAPH.QUERY"}},
        ]},
    }
    assert graph_write_tool_use_ids(assistant) == {"w1"}
    lower = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "w2", "name": "Bash",
         "input": {"command": "redis-cli graph.query orion_worldview 'CREATE (:Hop {n:2})'"}},
        {"type": "tool_use", "id": "r2", "name": "Bash",
         "input": {"command": "redis-cli graph.ro_query orion_worldview 'MATCH (h) RETURN h'"}},
    ]}}
    assert graph_write_tool_use_ids(lower) == {"w2"}
    user = {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "w1", "content": "ok"},
    ]}}
    assert completed_tool_use_ids(user) == {"w1"}
    assert graph_write_tool_use_ids(user) == set()
    assert completed_tool_use_ids(assistant) == set()


@pytest.mark.skipif(
    not os.environ.get("ORION_TEST_FALKORDB_PORT"),
    reason="needs a scratch FalkorDB (ORION_TEST_FALKORDB_PORT on 127.0.0.1); never point at production",
)
def test_real_falkordb_stamps_new_and_spares_legacy() -> None:
    import redis

    client = redis.Redis(host="127.0.0.1", port=int(os.environ["ORION_TEST_FALKORDB_PORT"]), decode_responses=True)
    g = "write_stamp_scratch"
    try:
        client.execute_command("GRAPH.QUERY", g, 'CREATE (:Hop {run_id:"old", n:1}), (:Prior {claim:"x"})')
        stamper = WriteStamper(client=client, graph_name=g)
        assert stamper.take_baseline()
        client.execute_command("GRAPH.QUERY", g, 'CREATE (:Hop {run_id:"new", n:1, note:"model forgot"})')
        assert stamper.stamp() == 1
        rows = client.execute_command(
            "GRAPH.QUERY", g, "MATCH (h:Hop) RETURN h.run_id, h.written_at, h.written_at_source ORDER BY h.run_id"
        )[1]
        assert rows[0] == ["new", rows[0][1], STAMP_SOURCE] and isinstance(rows[0][1], int)
        assert rows[1] == ["old", None, None]
    finally:
        client.execute_command("GRAPH.DELETE", g)
