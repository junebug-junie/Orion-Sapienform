"""Duplicate-identity repair: plan rules (always) and a live apply (opt-in).

The live test needs a THROWAWAY FalkorDB (never the production instance):
  docker run -d --rm --name falkor-dedupe-test -p 127.0.0.1:56380:6379 falkordb/falkordb:6.0.1
  ORION_DEDUPE_TEST_FALKOR_URI=redis://127.0.0.1:56380 pytest tests/test_repair_substrate_duplicate_identities.py
"""
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "repair_dupes", Path(__file__).resolve().parents[1] / "scripts" / "repair_substrate_duplicate_identities.py"
)
repair = importlib.util.module_from_spec(_SPEC)
sys.modules["repair_dupes"] = repair  # dataclasses resolve their module by name
_SPEC.loader.exec_module(repair)
EdgeRow = repair.EdgeRow


def _node(node_id, key="concept|world||label:coffee", refs=(), observed="2026-10-01T00:00:00+00:00"):
    return {"node_id": node_id, "identity_key": key, "evidence_refs_json": json.dumps(list(refs)),
            "observed_at": observed}


def _edge(edge_id, rel, s, t, key=None):
    return EdgeRow(edge_id=edge_id, rel_type=rel, source=s, target=t,
                   props={"edge_id": edge_id, "identity_key": key or f"{s}|{rel}|{t}"})


def test_keeper_is_smallest_node_id_and_others_are_deleted():
    plan = repair.build_plan([_node("c-run2"), _node("c-run1"), _node("c-run3"), _node("solo", key="other")], [], set())
    assert plan.groups == {"concept|world||label:coffee": ["c-run1", "c-run2", "c-run3"]}
    assert plan.nodes_to_delete == ["c-run2", "c-run3"]


def test_edges_repoint_drop_self_loops_and_parallels_and_map_identity():
    nodes = [_node("a1"), _node("a2"), _node("b1", key="k:b"), _node("b2", key="k:b")]
    edges = [
        _edge("e1", "supports", "ev", "a2"),          # -> ev->a1
        _edge("e2", "co_occurs_with", "a2", "b2"),    # dup->dup -> a1->b1
        _edge("e2", "co_occurs_with", "a2", "b2"),    # same edge listed from the other end
        _edge("e3", "co_occurs_with", "a2", "a1"),    # becomes a1->a1: dropped
        _edge("e4", "supports", "ev2", "a2"),         # keeper already has ev2->a1: dropped
    ]
    plan = repair.build_plan(nodes, edges, {("supports", "ev2", "a1")})
    rewired = {e.edge_id: (s, t, p["identity_key"]) for e, s, t, p in plan.rewire}
    assert rewired == {"e1": ("ev", "a1", "ev|supports|a1"), "e2": ("a1", "b1", "a1|co_occurs_with|b1")}
    assert [e.edge_id for e in plan.drop_self_loops] == ["e3"]
    assert [e.edge_id for e in plan.drop_parallel] == ["e4"]


def test_keeper_gets_union_of_evidence_and_latest_observed_at():
    nodes = [_node("a1", refs=["r1"], observed="2026-09-01"), _node("a2", refs=["r2", "r1"], observed="2026-10-13")]
    plan = repair.build_plan(nodes, [], set())
    assert json.loads(plan.node_updates["a1"]["evidence_refs_json"]) == ["r1", "r2"]
    assert plan.node_updates["a1"]["observed_at"] == "2026-10-13"


def test_apply_refuses_without_approval(capsys):
    assert repair.main(["--graph", "g", "--apply", "--uri", "redis://127.0.0.1:1"]) == 2


class _CappedClient:
    """Answers like FalkorDB with RESULTSET_SIZE: a result is cut at the cap."""

    def __init__(self, cap, rows_per_item):
        self.cap, self.rows_per_item, self.calls = cap, rows_per_item, []

    def graph_query(self, cypher, params):
        self.calls.append(len(params["items"]))
        rows = [{"item": i, "n": n} for i in params["items"] for n in range(self.rows_per_item)]
        return rows[:self.cap]


def test_chunk_at_the_server_cap_is_split_not_trusted(monkeypatch):
    monkeypatch.setattr(repair, "SERVER_RESULT_CAP", 10)
    client = _CappedClient(cap=10, rows_per_item=3)
    rows = repair._query_chunked(client, "q", list(range(8)), chunk=8)
    assert len(rows) == 24  # 8 items x 3, nothing lost to the cap
    assert client.calls[0] == 8 and max(client.calls[1:]) < 8


_LIVE = os.environ.get("ORION_DEDUPE_TEST_FALKOR_URI")


@pytest.mark.skipif(not _LIVE, reason="ORION_DEDUPE_TEST_FALKOR_URI (throwaway FalkorDB) not set")
def test_live_apply_merges_duplicates_on_a_throwaway_graph(tmp_path, monkeypatch):
    from orion.graph.falkor_client import RedisGraphQueryClient

    graph = f"t_dedupe_{uuid.uuid4().hex[:8]}"
    c = RedisGraphQueryClient(uri=_LIVE, graph_name=graph)
    try:
        c.graph_query(
            "CREATE (:SubstrateNode:Concept {node_id:'c1', identity_key:'k', evidence_refs_json:'[\"r1\"]', observed_at:'2026-09-01'}),"
            " (:SubstrateNode:Concept {node_id:'c2', identity_key:'k', evidence_refs_json:'[\"r2\"]', observed_at:'2026-10-01'}),"
            " (:SubstrateNode:Concept {node_id:'d1', identity_key:'kd'}),"
            " (:SubstrateNode:Concept {node_id:'d2', identity_key:'kd'}),"
            " (:SubstrateNode:Evidence {node_id:'ev', identity_key:'kev'})"
        )
        c.graph_query("MATCH (a {node_id:'ev'}),(b {node_id:'c2'}) CREATE (a)-[:supports {edge_id:'e1', identity_key:'ev|supports|c2', substrate_edge:true}]->(b)")
        c.graph_query("MATCH (a {node_id:'c2'}),(b {node_id:'d2'}) CREATE (a)-[:co_occurs_with {edge_id:'e2', identity_key:'c2|co_occurs_with|d2', substrate_edge:true, weight:3}]->(b)")
        c.graph_query("MATCH (a {node_id:'c1'}),(b {node_id:'c2'}) CREATE (a)-[:co_occurs_with {edge_id:'e3', substrate_edge:true}]->(b)")

        plan, summary = repair.dry_run(_LIVE, graph, tmp_path)
        assert summary["nodes_to_delete"] == 2 and summary["snapshot_written"]
        result = repair.apply_plan(_LIVE, graph, plan, tmp_path)
        assert result["errors"] == 0

        ids = sorted(r["id"] for r in c.graph_query("MATCH (n:SubstrateNode) RETURN n.node_id AS id"))
        assert ids == ["c1", "d1", "ev"]
        edges = {r["eid"]: r for r in c.graph_query(
            "MATCH (s)-[e]->(t) RETURN e.edge_id AS eid, s.node_id AS s, t.node_id AS t, "
            "e.identity_key AS k, e.weight AS w")}
        assert set(edges) == {"e1", "e2"}  # e3 became a self-loop
        assert (edges["e1"]["s"], edges["e1"]["t"], edges["e1"]["k"]) == ("ev", "c1", "ev|supports|c1")
        assert (edges["e2"]["s"], edges["e2"]["t"], edges["e2"]["w"]) == ("c1", "d1", 3)
        keeper = c.graph_query("MATCH (n {node_id:'c1'}) RETURN n.evidence_refs_json AS r, n.observed_at AS o")[0]
        assert json.loads(keeper["r"]) == ["r1", "r2"] and keeper["o"] == "2026-10-01"
        # idempotent: a second dry run finds nothing
        _, again = repair.dry_run(_LIVE, graph, tmp_path)
        assert again["duplicate_identities"] == 0
    finally:
        try:
            c._r.execute_command("GRAPH.DELETE", graph)
        finally:
            c.close()
