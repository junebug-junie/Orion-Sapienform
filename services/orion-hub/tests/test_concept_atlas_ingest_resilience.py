"""_CountingSubstrateStore must not let one unwritable node cost a run its edges.

THE INCIDENT, 2026-08-29. `FalkorSubstrateStore.upsert_node` raised on any
node_kind outside concept/evidence, and the topic-foundry adapter emits
`EntityNodeV1`. Because `SubstrateGraphMaterializer.apply_record` writes
incrementally, that exception aborted the whole ingest **before a single edge
was written**:

    concept_atlas_ingest_topic_foundry_store_write_failed
      error=... got node_kind='entity'
      -> concepts_written=18 entities_written=0 edges_written=0

Both graphs. 18 orphaned Evidence nodes on the substrate graph, 132 nodes and
0 edges on the AI Town graph, and no new structure on either.

Entity nodes are durable as of the same patch, which removes THIS trigger.
These tests cover the independent second half: any future kind the store
cannot persist must cost its own node, not the run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

for key, value in {
    "CHANNEL_VOICE_TRANSCRIPT": "orion:voice:transcript",
    "CHANNEL_VOICE_LLM": "orion:voice:llm",
    "CHANNEL_VOICE_TTS": "orion:voice:tts",
    "CHANNEL_COLLAPSE_INTAKE": "orion:collapse:intake",
    "CHANNEL_COLLAPSE_TRIAGE": "orion:collapse:triage",
}.items():
    os.environ.setdefault(key, value)


class _Node:
    def __init__(self, node_id, node_kind):
        self.node_id = node_id
        self.node_kind = node_kind


class _Ref:
    def __init__(self, node_id):
        self.node_id = node_id


class _Edge:
    def __init__(self, source_id, target_id, predicate="supports"):
        self.source = _Ref(source_id)
        self.target = _Ref(target_id)
        self.predicate = predicate


class _RefusingStore:
    """Accepts concept/evidence, raises on everything else -- the exact shape
    FalkorSubstrateStore had before entity nodes became durable."""

    def __init__(self, refuse_kinds=("entity",)):
        self.refuse_kinds = set(refuse_kinds)
        self.written_nodes = []
        self.written_edges = []

    def upsert_node(self, *, identity_key=None, node=None, skip_metadata_keys=None):
        if node.node_kind in self.refuse_kinds:
            raise ValueError(
                "FalkorSubstrateStore durable writes support concept and evidence nodes only; "
                f"got node_kind={node.node_kind!r}"
            )
        self.written_nodes.append(node.node_id)

    def upsert_edge(self, *, identity_key, edge):
        self.written_edges.append((edge.source.node_id, edge.target.node_id))


def _counting(store):
    _ensure_hub_scripts_import_path()
    from scripts.concept_atlas_routes import _CountingSubstrateStore

    return _CountingSubstrateStore(store)


def test_a_refused_node_does_not_stop_the_run():
    inner = _RefusingStore()
    cs = _counting(inner)

    cs.upsert_node(identity_key="c1", node=_Node("c1", "concept"))
    cs.upsert_node(identity_key="e1", node=_Node("e1", "entity"))  # refused
    cs.upsert_node(identity_key="ev1", node=_Node("ev1", "evidence"))
    cs.upsert_edge(identity_key="edge1", edge=_Edge("ev1", "c1"))

    assert cs.concepts_written == 1
    assert cs.evidence_nodes_written == 1
    assert cs.entities_written == 0
    assert cs.edges_written == 1, "the edge is the whole point -- it must survive"
    assert inner.written_edges == [("ev1", "c1")]


def test_the_refused_node_is_recorded_not_swallowed():
    cs = _counting(_RefusingStore())
    cs.upsert_node(identity_key="e1", node=_Node("e1", "entity"))

    assert len(cs.skipped_nodes) == 1
    entry = cs.skipped_nodes[0]
    assert entry["node_id"] == "e1"
    assert entry["node_kind"] == "entity"
    assert "node_kind='entity'" in entry["error"], "the real reason must be kept, not just a flag"


def test_an_edge_touching_a_refused_node_is_dropped_not_written():
    """upsert_edge's Cypher opens `MERGE (source:SubstrateNode {node_id})`,
    which CREATES a bare node when none exists. Writing the edge anyway would
    replace the refused node with a phantom carrying only a node_id -- which
    every decoder returns None for. Dropping the edge keeps the graph honest
    about what is missing."""
    inner = _RefusingStore()
    cs = _counting(inner)

    cs.upsert_node(identity_key="c1", node=_Node("c1", "concept"))
    cs.upsert_node(identity_key="e1", node=_Node("e1", "entity"))  # refused
    cs.upsert_edge(identity_key="m1", edge=_Edge("c1", "e1", predicate="mentions"))
    cs.upsert_edge(identity_key="m2", edge=_Edge("e1", "c1", predicate="mentions"))

    assert inner.written_edges == [], "no phantom node may be conjured by MERGE"
    assert cs.edges_written == 0
    assert cs.skipped_edges == 2


def test_edges_between_written_nodes_are_unaffected():
    inner = _RefusingStore()
    cs = _counting(inner)
    cs.upsert_node(identity_key="c1", node=_Node("c1", "concept"))
    cs.upsert_node(identity_key="c2", node=_Node("c2", "concept"))
    cs.upsert_node(identity_key="e1", node=_Node("e1", "entity"))  # refused
    cs.upsert_edge(identity_key="co", edge=_Edge("c1", "c2", predicate="co_occurs_with"))

    assert cs.edges_written == 1
    assert cs.skipped_edges == 0
    assert inner.written_edges == [("c1", "c2")]


def test_a_clean_run_records_no_skips():
    """The skip bookkeeping must stay silent when nothing is refused, or every
    healthy ingest reports itself as degraded."""
    inner = _RefusingStore(refuse_kinds=())
    cs = _counting(inner)
    cs.upsert_node(identity_key="c1", node=_Node("c1", "concept"))
    cs.upsert_node(identity_key="e1", node=_Node("e1", "entity"))
    cs.upsert_edge(identity_key="m", edge=_Edge("e1", "c1", predicate="mentions"))

    assert cs.skipped_nodes == []
    assert cs.skipped_edges == 0
    assert cs.entities_written == 1
    assert cs.edges_written == 1


def test_entity_nodes_now_reach_the_store_at_all():
    """The fix's other half: with a store that accepts entities, they are
    written and counted rather than raising."""
    inner = _RefusingStore(refuse_kinds=())
    cs = _counting(inner)
    cs.upsert_node(identity_key="e1", node=_Node("e1", "entity"))
    assert inner.written_nodes == ["e1"]
    assert cs.entities_written == 1


def test_a_node_with_no_id_records_a_skip_without_poisoning_the_set():
    """Defensive: the skip path must not itself raise on a malformed node, or
    the resilience mechanism reintroduces the abort it exists to prevent.

    And an empty id must NOT join _failed_node_ids: it would sit there as a
    sentinel matching any later edge whose endpoint ref is missing, dropping
    that edge and blaming `endpoint_not_written` for an unrelated cause."""
    class _Bad:
        node_kind = "entity"

    class _EdgeWithNoRefs:
        source = _Ref("")
        target = _Ref("")
        predicate = "supports"

    inner = _RefusingStore()
    cs = _counting(inner)
    cs.upsert_node(identity_key=None, node=_Bad())
    assert len(cs.skipped_nodes) == 1
    assert cs.skipped_nodes[0]["node_id"] == ""

    cs.upsert_edge(identity_key="x", edge=_EdgeWithNoRefs())
    assert cs.skipped_edges == 0, "an unrelated edge must not be blamed on the id-less skip"
    assert cs.edges_written == 1


def test_getattr_passthrough_still_works():
    """The wrapper delegates every other store operation, including the
    identity-resolver reads the materializer makes."""
    inner = _RefusingStore()
    inner.snapshot = lambda: "snap"
    assert _counting(inner).snapshot() == "snap"


# --- a dead store must not read as success ---------------------------------


class _DeadStore:
    """Every write fails -- FalkorDB unreachable."""

    def __init__(self):
        self.attempts = 0

    def upsert_node(self, *, identity_key=None, node=None, skip_metadata_keys=None):
        self.attempts += 1
        raise ConnectionError("Error 111 connecting to 127.0.0.1:6380. Connection refused.")

    def upsert_edge(self, *, identity_key, edge):
        raise ConnectionError("Error 111 connecting to 127.0.0.1:6380. Connection refused.")


def test_an_unbroken_run_of_failures_stops_being_swallowed():
    """Swallowing every failure turns an unreachable store into N caught
    exceptions, a normally-completing apply_record, and a route answering
    `available: true` with every count at zero -- strictly worse than the abort
    it replaced. Past the bound the exception propagates as before."""
    _ensure_hub_scripts_import_path()
    from scripts.concept_atlas_routes import _CountingSubstrateStore

    inner = _DeadStore()
    cs = _CountingSubstrateStore(inner)

    with pytest.raises(ConnectionError):
        for i in range(_CountingSubstrateStore.MAX_CONSECUTIVE_NODE_FAILURES + 5):
            cs.upsert_node(identity_key=f"c{i}", node=_Node(f"c{i}", "concept"))

    assert inner.attempts == _CountingSubstrateStore.MAX_CONSECUTIVE_NODE_FAILURES, (
        "must stop retrying a dead store, not pay N x connect-timeout"
    )


def test_a_success_resets_the_failure_run():
    """A single unwritable kind interleaved with successes must never trip the
    breaker, however many such nodes a run contains."""
    _ensure_hub_scripts_import_path()
    from scripts.concept_atlas_routes import _CountingSubstrateStore

    cs = _counting(_RefusingStore())
    for i in range(_CountingSubstrateStore.MAX_CONSECUTIVE_NODE_FAILURES * 3):
        cs.upsert_node(identity_key=f"e{i}", node=_Node(f"e{i}", "entity"))   # refused
        cs.upsert_node(identity_key=f"c{i}", node=_Node(f"c{i}", "concept"))  # ok

    assert cs.concepts_written == _CountingSubstrateStore.MAX_CONSECUTIVE_NODE_FAILURES * 3
    assert len(cs.skipped_nodes) == _CountingSubstrateStore.MAX_CONSECUTIVE_NODE_FAILURES * 3


def test_wrote_anything_is_false_only_when_nothing_landed():
    inner = _RefusingStore()
    cs = _counting(inner)
    cs.upsert_node(identity_key="e1", node=_Node("e1", "entity"))  # refused
    assert cs.wrote_anything is False

    cs.upsert_node(identity_key="c1", node=_Node("c1", "concept"))
    assert cs.wrote_anything is True


def test_wrote_anything_counts_an_edge_only_run():
    """A run that merged existing nodes and wrote only edges still produced
    something; reporting it as "wrote nothing" would be wrong."""
    cs = _counting(_RefusingStore())
    cs.upsert_edge(identity_key="co", edge=_Edge("a", "b", predicate="co_occurs_with"))
    assert cs.wrote_anything is True
