"""Tests for orion.graph.analytics.

The two things that can break here without anything going red are pinned
deliberately, because both were hit live while building this module:

  1. A WRONG PROCEDURE ARITY IS NOT ALWAYS AN ERROR. ``algo.pageRank()`` raises
     "requires 2 arguments, got 0", but ``algo.betweenness(null, null)`` returns
     a header and zero rows -- a silently empty ranking. So the emitted CALL
     text is asserted verbatim rather than only asserting that some rows came
     back from a stub that would have returned them regardless.

  2. A NON-READ-ONLY CYPHER STRING WOULD STILL PASS EVERY BEHAVIOURAL TEST here,
     since the stub client does not care. ``test_every_emitted_cypher_is_read_only``
     drives every public method and greps the recorded Cypher for mutating
     clauses, so adding a write to this module fails the suite rather than the
     production ``GRAPH.RO_QUERY``.
"""

from __future__ import annotations

import re

import pytest

from orion.graph.analytics import (
    MAX_NEIGHBOURHOOD_DEPTH,
    MAX_PATH_DEPTH,
    MEASURES,
    Component,
    GraphAnalytics,
    GraphAnalyticsError,
    StructureSummary,
)


class StubClient:
    """Records Cypher and replays scripted rows, in order."""

    def __init__(self, *responses: list[dict]) -> None:
        self.calls: list[tuple[str, dict | None]] = []
        self._responses = list(responses)

    def graph_query(self, cypher: str, params: dict | None = None):
        self.calls.append((cypher, params))
        if self._responses:
            return self._responses.pop(0)
        return []

    @property
    def cypher(self) -> str:
        return self.calls[-1][0]


class ExplodingClient:
    def graph_query(self, cypher: str, params: dict | None = None):
        raise RuntimeError("backend down")


# --- read-only enforcement --------------------------------------------------

_MUTATING = re.compile(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP)\b", re.IGNORECASE)


def _drive_every_method(client: StubClient) -> None:
    a = GraphAnalytics(client)
    a.node_count()
    a.node_count("Concept")
    a.edge_type_counts()
    a.components()
    a.communities()
    a.communities(rel_types=["co_occurs_with"])
    for measure in MEASURES:
        a.rank(measure)
    a.neighborhood("n1")
    a.neighborhood("n1", depth=2, rel_types=["supports"])
    a.path("n1", "n2")
    a.path("n1", "n2", rel_types=["supports"])
    a.summary()


def test_every_emitted_cypher_is_read_only():
    client = StubClient()
    _drive_every_method(client)
    assert client.calls, "no Cypher was emitted; the test would pass vacuously"
    offenders = [c for c, _ in client.calls if _MUTATING.search(c)]
    assert offenders == [], f"non-read-only Cypher emitted: {offenders}"


def test_read_only_check_would_catch_a_write():
    """The guard above is only worth having if it can fail."""
    assert _MUTATING.search("MATCH (n) SET n.x = 1")
    assert _MUTATING.search("CREATE (:Tmp)")
    assert not _MUTATING.search("MATCH (n) RETURN count(n) AS n")


# --- procedure arity is pinned ---------------------------------------------

def test_measure_call_forms_are_the_verified_ones():
    """Verified live 2026-08-29 on FalkorDB 4.18.11 -- see MEASURES' comment.

    pageRank RAISES on the wrong arity; betweenness returns silently empty. A
    change here must be re-verified against a live engine, not guessed.
    """
    assert MEASURES["pagerank"] == "algo.pageRank(null, null)"
    assert MEASURES["betweenness"] == "algo.betweenness()"
    assert MEASURES["harmonic"] == "algo.HarmonicCentrality()"


@pytest.mark.parametrize("measure,expected", sorted(MEASURES.items()))
def test_rank_emits_the_pinned_call_form(measure, expected):
    client = StubClient([])
    GraphAnalytics(client).rank(measure)
    assert f"CALL {expected} YIELD node, score" in client.cypher
    # the bug this catches: an extra "()" appended to an already-complete form
    assert "()()" not in client.cypher
    assert "(null, null)()" not in client.cypher


def test_unknown_measure_raises_rather_than_silently_returning_nothing():
    with pytest.raises(ValueError, match="unknown measure"):
        GraphAnalytics(StubClient()).rank("eigenvector")


# --- injection boundaries ---------------------------------------------------

@pytest.mark.parametrize(
    "bad",
    ["co_occurs_with) RETURN 1 //", "a|b", "has-dash", "1leading_digit", "with space", "sem;colon"],
)
def test_relationship_types_reject_non_identifiers(bad):
    with pytest.raises(ValueError, match="non-identifier relationship types"):
        GraphAnalytics(StubClient()).neighborhood("n1", rel_types=[bad])


def test_communities_rel_types_reject_non_identifiers():
    with pytest.raises(ValueError, match="non-identifier relationship types"):
        GraphAnalytics(StubClient()).communities(rel_types=["x) RETURN 1 //"])


@pytest.mark.parametrize("field", ["id_property", "label_property"])
def test_property_names_reject_non_identifiers(field):
    with pytest.raises(ValueError, match=f"non-identifier {field}"):
        GraphAnalytics(StubClient(), **{field: "label) RETURN 1 //"})


def test_node_count_label_rejects_non_identifier():
    with pytest.raises(ValueError, match="non-identifier label"):
        GraphAnalytics(StubClient()).node_count("Concept) RETURN 1 //")


# --- configurable identity/display properties -------------------------------

def test_custom_properties_reach_the_cypher():
    """orion_recall uses turn_id and has no `label` at all; hardcoding
    node.label there returns None for every node while looking like it worked.
    """
    client = StubClient([])
    a = GraphAnalytics(client, id_property="turn_id", label_property="source_kind")
    a.rank("pagerank")
    assert "node.turn_id AS node_id" in client.cypher
    assert "node.source_kind AS label" in client.cypher
    assert "node.node_id" not in client.cypher
    assert "node.label" not in client.cypher

    a.neighborhood("t1", depth=1)
    assert "s.turn_id = $node_id" in client.cypher
    assert "m.turn_id AS node_id" in client.cypher

    a.path("t1", "t2")
    assert "a.turn_id = $source" in client.cypher
    assert "b.turn_id = $target" in client.cypher
    assert "n.source_kind" in client.cypher

    a.components()
    assert "collect(node.source_kind)" in client.cypher


def test_defaults_are_the_substrate_properties():
    client = StubClient([])
    GraphAnalytics(client).rank("pagerank")
    assert "node.node_id AS node_id" in client.cypher
    assert "node.label AS label" in client.cypher


# --- parsing ----------------------------------------------------------------

def test_components_parse_and_sort_by_size():
    rows = [
        {"componentId": "1", "n": 116, "sample": ["Orion", "Juniper"]},
        {"componentId": "56", "n": 10, "sample": ["Burst Test Conversations"]},
        {"componentId": "4", "n": 1, "sample": ["Chat prediction error"]},
    ]
    comps = GraphAnalytics(StubClient(rows)).components()
    assert [c.size for c in comps] == [116, 10, 1]
    assert comps[0].sample_labels == ("Orion", "Juniper")
    assert comps[0].is_singleton is False
    assert comps[2].is_singleton is True


def test_components_tolerate_a_component_with_no_labelled_members():
    """80 of orion_substrate's 136 nodes are Evidence nodes with no label
    property, so `collect(node.label)` legitimately yields an empty list."""
    comps = GraphAnalytics(StubClient([{"componentId": "9", "n": 3, "sample": []}])).components()
    assert comps[0].size == 3
    assert comps[0].sample_labels == ()


def test_rank_parses_rows_and_tolerates_a_missing_label():
    rows = [
        {"node_id": "a", "label": "Orion", "score": 0.0993},
        {"node_id": "b", "label": None, "score": 0.0384},
    ]
    ranked = GraphAnalytics(StubClient(rows)).rank("pagerank")
    assert (ranked[0].node_id, ranked[0].label) == ("a", "Orion")
    assert ranked[0].score == pytest.approx(0.0993)
    assert ranked[1].label is None


def test_path_returns_hops_in_order_with_index_as_score():
    rows = [{"ids": ["a", "b", "c"], "labels": ["Orion", "Model Development", "Juniper"]}]
    hops = GraphAnalytics(StubClient(rows)).path("a", "c")
    assert [h.label for h in hops] == ["Orion", "Model Development", "Juniper"]
    assert [h.score for h in hops] == [0.0, 1.0, 2.0]


def test_path_tolerates_labels_shorter_than_ids():
    rows = [{"ids": ["a", "b", "c"], "labels": ["Orion"]}]
    hops = GraphAnalytics(StubClient(rows)).path("a", "c")
    assert [h.node_id for h in hops] == ["a", "b", "c"]
    assert [h.label for h in hops] == ["Orion", None, None]


def test_path_empty_means_no_path_within_depth_not_disconnected():
    assert GraphAnalytics(StubClient([])).path("a", "z") == ()


def test_neighborhood_carries_hop_distance_as_score():
    rows = [
        {"node_id": "b", "label": "AI Model Transparency", "hops": 1},
        {"node_id": "c", "label": "Far", "hops": 2},
    ]
    nb = GraphAnalytics(StubClient(rows)).neighborhood("a", depth=2)
    assert [n.score for n in nb] == [1.0, 2.0]


def test_edge_type_counts_drops_null_types_but_keeps_zero_counts():
    rows = [{"edge_type": "co_occurs_with", "n": 307}, {"edge_type": None, "n": 5}]
    assert GraphAnalytics(StubClient(rows)).edge_type_counts() == {"co_occurs_with": 307}


# --- bounds -----------------------------------------------------------------

@pytest.mark.parametrize("asked,clamped", [(0, 1), (1, 1), (99, MAX_NEIGHBOURHOOD_DEPTH)])
def test_neighborhood_depth_is_clamped(asked, clamped):
    client = StubClient([])
    GraphAnalytics(client).neighborhood("a", depth=asked)
    assert f"*1..{clamped}]" in client.cypher


@pytest.mark.parametrize("asked,clamped", [(0, 1), (3, 3), (99, MAX_PATH_DEPTH)])
def test_path_depth_is_clamped(asked, clamped):
    """path() measured 143ms at depth 4 on a 136-node graph; the cap is a real
    cost control, not decoration."""
    client = StubClient([])
    GraphAnalytics(client).path("a", "b", max_depth=asked)
    assert f"*1..{clamped}]" in client.cypher


def test_rank_top_n_floors_at_one():
    client = StubClient([])
    GraphAnalytics(client).rank("pagerank", top_n=0)
    assert "LIMIT 1" in client.cypher


def test_rel_types_narrow_the_traversal():
    client = StubClient([])
    GraphAnalytics(client).neighborhood("a", rel_types=["supports", "associated_with"])
    assert "[:supports|associated_with*1..1]" in client.cypher


def test_no_rel_types_leaves_the_traversal_untyped():
    client = StubClient([])
    GraphAnalytics(client).neighborhood("a")
    assert "[*1..1]" in client.cypher


# --- summary ----------------------------------------------------------------

def _substrate_summary() -> StructureSummary:
    """The real live shape of orion_substrate on 2026-08-29."""
    return StructureSummary(
        node_count=136,
        edge_count=461,
        edge_type_counts={"co_occurs_with": 307, "supports": 80, "associated_with": 74},
        components=(
            Component("1", 116, ("Orion", "Juniper")),
            Component("56", 10, ("Burst Test Conversations",)),
            *[Component(str(i), 1, ("Chat prediction error",)) for i in range(4, 14)],
        ),
    )


def test_summary_properties_against_the_real_live_shape():
    s = _substrate_summary()
    assert s.component_count == 12
    assert s.largest_component_size == 116
    assert s.singleton_count == 10
    assert s.dominant_edge_type == "co_occurs_with"


def test_saturation_is_edges_over_possible_pairs():
    """Hand-computed: 307 co_occurs_with edges over 56 concepts.
    56*55/2 = 1540 possible pairs; 307/1540 = 0.19935...
    """
    s = StructureSummary(node_count=56, edge_count=307)
    assert s.saturation() == pytest.approx(307 / 1540)
    assert s.saturation() == pytest.approx(0.1994, abs=1e-4)


def test_saturation_denominator_can_be_overridden_to_the_real_subpopulation():
    """The graph holds 136 nodes but only 56 of them can carry a concept-concept
    edge, so the graph-wide denominator understates saturation ~6x."""
    s = _substrate_summary()
    graph_wide = s.saturation()
    concepts_only = StructureSummary(
        node_count=56, edge_count=s.edge_type_counts["co_occurs_with"]
    ).saturation()
    assert graph_wide == pytest.approx(461 / (136 * 135 / 2))
    assert concepts_only == pytest.approx(0.1994, abs=1e-4)
    assert concepts_only > graph_wide * 3


@pytest.mark.parametrize("n", [0, 1])
def test_saturation_undefined_below_two_nodes(n):
    assert StructureSummary(node_count=n, edge_count=0).saturation() is None


def test_summary_edge_count_is_the_sum_of_typed_edges():
    client = StubClient(
        [{"edge_type": "co_occurs_with", "n": 307}, {"edge_type": "supports", "n": 80}],
        [{"n": 136}],
        [{"componentId": "1", "n": 136, "sample": []}],
    )
    s = GraphAnalytics(client).summary()
    assert s.edge_count == 387
    assert s.node_count == 136


def test_empty_graph_summary_is_all_zeros_not_an_error():
    s = GraphAnalytics(StubClient([], [], [])).summary()
    assert (s.node_count, s.edge_count, s.component_count) == (0, 0, 0)
    assert s.dominant_edge_type is None
    assert s.largest_component_size == 0


def test_a_zero_edge_graph_reads_as_all_singletons():
    """orion_worldview live 2026-08-29: 48 nodes, 0 relationships of any type.
    The honest read is 48 singleton components, not one graph."""
    client = StubClient(
        [],
        [{"n": 48}],
        [{"componentId": str(i), "n": 1, "sample": []} for i in range(48)],
    )
    s = GraphAnalytics(client).summary()
    assert s.edge_count == 0
    assert s.singleton_count == 48
    assert s.largest_component_size == 1
    assert s.dominant_edge_type is None


# --- failure ----------------------------------------------------------------

def test_backend_failure_surfaces_as_a_typed_error():
    with pytest.raises(GraphAnalyticsError, match="graph query failed"):
        GraphAnalytics(ExplodingClient()).components()
