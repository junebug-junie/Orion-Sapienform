"""Unit tests for ``orion/structural_mass/graph_delta.py``."""

from __future__ import annotations

import pytest

from orion.structural_mass.graph_delta import graph_structural_delta
from orion.structural_mass.snapshot_history import GraphSnapshotStats


def _snap(
    *,
    node_count: int = 0,
    edge_count: int = 0,
    community_count: int = 0,
    god_nodes: tuple[str, ...] | None = (),
) -> GraphSnapshotStats:
    return GraphSnapshotStats(
        node_count=node_count,
        edge_count=edge_count,
        community_count=community_count,
        god_nodes=god_nodes,
    )


def test_zero_delta_when_identical_snapshots() -> None:
    snap = _snap(node_count=100, edge_count=200, community_count=10, god_nodes=("A", "B"))
    delta = graph_structural_delta(snap, snap)
    assert delta.node_count_delta == 0
    assert delta.edge_count_delta == 0
    assert delta.community_count_delta == 0
    assert delta.god_node_jaccard_similarity == pytest.approx(1.0)
    assert delta.god_nodes_entered == ()
    assert delta.god_nodes_exited == ()


def test_counts_positive_deltas() -> None:
    prev = _snap(node_count=100, edge_count=200, community_count=10)
    curr = _snap(node_count=150, edge_count=260, community_count=12)
    delta = graph_structural_delta(prev, curr)
    assert delta.node_count_delta == 50
    assert delta.edge_count_delta == 60
    assert delta.community_count_delta == 2


def test_counts_negative_deltas() -> None:
    """A shrinking graph (a deletion arc, e.g. PR #879->#1486's drives-system
    removal) is real signal, not clamped to zero."""
    prev = _snap(node_count=100, edge_count=200, community_count=10)
    curr = _snap(node_count=60, edge_count=90, community_count=7)
    delta = graph_structural_delta(prev, curr)
    assert delta.node_count_delta == -40
    assert delta.edge_count_delta == -110
    assert delta.community_count_delta == -3


def test_god_node_jaccard_identical_sets_is_one() -> None:
    prev = _snap(god_nodes=("A", "B", "C"))
    curr = _snap(god_nodes=("C", "B", "A"))  # same set, different order
    delta = graph_structural_delta(prev, curr)
    assert delta.god_node_jaccard_similarity == pytest.approx(1.0)
    assert delta.god_nodes_entered == ()
    assert delta.god_nodes_exited == ()


def test_god_node_jaccard_completely_disjoint_sets_is_zero() -> None:
    prev = _snap(god_nodes=("A", "B"))
    curr = _snap(god_nodes=("C", "D"))
    delta = graph_structural_delta(prev, curr)
    assert delta.god_node_jaccard_similarity == pytest.approx(0.0)
    assert set(delta.god_nodes_entered) == {"C", "D"}
    assert set(delta.god_nodes_exited) == {"A", "B"}


def test_god_node_jaccard_partial_overlap() -> None:
    prev = _snap(god_nodes=("A", "B", "C", "D"))
    curr = _snap(god_nodes=("C", "D", "E", "F"))
    delta = graph_structural_delta(prev, curr)
    # intersection {C, D} = 2, union {A,B,C,D,E,F} = 6 -> 2/6
    assert delta.god_node_jaccard_similarity == pytest.approx(2 / 6)
    assert set(delta.god_nodes_entered) == {"E", "F"}
    assert set(delta.god_nodes_exited) == {"A", "B"}


def test_god_node_jaccard_both_empty_is_one_not_zero() -> None:
    """Two snapshots with zero god nodes each represent 'nothing changed,' not
    'totally different' -- an empty-set/empty-set comparison must not
    misreport maximal churn via a 0/0 division artifact."""
    prev = _snap(god_nodes=())
    curr = _snap(god_nodes=())
    delta = graph_structural_delta(prev, curr)
    assert delta.god_node_jaccard_similarity == pytest.approx(1.0)


def test_god_node_jaccard_one_empty_one_populated_is_zero() -> None:
    prev = _snap(god_nodes=())
    curr = _snap(god_nodes=("A",))
    delta = graph_structural_delta(prev, curr)
    assert delta.god_node_jaccard_similarity == pytest.approx(0.0)
    assert delta.god_nodes_entered == ("A",)


def test_god_nodes_none_on_either_side_yields_none_not_fabricated_value() -> None:
    """Regression guard for the 2026-07-30 code-review finding: when a
    snapshot's god_nodes could not be determined (None -- unparseable
    GRAPH_REPORT.md), the delta must say "unknown" (None), not silently
    report 1.0 ("nothing changed") or 0.0 ("totally different") -- either
    fabricated value would misrepresent a parse failure as a real reading.
    Count deltas (which don't depend on god-node parsing) must still compute
    normally."""
    prev = _snap(node_count=100, edge_count=200, god_nodes=None)
    curr = _snap(node_count=110, edge_count=210, god_nodes=("A", "B"))
    delta = graph_structural_delta(prev, curr)
    assert delta.god_node_jaccard_similarity is None
    assert delta.god_nodes_entered is None
    assert delta.god_nodes_exited is None
    assert delta.node_count_delta == 10
    assert delta.edge_count_delta == 10


def test_god_nodes_none_on_curr_side_also_yields_none() -> None:
    prev = _snap(god_nodes=("A", "B"))
    curr = _snap(god_nodes=None)
    delta = graph_structural_delta(prev, curr)
    assert delta.god_node_jaccard_similarity is None


def test_god_nodes_both_none_yields_none_not_calm() -> None:
    """Two consecutive parse failures must not average out to 'calm' -- this
    is exactly the scenario that would have masked the real 2026-07-14
    destructive-graph-update incident had the god-node section itself been
    unparseable during that window (it wasn't, in the real replay -- but the
    code must not depend on that being true by luck)."""
    prev = _snap(node_count=100, god_nodes=None)
    curr = _snap(node_count=5, god_nodes=None)  # a real ~95% node-count crash
    delta = graph_structural_delta(prev, curr)
    assert delta.god_node_jaccard_similarity is None
    assert delta.node_count_delta == -95  # still correctly caught by the count delta
