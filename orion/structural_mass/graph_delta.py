"""Graphify structural-graph deltas -- the ``graph_delta`` piece of the
codebase-mass domain (docs/superpowers/specs/2026-07-30-codebase-mass-signal-
design.md, Phase 1). Diffs two ``GraphSnapshotStats``
(``orion/structural_mass/snapshot_history.py``) into node/edge/community count
deltas (design spec idea #2) and god-node Jaccard churn (idea #3).

**Hop-distance drift (idea #6) is deliberately not implemented in this patch.**
Unlike the other two ideas, which are pure functions over already-computed
snapshot summaries, hop-distance drift requires live shortest-path traversal
calls (``graphify path``) over a fixed sample of node pairs -- a materially
different cost/complexity class, and not yet validated as non-degenerate on
real data the way the two ideas here now are (see
``scripts/analysis/measure_graph_structural_delta.py``). Building it alongside
these two before either is proven would be exactly the kind of premature
scope-stacking this repo's "thin seams, not ornamental layers" rule exists to
prevent -- a real follow-up, not dropped, just sequenced after this slice
measures clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orion.structural_mass.snapshot_history import GraphSnapshotStats


@dataclass(frozen=True)
class GraphStructuralDelta:
    node_count_delta: int
    edge_count_delta: int
    community_count_delta: int
    # 0.0 (completely different god-node sets) to 1.0 (identical sets) --
    # Jaccard similarity of prev/curr god-node label sets. 1.0 (not 0.0) when
    # both snapshots have zero *real* god nodes -- "nothing changed" is the
    # correct read for two empty sets, not "totally different." None (not a
    # fabricated 1.0 or 0.0) when either snapshot's god_nodes could not be
    # determined at all (GraphSnapshotStats.god_nodes is None) -- see
    # parse_graph_report_god_nodes()'s docstring for why collapsing "unknown"
    # into "unchanged" is exactly the failure this repo has been burned by
    # twice before (CLAUDE.md's metric-quality-gate step 4).
    god_node_jaccard_similarity: float | None
    god_nodes_entered: tuple[str, ...] | None = field(default_factory=tuple)
    god_nodes_exited: tuple[str, ...] | None = field(default_factory=tuple)


def graph_structural_delta(
    prev: GraphSnapshotStats,
    curr: GraphSnapshotStats,
) -> GraphStructuralDelta:
    """Pure diff of two graph snapshots -- no I/O, no state, mirrors
    ``git_churn_delta``'s and ``pr_lifecycle_delta``'s shape (a caller-owned
    pair of "before"/"after" states in, a plain delta out).

    God-node churn is Jaccard similarity of the two label *sets* (order-
    independent) rather than a positional diff -- a god node re-ranking from
    #3 to #7 without leaving the top-10 list entirely is real churn by rank
    but not by *set membership*, and this program's SSP §7 "reuse the live
    pipeline" framing treats the top-10 list itself (not its internal order)
    as the meaningful graphify-computed artifact to track. ``god_nodes_
    entered``/``god_nodes_exited`` name exactly which labels moved, for a
    future concept-classification consumer that wants to say *which* god node
    changed, not just that churn happened.

    **``god_node_jaccard_similarity`` (and ``entered``/``exited``) are
    ``None``, not a fabricated value, whenever either side's ``god_nodes`` is
    ``None``** (unparseable/missing GRAPH_REPORT.md section) -- live-confirmed
    during code review 2026-07-30: during this repo's own real
    graph-shrink-then-revert incident (CLAUDE.md's documented 2026-07-14
    destructive-update bug), the god-node top-10 list happened to survive
    intact even while node/edge counts dropped ~95%, so jaccard alone would
    have read "1.0, nothing changed" through the entire incident regardless
    of this fix -- ``node_count_delta``/``edge_count_delta`` are what actually
    catch that case. This ``None`` handling protects a *different* failure
    mode: two snapshots whose god-node data genuinely could not be read at
    all must not silently report "calm" either.
    """
    node_count_delta = curr.node_count - prev.node_count
    edge_count_delta = curr.edge_count - prev.edge_count
    community_count_delta = curr.community_count - prev.community_count

    if prev.god_nodes is None or curr.god_nodes is None:
        return GraphStructuralDelta(
            node_count_delta=node_count_delta,
            edge_count_delta=edge_count_delta,
            community_count_delta=community_count_delta,
            god_node_jaccard_similarity=None,
            god_nodes_entered=None,
            god_nodes_exited=None,
        )

    prev_set = set(prev.god_nodes)
    curr_set = set(curr.god_nodes)
    union = prev_set | curr_set
    jaccard = 1.0 if not union else len(prev_set & curr_set) / len(union)

    return GraphStructuralDelta(
        node_count_delta=node_count_delta,
        edge_count_delta=edge_count_delta,
        community_count_delta=community_count_delta,
        god_node_jaccard_similarity=jaccard,
        god_nodes_entered=tuple(curr_set - prev_set),
        god_nodes_exited=tuple(prev_set - curr_set),
    )
