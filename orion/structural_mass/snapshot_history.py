"""Graphify structural-snapshot extraction and an append-only history log --
the ``snapshot_history`` piece of the codebase-mass domain
(docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md, Phase 1).

graphify itself retains no time series (each ``graphify update`` overwrites
``graphify-out/graph.json`` in place) -- this module is new state that fills
that gap, either going forward (``scripts/safe_graphify_update.sh`` appending
after each successful, non-reverted update -- not wired in this patch, see the
design spec's Phase 1 bullet) or backward (reconstructed from git history via
``git show <sha>:graphify-out/graph.json``, see
``scripts/analysis/measure_graph_structural_delta.py``).

**God nodes are read from ``GRAPH_REPORT.md``'s own "God Nodes" section, not
recomputed from raw edges here.** graphify already computes this (its own
degree-ranking logic, already tested, already the thing a human reads) --
duplicating that computation in this module would risk drifting from
graphify's real definition (weighting, tie-breaking, top-N cutoff) and
violates this program's "reuse the live pipeline, don't parallel it" rule
(SSP §7) one layer down from where it's usually invoked.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Consumes the *whole* header line (e.g. the "(most connected - your core
# abstractions)" suffix), not just the "## God Nodes" prefix -- otherwise the
# leftover suffix text would count as non-empty "section content" and make
# a legitimately empty god-nodes list misfire the format-changed/None check
# below (confirmed live while fixing that check).
_GOD_NODES_HEADER_RE = re.compile(r"^## God Nodes[^\n]*\n?", re.MULTILINE)
_GOD_NODES_LINE_RE = re.compile(r"^\d+\.\s+`([^`]+)`")


@dataclass(frozen=True)
class GraphSnapshotStats:
    node_count: int
    edge_count: int
    community_count: int
    # None means "could not determine god nodes for this snapshot" (missing
    # or unparseable GRAPH_REPORT.md section) -- distinct from `()`, a real
    # "zero god nodes." See parse_graph_report_god_nodes()'s docstring for why
    # this distinction matters (a silently-collapsed None->() would make
    # graph_structural_delta() misreport a parse failure as "nothing
    # changed").
    god_nodes: tuple[str, ...] | None = field(default_factory=tuple)
    # Provenance -- which commit this snapshot reflects (git SHA) and whether
    # it was reconstructed from history vs. captured live going forward.
    commit_sha: str | None = None
    backfilled: bool = False

    def to_json_dict(self) -> dict:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "community_count": self.community_count,
            "god_nodes": None if self.god_nodes is None else list(self.god_nodes),
            "commit_sha": self.commit_sha,
            "backfilled": self.backfilled,
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "GraphSnapshotStats":
        raw_god_nodes = data.get("god_nodes")
        return cls(
            node_count=int(data["node_count"]),
            edge_count=int(data["edge_count"]),
            community_count=int(data["community_count"]),
            god_nodes=None if raw_god_nodes is None else tuple(raw_god_nodes),
            commit_sha=data.get("commit_sha"),
            backfilled=bool(data.get("backfilled", False)),
        )


def parse_graph_report_god_nodes(report_text: str) -> tuple[str, ...] | None:
    """Extracts the ordered god-node label list from GRAPH_REPORT.md's
    ``## God Nodes`` section (``N. \\`Label\\` - D edges``), stopping at the
    next ``##`` heading.

    **Returns ``None``, not ``()``, whenever god-node data cannot actually be
    determined -- distinct from a real, empty result.** Two failure shapes
    both matter here, confirmed live by code review 2026-07-30: no
    ``## God Nodes`` header at all (report predates this section, or the
    report is missing/truncated), *or* the header is present with real
    section content but zero lines match the expected ``N. \\`Label\\```
    pattern (graphify's own report format changed). Both would otherwise
    silently collapse to the same ``()`` as "genuinely zero god nodes exist" --
    live-reproduced during review: two consecutive parse failures would make
    ``graph_structural_delta()`` report ``god_node_jaccard_similarity=1.0``
    ("nothing changed"), the exact "looks calm, isn't" failure shape
    CLAUDE.md's metric-quality-gate step 4 already names twice by incident
    (the permanent-0.27 bus_synaptic floor, and node:substrate.route's
    decay-to-zero). A header present with a genuinely *empty* body (real "no
    god nodes yet," e.g. a brand-new/tiny graph) is not conflated with a
    parse failure -- that case still returns ``()``.
    """
    header_match = _GOD_NODES_HEADER_RE.search(report_text)
    if header_match is None:
        return None
    section_start = header_match.end()
    next_header = report_text.find("\n## ", section_start)
    section = report_text[section_start : next_header if next_header != -1 else len(report_text)]
    labels: list[str] = []
    for line in section.splitlines():
        match = _GOD_NODES_LINE_RE.match(line.strip())
        if match:
            labels.append(match.group(1))
    if not labels and section.strip():
        # Header present, non-empty body, nothing matched -- the expected
        # format likely changed. Not the same state as a genuinely empty body.
        return None
    return tuple(labels)


def graph_snapshot_stats_from_text(
    graph_json_text: str,
    graph_report_text: str,
    *,
    commit_sha: str | None = None,
    backfilled: bool = False,
) -> GraphSnapshotStats:
    """Pure function: compute node/edge/community counts from a ``graph.json``
    payload's text, and the god-node list from a ``GRAPH_REPORT.md`` payload's
    text -- both taken as raw text (not file paths) so this works identically
    whether the caller read a live file or a ``git show`` blob."""
    graph = json.loads(graph_json_text)
    nodes = graph.get("nodes") or []
    links = graph.get("links") or []
    communities = {node.get("community") for node in nodes if isinstance(node, dict) and "community" in node}
    return GraphSnapshotStats(
        node_count=len(nodes),
        edge_count=len(links),
        community_count=len(communities),
        god_nodes=parse_graph_report_god_nodes(graph_report_text),
        commit_sha=commit_sha,
        backfilled=backfilled,
    )


def append_snapshot(path: str | Path, snapshot: GraphSnapshotStats) -> None:
    """Append one snapshot as a JSONL line. Append-only by design -- this is a
    history log, not a mutable table; a wrong entry gets corrected by
    appending a corrected one, not by editing history in place."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot.to_json_dict()) + "\n")


def read_snapshots(path: str | Path) -> list[GraphSnapshotStats]:
    """Reads the full JSONL history log in append order (oldest first). Missing
    file returns an empty list -- "no history captured yet" is a real, valid
    state for a brand-new log, not an error."""
    p = Path(path)
    if not p.exists():
        return []
    snapshots: list[GraphSnapshotStats] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            snapshots.append(GraphSnapshotStats.from_json_dict(json.loads(line)))
    return snapshots
