#!/usr/bin/env python3
"""Read-only replay of ``graph_structural_delta()`` against this repo's real,
sparse ``graphify-out/graph.json``/``GRAPH_REPORT.md`` git history -- SSP §7's
"measure before minting" discipline (docs/superpowers/specs/2026-07-30-
codebase-mass-signal-design.md, "Backfill" section).

graphify retains no time series of its own (each ``graphify update`` overwrites
the snapshot in place) -- but git itself retains every historical commit that
touched these two files, so this script reconstructs each one via
``git show <sha>:path`` and diffs successive reconstructions. Real data,
irregularly spaced, none before graphify's first commit to this repo
(2026-07-14, PR #1034) -- sparse but real, exactly as the design spec's
Backfill section describes, not a dense synthetic series.

Run:
    python scripts/analysis/measure_graph_structural_delta.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from orion.structural_mass.graph_delta import graph_structural_delta  # noqa: E402
from orion.structural_mass.snapshot_history import (  # noqa: E402
    graph_snapshot_stats_from_text,
)

_GRAPH_JSON_PATH = "graphify-out/graph.json"
_GRAPH_REPORT_PATH = "graphify-out/GRAPH_REPORT.md"


def _git_show(sha: str, path: str, repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{sha}:{path}"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "does not exist" in stderr or "exists on disk, but not" in stderr:
            # Real, expected case: GRAPH_REPORT.md (or, in principle,
            # graph.json) not present yet at an early commit -- not every
            # commit that touches one necessarily touches the other.
            return ""
        # Any other failure (corrupted object, ambiguous ref, etc.) is not
        # the same as "file not present yet" -- surface it loudly rather than
        # silently treating it as an empty/missing file, which this replay's
        # whole purpose (trustworthy live-data sanity checking) depends on
        # not doing quietly.
        print(f"WARNING: git show {sha}:{path} failed unexpectedly: {stderr}", file=sys.stderr)
        return ""
    return result.stdout


def _commits_touching(path: str, repo_path: Path) -> list[str]:
    result = subprocess.run(
        ["git", "log", "--reverse", "--format=%H", "--follow", "--", path],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    repo_path = _REPO_ROOT
    commit_shas = _commits_touching(_GRAPH_JSON_PATH, repo_path)
    if len(commit_shas) < 2:
        print(f"Not enough historical commits touching {_GRAPH_JSON_PATH} to replay "
              f"(found {len(commit_shas)}).")
        return 1

    print(f"Replaying {len(commit_shas) - 1} delta(s) across {len(commit_shas)} "
          f"real historical commits touching {_GRAPH_JSON_PATH}\n")

    snapshots = []
    for sha in commit_shas:
        graph_json_text = _git_show(sha, _GRAPH_JSON_PATH, repo_path)
        graph_report_text = _git_show(sha, _GRAPH_REPORT_PATH, repo_path)
        if not graph_json_text.strip():
            print(f"{sha[:10]}: {_GRAPH_JSON_PATH} empty/missing at this commit -- skipping")
            continue
        snapshot = graph_snapshot_stats_from_text(
            graph_json_text, graph_report_text, commit_sha=sha, backfilled=True
        )
        snapshots.append(snapshot)
        god_nodes_repr = "N/A (unparseable)" if snapshot.god_nodes is None else str(len(snapshot.god_nodes))
        print(f"{sha[:10]}: nodes={snapshot.node_count:>6} edges={snapshot.edge_count:>6} "
              f"communities={snapshot.community_count:>5} god_nodes={god_nodes_repr}")

    print(f"\n--- Deltas between {len(snapshots)} reconstructed snapshots ---")
    node_deltas = []
    jaccards = []
    unparseable_god_node_pairs = 0
    for prev, curr in zip(snapshots, snapshots[1:]):
        delta = graph_structural_delta(prev, curr)
        node_deltas.append(delta.node_count_delta)
        if delta.god_node_jaccard_similarity is None:
            unparseable_god_node_pairs += 1
            jaccard_repr = "N/A (unparseable)"
        else:
            jaccards.append(delta.god_node_jaccard_similarity)
            jaccard_repr = f"{delta.god_node_jaccard_similarity:.3f}"
        print(
            f"{prev.commit_sha[:10]} -> {curr.commit_sha[:10]}: "
            f"nodes {delta.node_count_delta:+d}  edges {delta.edge_count_delta:+d}  "
            f"communities {delta.community_count_delta:+d}  "
            f"god_node_jaccard {jaccard_repr}  "
            f"entered={delta.god_nodes_entered}  exited={delta.god_nodes_exited}"
        )

    print("\n--- Distributional sanity (CLAUDE.md metric-quality-gate step 4) ---")
    if len(snapshots) < 2:
        print("FLAG: fewer than 2 real reconstructed snapshots -- nothing to diff.")
        return 1
    if len(set(node_deltas)) == 1 and len(node_deltas) > 1:
        print("FLAG: every node-count delta identical -- degenerate, do not trust yet.")
        return 1
    print(f"node_count_delta: min={min(node_deltas)} max={max(node_deltas)} "
          f"(non-degenerate: {len(set(node_deltas)) > 1})")
    if jaccards:
        print(f"god_node_jaccard_similarity: min={min(jaccards):.3f} max={max(jaccards):.3f} "
              f"({unparseable_god_node_pairs} pair(s) unparseable, excluded)")
    else:
        print(f"god_node_jaccard_similarity: no parseable pairs "
              f"({unparseable_god_node_pairs} pair(s) unparseable)")
    print("\nOK: real, sparse, non-degenerate deltas across genuine historical "
          "graphify commits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
