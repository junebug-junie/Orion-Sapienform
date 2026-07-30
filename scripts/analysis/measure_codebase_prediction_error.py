#!/usr/bin/env python3
"""Read-only replay of ``codebase_prediction_error()`` against this repo's own real
git/GitHub/graphify history -- SSP §7's "measure before minting" discipline, same
order every other domain in ``orion/substrate/prediction_error.py`` has followed
(see docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md). No bus
wiring, no NODE_CHANNELS registration -- this script only proves the composite
scoring function is non-degenerate before either of those exist.

Walks ``git log --reverse`` on the given branch/range, one tick per real commit.
Each tick always has a real ``git_churn_delta``. It also carries a real
``pr_lifecycle_delta`` for the window between the two commits' own timestamps
(requires a real authenticated ``gh`` CLI -- this is intentionally not mocked).
``graph_delta`` is only attached on ticks whose commit is one of the real,
sparse set of commits that actually touched ``graphify-out/graph.json``
(confirmed live: ~7 such commits exist in this repo's history) -- every other
tick correctly passes ``graph_delta=None``, exercising the composite function's
"not every domain fires every tick" contract for real, not synthetically.

Run:
    python scripts/analysis/measure_codebase_prediction_error.py
    python scripts/analysis/measure_codebase_prediction_error.py --max-commits 49
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from orion.structural_mass.git_delta import git_churn_delta  # noqa: E402
from orion.structural_mass.graph_delta import graph_structural_delta  # noqa: E402
from orion.structural_mass.pr_lifecycle import (  # noqa: E402
    fetch_recent_prs,
    pr_lifecycle_delta,
)
from orion.structural_mass.snapshot_history import (  # noqa: E402
    graph_snapshot_stats_from_text,
)
from orion.substrate.prediction_error import (  # noqa: E402
    CodebaseMassBaseline,
    codebase_prediction_error,
)

_GRAPH_JSON_PATH = "graphify-out/graph.json"
_GRAPH_REPORT_PATH = "graphify-out/GRAPH_REPORT.md"


def _commit_shas(repo_path: Path, since_sha: str | None, until_sha: str, max_commits: int) -> list[str]:
    args = ["log", "--reverse", "--format=%H"]
    if since_sha:
        args.append(f"{since_sha}..{until_sha}")
    else:
        args.append(until_sha)
    result = subprocess.run(
        ["git", *args], cwd=repo_path, capture_output=True, text=True, check=True
    )
    shas = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if max_commits and len(shas) > max_commits:
        shas = shas[-max_commits:]
    return shas


def _all_commit_timestamps(repo_path: Path) -> dict[str, datetime]:
    """One `git log` call for every commit's timestamp, not one `git show`
    per commit -- 2 subprocess spawns total instead of 2x the tick count
    (confirmed live: the naive per-commit version took 5+ minutes replaying
    1400 ticks; this is the same data, just fetched in bulk)."""
    result = subprocess.run(
        ["git", "log", "--format=%H|%cI"], cwd=repo_path, capture_output=True, text=True, check=True,
    )
    timestamps: dict[str, datetime] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, ts = line.partition("|")
        timestamps[sha] = datetime.fromisoformat(ts)
    return timestamps


def _graph_touching_shas(repo_path: Path) -> set[str]:
    result = subprocess.run(
        ["git", "log", "--reverse", "--format=%H", "--follow", "--", _GRAPH_JSON_PATH],
        cwd=repo_path, capture_output=True, text=True, check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _git_show(sha: str, path: str, repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{sha}:{path}"], cwd=repo_path, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", default=str(_REPO_ROOT))
    parser.add_argument("--since-sha", default=None, help="Exclusive lower bound; default walks full local history.")
    parser.add_argument("--until-sha", default="HEAD")
    parser.add_argument("--max-commits", type=int, default=500, help="Cap on ticks replayed (0 = no cap).")
    parser.add_argument("--owner", default="junebug-junie")
    parser.add_argument("--repo", default="Orion-Sapienform")
    parser.add_argument("--pr-fetch-limit", type=int, default=300)
    args = parser.parse_args()

    repo_path = Path(args.repo_path)
    shas = _commit_shas(repo_path, args.since_sha, args.until_sha, args.max_commits)
    if len(shas) < 2:
        print(f"Not enough commits to replay (found {len(shas)}) -- widen the range.")
        return 1

    graph_touching = _graph_touching_shas(repo_path)
    prs = fetch_recent_prs(args.owner, args.repo, limit=args.pr_fetch_limit)
    timestamps = _all_commit_timestamps(repo_path)
    print(f"Fetched {len(prs)} real PR record(s); {len(graph_touching)} real commit(s) "
          f"touch {_GRAPH_JSON_PATH}\n")

    baseline = CodebaseMassBaseline()
    scores: list[float] = []
    git_magnitudes: list[float] = []
    pr_magnitudes: list[float] = []
    graph_magnitudes: list[float] = []
    graph_ticks = 0
    git_variances_warm: list[float] = []
    pr_variances_warm: list[float] = []
    graph_variances_warm: list[float] = []

    print(f"Replaying {len(shas) - 1} tick(s) across {len(shas)} commits "
          f"({shas[0][:10]}..{shas[-1][:10]})\n")
    print(f"{'sha':<12} {'git_lines':>10} {'pr_events':>10} {'graph_mag':>10} {'score':>8}")

    for prev_sha, head_sha in zip(shas, shas[1:]):
        git_delta = git_churn_delta(prev_sha, head_sha, repo_path=repo_path)
        git_magnitudes.append(float(git_delta.lines_changed))

        since_ts = timestamps[prev_sha]
        until_ts = timestamps[head_sha]
        pr_delta = pr_lifecycle_delta(prs, since=since_ts, until=until_ts, fetch_limit=args.pr_fetch_limit)
        pr_magnitude = pr_delta.submitted_count + pr_delta.merged_count + pr_delta.closed_without_merge_count
        pr_magnitudes.append(float(pr_magnitude))

        graph_delta = None
        graph_magnitude_repr = "-"
        if head_sha in graph_touching:
            # Reconstruct against the *previous* graph-touching commit at or
            # before prev_sha, not necessarily prev_sha itself (graph-touching
            # commits are sparse -- most ticks aren't one).
            prior_graph_shas = [s for s in graph_touching if s in shas[: shas.index(head_sha)]]
            if prior_graph_shas:
                prior_sha = max(prior_graph_shas, key=shas.index)
                prev_snapshot = graph_snapshot_stats_from_text(
                    _git_show(prior_sha, _GRAPH_JSON_PATH, repo_path),
                    _git_show(prior_sha, _GRAPH_REPORT_PATH, repo_path),
                    commit_sha=prior_sha, backfilled=True,
                )
                curr_snapshot = graph_snapshot_stats_from_text(
                    _git_show(head_sha, _GRAPH_JSON_PATH, repo_path),
                    _git_show(head_sha, _GRAPH_REPORT_PATH, repo_path),
                    commit_sha=head_sha, backfilled=True,
                )
                graph_delta = graph_structural_delta(prev_snapshot, curr_snapshot)
                magnitude = abs(graph_delta.node_count_delta) + abs(graph_delta.edge_count_delta)
                graph_magnitudes.append(float(magnitude))
                graph_magnitude_repr = str(magnitude)
                graph_ticks += 1

        result = codebase_prediction_error(
            git_delta=git_delta, pr_delta=pr_delta, graph_delta=graph_delta, baseline=baseline
        )
        scores.append(result.score)
        baseline = result.baseline
        # git/pr fire every tick, so baseline.X.n >= 5 always coincides with a
        # real observation this tick -- but graph_delta is often None (most
        # ticks aren't one of the sparse graph-touching commits), and
        # _domain_zscore leaves a non-firing domain's baseline frozen
        # unchanged. Without the `graph_delta is not None` guard here, every
        # tick between two real graph observations would re-append the same
        # stale, already-recorded variance value -- confirmed live during
        # code review: this inflated 6 real graph observations into a
        # reported "n=240" of mostly-duplicated values before this fix.
        if baseline.git.n >= 5:
            git_variances_warm.append(baseline.git.variance)
        if baseline.pr.n >= 5:
            pr_variances_warm.append(baseline.pr.variance)
        if graph_delta is not None and baseline.graph.n >= 5:
            graph_variances_warm.append(baseline.graph.variance)
        print(f"{head_sha[:10]:<12} {git_delta.lines_changed:>10} {pr_magnitude:>10} "
              f"{graph_magnitude_repr:>10} {result.score:>8.4f}")

    print("\n--- Distributional sanity (CLAUDE.md metric-quality-gate step 4) ---")
    print(f"git magnitude:   min={min(git_magnitudes):.0f} max={max(git_magnitudes):.0f} "
          f"mean={statistics.fmean(git_magnitudes):.1f}")
    print(f"pr magnitude:    min={min(pr_magnitudes):.0f} max={max(pr_magnitudes):.0f} "
          f"mean={statistics.fmean(pr_magnitudes):.1f}")
    if graph_magnitudes:
        print(f"graph magnitude: min={min(graph_magnitudes):.0f} max={max(graph_magnitudes):.0f} "
              f"mean={statistics.fmean(graph_magnitudes):.1f} ({graph_ticks} real tick(s) with graph data)")
    else:
        print("graph magnitude: no real graph-touching commits in this replay window")
    print(f"composite score: min={min(scores):.4f} max={max(scores):.4f} "
          f"mean={statistics.fmean(scores):.4f}")

    print("\n--- Per-domain warmed-up EWMA variance (n>=5) -- for min_variance calibration ---")
    for label, values in (("git", git_variances_warm), ("pr", pr_variances_warm), ("graph", graph_variances_warm)):
        if values:
            print(f"{label:>6}: min={min(values):.4f} max={max(values):.4f} mean={statistics.fmean(values):.4f} n={len(values)}")
        else:
            print(f"{label:>6}: no warmed-up samples (fewer than 5 real ticks for this domain)")

    if len(set(scores)) == 1:
        print("\nFLAG: every score identical -- degenerate, do not trust this domain yet.")
        return 1
    if min(scores) > 0.05 and len(scores) > 5:
        print("\nFLAG: never reads near-zero across the replayed window -- check for a "
              "structurally-incapable-of-calm bug (see bus_synaptic_prediction_error's "
              "2026-07-26 floor-bias history).")
        return 1

    print("\nOK: composite scores are non-degenerate and include near-zero readings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
