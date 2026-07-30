#!/usr/bin/env python3
"""Read-only replay of ``codebase_prediction_error()`` against this repo's own real
commit history -- SSP §7's "measure before minting" discipline, same order every
other domain in ``orion/substrate/prediction_error.py`` has followed (see
docs/superpowers/specs/2026-07-30-codebase-mass-signal-design.md, "Recommended next
patch"). No bus wiring, no NODE_CHANNELS registration -- this script only proves the
function is non-degenerate before either of those exist.

Walks ``git log --reverse`` on the given branch/range, computes one
``git_churn_delta`` per consecutive commit pair (one tick per real commit), and
reports whether the resulting scores are non-degenerate (real variance, not flat)
and whether the domain can genuinely rest near a low/zero reading, per CLAUDE.md's
metric-quality-gate step 4.

Run:
    python scripts/analysis/measure_codebase_prediction_error.py
    python scripts/analysis/measure_codebase_prediction_error.py --max-commits 200
    python scripts/analysis/measure_codebase_prediction_error.py --since-sha <sha> --until-sha <sha>
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from orion.structural_mass.git_delta import git_churn_delta  # noqa: E402
from orion.substrate.prediction_error import (  # noqa: E402
    CodebaseMassBaseline,
    codebase_prediction_error,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", default=str(_REPO_ROOT))
    parser.add_argument("--since-sha", default=None, help="Exclusive lower bound; default walks full local history.")
    parser.add_argument("--until-sha", default="HEAD")
    parser.add_argument("--max-commits", type=int, default=500, help="Cap on ticks replayed (0 = no cap).")
    args = parser.parse_args()

    repo_path = Path(args.repo_path)
    shas = _commit_shas(repo_path, args.since_sha, args.until_sha, args.max_commits)
    if len(shas) < 2:
        print(f"Not enough commits to replay (found {len(shas)}) -- widen the range.")
        return 1

    baseline = CodebaseMassBaseline()
    raw_magnitudes: list[float] = []
    scores: list[float] = []
    variances_after_warmup: list[float] = []

    print(f"Replaying {len(shas) - 1} tick(s) across {len(shas)} commits "
          f"({shas[0][:10]}..{shas[-1][:10]})\n")
    print(f"{'sha':<12} {'commits':>7} {'+files':>6} {'-files':>6} {'~files':>6} "
          f"{'lines':>8} {'zscore/score':>14}")

    for prev_sha, head_sha in zip(shas, shas[1:]):
        delta = git_churn_delta(prev_sha, head_sha, repo_path=repo_path)
        result = codebase_prediction_error(delta, baseline)
        raw_magnitudes.append(float(delta.lines_changed))
        scores.append(result.score)
        if baseline.n >= 5:  # "warmed up" -- ignore cold-start variance=0.0 entries
            variances_after_warmup.append(result.baseline.variance)
        baseline = result.baseline
        print(
            f"{head_sha[:10]:<12} {delta.commit_count:>7} {delta.files_added:>6} "
            f"{delta.files_deleted:>6} {delta.files_modified:>6} "
            f"{delta.lines_changed:>8} {result.score:>14.4f}"
        )

    print("\n--- Distributional sanity (CLAUDE.md metric-quality-gate step 4) ---")
    print(f"raw magnitude (lines_changed): min={min(raw_magnitudes):.0f} "
          f"max={max(raw_magnitudes):.0f} mean={statistics.fmean(raw_magnitudes):.1f} "
          f"stdev={statistics.pstdev(raw_magnitudes):.1f}")
    print(f"score: min={min(scores):.4f} max={max(scores):.4f} "
          f"mean={statistics.fmean(scores):.4f}")
    n_zero_magnitude = sum(1 for m in raw_magnitudes if m == 0.0)
    print(f"zero-churn ticks (real no-op commits): {n_zero_magnitude}/{len(raw_magnitudes)}")
    if variances_after_warmup:
        print(f"warmed-up EWMA variance (n>=5): min={min(variances_after_warmup):.2f} "
              f"max={max(variances_after_warmup):.2f} "
              f"mean={statistics.fmean(variances_after_warmup):.2f}")

    if len(set(scores)) == 1:
        print("\nFLAG: every score identical -- degenerate, do not trust this domain yet.")
        return 1
    if min(scores) > 0.05 and len(scores) > 5:
        print("\nFLAG: never reads near-zero across the replayed window -- check for a "
              "structurally-incapable-of-calm bug (see bus_synaptic_prediction_error's "
              "2026-07-26 floor-bias history).")
        return 1

    print("\nOK: scores are non-degenerate and include near-zero readings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
