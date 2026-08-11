#!/usr/bin/env python3
"""Read-only empirical streak-length distribution over real
`goal_provenance_streak_ticks` history.

Part H of `docs/superpowers/specs/2026-07-30-goal-system-remaining-gaps-design.md`
(Missing Question 5): is `ORION_GOAL_PROVENANCE_MIN_STREAK`'s placeholder value of 3
the right debounce for `orion-attention-runtime`'s goal-provenance producer, or too
strict/loose? `FieldGoalProvenanceV1` (published to `orion:memory:goals:proposed`) alone
cannot answer this -- it only exists once a streak has *already* survived `min_streak`
ticks, a censored sample that can show a survival rate, never a rejection rate. This
script reads the real, uncensored per-tick telemetry added 2026-08-11
(`DominanceStreakTickV1` / `orion:debug:attention:streak_tick` /
`goal_provenance_streak_ticks`) and reconstructs the true streak-length distribution: how
long each real node-target dominance run actually lasted, regardless of whether it ever
crossed the debounce.

This is the "measure before minting" precedent SSP's own item 5 (emergent-clustering
probe, see `measure_emergent_clustering_probe.py`) already established for this repo --
same shape, applied to a different real signal.

Run:
    python scripts/analysis/measure_goal_provenance_streak_distribution.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Explicit path insert (not a plain relative import) so this module loads the same way
# whether run directly (`python scripts/analysis/measure_goal_provenance_streak_distribution.py`)
# or loaded by file path via importlib, as this module's own test file does.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pg_readonly import open_readonly_connection  # noqa: E402

logger = logging.getLogger("orion.analysis.goal_provenance_streak_distribution")

MIN_TOTAL_ROWS: int = 50
MAX_ROWS: int = 500_000

OUTPUT_DIR = Path("/tmp/goal-provenance-streak-distribution")
REPORT_PATH = OUTPUT_DIR / "report.md"
CSV_PATH = OUTPUT_DIR / "runs.csv"

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"


# ===========================================================================
# Pure layer -- no I/O. Reconstructs streak runs from a time-ordered sequence
# of per-tick rows and computes distributions. Exercised directly by unit
# tests with synthetic fixtures, no DB.
# ===========================================================================


@dataclass(frozen=True)
class StreakTickRow:
    observed_at: datetime
    target_id: Optional[str]
    streak_count: int
    min_streak_at_tick: int
    qualified: bool


@dataclass(frozen=True)
class StreakRun:
    target_id: str
    max_count: int
    n_ticks: int
    start_ts: datetime
    end_ts: datetime
    # True only for the run still active on the very last observed row -- its final
    # length is unknown (right-censored), not a real completed measurement. Excluded
    # from the distribution by default; see `reconstruct_streak_runs`'s docstring.
    ongoing: bool
    # True only for the FIRST run when the caller knows older history existed beyond
    # what was fetched (left-censored: this run may have started before the fetched
    # window, so its true start/full length is unknown, same rationale as `ongoing`
    # for the last run). Always False when the full real history was fetched.
    left_censored: bool
    # True if this run ever crossed its own min_streak_at_tick while it was live --
    # i.e. it would have (or did) trigger a real FieldGoalProvenanceV1 emission.
    qualified: bool


def reconstruct_streak_runs(rows: list[StreakTickRow], *, first_run_left_censored: bool = False) -> list[StreakRun]:
    """Group a time-ordered sequence of per-tick rows into maximal runs of
    consecutive ticks sharing the same non-null `target_id`.

    A run ends (and the next one begins) whenever `target_id` changes from the
    previous row, or is `None` (`update_dominance_streak`'s own no-winner reset
    case -- rows with `target_id=None` are dropped entirely, they are not a run
    of length zero). The LAST run in the input is marked `ongoing=True` and its
    `max_count` is a lower bound, not its true final length -- the streak may
    keep growing on ticks that haven't happened yet. If `first_run_left_censored`
    is True (the caller fetched a bounded window and knows older history exists
    before it), the FIRST run is marked `left_censored=True` for the same reason
    in reverse: it may have started before the fetched window. Rows must already
    be sorted ascending by `observed_at`; this function does not sort (the
    caller's SQL query is the single source of ordering truth).
    """
    runs: list[StreakRun] = []
    current: Optional[dict[str, Any]] = None

    for row in rows:
        if row.target_id is None:
            if current is not None:
                runs.append(
                    StreakRun(
                        target_id=current["target_id"],
                        max_count=current["max_count"],
                        n_ticks=current["n_ticks"],
                        start_ts=current["start_ts"],
                        end_ts=current["end_ts"],
                        ongoing=False,
                        left_censored=False,
                        qualified=current["qualified"],
                    )
                )
                current = None
            continue
        if current is None or row.target_id != current["target_id"]:
            if current is not None:
                runs.append(
                    StreakRun(
                        target_id=current["target_id"],
                        max_count=current["max_count"],
                        n_ticks=current["n_ticks"],
                        start_ts=current["start_ts"],
                        end_ts=current["end_ts"],
                        ongoing=False,
                        left_censored=False,
                        qualified=current["qualified"],
                    )
                )
            current = {
                "target_id": row.target_id,
                "max_count": row.streak_count,
                "n_ticks": 1,
                "start_ts": row.observed_at,
                "end_ts": row.observed_at,
                "qualified": row.qualified,
            }
        else:
            current["max_count"] = max(current["max_count"], row.streak_count)
            current["n_ticks"] += 1
            current["end_ts"] = row.observed_at
            current["qualified"] = current["qualified"] or row.qualified

    if current is not None:
        runs.append(
            StreakRun(
                target_id=current["target_id"],
                max_count=current["max_count"],
                n_ticks=current["n_ticks"],
                start_ts=current["start_ts"],
                end_ts=current["end_ts"],
                ongoing=True,
                left_censored=False,
                qualified=current["qualified"],
            )
        )

    if first_run_left_censored and runs:
        import dataclasses

        runs[0] = dataclasses.replace(runs[0], left_censored=True)
    return runs


def _is_completed(run: StreakRun) -> bool:
    """A run is a real, fully-measured completed run only if it is neither
    right-censored (`ongoing` -- may still be growing) nor left-censored
    (`left_censored` -- may have started before a truncated fetch window)."""
    return not run.ongoing and not run.left_censored


def length_distribution(runs: list[StreakRun]) -> Counter:
    """Histogram of completed runs' `max_count`. Censored runs (either end) are
    excluded -- their true final length is unknown, and including a lower-bound
    value would understate the real distribution's tail."""
    return Counter(r.max_count for r in runs if _is_completed(r))


def qualification_rate_at(runs: list[StreakRun], min_streak: int) -> Optional[float]:
    """Fraction of completed runs whose `max_count >= min_streak` -- i.e. what
    share of real dominance runs would clear this candidate debounce. `None` if
    there are no completed runs to measure (not zero -- zero would falsely read
    as 'nothing ever qualifies')."""
    completed = [r for r in runs if _is_completed(r)]
    if not completed:
        return None
    return sum(1 for r in completed if r.max_count >= min_streak) / len(completed)


def target_id_distribution(runs: list[StreakRun]) -> Counter:
    """Which target_ids actually won real dominance runs, and how often --
    completed runs only, same censoring rationale as `length_distribution`."""
    return Counter(r.target_id for r in runs if _is_completed(r))


# ===========================================================================
# I/O layer -- psycopg2 read-only. open_readonly_connection is shared (see
# _pg_readonly.py); its connection contract still mirrors
# measure_emergent_clustering_probe.py's own (pre-existing, un-migrated) copy
# exactly (refuses a non-read-only session).
# ===========================================================================


def fetch_streak_tick_rows(conn, max_rows: int = MAX_ROWS) -> tuple[list[StreakTickRow], bool, bool]:
    """The NEWEST `max_rows` real rows, returned in ASC order (oldest-first) for
    `reconstruct_streak_runs`.

    Fetches DESC (most recent first) then reverses in Python -- deliberately the
    newest slice, not the oldest (review fix, 2026-08-11): fetching the oldest
    `max_rows` via `ORDER BY ... ASC LIMIT` would mean every run past
    MAX_ROWS accumulated silently re-analyzes the identical stale slice forever
    and never sees new data once the table outgrows the limit. The newest slice
    is also the one that actually matters for a "should we change the live
    default now" calibration question.

    Returns `(rows, truncated, ok)`: `ok=False` means the query itself failed
    (bad connection, missing table, schema drift) -- distinct from `rows=[]`
    with `ok=True`, which means the query succeeded and the table is genuinely
    (near-)empty. Conflating these two would report "insufficient data, re-run
    later" for a broken query that will never self-heal by waiting.
    """
    if conn is None:
        return [], False, False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT observed_at, target_id, streak_count, min_streak_at_tick, qualified
                FROM goal_provenance_streak_ticks
                ORDER BY observed_at DESC, created_at DESC
                LIMIT %s
                """,
                (max_rows,),
            )
            rows = cur.fetchall()
    except Exception:
        logger.error("failed to fetch goal_provenance_streak_ticks rows", exc_info=True)
        return [], False, False
    out: list[StreakTickRow] = []
    for observed_at, target_id, streak_count, min_streak_at_tick, qualified in rows:
        if observed_at is None:
            continue
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        out.append(
            StreakTickRow(
                observed_at=observed_at,
                target_id=target_id,
                streak_count=int(streak_count),
                min_streak_at_tick=int(min_streak_at_tick),
                qualified=bool(qualified),
            )
        )
    out.reverse()  # DESC fetch -> ASC order for reconstruction
    return out, len(rows) >= max_rows, True


# ===========================================================================
# Report rendering + orchestration.
# ===========================================================================


def write_runs_csv(path: Path, runs: list[StreakRun]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["target_id", "max_count", "n_ticks", "start_ts", "end_ts", "ongoing", "left_censored", "qualified"]
        )
        for r in runs:
            writer.writerow(
                [
                    r.target_id,
                    r.max_count,
                    r.n_ticks,
                    r.start_ts.isoformat(),
                    r.end_ts.isoformat(),
                    r.ongoing,
                    r.left_censored,
                    r.qualified,
                ]
            )


def render_report(
    *,
    total_rows: int,
    runs: list[StreakRun],
    current_min_streak: Optional[int],
    caveats: list[str],
) -> str:
    completed = [r for r in runs if _is_completed(r)]
    dist = length_distribution(runs)
    targets = target_id_distribution(runs)

    lines = [
        "# Goal-Provenance Streak-Length Distribution (Part H, Missing Question 5)",
        "",
        "Read-only. Reconstructs real node-target dominance runs from "
        "`goal_provenance_streak_ticks` (every real tick, not just qualifying "
        "`FieldGoalProvenanceV1` emissions -- see that table's own comment for why the "
        "qualifying-only channel alone cannot answer this question).",
        "",
        f"- Total per-tick rows: {total_rows}",
        f"- Reconstructed runs: {len(runs)} ({len(completed)} completed, "
        f"{sum(1 for r in runs if r.ongoing)} still ongoing as of the last observed row, "
        f"{sum(1 for r in runs if r.left_censored)} left-censored -- may have started before "
        f"the fetched window)",
        "",
    ]

    if not completed:
        lines.extend(
            [
                "## INSUFFICIENT REAL DATA",
                "",
                "No completed runs yet -- either too little history has accumulated, or the "
                "single dominant target has never changed since telemetry started (an ongoing "
                "run with no completed sibling to measure). No distribution or calibration "
                "recommendation is reported; re-run once real data has accumulated for a few "
                "days, per this instrumentation's own design intent.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Completed-run length distribution",
                "",
                "| streak length (max_count reached) | # runs | share of completed runs |",
                "| --- | --- | --- |",
            ]
        )
        for length in sorted(dist):
            count = dist[length]
            lines.append(f"| {length} | {count} | {count / len(completed) * 100:.2f}% |")

        lines.extend(
            [
                "",
                "## Candidate min_streak qualification rates",
                "",
                "What share of real completed runs would clear each candidate debounce -- "
                "i.e. the true survival rate `orion:memory:goals:proposed` alone can never show, "
                "since it only ever sees runs that already survived whatever `min_streak` was "
                "active at the time.",
                "",
                "| candidate min_streak | qualification rate |",
                "| --- | --- |",
            ]
        )
        for candidate in range(1, max(dist) + 2):
            rate = qualification_rate_at(runs, candidate)
            marker = " (live default)" if current_min_streak == candidate else ""
            lines.append(f"| {candidate}{marker} | {rate * 100:.2f}% |" if rate is not None else f"| {candidate}{marker} | n/a |")

        lines.extend(
            [
                "",
                "## Which node-targets actually won a completed dominance run",
                "",
                "| target_id | completed runs won | share |",
                "| --- | --- | --- |",
            ]
        )
        for target_id, count in targets.most_common():
            lines.append(f"| `{target_id}` | {count} | {count / len(completed) * 100:.2f}% |")
        lines.append("")

    lines.extend(["## Coverage caveats", ""])
    if caveats:
        lines.extend(f"- {c}" for c in caveats)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def run() -> int:
    dsn = os.environ.get("POSTGRES_URI", DEFAULT_POSTGRES_URI)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    caveats: list[str] = []

    conn = open_readonly_connection(dsn)
    if conn is None:
        caveats.append("postgres unavailable or not read-only; nothing measured")
        report = render_report(total_rows=0, runs=[], current_min_streak=None, caveats=caveats)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        return 2

    rows, truncated, query_ok = fetch_streak_tick_rows(conn)
    try:
        conn.close()
    except Exception:
        pass
    if not query_ok:
        caveats.append(
            "the goal_provenance_streak_ticks query itself failed (missing table, broken "
            "query, permission error) -- this is NOT 'insufficient data, wait and re-run': "
            "see the logged exception for the real cause"
        )
        report = render_report(total_rows=0, runs=[], current_min_streak=None, caveats=caveats)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        return 2
    if truncated:
        caveats.append(
            f"rows truncated at MAX_ROWS={MAX_ROWS} -- fetched the newest {MAX_ROWS} rows "
            "(ORDER BY observed_at DESC), so older history beyond that was dropped; the "
            "first reconstructed run is marked left_censored (see report) since it may have "
            "started before this fetch window"
        )

    total_rows = len(rows)
    if total_rows < MIN_TOTAL_ROWS:
        caveats.append(
            f"insufficient real goal_provenance_streak_ticks history (rows={total_rows}, "
            f"min_required={MIN_TOTAL_ROWS}); cannot run a meaningful measurement yet"
        )
        report = render_report(total_rows=total_rows, runs=[], current_min_streak=None, caveats=caveats)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)
        return 2

    current_min_streak = rows[-1].min_streak_at_tick
    runs = reconstruct_streak_runs(rows, first_run_left_censored=truncated)
    write_runs_csv(CSV_PATH, runs)
    report = render_report(total_rows=total_rows, runs=runs, current_min_streak=current_min_streak, caveats=caveats)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nartifacts: {REPORT_PATH}, {CSV_PATH}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Read-only empirical streak-length distribution over real goal_provenance_streak_ticks history."
    )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_arg_parser().parse_args(argv)
    return run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
