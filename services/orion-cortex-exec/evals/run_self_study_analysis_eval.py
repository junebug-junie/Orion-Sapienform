#!/usr/bin/env python3
"""Live-data eval for `skills.self_study.analyze.v1`.

Unit tests prove each rule fires and refuses on constructed windows. They
cannot prove the thing that actually matters about this action in production:
that against REAL rows it can both FIRE and REST. An analysis that fires on
every window is digest spam wearing a gate; one that never fires is dead code
with a passing test suite. This is CLAUDE.md section 0A's live-data sanity
check, made runnable rather than eyeballed once.

Read-only. Never publishes -- `bus=None`, so nothing reaches the journal.

    python services/orion-cortex-exec/evals/run_self_study_analysis_eval.py
    ... --dsn postgresql://postgres:postgres@localhost:55432/conjourney

Exit 0 = pass. Exit 1 = a real gate failed. Exit 2 = could not run at all
(no DSN, no database) -- deliberately distinct, because "cannot measure" must
never be reported as "measured and fine".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "services" / "orion-cortex-exec"))

from app.self_study_analysis import (  # noqa: E402
    ANALYSIS_CELLS,
    SOURCE_SPECS,
    WINDOW_LADDER,
    run_self_study_analysis,
)
from orion.core.bus.bus_schemas import ServiceRef  # noqa: E402

# THE EVAL GRID IS PRODUCTION'S GRID, by construction, and that is the fix for
# a real miss rather than a tidiness preference. The first deployed version was
# pinned to a single 6h window while this eval swept 1/6/24/72h; it PASSED,
# because some cell fired -- and then produced 48 consecutive
# `skipped_not_notable` runs live, because 6h was the one width at which
# nothing fired. The eval was measuring a configuration production did not run.
# Importing `ANALYSIS_CELLS` makes that divergence impossible: whatever the
# action actually rotates over is exactly what gets graded here.
WINDOWS = WINDOW_LADDER

SOURCE_REF = ServiceRef(name="orion-cortex-exec-eval", version="0.1.0", node="athena")

# GATES, and why each one is here.
#
# 1. Every source must be READABLE. An `unavailable` reading means a table was
#    renamed or a column dropped out from under the spec -- the failure mode
#    that turns this action into a silent no-op while every test still passes.
# 2. The action must REST somewhere. If every window on every source fires, the
#    bars are decorative.
# 3. The action must FIRE somewhere across the whole grid. If nothing ever
#    fires against a live corpus with real gaps and real volume swings in it,
#    the bars are unreachable and the action cannot ever say anything.
MIN_QUIET_FRACTION = 0.25
MIN_FIRING_CELLS = 1


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=None, help="Postgres DSN; defaults to the service's own chain.")
    parser.add_argument("--json", action="store_true", help="Emit the grid as JSON.")
    args = parser.parse_args()

    if args.dsn:
        os.environ["SUBSTRATE_FELT_STATE_DATABASE_URL"] = args.dsn

    rows: list[dict] = []
    for source, hours in ANALYSIS_CELLS:
        if True:
            result = await run_self_study_analysis(
                bus=None,
                source_ref=SOURCE_REF,
                source=source,
                window_hours=hours,
                correlation_id=f"eval:{source}:{hours}",
            )
            rows.append(
                {
                    "source": source,
                    "table": SOURCE_SPECS[source].table,
                    "window_hours": hours,
                    "status": result.status,
                    "recent_rows": result.recent_rows,
                    "baseline_rows": result.baseline_rows,
                    "fired": [f.rule for f in result.findings],
                    "unavailable_reason": result.unavailable_reason,
                }
            )

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"{'source':20s} {'win':>6s} {'status':26s} {'recent':>7s} {'base':>7s}  fired")
        for row in rows:
            print(
                f"{row['source']:20s} {row['window_hours']:6.1f} {row['status']:26s} "
                f"{row['recent_rows']:7d} {row['baseline_rows']:7d}  {','.join(row['fired']) or '-'}"
            )

    if all(row["unavailable_reason"] == "database_url_unset" for row in rows):
        print("\nCANNOT RUN: no Postgres DSN reachable. This is not a pass.", file=sys.stderr)
        return 2

    failures: list[str] = []

    unreadable = [r for r in rows if r["status"] == "unavailable"]
    if unreadable:
        for row in unreadable:
            failures.append(
                f"GATE 1 unreadable: {row['source']} @ {row['window_hours']}h -> "
                f"{row['unavailable_reason']}"
            )

    # A `journal_failed` reading here is EXPECTED and is a fire, not a failure:
    # this eval passes bus=None on purpose, so any cell that would have
    # journalled reports that instead.
    fired_cells = [r for r in rows if r["fired"]]
    quiet_cells = [r for r in rows if not r["fired"] and r["status"] != "unavailable"]
    measurable = len(fired_cells) + len(quiet_cells)

    if measurable and (len(quiet_cells) / measurable) < MIN_QUIET_FRACTION:
        failures.append(
            f"GATE 2 never rests: only {len(quiet_cells)}/{measurable} cells quiet "
            f"(bar: {MIN_QUIET_FRACTION:.0%}). The notability bars are decorative."
        )
    if len(fired_cells) < MIN_FIRING_CELLS:
        failures.append(
            f"GATE 3 never fires: {len(fired_cells)} cells fired across "
            f"{measurable} measurable cells. The bars are unreachable."
        )

    print(
        f"\n{len(fired_cells)} fired / {len(quiet_cells)} quiet / {len(unreadable)} unreadable"
    )
    # Per-window breakdown, printed rather than gated. A window that is quiet
    # across all four sources TODAY is not necessarily a dead width -- "dead
    # today" is not "dead" -- but a width that is persistently silent here is
    # the signature of the miss described above, and it should be visible
    # rather than buried in an aggregate that passes.
    print("per-window fire counts (a persistently silent width is worth a look):")
    for hours in WINDOWS:
        at_window = [r for r in rows if r["window_hours"] == hours]
        fired_here = sum(1 for r in at_window if r["fired"])
        print(f"  {hours:6.1f}h  {fired_here}/{len(at_window)} sources fired")
    if failures:
        print("\nFAIL", file=sys.stderr)
        for line in failures:
            print("  " + line, file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
