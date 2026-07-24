#!/usr/bin/env python3
"""Read-only measurement: is the AST/HOT reducer's `field_salience_only`
branch structurally starved by the broadcast-lane branch above it?

Context: `orion/substrate/attention_self_model.py::reduce_attention_self_model`
branches on `attention_reason` in strict elif order --
`top_down_override` > `bottom_up_salience` > `field_salience_only` > `no_data`.
The new Active-Inference confidence formula (PR #1301/#1304,
`_aggregate_prediction_error_confidence`) only computes/surfaces inside the
`field_salience_only` branch. A live run of `measure_ast_hot_reducer.py`
(2026-07-24, 8h window, 9208 ticks) found that branch fires on only 4 ticks
(0.04%) -- this script traces *why*, quantitatively, instead of guessing from
reading the branch order alone.

No DB, no bus, no I/O beyond reading the CSV artifact
`measure_ast_hot_reducer.py` already writes (`/tmp/ast-hot-reducer/ticks.csv`
by default) -- this is a secondary analysis pass over that script's own
real-data replay output, not a duplicate Postgres-querying pipeline. Per
program-charter section 7 ("measure before minting"): this exists to check
whether item 2's Active-Inference confidence claim (Sentience Striving
Program charter section 6 objective 2) is real in the sense of "computed
correctly when it runs" AND real in the sense of "actually exercised by live
traffic" -- these are different claims, and only the first has been checked
so far.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

DEFAULT_TICKS_CSV = "/tmp/ast-hot-reducer/ticks.csv"


def _parse_bool(value: str) -> bool | None:
    v = (value or "").strip().lower()
    if v in ("true", "1"):
        return True
    if v in ("false", "0"):
        return False
    return None


def analyze_branch_starvation(rows: list[dict]) -> dict:
    """Pure function: given parsed tick rows (dicts with at least
    `attention_reason`, `broadcast_lane_present`, `broadcast_lane_stale`),
    quantify how often the broadcast lane is fresh (present and not stale)
    -- the condition that structurally preempts `field_salience_only` -- and
    cross-check that the reducer's own elif ordering was actually honored in
    this real data (a broadcast-fresh tick should never show
    `field_salience_only`; a contradiction here would mean either this
    script's row-parsing is wrong or the reducer's branch logic changed
    since this script was written).

    Never raises on malformed rows: an unparseable bool is treated as
    "unknown" and excluded from the fresh/stale denominator, not defaulted
    to True or False (a silent default here would quietly bias the
    percentage this script exists to report honestly).
    """
    total = len(rows)
    reason_counts: dict[str, int] = {}
    broadcast_fresh = 0
    broadcast_known = 0
    contradictions: list[dict] = []

    for row in rows:
        reason = row.get("attention_reason") or "unknown"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        present = _parse_bool(row.get("broadcast_lane_present", ""))
        stale = _parse_bool(row.get("broadcast_lane_stale", ""))
        if present is None or stale is None:
            continue
        broadcast_known += 1
        fresh = present and not stale
        if fresh:
            broadcast_fresh += 1
            if reason == "field_salience_only":
                contradictions.append(row)

    fresh_pct = (broadcast_fresh / broadcast_known * 100.0) if broadcast_known else 0.0
    field_salience_only_count = reason_counts.get("field_salience_only", 0)
    field_salience_only_pct = (field_salience_only_count / total * 100.0) if total else 0.0

    return {
        "total_ticks": total,
        "reason_counts": reason_counts,
        "broadcast_known_ticks": broadcast_known,
        "broadcast_fresh_ticks": broadcast_fresh,
        "broadcast_fresh_pct": round(fresh_pct, 4),
        "field_salience_only_count": field_salience_only_count,
        "field_salience_only_pct": round(field_salience_only_pct, 4),
        "elif_ordering_contradictions": len(contradictions),
        "starvation_confirmed": (
            broadcast_known > 0
            and fresh_pct > 90.0
            and field_salience_only_pct < 10.0
            and len(contradictions) == 0
        ),
    }


def render_report(result: dict, csv_path: str) -> str:
    lines = [
        "# AST/HOT reducer: field_salience_only branch-starvation check",
        "",
        f"Source: `{csv_path}` (real replay output from `measure_ast_hot_reducer.py`)",
        f"Total ticks: {result['total_ticks']}",
        "",
        "## attention_reason distribution",
        "",
        "| reason | count | pct |",
        "| --- | --- | --- |",
    ]
    total = result["total_ticks"] or 1
    for reason, count in sorted(result["reason_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {reason} | {count} | {count / total * 100:.2f}% |")

    lines += [
        "",
        "## Broadcast-lane freshness (the gating condition)",
        "",
        f"- Ticks with known broadcast presence/staleness: {result['broadcast_known_ticks']}",
        f"- Fresh (present AND not stale): {result['broadcast_fresh_ticks']} "
        f"({result['broadcast_fresh_pct']}%)",
        "",
        "## Elif-ordering sanity check",
        "",
        f"- Contradictions (a broadcast-fresh tick showing `field_salience_only`): "
        f"{result['elif_ordering_contradictions']}",
        (
            "  - Zero contradictions: the reducer's own elif order "
            "(`top_down_override` > `bottom_up_salience` > `field_salience_only`) "
            "was honored exactly as coded in this real data."
            if result["elif_ordering_contradictions"] == 0
            else "  - NONZERO -- either this script's parsing is wrong or the reducer's "
            "branch order has changed since this script was written. Investigate before "
            "trusting the conclusion below."
        ),
        "",
        "## Conclusion",
        "",
    ]
    if result["starvation_confirmed"]:
        lines.append(
            f"**Confirmed: `field_salience_only` is structurally starved, not broken.** "
            f"The broadcast/GWT-dispatch lane is fresh on {result['broadcast_fresh_pct']}% of "
            f"real ticks, which -- because of the reducer's elif branch order -- preempts "
            f"`field_salience_only` (and therefore the new Active-Inference confidence formula, "
            f"PR #1301/#1304) on all but {result['field_salience_only_pct']}% of ticks. This is "
            f"not a bug: the code and its own docstrings document this priority deliberately. "
            f"It is, however, a real gap in claiming Sentience Striving Program charter section 6 "
            f"objective 2 (the AST/HOT reducer) is 'proven' -- the new confidence formula is "
            f"correct when it runs (unit-tested, live-verified this session) but is almost never "
            f"the formula actually driving `AttentionSelfModelV1.confidence` in production."
        )
    else:
        lines.append(
            "NOT confirmed as branch-starvation by this data -- either the broadcast lane is "
            "not as consistently fresh as hypothesized, `field_salience_only` fires more than "
            "expected, or an elif-ordering contradiction was found (see above). Re-read the "
            "numbers above before assuming the starvation hypothesis holds."
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ticks-csv",
        default=DEFAULT_TICKS_CSV,
        help=f"Path to measure_ast_hot_reducer.py's ticks.csv artifact (default {DEFAULT_TICKS_CSV})",
    )
    args = parser.parse_args(argv)

    path = Path(args.ticks_csv)
    if not path.exists():
        print(
            f"ERROR: {path} not found. Run scripts/analysis/measure_ast_hot_reducer.py first "
            "(it writes this CSV as one of its artifacts).",
            file=sys.stderr,
        )
        return 1

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print(f"ERROR: {path} has no data rows.", file=sys.stderr)
        return 1

    result = analyze_branch_starvation(rows)
    print(render_report(result, str(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
