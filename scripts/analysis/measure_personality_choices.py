#!/usr/bin/env python3
"""Read-only: reconstruct Orion's choices (picked + not picked) from the two lanes that
already store both, and summarize how often Orion follows the order it is shown.

## Why this exists

P0 of the dynamic personality graph design (PR #2726, section 1 "Evidence"). The evidence
unit is a choice plus its alternatives. Before anything persists choices, this proves the
two existing lanes yield real, non-degenerate choices, using the pure reducers in
``orion/personality/choices.py``.

## What it reads

- ``curiosity_offer_decisions`` LEFT JOIN ``curiosity_run_outcomes`` on ``run_id``, offers
  decided within ``--window-days``. Chooser: Orion (the LLM).
- ``substrate_attention_broadcast_log`` rows within ``--window-days`` whose
  ``frame.voluntary_override`` is an object. Chooser: the goal-biased attention formula.
  Rows with no override are not choices and are not fetched; their count is reported.

## What it writes

Nothing to any DB (read-only session, ``orion.db_readonly.open_readonly_connection``).
Writes ``choices.jsonl`` and ``summary.txt`` to ``--out-dir``.

Usage::

    POSTGRES_URI=postgresql://postgres:postgres@localhost:55432/conjourney \\
        python3 scripts/analysis/measure_personality_choices.py --window-days 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orion.personality.choices import (  # noqa: E402
    LANE_CURIOSITY,
    LANE_OVERRIDE,
    explain_curiosity_choice,
    explain_voluntary_override_choice,
)
from orion.schemas.personality import PersonalityChoiceV1  # noqa: E402

LANES = (LANE_CURIOSITY, LANE_OVERRIDE)

SQL_CURIOSITY = """
SELECT d.run_id, d.decided_at, d.arm, d.offered, d.stale_offered, d.material_ids,
       d.turn_started_at,
       o.run_id AS outcome_run_id, o.completed_at, o.turn_ok, o.realized_nats, o.per_prior
FROM curiosity_offer_decisions d
LEFT JOIN curiosity_run_outcomes o ON o.run_id = d.run_id
WHERE d.decided_at >= now() - make_interval(days => %s)
ORDER BY d.decided_at
"""

SQL_OVERRIDE = """
SELECT log_id, generated_at, projection_json
FROM substrate_attention_broadcast_log
WHERE generated_at >= now() - make_interval(days => %s)
  AND jsonb_typeof(projection_json->'frame'->'voluntary_override') = 'object'
ORDER BY generated_at
"""

SQL_BROADCAST_TOTAL = """
SELECT count(*) AS n FROM substrate_attention_broadcast_log
WHERE generated_at >= now() - make_interval(days => %s)
"""


def _dsn() -> str:
    for key in ("POSTGRES_URI", "DATABASE_URL", "ORION_SQL_DSN"):
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return "postgresql://postgres:postgres@localhost:55432/conjourney"


# ------------------------------------------------------------------ pure


def split_curiosity_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """One joined row -> (offer, outcome or None when the LEFT JOIN found nothing)."""
    offer = {k: row.get(k) for k in ("run_id", "decided_at", "arm", "offered",
                                     "stale_offered", "material_ids", "turn_started_at")}
    if row.get("outcome_run_id") is None:
        return offer, None
    outcome = {"run_id": row.get("outcome_run_id"), "completed_at": row.get("completed_at"),
               "turn_ok": row.get("turn_ok"), "realized_nats": row.get("realized_nats"),
               "per_prior": row.get("per_prior")}
    return offer, outcome


def reduce_rows(
    curiosity_rows: Iterable[Dict[str, Any]], override_rows: Iterable[Dict[str, Any]]
) -> Tuple[List[PersonalityChoiceV1], Dict[str, Counter]]:
    """-> (choices, {lane: Counter(reason for each row that yielded None)})."""
    choices: List[PersonalityChoiceV1] = []
    skipped: Dict[str, Counter] = {lane: Counter() for lane in LANES}
    for row in curiosity_rows:
        choice, reason = explain_curiosity_choice(*split_curiosity_row(row))
        if choice is None:
            skipped[LANE_CURIOSITY][reason or "unknown"] += 1
        else:
            choices.append(choice)
    for row in override_rows:
        choice, reason = explain_voluntary_override_choice(row)
        if choice is None:
            skipped[LANE_OVERRIDE][reason or "unknown"] += 1
        else:
            choices.append(choice)
    return choices, skipped


def summarize(
    choices: Sequence[PersonalityChoiceV1],
    skipped: Dict[str, Counter],
    *,
    window_days: int,
    rows_read: Dict[str, int],
) -> Dict[str, Any]:
    lanes: Dict[str, Any] = {}
    for lane in LANES:
        mine = [c for c in choices if c.lane == lane]
        n = len(mine)
        ranks = Counter(c.chosen_best_rank for c in mine)
        unranked_picks = sum(1 for c in mine for o in c.chosen if o.rank is None)
        total_picks = sum(len(c.chosen) for c in mine)
        costs_s = [c.cost_seconds for c in mine if c.cost_seconds is not None]
        costs_n = [c.cost_nats for c in mine if c.cost_nats is not None]
        costs_e = [c.cost_effort for c in mine if c.cost_effort is not None]
        lanes[lane] = {
            "rows_read": rows_read.get(lane, 0),
            "choices": n,
            "skipped": dict(sorted(skipped.get(lane, Counter()).items())),
            "chosen_best_rank": {
                ("none" if k is None else str(k)): v
                for k, v in sorted(ranks.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
            },
            "followed_order_rank1": ranks.get(1, 0),
            "picks_total": total_picks,
            "picks_outside_ordered_list": unranked_picks,
            "mean_chosen_per_choice": round(total_picks / n, 3) if n else None,
            "mean_alternatives_per_choice": (
                round(sum(len(c.alternatives) for c in mine) / n, 3) if n else None
            ),
            "mean_cost_seconds": round(sum(costs_s) / len(costs_s), 1) if costs_s else None,
            "mean_cost_nats": round(sum(costs_n) / len(costs_n), 4) if costs_n else None,
            "mean_cost_effort": round(sum(costs_e) / len(costs_e), 4) if costs_e else None,
        }
    return {"window_days": window_days, "rows_read": rows_read, "lanes": lanes}


def render_summary(report: Dict[str, Any]) -> str:
    lines = [f"personality choices, window {report['window_days']} days", ""]
    extra = report["rows_read"].get("broadcast_rows_total")
    for lane, s in report["lanes"].items():
        lines.append(f"[{lane}]")
        lines.append(f"  rows read: {s['rows_read']}   choices: {s['choices']}")
        if lane == LANE_OVERRIDE and extra is not None:
            lines.append(f"  (broadcast rows in window, override or not: {extra})")
        skipped = s["skipped"]
        lines.append(
            "  yielded None: "
            + (", ".join(f"{k}={v}" for k, v in skipped.items()) if skipped else "0")
        )
        lines.append(
            "  chosen_best_rank: "
            + (", ".join(f"{k}={v}" for k, v in s["chosen_best_rank"].items()) or "-")
        )
        if s["choices"]:
            pct = 100.0 * s["followed_order_rank1"] / s["choices"]
            lines.append(f"  took rank 1: {s['followed_order_rank1']}/{s['choices']} ({pct:.0f}%)")
        lines.append(
            f"  picks: {s['picks_total']} total, {s['picks_outside_ordered_list']} outside the "
            f"ordered list; mean chosen/choice {s['mean_chosen_per_choice']}, "
            f"mean alternatives/choice {s['mean_alternatives_per_choice']}"
        )
        lines.append(
            f"  mean cost: seconds={s['mean_cost_seconds']} nats={s['mean_cost_nats']} "
            f"effort={s['mean_cost_effort']}"
        )
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ io


def _fetch(conn, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window-days", type=int, default=14)
    ap.add_argument("--out-dir", default="/tmp/personality-choices/")
    args = ap.parse_args(argv)

    from orion.db_readonly import open_readonly_connection

    conn = open_readonly_connection(_dsn(), connect_timeout=10, statement_timeout_ms=60_000)
    if conn is None:
        print("UNKNOWN: could not open a confirmed read-only DB session", file=sys.stderr)
        return 2
    try:
        cur_rows = _fetch(conn, SQL_CURIOSITY, (args.window_days,))
        ovr_rows = _fetch(conn, SQL_OVERRIDE, (args.window_days,))
        total = _fetch(conn, SQL_BROADCAST_TOTAL, (args.window_days,))[0]["n"]
    finally:
        conn.close()

    choices, skipped = reduce_rows(cur_rows, ovr_rows)
    report = summarize(
        choices, skipped, window_days=args.window_days,
        rows_read={LANE_CURIOSITY: len(cur_rows), LANE_OVERRIDE: len(ovr_rows),
                   "broadcast_rows_total": int(total)},
    )
    summary = render_summary(report)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "choices.jsonl").open("w") as fh:
        for c in choices:
            fh.write(c.model_dump_json() + "\n")
    (out / "summary.txt").write_text(summary)
    print(summary, end="")
    print(f"wrote {out / 'choices.jsonl'} and {out / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
