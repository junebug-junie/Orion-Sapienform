#!/usr/bin/env python3
"""What the self-question door did, week by week (read-only; no model calls).

Spec: docs/superpowers/specs/2026-11-25-orion-self-reflection-drift-design.md, acceptance checks 3-5. Orion
asked (session thread on PR #3908, question 5) whether anything tracks dressing as a rate, not only as a share.
This prints, per week of ``door_judged_at`` for Orion-written questions:

* admitted, routed, unjudged, still held, and demoted (by week of the demotion's judged question);
* the dressing rate: demoted / admitted. A rising rate means more plumbing questions in first-person words get
  through the door and are caught only after a sitting. Rest point: None when nothing was admitted that week;
* the unjudged share of judged rows (check 5: under 5%);
* routed questions offered to the investigate line, and how many a run took (check 4).

Every read is SELECT inside a read-only transaction (``PGOPTIONS default_transaction_read_only=on``).

    python3 scripts/analysis/self_question_door_report.py [--weeks 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reflection_register_sources import psql_json  # noqa: E402

WEEKS_SQL = """
SELECT json_build_object('week', to_char(date_trunc('week', door_judged_at), 'YYYY-MM-DD'), 'door', door, 'n', count(*))
FROM curiosity_self_questions
WHERE minted_by = 'orion' AND door IS NOT NULL AND door_judged_at > now() - make_interval(weeks => {weeks})
GROUP BY date_trunc('week', door_judged_at), door
"""
NOW_SQL = """
SELECT json_build_object('door', coalesce(door, 'null (before the door)'), 'n', count(*))
FROM curiosity_self_questions WHERE minted_by = 'orion' GROUP BY door
"""
OFFERS_SQL = """
SELECT json_build_object('offers', count(*), 'taken', count(*) FILTER (WHERE offered_run_id IS NOT NULL),
  'pending', count(*) FILTER (WHERE status = 'pending'), 'replaced_or_expired',
  count(*) FILTER (WHERE status IN ('replaced', 'expired')))
FROM curiosity_focus_request WHERE left(chat_correlation_id, 19) = 'self_question_door:'
"""


def weekly(rows: list[dict]) -> list[dict]:
    """Per week: counts by door, dressing rate (demoted / (admitted + demoted)), unjudged share. Pure.

    A demoted row was admitted first, so the admitted base counts both."""
    weeks: dict[str, dict[str, int]] = {}
    for r in rows:
        weeks.setdefault(r["week"], {})[r["door"]] = int(r["n"])
    out = []
    for week in sorted(weeks):
        c = weeks[week]
        admitted_base = c.get("admitted", 0) + c.get("demoted", 0)
        judged = sum(v for k, v in c.items() if k != "held")
        out.append({"week": week, **{k: c.get(k, 0) for k in ("admitted", "demoted", "routed", "unjudged", "held")},
                    "dressing_rate": (c.get("demoted", 0) / admitted_base) if admitted_base else None,
                    "unjudged_share": (c.get("unjudged", 0) / judged) if judged else None})
    return out


def _pct(v):
    return "None" if v is None else f"{v:.0%}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weeks", type=int, default=8)
    args = ap.parse_args(argv)
    print("Orion-written self-questions now:", {r["door"]: r["n"] for r in psql_json(NOW_SQL)})
    print("| Week of | Admitted | Demoted | Routed | Unjudged | Held | Dressing rate (demoted / admitted) | Unjudged share |")
    print("|---|---|---|---|---|---|---|---|")
    for w in weekly(psql_json(WEEKS_SQL.format(weeks=int(args.weeks)))):
        print(f"| {w['week']} | {w['admitted']} | {w['demoted']} | {w['routed']} | {w['unjudged']} | {w['held']} | "
              f"{_pct(w['dressing_rate'])} | {_pct(w['unjudged_share'])} |")
    print("Door offers to the investigate line:", psql_json(OFFERS_SQL)[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
