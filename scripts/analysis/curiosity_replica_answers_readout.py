"""Readout of the curiosity replica's answers to Prime (design 2026-11-04 section 9, patch 4). Read-only.

Reads the replica's answer acts from ``grammar_acts`` (actor ``curiosity.replica.answer``, verb ``ask``). Each act's close
note starts with its stance (``refuse: <reason> | ...``, orion/curiosity/replica/answers.py ``answer_note``) and its
``test_before`` names its holdback arm. Reports:

* answers by stance and arm, and how many treated refusals and challenges were delivered;
* the guard against an organ that always says no: the refusal rate over the last 20 answers, as of the end of
  each of the last 14 days. ALARM when every one of those 14 days reads above 50% -- bring it back to Juniper
  with these numbers. A day with no answers yet is unknown and breaks the run (never an alarm by itself).

    python scripts/analysis/curiosity_replica_answers_readout.py            # live Postgres (read-only)
    python scripts/analysis/curiosity_replica_answers_readout.py --json

Exit code 0, or 3 on ALARM, or 2 when Postgres could not be read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pg_readonly import open_readonly_connection  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.curiosity.replica.answers import (  # noqa: E402
    REFUSAL_ALARM_DAYS,
    REFUSAL_RATE_ALARM,
    REFUSAL_RATE_WINDOW,
    refusal_alarm,
    stance_from_note,
)

from orion.schemas.curiosity_replica import ANSWER_ACTOR  # noqa: E402

DEFAULT_POSTGRES_URI = "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
PG_SQL = """
SELECT act_id, parent_act_id, test_before, opened_at, outcome_value, outcome_note
FROM grammar_acts
WHERE actor = %(actor)s AND verb = 'ask' AND opened_at > now() - interval '120 days'
ORDER BY opened_at ASC
"""


def parse_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One answer per act, with its stance (None when the note does not name one: still open or foreign)."""
    out = []
    for r in rows:
        tb = str(r.get("test_before") or "")
        out.append({"act_id": r.get("act_id"), "at": r.get("opened_at"), "stance": stance_from_note(r.get("outcome_note")),
                    "arm": tb.split(":", 1)[1] if tb.startswith("holdback:") else None,
                    "produced": r.get("outcome_value"), "note": str(r.get("outcome_note") or "")})
    return out


def daily_rates(answers: Sequence[dict[str, Any]], *, today: datetime, days: int = REFUSAL_ALARM_DAYS,
                window: int = REFUSAL_RATE_WINDOW) -> list[tuple[str, Optional[float]]]:
    """(UTC day, refusal share over the last ``window`` answers as of that day's end), oldest first."""
    known = sorted((a for a in answers if a["stance"] and a["at"] is not None), key=lambda a: a["at"])
    out: list[tuple[str, Optional[float]]] = []
    for back in range(days - 1, -1, -1):
        day = (today - timedelta(days=back)).date()
        end = datetime.combine(day, time.max, tzinfo=timezone.utc)
        last = [a for a in known if a["at"] <= end][-window:]
        out.append((day.isoformat(), (sum(a["stance"] == "refuse" for a in last) / len(last)) if last else None))
    return out


def summarize(answers: Sequence[dict[str, Any]], *, today: datetime) -> dict[str, Any]:
    by = Counter((a["stance"] or "open", a["arm"] or "?") for a in answers)
    delivered = sum(1 for a in answers if a["arm"] == "treated" and a["stance"] in ("refuse", "challenge")
                    and a["note"].endswith("| delivered"))
    treated = sum(1 for a in answers if a["arm"] == "treated" and a["stance"] in ("refuse", "challenge"))
    rates = daily_rates(answers, today=today)
    alarm = refusal_alarm(rates)
    return {"answers": len(answers), "by_stance_arm": {f"{s}/{a}": n for (s, a), n in sorted(by.items())},
            "treated_challenge_or_refuse": treated, "delivered": delivered, "daily_refusal_rate": rates,
            "alarm": alarm,
            "verdict": (f"ALARM: refusal rate above {REFUSAL_RATE_ALARM:.0%} on each of the last {REFUSAL_ALARM_DAYS} "
                        "days -- bring this to Juniper" if alarm else
                        "ok" if answers else "NO DATA: the replica has not answered Prime yet")}


def read_pg(dsn: str) -> Optional[list[dict[str, Any]]]:
    conn = open_readonly_connection(dsn, connect_timeout=10, statement_timeout_ms=60_000)
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(PG_SQL, {"actor": ANSWER_ACTOR})
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    dsn = os.environ.get("POSTGRES_URI") or DEFAULT_POSTGRES_URI
    rows = read_pg(dsn)
    if rows is None:
        print("UNKNOWN: Postgres unreadable", file=sys.stderr)
        return 2
    report = summarize(parse_rows(rows), today=datetime.now(timezone.utc))
    if args.json:
        print(json.dumps(report, indent=1, default=str))
    else:
        print(f"answers: {report['answers']}  {report['by_stance_arm']}")
        print(f"treated challenges+refusals: {report['treated_challenge_or_refuse']}, delivered: {report['delivered']}")
        for day, rate in report["daily_refusal_rate"]:
            print(f"  {day}: {'-' if rate is None else f'{rate:.0%}'}")
        print(report["verdict"])
    return 3 if report["alarm"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
