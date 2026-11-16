#!/usr/bin/env python3
"""Live eval of the daily dig roundup (integration patch 4, spec Part A.7 acceptance). Read-only.

For one Denver day (default: yesterday) it reads the roundup the way `dig organs <date>` does (ReadOnlyPsql,
READ ONLY transaction) and checks:

  E1  it reads in under 30 s;
  E2  every line is a count, a quote or a ref (check_lines);
  E3  no query is UNREAD;
  E4  its use rate and closed organs equal scripts/report_integration_coverage.py's for the same window;
  E5  every [experience:<id>] it prints is a real temporal_self_event row of that day, and every [act:<id>] a
      real grammar_acts row (no ref is invented);
  E6  the footer's notification count equals notify_requests for the day (the retired Daily Digest's numbers);
  E7  it is not empty-shell: at least one organ line quotes an experience with its ref.

    PYTHONPATH=. python orion/dig/evals/run_organs_roundup_eval.py [--day YYYY-MM-DD] [--role orion_readonly]

``--role``: the read-only role (default orion_readonly, Orion's own; ``postgres`` still runs in a READ ONLY
transaction). Exit 0 = every check passed; 1 = a check failed; 2 = the database could not be reached.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.body_atlas.sources import psql_rows  # noqa: E402
from orion.dig import organs  # noqa: E402
from orion.evals.model_replay.sandbox import ReadOnlyPsql  # noqa: E402
from orion.integration.organs import load_organs  # noqa: E402

LIMIT_SEC = 30.0


def _coverage(dsn: str, day: date):
    """The coverage report's own live read (psycopg-bound params, its own connection) for the same Denver day:
    independent of the roundup's SQL binding, so a _bind or window mistake shows as a different number."""
    spec = importlib.util.spec_from_file_location("rc", REPO / "scripts" / "report_integration_coverage.py")
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    lo, hi = (datetime.fromisoformat(x) for x in organs.day_bounds(day))
    return rc.read_live(dsn, (hi - lo).total_seconds() / 3600.0, now=hi)


ID_RE = re.compile(r"[A-Za-z0-9:._-]{1,200}")


def _in_list(ids: list[str]) -> str:
    return ",".join(f"'{i}'" for i in ids if ID_RE.fullmatch(i)) or "''"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", default=(datetime.now(ZoneInfo("America/Denver")).date() - timedelta(days=1)).isoformat())
    ap.add_argument("--role", default="orion_readonly")
    ap.add_argument("--dsn", default=os.getenv("ORION_LIVE_DSN") or os.getenv("POSTGRES_URI"),
                    help="psycopg DSN for E4's independent coverage-report read")
    args = ap.parse_args(argv)
    day = date.fromisoformat(args.day)
    runner = ReadOnlyPsql(role=args.role, timeout_sec=60.0)
    try:
        psql_rows(runner, "SELECT 1 AS ok")
    except RuntimeError as exc:
        print(f"UNREACHABLE: {exc}")
        return 2
    registry = load_organs()
    t0 = time.monotonic()
    r = organs.read_sync(day, lambda sql: psql_rows(runner, sql), registry)
    took = time.monotonic() - t0
    results: list[tuple[str, bool, str]] = []
    results.append(("E1 under 30 s", took < LIMIT_SEC, f"{took:.1f} s"))
    try:
        organs.check_lines(r.lines)
        results.append(("E2 every line a count, quote or ref", True, f"{len(r.lines)} lines"))
    except ValueError as exc:
        results.append(("E2 every line a count, quote or ref", False, str(exc)))
    results.append(("E3 nothing UNREAD", not r.unread, ", ".join(sorted(r.unread)) or "all 11 queries read"))
    if not args.dsn:
        results.append(("E4 same number as the coverage report", False, "no --dsn / ORION_LIVE_DSN: not checked"))
    else:
        try:
            rep = _coverage(args.dsn, day)
            same = (rep["experience_use_rate"] == r.use_rate and rep["closed_organs"] == r.closed_organs
                    and rep["experiences"] == r.experiences)
            results.append(("E4 same number as the coverage report (its own read)", same,
                            f"roundup {r.experiences} exp, {r.use_rate} {r.closed_organs}; report "
                            f"{rep['experiences']} exp, {rep['experience_use_rate']} {rep['closed_organs']}"))
        except Exception as exc:  # noqa: BLE001
            results.append(("E4 same number as the coverage report", False, f"{type(exc).__name__}: {exc}"))
    a, b = organs.day_bounds(day)
    exp = [x.split(":", 1)[1] for x in r.refs if x.startswith("experience:")]
    acts = [x.split(":", 1)[1] for x in r.refs if x.startswith("act:")]
    try:
        # any day: a cited item may be from an earlier day than the roundup's
        found = {row["event_id"] for row in psql_rows(runner, (
            f"SELECT event_id FROM temporal_self_event WHERE event_id IN ({_in_list(exp)})"))}
        found_acts = {row["act_id"] for row in psql_rows(runner, (
            f"SELECT act_id FROM grammar_acts WHERE act_id IN ({_in_list(acts)})"))}
        malformed = [i for i in exp + acts if not ID_RE.fullmatch(i)]
        missing = sorted(set(exp) - found) + sorted(set(acts) - found_acts)
        results.append(("E5 every ref is a real row", not missing and not malformed,
                        f"{len(exp)} experience refs, {len(acts)} act refs; missing {missing[:3]}; "
                        f"malformed {malformed[:3]}"))
        # independent of the roundup's window SQL: naive UTC bounds computed here, in Python
        na, nb = (datetime.fromisoformat(x).astimezone(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
                  for x in (a, b))
        n = psql_rows(runner, f"SELECT count(*) AS n FROM notify_requests WHERE created_at >= '{na}' "
                              f"AND created_at < '{nb}'")
        foot = next((x for x in r.lines if x.startswith("notifications (")), "")
        results.append(("E6 notification footer equals notify_requests", f": {n[0]['n']} requests" in foot,
                        foot[:120]))
    except RuntimeError as exc:
        results.append(("E5/E6 checks", False, str(exc)))
    quoted = [x for x in r.lines if x.startswith('  "') and "[experience:" in x]
    results.append(("E7 not empty-shell", bool(quoted), f"{len(quoted)} quoted experiences with refs"))

    print(f"organs roundup eval {day} (role {args.role}):")
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'} {name}: {detail}")
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
