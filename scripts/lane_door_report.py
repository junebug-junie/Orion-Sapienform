#!/usr/bin/env python3
"""Daily lane door report: is it safe to make a lane's door refuse unkeyed calls? (read-only)

Leased-evals design, Phase 2 -> Phase 3 (docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md;
runbook docs/runbooks/2026-11-14-lane-door.md). Reads the doors' per-call JSON lines
(``<lane>-YYYY-MM-DD.jsonl``, one line per model call; no prompt text) and prints, per lane per UTC day:

* keyed calls (the gateway's), unkeyed calls (``missing`` key) and wrong-key calls (``bad``);
* who sent the unkeyed ones (source address + path);
* with ``--dsn``: keyed calls whose lease id is NOT in ``gpu_pool_events`` (a key used outside a grant), and
  the gateway's own count of calls it sent to that lane (``llm_call_usage.pool_role``), with the gap in %.

Then a verdict per lane. READY TO ENFORCE needs, for the last ``--streak-days`` (default 7) UTC days in a
row, each day present in the log (a missing day is unknown, never zero) with:
  unkeyed = 0, bad = 0, keyed > 0, and with --dsn: every keyed lease id found in gpu_pool_events and the
  keyed count within --max-gap-pct (default 1 %) of the gateway's count.
Without --dsn the verdict says the lease join was not run.

    # on athena, logs copied from circe (read-only):
    ssh circe@circe 'cat /mnt/telemetry/lane-door/*.jsonl' | \\
      python3 scripts/lane_door_report.py - --dsn postgresql://postgres:postgres@localhost:55432/conjourney

    python3 scripts/lane_door_report.py /path/to/lane-door/            # a folder of .jsonl files
    python3 scripts/lane_door_report.py --stats http://100.112.254.99:8011 ...   # live /door/stats

Exit 0 always (a report, not a gate); ``--json`` for machine output.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


@dataclass
class Day:
    ok: int = 0
    missing: int = 0
    bad: int = 0
    unkeyed_from: Counter = field(default_factory=Counter)   # "src path" -> n
    lease_ids: set = field(default_factory=set)
    no_lease_id: int = 0                                     # keyed calls without a lease id
    unmatched_leases: Optional[int] = None                   # --dsn: keyed lease ids not in gpu_pool_events
    gateway_calls: Optional[int] = None                      # --dsn: llm_call_usage rows for this lane


def iter_lines(sources: list[str]) -> Iterator[str]:
    for src in sources:
        if src == "-":
            yield from sys.stdin
            continue
        path = Path(src)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for f in files:
            with f.open(encoding="utf-8", errors="replace") as fh:
                yield from fh


def aggregate(lines: Iterable[str]) -> tuple[dict[str, dict[str, Day]], int]:
    """lane -> UTC day (YYYY-MM-DD) -> Day. Returns also the number of unreadable lines."""
    out: dict[str, dict[str, Day]] = defaultdict(dict)
    bad_lines = 0
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
            lane, day, key = str(row["lane"]), str(row["ts"])[:10], str(row["key"])
        except (ValueError, KeyError, TypeError):
            bad_lines += 1
            continue
        d = out[lane].setdefault(day, Day())
        if key == "ok":
            d.ok += 1
            lease = row.get("lease_id")
            if lease and lease != "invalid":
                d.lease_ids.add(str(lease))
            else:
                d.no_lease_id += 1
        else:
            if key == "bad":
                d.bad += 1
            else:
                d.missing += 1
            d.unkeyed_from[f"{row.get('src', '?')} {row.get('method', '?')} {row.get('path', '?')}"] += 1
    return out, bad_lines


def join_db(report: dict[str, dict[str, Day]], dsn: str) -> None:
    """Fill unmatched_leases and gateway_calls from Postgres (read-only)."""
    try:
        import psycopg2  # noqa: PLC0415
    except ImportError:   # pragma: no cover
        raise SystemExit("--dsn needs psycopg2 (the repo .venv has it)")
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        for lane, days in report.items():
            for day, d in days.items():
                ids = sorted(d.lease_ids)
                found: set[str] = set()
                for i in range(0, len(ids), 5000):
                    cur.execute("SELECT DISTINCT lease_id FROM gpu_pool_events WHERE lease_id = ANY(%s)",
                                (ids[i:i + 5000],))
                    found.update(r[0] for r in cur.fetchall())
                d.unmatched_leases = len(set(ids) - found) + d.no_lease_id
                cur.execute("SELECT count(*) FROM llm_call_usage WHERE pool_role = %s "
                            "AND created_at >= %s::date AND created_at < %s::date + 1", (lane, day, day))
                d.gateway_calls = int(cur.fetchone()[0])


def verdict(days: dict[str, Day], *, streak_days: int, max_gap_pct: float, joined: bool,
            today: Optional[str] = None) -> tuple[bool, str]:
    """READY only on ``streak_days`` consecutive complete UTC days, ending yesterday (today is partial)."""
    end = date.fromisoformat(today) - timedelta(days=1) if today else None
    if end is None:
        if not days:
            return False, "no door log at all"
        end = date.fromisoformat(max(days))
    for i in range(streak_days):
        day = (end - timedelta(days=i)).isoformat()
        d = days.get(day)
        if d is None:
            return False, f"{day}: no log (door down, or not deployed yet) -- unknown is not zero"
        if d.missing or d.bad:
            return False, f"{day}: {d.missing} unkeyed + {d.bad} wrong-key calls"
        if d.ok == 0:
            return False, f"{day}: no keyed calls (nothing proves the gateway's traffic passes)"
        if joined:
            if d.unmatched_leases:
                return False, f"{day}: {d.unmatched_leases} keyed calls with no matching pool lease"
            if d.gateway_calls is not None and d.gateway_calls > 0:
                gap = abs(d.ok - d.gateway_calls) * 100.0 / d.gateway_calls
                if gap > max_gap_pct:
                    return False, f"{day}: door saw {d.ok} keyed calls, gateway sent {d.gateway_calls} ({gap:.1f} % apart)"
    tail = "" if joined else " (lease join NOT run: add --dsn before flipping)"
    return True, f"{streak_days} clean days ending {end.isoformat()}{tail}"


def fetch_stats(urls: list[str]) -> list[dict[str, Any]]:
    out = []
    for url in urls:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/door/stats", timeout=5) as resp:
                out.append({"url": url, **json.load(resp)})
        except (OSError, ValueError) as exc:
            out.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    return out


def render(report: dict[str, dict[str, Day]], verdicts: dict[str, tuple[bool, str]], joined: bool) -> str:
    lines = []
    for lane in sorted(report):
        ok, why = verdicts[lane]
        lines.append(f"== lane {lane}: {'READY TO ENFORCE' if ok else 'NOT READY'} -- {why}")
        head = f"   {'day':10} {'keyed':>7} {'unkeyed':>8} {'badkey':>7}"
        if joined:
            head += f" {'no-lease':>9} {'gateway':>8} {'gap%':>6}"
        lines.append(head)
        for day in sorted(report[lane]):
            d = report[lane][day]
            row = f"   {day:10} {d.ok:7d} {d.missing:8d} {d.bad:7d}"
            if joined:
                gap = (f"{abs(d.ok - d.gateway_calls) * 100.0 / d.gateway_calls:6.1f}"
                       if d.gateway_calls else "     -")
                row += f" {d.unmatched_leases or 0:9d} {d.gateway_calls if d.gateway_calls is not None else '-':>8} {gap}"
            lines.append(row)
            for who, n in d.unkeyed_from.most_common(5):
                lines.append(f"      unkeyed x{n}: {who}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("sources", nargs="*", help=".jsonl files, folders of them, or - for stdin")
    ap.add_argument("--dsn", default=None, help="Postgres DSN for the lease join + gateway counts (read-only)")
    ap.add_argument("--streak-days", type=int, default=7)
    ap.add_argument("--max-gap-pct", type=float, default=1.0)
    ap.add_argument("--today", default=None,
                    help="UTC day treated as today (partial, so excluded). Default: the real UTC date")
    ap.add_argument("--stats", nargs="*", default=None, help="door base URLs: print their live /door/stats")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.stats:
        stats = fetch_stats(args.stats)
        print(json.dumps(stats, indent=1))
        if not args.sources:
            return 0
    report, bad_lines = aggregate(iter_lines(args.sources or ["-"]))
    joined = bool(args.dsn)
    if joined:
        join_db(report, args.dsn)
    today = args.today or datetime.now(timezone.utc).date().isoformat()
    verdicts = {lane: verdict(days, streak_days=args.streak_days, max_gap_pct=args.max_gap_pct, joined=joined,
                              today=today)
                for lane, days in report.items()}
    if args.json:
        print(json.dumps({"lanes": {lane: {"ready": verdicts[lane][0], "why": verdicts[lane][1], "days": {
            day: {"keyed": d.ok, "unkeyed": d.missing, "bad_key": d.bad, "unkeyed_from": dict(d.unkeyed_from),
                  "unmatched_leases": d.unmatched_leases, "gateway_calls": d.gateway_calls}
            for day, d in sorted(days.items())}} for lane, days in report.items()},
            "unreadable_lines": bad_lines, "lease_join": joined}, indent=1))
        return 0
    if not report:
        print("no lane door log lines read")
    else:
        print(render(report, verdicts, joined))
    if bad_lines:
        print(f"({bad_lines} unreadable lines skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
