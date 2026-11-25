#!/usr/bin/env python3
"""Daily lane door report: is it safe to make a lane's door refuse unleased calls? (read-only)

Leased-evals design, Phase 2 -> Phase 3 (docs/superpowers/specs/2026-11-12-leased-evals-and-server-refusal-design.md;
runbook docs/runbooks/2026-11-14-lane-door.md). Reads the doors' per-call JSON lines
(``<lane>-YYYY-MM-DD.jsonl``, one line per model call; no prompt text) and prints, per lane per UTC day:

* leased calls (carry a lease id: the gateway's), unleased calls (no lease id) and invalid ones (a lease
  header that is not a lease id);
* who sent the unleased and invalid ones (source address + path);
* with ``--dsn``: leased calls whose lease id is NOT in ``gpu_pool_events`` (a made-up id: the door itself
  cannot tell), and the gateway's own count of calls it sent to that lane (``llm_call_usage.pool_role``),
  with the gap in %.

There is no shared secret: a lease id is what marks a gateway call, and the join to ``gpu_pool_events`` is
the check that the id is real. The door catches accidental bypass by Orion's own evals and scripts.

Then a verdict per lane. READY TO ENFORCE needs, for the last ``--streak-days`` (default 7) UTC days in a
row, each day present in the log (a missing day is unknown, never zero) with:
  unleased = 0, invalid = 0, leased > 0, and with --dsn: every leased call's id found in gpu_pool_events
  and the number of distinct lease ids within --max-gap-pct (default 1 %) of the gateway's count. The gateway
  writes one llm_call_usage row per lease, but a call sent again under the same lease reaches the door twice,
  so raw leased calls overcount (#3919); they stay in the table for information. Calls an enforcing door
  already refused are listed, not counted against the lane (they never reached the engine).
Without --dsn the verdict says the lease join was not run.

    # on athena, logs copied from circe (read-only):
    ssh circe@circe 'cat /mnt/telemetry/lane-door/*.jsonl' | \\
      python3 scripts/lane_door_report.py - --dsn postgresql://postgres:postgres@localhost:55432/conjourney

    python3 scripts/lane_door_report.py /path/to/lane-door/            # a folder of .jsonl files
    python3 scripts/lane_door_report.py --stats http://<circe-address>:<lane-port> ...   # live /door/stats (the doors' own counters)

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
    leased: int = 0
    unleased: int = 0
    invalid: int = 0
    unleased_from: Counter = field(default_factory=Counter)  # "src method path" -> n (unleased + invalid)
    lease_ids: set = field(default_factory=set)
    refused: int = 0                                         # Phase 3: calls an enforcing door refused
    refused_from: Counter = field(default_factory=Counter)
    unmatched_leases: Optional[int] = None                   # --dsn: leased ids not in gpu_pool_events
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


def aggregate(lines: Iterable[str], counts: Optional[dict] = None) -> tuple[dict[str, dict[str, Day]], int]:
    """lane -> UTC day (YYYY-MM-DD) -> Day. Returns also the number of unreadable lines."""
    out: dict[str, dict[str, Day]] = defaultdict(dict)
    bad_lines = 0
    pre_revision = 0
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
            if "lease" not in row and "key" in row:
                pre_revision += 1   # written by the first (keyed) door build; not evidence either way
                continue
            lane, day, state = str(row["lane"]), str(row["ts"])[:10], str(row["lease"])
        except (ValueError, KeyError, TypeError):
            bad_lines += 1
            continue
        d = out[lane].setdefault(day, Day())
        who = f"{row.get('src', '?')} {row.get('method', '?')} {row.get('path', '?')}"
        if row.get("refused"):
            # Already refused by an enforcing door: never reached the engine. Shown, not counted.
            d.refused += 1
            d.refused_from[who] += 1
        elif state == "leased":
            d.leased += 1
            if row.get("lease_id"):
                d.lease_ids.add(str(row["lease_id"]))
        else:
            if state == "invalid":
                d.invalid += 1
            else:
                d.unleased += 1
            d.unleased_from[who] += 1
    if counts is not None:
        counts["pre_revision"] = pre_revision
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
                d.unmatched_leases = len(set(ids) - found)
                cur.execute("SELECT count(*) FROM llm_call_usage WHERE pool_role = %s "
                            "AND created_at >= (%s::date)::timestamp AT TIME ZONE 'UTC' "
                            "AND created_at < (%s::date + 1)::timestamp AT TIME ZONE 'UTC'", (lane, day, day))
                d.gateway_calls = int(cur.fetchone()[0])


def gap_pct(d: Day) -> Optional[float]:
    """Distinct lease ids vs the gateway's own call count, in %. None without a gateway count."""
    if not d.gateway_calls:
        return None
    return abs(len(d.lease_ids) - d.gateway_calls) * 100.0 / d.gateway_calls


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
        if d.unleased or d.invalid:
            return False, f"{day}: {d.unleased} unleased + {d.invalid} invalid-lease calls"
        if d.leased == 0:
            return False, f"{day}: no leased calls (nothing proves the gateway's traffic passes)"
        if joined:
            if d.unmatched_leases:
                return False, f"{day}: {d.unmatched_leases} lease ids not in gpu_pool_events (made up?)"
            if not d.gateway_calls:
                return False, (f"{day}: door saw {d.leased} leased calls but the gateway recorded none for pool_role "
                               "of the same name (a lane that does not match pool_role?)")
            else:
                gap = gap_pct(d)
                if gap > max_gap_pct:
                    return False, (f"{day}: door saw {len(d.lease_ids)} distinct leases ({d.leased} leased calls), "
                                   f"gateway sent {d.gateway_calls} ({gap:.1f} % apart)")
    if not joined:
        return False, (f"{streak_days} clean days ending {end.isoformat()} in the door log, but the lease join was "
                       "not run: add --dsn (READY needs the leases found and the gateway's own count)")
    return True, f"{streak_days} clean days ending {end.isoformat()}"


def fetch_stats(urls: list[str]) -> list[dict[str, Any]]:
    out = []
    for url in urls:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/door/stats", timeout=5) as resp:
                out.append({"url": url, **json.load(resp)})
        except (OSError, ValueError) as exc:
            out.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    return out


EXPECTED_LANES = ("chat", "agent", "agent-gpu2", "fast", "agent-deep")   # the door compose services


def render(report: dict[str, dict[str, Day]], verdicts: dict[str, tuple[bool, str]], joined: bool) -> str:
    lines = []
    for lane in sorted(report):
        ok, why = verdicts[lane]
        lines.append(f"== lane {lane}: {'READY TO ENFORCE' if ok else 'NOT READY'} -- {why}")
        if not report[lane]:
            lines.append("   no log lines for this lane (door not deployed, or no traffic)")
            continue
        head = f"   {'day':10} {'leased':>7} {'unleased':>8} {'invalid':>7} {'refused':>8}"
        if joined:
            head += f" {'leases':>7} {'no-lease':>9} {'gateway':>8} {'gap%':>6}"
        lines.append(head)
        for day in sorted(report[lane]):
            d = report[lane][day]
            row = f"   {day:10} {d.leased:7d} {d.unleased:8d} {d.invalid:7d} {d.refused:8d}"
            if joined:
                g = gap_pct(d)
                gap = f"{g:6.1f}" if g is not None else "     -"
                row += f" {len(d.lease_ids):7d} {d.unmatched_leases or 0:9d} {d.gateway_calls if d.gateway_calls is not None else '-':>8} {gap}"
            lines.append(row)
            for who, n in d.unleased_from.most_common(5):
                lines.append(f"      unleased x{n}: {who}")
            for who, n in d.refused_from.most_common(3):
                lines.append(f"      refused x{n}: {who}")
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
    counts: dict = {}
    report, bad_lines = aggregate(iter_lines(args.sources or ["-"]), counts)
    for lane in EXPECTED_LANES:   # a lane with no log line at all is reported, never silently absent
        report.setdefault(lane, {})
    joined = bool(args.dsn)
    if joined:
        join_db(report, args.dsn)
    today = args.today or datetime.now(timezone.utc).date().isoformat()
    verdicts = {lane: verdict(days, streak_days=args.streak_days, max_gap_pct=args.max_gap_pct, joined=joined,
                              today=today)
                for lane, days in report.items()}
    if args.json:
        print(json.dumps({"lanes": {lane: {"ready": verdicts[lane][0], "why": verdicts[lane][1], "days": {
            day: {"leased": d.leased, "unleased": d.unleased, "invalid": d.invalid, "refused": d.refused,
                  "unleased_from": dict(d.unleased_from),
                  "refused_from": dict(d.refused_from),
                  "unmatched_leases": d.unmatched_leases, "gateway_calls": d.gateway_calls}
            for day, d in sorted(days.items())}} for lane, days in report.items()},
            "unreadable_lines": bad_lines, "pre_revision_lines": counts.get("pre_revision", 0),
            "lease_join": joined}, indent=1))
        return 0
    if not report:
        print("no lane door log lines read")
    else:
        print(render(report, verdicts, joined))
    if bad_lines:
        print(f"({bad_lines} unreadable lines skipped)")
    if counts.get("pre_revision"):
        print(f"({counts['pre_revision']} lines from the first, keyed door build skipped: not evidence either way)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
