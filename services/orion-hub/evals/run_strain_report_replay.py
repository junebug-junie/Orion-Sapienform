#!/usr/bin/env python3
"""Live replay, read-only: over saved arousal history, how many strain reports would Orion have
sent Juniper per local day?

    python3 services/orion-hub/evals/run_strain_report_replay.py [--since 2026-10-10] [--json]
        [--min-episode-sec 3600] [--quiet 23 8] [--tz America/Denver] [--rows-json FILE]

Source: ``temporal_self_event`` rows with ``source_kind='arousal_transition'``, level rows only
(``event_id NOT LIKE '%:at:%'``; the ``:at:`` rows are cause/keepalive rows inside an episode).
Read with ``docker exec orion-athena-sql-db psql ... -Atc`` (SELECT only), or from ``--rows-json``
(a list of ``{"occurred_at", "payload_json"}``) for the fixture test.

Runs the production gate (``scripts.strain_report.strain_report_block`` and
``StrainReportHistory``) minute by minute through each strained episode, plus quiet hours
(``scripts.endogenous_outreach.in_quiet_hours``). Not replayed: turn_in_flight, recent_chat, the
daily cap and the cooldowns (their history is not reconstructable here); those only delay or
remove a report, so the count is an upper bound. Orion's words and PASS are not replayed either.

An episode starts at a strained level row's ``reading.since`` and ends at the next level row
(any level, or a strained row with a new ``since``: a restart). The last open episode runs to now.
Exit 0 on a completed replay; 2 when the rows cannot be read.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

_HUB = Path(__file__).resolve().parents[1]
for key in [k for k in sys.modules if k == "scripts" or k.startswith("scripts.")]:
    del sys.modules[key]
for p in (str(_HUB.parents[1]), str(_HUB)):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

SQL = (
    "SELECT json_build_object('occurred_at', occurred_at, 'payload_json', payload_json) "
    "FROM temporal_self_event WHERE source_kind = 'arousal_transition' "
    "AND event_id NOT LIKE '%%:at:%%' AND occurred_at >= '%s' ORDER BY occurred_at"
)


@dataclass
class Episode:
    since: datetime
    end: datetime
    still_open: bool


def _dt(v: Any) -> datetime:
    d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def load_rows_psql(since: str) -> List[Dict[str, Any]]:
    out = subprocess.run(
        ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney",
         "-Atc", SQL % since],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def episodes_from_rows(rows: List[Dict[str, Any]], now: datetime) -> List[Episode]:
    eps: List[Episode] = []
    cur: Optional[datetime] = None
    for row in rows:
        payload = row["payload_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        at = _dt(row["occurred_at"])
        to = payload.get("to")
        since = _dt((payload.get("reading") or {}).get("since") or at)
        if to == "strained":
            if cur is not None and since != cur:
                eps.append(Episode(cur, at, False))
                cur = None
            if cur is None:
                cur = since
            continue
        if cur is not None:
            eps.append(Episode(cur, at, False))
            cur = None
    if cur is not None:
        eps.append(Episode(cur, now, True))
    return eps


def replay(eps: List[Episode], *, min_episode_sec: float, quiet: tuple, tz: Any, step_sec: float = 60.0) -> Dict[str, Any]:
    from scripts.endogenous_outreach import in_quiet_hours
    from scripts.strain_report import StrainReportHistory, strain_report_block

    history = StrainReportHistory()
    days: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"episodes_started": 0, "long_episodes": 0,
                                                            "reports": [], "held_by_day_cap": 0,
                                                            "held_by_quiet_hours": 0})
    for ep in eps:
        d = days[ep.since.astimezone(tz).date().isoformat()]
        d["episodes_started"] += 1
        if (ep.end - ep.since).total_seconds() < min_episode_sec:
            continue
        d["long_episodes"] += 1
        t = ep.since + timedelta(seconds=min_episode_sec)
        waited_quiet = False
        last_block = None
        while t <= ep.end:
            if in_quiet_hours(t.astimezone(tz).hour, quiet[0], quiet[1]):
                waited_quiet = True
                t += timedelta(seconds=step_sec)
                continue
            block = strain_report_block(level="strained", since=ep.since, now=t,
                                        min_episode_sec=min_episode_sec, history=history, tz=tz)
            if block is None:
                from scripts.strain_report import episode_key

                history.note(episode=episode_key(ep.since), reason="sent", at=t, tz=tz)
                local = t.astimezone(tz)
                days[local.date().isoformat()]["reports"].append({
                    "sent_at_local": local.strftime("%Y-%m-%d %H:%M"),
                    "episode_since_local": ep.since.astimezone(tz).strftime("%Y-%m-%d %H:%M"),
                    "episode_len_h": round((ep.end - ep.since).total_seconds() / 3600, 2),
                    "waited_for_quiet_hours": waited_quiet,
                })
                break
            last_block = block
            t += timedelta(seconds=step_sec)
        else:
            if last_block == "already_reported_today":
                d["held_by_day_cap"] += 1
            elif waited_quiet:
                d["held_by_quiet_hours"] += 1
    total = sum(len(v["reports"]) for v in days.values())
    return {"episodes": len(eps), "reports_total": total, "days": dict(sorted(days.items()))}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-10")
    ap.add_argument("--min-episode-sec", type=float, default=3600.0)
    ap.add_argument("--quiet", type=int, nargs=2, default=(23, 8))
    ap.add_argument("--tz", default="America/Denver")
    ap.add_argument("--rows-json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    tz = ZoneInfo(args.tz)
    try:
        rows = json.loads(Path(args.rows_json).read_text()) if args.rows_json else load_rows_psql(args.since)
    except Exception as exc:  # noqa: BLE001
        print(f"rows unavailable: {exc}", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    eps = episodes_from_rows(rows, now)
    result = replay(eps, min_episode_sec=args.min_episode_sec, quiet=tuple(args.quiet), tz=tz)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"level rows={len(rows)} strained episodes={result['episodes']} reports={result['reports_total']} "
          f"(min_episode={args.min_episode_sec:.0f}s quiet={args.quiet[0]}-{args.quiet[1]} tz={args.tz})")
    print(f"{'local day':<12} {'eps':>4} {'>=min':>6} {'reports':>8} {'day-cap held':>13} {'quiet held':>11}  sends")
    for day, v in result["days"].items():
        sends = "; ".join(
            f"{r['sent_at_local'][11:]} (episode from {r['episode_since_local']}, {r['episode_len_h']} h"
            f"{', waited quiet hours' if r['waited_for_quiet_hours'] else ''})" for r in v["reports"]
        )
        print(f"{day:<12} {v['episodes_started']:>4} {v['long_episodes']:>6} {len(v['reports']):>8} "
              f"{v['held_by_day_cap']:>13} {v['held_by_quiet_hours']:>11}  {sends}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
