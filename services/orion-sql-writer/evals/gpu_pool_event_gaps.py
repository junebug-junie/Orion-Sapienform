#!/usr/bin/env python3
"""Eval (#3920): are any lease grants missing from gpu_pool_events, and do they line up with
sql-writer restarts?

The pool's own projection ``gpu_pool_leases`` (one row per lease, written by the pool directly)
is the ground truth for "a grant happened". ``gpu_pool_events`` is the history sql-writer writes
from the pool's event stream. A lease with ``granted_at`` in the window and no ``granted`` row
is a lost event.

Each missing grant is tagged with the sql-writer rebuild window it falls in, read from
``~/.orion/post_pull_rebuild.log`` (``START``/``END ... orion-sql-writer up``). ``END`` is when
compose returns, not when the new container subscribes: the service boots for another ~20-25 s
(gpu_pool_state_history resumed 06:36:22Z after the 06:36:07Z END on 2026-11-25), so the window
runs to ``END + --pad-after-sec``. Manual redeploys are not in that log; they show as "outside".

GATE (after #3920 is deployed): 0 missing grants in a window that contains at least one
sql-writer rebuild. Before the fix, 2026-11-24..25 had 69 missing, 68 inside a rebuild window.

    python3 services/orion-sql-writer/evals/gpu_pool_event_gaps.py --since 2026-11-24 --until 2026-11-25T07:00
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine, text

DEFAULT_URI = "postgresql://postgres:postgres@localhost:55432/conjourney"
LOG = Path.home() / ".orion" / "post_pull_rebuild.log"
LINE = re.compile(r"^(\S+Z) .* (START|END)\b.* orion-sql-writer up\b")

MISSING_SQL = text("""
    SELECT l.lease_id, l.granted_at, l.holder, l.work_class, l.role
    FROM gpu_pool_leases l
    WHERE l.granted_at >= :since AND l.granted_at < :until
      AND NOT EXISTS (SELECT 1 FROM gpu_pool_events e
                      WHERE e.lease_id = l.lease_id AND e.event = 'granted'
                        AND e.generated_at >= CAST(:since AS timestamptz) - interval '1 day')
    ORDER BY l.granted_at
""")
GRANTED_SQL = text("SELECT count(*) FROM gpu_pool_leases WHERE granted_at >= :since AND granted_at < :until")


def _ts(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def rebuild_windows(log_text: str, before: timedelta, after: timedelta) -> list[tuple[datetime, datetime]]:
    """Paired START/END lines for orion-sql-writer up: [START - before, END + after]."""
    out, start = [], None
    for line in log_text.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        if m.group(2) == "START":
            start = _ts(m.group(1))
        elif start is not None:
            out.append((start - before, _ts(m.group(1)) + after))
            start = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", default=None)
    ap.add_argument("--pad-before-sec", type=float, default=5.0)
    ap.add_argument("--pad-after-sec", type=float, default=40.0)
    args = ap.parse_args()
    since = _ts(args.since)
    until = _ts(args.until) if args.until else datetime.now(timezone.utc)
    windows = rebuild_windows(LOG.read_text() if LOG.exists() else "", timedelta(seconds=args.pad_before_sec),
                              timedelta(seconds=args.pad_after_sec))
    windows = [w for w in windows if w[1] >= since and w[0] < until]

    engine = create_engine(os.environ.get("ORION_ANALYSIS_PG_URI", DEFAULT_URI))
    with engine.connect() as conn:
        granted = conn.execute(GRANTED_SQL, {"since": since, "until": until}).scalar() or 0
        missing = conn.execute(MISSING_SQL, {"since": since, "until": until}).all()

    inside = [r for r in missing if any(a <= r.granted_at <= b for a, b in windows)]
    print(f"window {since.isoformat()} .. {until.isoformat()}")
    print(f"grants (gpu_pool_leases): {granted}")
    print(f"sql-writer rebuild windows in range: {len(windows)}")
    print(f"missing 'granted' rows in gpu_pool_events: {len(missing)} "
          f"(inside a rebuild window: {len(inside)}, outside: {len(missing) - len(inside)})")
    for r in missing:
        tag = "rebuild" if r in inside else "outside"
        print(f"  {r.granted_at.isoformat()} {tag:8} {r.work_class or '-':16} {r.role or '-':12} {r.lease_id}")
    verdict = "PASS" if not missing else "FAIL"
    print(f"verdict: {verdict}" + ("" if windows else " (no sql-writer rebuild in range: weak evidence)"))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
