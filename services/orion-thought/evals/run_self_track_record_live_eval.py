"""Render the track-record cue from Orion's LIVE self-monitor state and truth-check it (read-only).

    python services/orion-thought/evals/run_self_track_record_live_eval.py [--source auto|redis|report]

``redis``: the published state, Redis ``orion:self_monitor:latest`` on ``ORION_BUS_URL`` (default the
athena bus), exactly what orion-thought reads. ``report``: no published state yet (before
self-monitor patch 2 is deployed) -- build the same ``SelfMonitorStateV1`` from the bet tables with
``orion.self_monitor.report.collect`` (``POSTGRES_URI``, default the host-published athena Postgres).
Note: report-built states have no hysteresis history, so every status is its first-evaluation
candidate. ``auto`` (default): redis when the key exists, else report.

Prints the source, the state age, every notable faculty, the cue exactly as the prompt gets it,
and ``self_track_record_truth.check_cue`` problems. Exit 1 on a truth problem.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE))

from orion.self_monitor.cue import NOTABLE, project_track_record_cue, sentence  # noqa: E402
from orion.self_monitor.view import SELF_MONITOR_REDIS_KEY, self_monitor_view  # noqa: E402
from self_track_record_truth import check_cue  # noqa: E402


def _from_redis(url: str):
    import redis

    return redis.Redis.from_url(url, socket_timeout=3, socket_connect_timeout=3).get(SELF_MONITOR_REDIS_KEY)


def _from_report(dsn: str, now: datetime) -> str:
    from orion.self_monitor.record import WINDOW_DAYS
    from orion.self_monitor.report import collect

    return collect(dsn, now=now, window_days=WINDOW_DAYS).model_dump_json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("auto", "redis", "report"), default="auto")
    args = ap.parse_args()
    now = datetime.now(timezone.utc)
    url = os.getenv("ORION_BUS_URL", "redis://100.92.216.81:6379/0")
    dsn = os.getenv("POSTGRES_URI", "postgresql://postgres:postgres@localhost:55432/conjourney")

    raw, source = None, args.source
    if source in ("auto", "redis"):
        try:
            raw = _from_redis(url)
        except Exception as exc:  # noqa: BLE001
            print(f"redis read failed: {type(exc).__name__}: {exc}")
        if raw is None and source == "auto":
            print(f"{SELF_MONITOR_REDIS_KEY} absent on the bus (patch 2 not deployed?); building from the report")
            source = "report"
        elif source == "auto":
            source = "redis"
    if source == "report":
        raw = _from_report(dsn, now)

    view = self_monitor_view(raw, now=now)
    print(f"source={source} known={view.known} reason={view.reason} age_sec={view.age_sec}")
    for record in sorted(view.faculties.values(), key=lambda r: r.faculty):
        if record.status in NOTABLE:
            print(f"  {record.status:13s} {record.faculty:45s} -> {sentence(record)}")
    cue = project_track_record_cue(view, now=now)
    print(f"\ncue ({len(cue.text)} chars, skip_reason={cue.skip_reason}, shown={len(cue.shown)}, "
          f"dropped={cue.lines_dropped}):\n{cue.text}\n")
    problems = check_cue(cue, view)
    for p in problems:
        print(f"TRUTH PROBLEM: {p}")
    print("truth: ok" if not problems else f"truth: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
