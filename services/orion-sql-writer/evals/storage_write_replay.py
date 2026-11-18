#!/usr/bin/env python3
"""Replay real write history through the storage-write organ, end to end.

Per minute, per table family: rows that landed (counted from the family's own
table timestamps) and writes that did not (``bus_fallback_log`` rows, classified
by the writer's own classifier from the stored error text). Each minute becomes
one writer window built by the REAL emitter (``app/write_health.py``
``build_window_events``) and reduced by the REAL reducer
(``orion/substrate/storage_write_loop``), so the printed timeline is what
``write_failure_pressure`` would have read live.

    # pull from the live DB (read-only) and print the timelines
    python services/orion-sql-writer/evals/storage_write_replay.py --live
    # also refresh the committed fixture the CI eval asserts on
    python services/orion-sql-writer/evals/storage_write_replay.py --live --write-fixture

Limits, stated rather than hidden:
- ``committed`` counts rows present now. Rows later deleted by retention are not
  counted, so a period older than a table's retention under-counts commits and
  the replayed reading is an upper bound. ``grammar_events`` is retained ~3 days,
  so the 2026-09-21 Postgres-restart minute has no surviving denominator and is
  not replayed (only its failure count is reported).
- Duplicates (idempotent skips) left no row and no fallback, so they are absent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orion.substrate.storage_write_loop.pipeline import empty_storage_write_projection  # noqa: E402
from orion.substrate.storage_write_loop.reducer import reduce_storage_write_trace_events  # noqa: E402

FIXTURE = HERE / "fixtures" / "storage_write_replay.json"


def _load_write_health():
    path = HERE.parent / "app" / "write_health.py"
    spec = importlib.util.spec_from_file_location("sqlw_write_health_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


wh = _load_write_health()

# Bus kind -> table family, for the kinds that appear in the replayed periods.
# (The live writer derives this from its route map: worker._write_family_for_kind.)
KIND_FAMILY = {
    "home.cooling.sample.v1": "home_cooling_sample",
    "cockpit.hop.v1": "cockpit_turn_sighting",
    "grammar.event.v1": "grammar_events",
    "durable.run.state.v1": "durable_run_state",
    "self_concept.history.write.v1": "self_concept_history",
}

# (label, start, end, [(family, table, ts_column)]) -- the committed-row sources.
PERIODS = [
    (
        "calm_recent_3h",
        None,  # filled at run time: now-3h .. now-5min
        None,
        [
            ("grammar_events", "grammar_events", "created_at"),
            ("gpu_pool_events", "gpu_pool_events", "created_at"),
            ("home_cooling_sample", "home_cooling_sample", "ts"),
            ("cockpit_turn_sighting", "cockpit_turn_sighting", "created_at"),
            ("metacognition_ticks", "metacognition_ticks", "generated_at"),
        ],
    ),
    (
        "home_cooling_serialization_2026-09-26",
        "2026-09-26 02:30:00+00",
        "2026-09-26 05:30:00+00",
        [("home_cooling_sample", "home_cooling_sample", "ts")],
    ),
    (
        "cockpit_validation_2026-09-07",
        "2026-09-07 07:30:00+00",
        "2026-09-07 19:30:00+00",
        [("cockpit_turn_sighting", "cockpit_turn_sighting", "created_at")],
    ),
]


def _psql(sql: str) -> list[list[str]]:
    out = subprocess.run(
        ["docker", "exec", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney",
         "-AtF", "\t", "-c", sql],
        check=True, capture_output=True, text=True, timeout=300,
    ).stdout
    return [line.split("\t") for line in out.splitlines() if line.strip()]


def pull_live() -> dict:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    periods = []
    for label, start, end, sources in PERIODS:
        if start is None:
            start = (now - timedelta(hours=3)).isoformat()
            end = (now - timedelta(minutes=5)).isoformat()
        minutes: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        for family, table, col in sources:
            # ``timestamp without time zone`` columns are stored UTC by the writer.
            rows = _psql(
                f"select to_char(date_trunc('minute', {col}) at time zone 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:00\"Z\"'), count(*) "
                f"from {table} where {col} >= '{start}' and {col} < '{end}' group by 1"
            )
            for minute, n in rows:
                minutes[minute][family]["committed"] = int(n)
        failures = _psql(
            "select to_char(date_trunc('minute', created_at_ts) at time zone 'UTC', 'YYYY-MM-DD\"T\"HH24:MI:00\"Z\"'), "
            "kind, left(regexp_replace(coalesce(error, ''), '\\s+', ' ', 'g'), 400), count(*) from bus_fallback_log "
            f"where created_at_ts >= '{start}' and created_at_ts < '{end}' group by 1, 2, 3"
        )
        for minute, kind, error, n in failures:
            family = KIND_FAMILY.get(kind, kind.replace(".", "_"))
            cls = wh.classify_write_error(error)
            bucket = minutes[minute][family]
            bucket[cls] = bucket.get(cls, 0) + int(n)
        periods.append({
            "label": label,
            "start": str(start),
            "end": str(end),
            "minutes": {m: {f: dict(c) for f, c in fams.items()} for m, fams in sorted(minutes.items())},
        })
    return {"pulled_at": now.isoformat(), "periods": periods}


def _minute_range(start: str, end: str) -> list[datetime]:
    s = datetime.fromisoformat(start).astimezone(timezone.utc).replace(second=0, microsecond=0)
    e = datetime.fromisoformat(end).astimezone(timezone.utc)
    out = []
    while s < e:
        out.append(s)
        s += timedelta(minutes=1)
    return out


def replay_period(period: dict) -> list[dict]:
    """Every minute of the period (idle minutes included) through emitter + reducer."""
    projection = None
    timeline = []
    for minute in _minute_range(period["start"], period["end"]):
        key = minute.strftime("%Y-%m-%dT%H:%M:00Z")
        families = period["minutes"].get(key, {})
        clock = [minute.timestamp()]
        rec = wh.WriteHealthRecorder(clock=lambda: clock[0])
        for family, classes in families.items():
            for cls, n in classes.items():
                rec.record(family, cls, count=int(n))
        clock[0] = minute.timestamp() + 60.0
        s, e, buckets, q = rec.drain()
        events = wh.build_window_events(writer_node="athena", window_start=s, window_end=e, buckets=buckets,
                                        grammar_queue_max=q)
        projection = projection or empty_storage_write_projection(now=minute)
        projection, receipt = reduce_storage_write_trace_events(events=events, projection=projection,
                                                                now=minute + timedelta(seconds=60))
        after = receipt.state_deltas[0].after
        reading = after["failure_window"]
        timeline.append({
            "minute": key,
            "pressure": after["pressure_hints"].get("write_failure_pressure"),
            "scope": reading["scope"],
            "failed": reading["failed"],
            "attempted": reading["attempted"],
        })
    return timeline


def summarize(label: str, timeline: list[dict]) -> dict:
    measured = [t for t in timeline if t["pressure"] is not None]
    nonzero = [t for t in measured if t["pressure"] > 0]
    peak = max(measured, key=lambda t: t["pressure"], default=None)
    return {
        "label": label,
        "minutes": len(timeline),
        "measured_minutes": len(measured),
        "unmeasured_minutes": len(timeline) - len(measured),
        "nonzero_minutes": len(nonzero),
        "first_nonzero": nonzero[0]["minute"] if nonzero else None,
        "last_nonzero": nonzero[-1]["minute"] if nonzero else None,
        "peak": peak["pressure"] if peak else None,
        "peak_minute": peak["minute"] if peak else None,
        "peak_scope": peak["scope"] if peak else None,
        "zero_minutes": len(measured) - len(nonzero),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="pull from the live DB (read-only)")
    ap.add_argument("--write-fixture", action="store_true", help="save the live pull as the CI fixture")
    ap.add_argument("--timeline", action="store_true", help="print every nonzero minute")
    args = ap.parse_args()
    data = pull_live() if args.live else json.loads(FIXTURE.read_text())
    if args.live and args.write_fixture:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        print(f"wrote {FIXTURE}")
    for period in data["periods"]:
        timeline = replay_period(period)
        print(json.dumps(summarize(period["label"], timeline)))
        if args.timeline:
            for t in timeline:
                if t["pressure"]:
                    print("   ", json.dumps(t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
