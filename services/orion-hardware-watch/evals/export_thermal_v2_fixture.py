"""Export the thermal-controller-v2 replay fixture from production Postgres (READ-ONLY).

Spec: docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md, Acceptance check 1.

What it reads (SELECT/COPY TO STDOUT only; nothing is written to the database):
- every ``home_cooling_sample`` row in [since, until)              -> ``c`` rows
- ``orion_biometrics_summary`` for athena and circe in [since, until):
  ``cabinet_temp_c``, ``temp_c_max``, ``gpu{N}_temp_c``, ``gpu_watts_total``  -> ``t`` rows
- every real shed episode that started in [since, until) (D11's input)                -> ``e`` rows:
  ``hardware_watch_incident`` rows with shed_requested (v1 reflex: shed_requested_at -> resolved_at) and
  ``gpu_pool_orion_shed`` rows that started (Orion's learned shed: started_at -> ended_at)

Size rule (keep the committed fixture small): when the gzipped file would exceed ``--max-bytes``
(default 5 MB), AC rows are thinned to one per ``--thin-sec`` OUTSIDE the full-resolution windows
the gate asserts on (10-06 00:00-04:00 and 10-03 21:00-10-04 02:00). Temperature rows are never
thinned (~30 s cadence already). The header line records what was done.

Usage (athena, from services/orion-hardware-watch):
  python evals/export_thermal_v2_fixture.py                      # defaults below
  python evals/export_thermal_v2_fixture.py --until 2026-10-06T04:30:00+00:00
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "thermal_v2_replay.csv.gz"
PSQL = ["docker", "exec", "-i", "orion-athena-sql-db", "psql", "-U", "postgres", "-d", "conjourney", "-At"]
FULL_RES_WINDOWS = (
    ("2026-10-06T00:00:00+00:00", "2026-10-06T04:00:00+00:00"),
    ("2026-10-03T21:00:00+00:00", "2026-10-04T02:00:00+00:00"),
)
KEYS_SQL = "(k IN ('temp_c_max', 'cabinet_temp_c', 'gpu_watts_total') OR k ~ '^gpu[0-9]+_temp_c$')"


def _copy(sql: str) -> str:
    return subprocess.run(PSQL + ["-c", sql], capture_output=True, text=True, check=True).stdout


def _thin(cooling_csv: str, thin_sec: float) -> tuple[str, int, int]:
    windows = [(datetime.fromisoformat(a).timestamp(), datetime.fromisoformat(b).timestamp())
               for a, b in FULL_RES_WINDOWS]
    out, kept, total, last = io.StringIO(), 0, 0, None
    w = csv.writer(out, lineterminator="\n")
    for row in csv.reader(io.StringIO(cooling_csv)):
        total += 1
        t = float(row[1])
        full = any(a <= t < b for a, b in windows)
        # Keep every non-live row too (offline / stale / null watts): those ARE the events.
        notable = row[2] == "" or row[3] == "t" or row[4] != "t" or row[5] != "t"
        if full or notable or last is None or t - last >= thin_sec:
            w.writerow(row)
            kept += 1
            last = t
    return out.getvalue(), kept, total


def export(since: datetime, until: datetime, max_bytes: int, thin_sec: float) -> Path:
    s, u = since.strftime("%Y-%m-%d %H:%M:%S"), until.strftime("%Y-%m-%d %H:%M:%S")
    cooling = _copy(
        "COPY (SELECT 'c', round(extract(epoch FROM ts)::numeric, 3), cooling_watts, stale, device_online, "
        f"controller_ready FROM home_cooling_sample WHERE ts >= '{s}+00' AND ts < '{u}+00' ORDER BY ts) "
        "TO STDOUT WITH CSV")
    temps = _copy(
        "COPY (SELECT 't', node, k, round(extract(epoch FROM timestamp::timestamptz)::numeric, 3), "
        "round((measurements->>k)::numeric, 2) FROM orion_biometrics_summary, jsonb_object_keys(measurements) k "
        f"WHERE timestamp >= '{s}' AND timestamp < '{u}' AND node IN ('athena', 'circe') AND {KEYS_SQL} "
        "ORDER BY node, k, timestamp) TO STDOUT WITH CSV")

    episodes = _copy(
        "COPY (SELECT 'e', 'hardware_watch_incident', incident_id, coalesce(shed_reason, ''), "
        "round(extract(epoch FROM shed_requested_at)::numeric, 3), round(extract(epoch FROM resolved_at)::numeric, 3) "
        f"FROM hardware_watch_incident WHERE shed_requested AND shed_requested_at >= '{s}+00' "
        f"AND shed_requested_at < '{u}+00' UNION ALL SELECT 'e', 'gpu_pool_orion_shed', shed_id, reason, "
        "round(extract(epoch FROM started_at)::numeric, 3), round(extract(epoch FROM ended_at)::numeric, 3) "
        f"FROM gpu_pool_orion_shed WHERE started_at >= '{s}+00' AND started_at < '{u}+00' ORDER BY 5) "
        "TO STDOUT WITH CSV")

    def build(cooling_body: str, note: str) -> bytes:
        head = (f"# since={since.isoformat()} cutoff={until.isoformat()} "
                f"exported_at={datetime.now(timezone.utc).isoformat()} {note}\n")
        return gzip.compress((head + cooling_body + temps + episodes).encode(), compresslevel=9)

    n_c = cooling.count("\n")
    blob = build(cooling, f"cooling_rows={n_c} thinned=no")
    if len(blob) > max_bytes:
        thinned, kept, total = _thin(cooling, thin_sec)
        blob = build(thinned, f"cooling_rows={kept}/{total} thinned={thin_sec:g}s_outside_full_res_windows")
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_bytes(blob)
    print(f"wrote {FIXTURE} ({len(blob)} bytes)")
    return FIXTURE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-09-29T00:00:00+00:00")
    ap.add_argument("--until", default="2026-10-06T04:30:00+00:00")
    ap.add_argument("--max-bytes", type=int, default=5_000_000)
    ap.add_argument("--thin-sec", type=float, default=30.0)
    a = ap.parse_args()
    export(datetime.fromisoformat(a.since).astimezone(timezone.utc),
           datetime.fromisoformat(a.until).astimezone(timezone.utc), a.max_bytes, a.thin_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
