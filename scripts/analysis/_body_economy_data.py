"""Read-only extract shared by fit_heat_sensitivity.py and replay_body_economy.py.

Three live tables, SELECT only, in a read-only session (``SET default_transaction_read_only``):

- ``orion_biometrics_cluster`` (~1 row / 15 s, 30-day retention): athena ``cabinet_temp_c``
  (the cabinet sensor sits on athena), per-host ``chassis_watts``, compute watts
  (``cpu_watts_total + gpu_watts_total``) and ``fan_pct_max`` from ``measurements_by_node``.
  Binned to 15 minutes.
- ``orion_energy_hourly``: per-source hourly kWh (athena, circe, hecate, cooling = the AC plug).
- ``prediction_weather_hourly``: outdoor temperature (deg F) persisted by orion-prediction from
  Open-Meteo (``orion/weather/openmeteo.py``). Observed rows only (``is_forecast = false``).

Access: ``psql`` through ``docker exec`` into ``orion-athena-sql-db`` by default; override the
command prefix with ``ORION_ANALYSIS_PSQL``. ``--cache`` writes/reads the extract as JSON so a run
is reproducible offline.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_PSQL = "docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -At -q -v ON_ERROR_STOP=1"
HOSTS = ("athena", "circe", "hecate")
# hecate reports only chassis/PDU watts, no CPU/GPU split or fans.
COMPUTE_HOSTS = ("athena", "circe")


def _bins_sql(start: str, end: str) -> str:
    w = ",\n".join(
        f"avg((measurements_by_node->'{h}'->>'chassis_watts')::float) as w_{h}" for h in HOSTS
    )
    w += ",\n" + ",\n".join(
        f"avg((measurements_by_node->'{h}'->>'cpu_watts_total')::float"
        f" + (measurements_by_node->'{h}'->>'gpu_watts_total')::float) as cw_{h},\n"
        f"avg((measurements_by_node->'{h}'->>'fan_pct_max')::float) as fan_{h}" for h in COMPUTE_HOSTS
    )
    return f"""
select coalesce(json_agg(r order by r.t), '[]'::json) from (
  select to_char(date_bin('15 minutes', observed_at, timestamptz '2026-01-01') at time zone 'UTC',
                 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as t,
         avg((measurements_by_node->'athena'->>'cabinet_temp_c')::float) as cabinet_c,
         {w},
         count(*) as n
  from orion_biometrics_cluster
  where observed_at >= timestamptz '{start}' and observed_at < timestamptz '{end}'
  group by 1
) r;"""


def _energy_sql(start: str, end: str) -> str:
    return f"""
select coalesce(json_agg(r order by r.t, r.source), '[]'::json) from (
  select to_char(hour_start at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as t,
         source, kwh, covered_sec, sample_count
  from orion_energy_hourly
  where hour_start >= timestamptz '{start}' and hour_start < timestamptz '{end}'
) r;"""


def _weather_sql(start: str, end: str) -> str:
    return f"""
select coalesce(json_agg(r order by r.t), '[]'::json) from (
  select to_char(hour_ts at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as t, temp_f, source
  from prediction_weather_hourly
  where is_forecast = false and hour_ts >= timestamptz '{start}' and hour_ts < timestamptz '{end}'
) r;"""


def _cooling_sql(start: str, end: str) -> str:
    # The portable cabinet AC, as orion-prediction reads it (series athena.cabinet_cooling_watts:
    # home_cooling_sample, role cabinet_cooling).
    return f"""
select coalesce(json_agg(r order by r.t), '[]'::json) from (
  select to_char(date_bin('15 minutes', ts, timestamptz '2026-01-01') at time zone 'UTC',
                 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as t, avg(cooling_watts) as w
  from home_cooling_sample
  where role = 'cabinet_cooling' and ts >= timestamptz '{start}' and ts < timestamptz '{end}'
  group by 1
) r;"""


def _house_sql(start: str, end: str) -> str:
    # Whole-house meter (Rocky Mountain Power, hourly, whole kWh, ~3 days behind): the signal
    # orion-prediction's run_house_heat_balance_eval.py uses to validate the central-AC proxy.
    return f"""
select coalesce(json_agg(r order by r.t), '[]'::json) from (
  select to_char(interval_start at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as t, energy_kwh as kwh
  from energy_usage_interval
  where source = 'rockymountain_power' and interval_start >= timestamptz '{start}' and interval_start < timestamptz '{end}'
) r;"""


# orion-prediction's change events (#2780): the same rows its engine folds into training
# (services/orion-prediction/app/store.py change_events, app/events.py applies()).
CHANGE_EVENTS_SQL = """
select coalesce(json_agg(r order by r.started_at), '[]'::json) from (
  select scope, effect, title, notes,
         to_char(started_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as started_at,
         to_char(ended_at at time zone 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS"Z"') as ended_at
  from prediction_change_event
  where retracted_at is null and scope in ('*', 'athena.cabinet_temp_c')
) r;"""


def _validate_ts(s: str) -> str:
    # Only ISO dates/timestamps reach the SQL text.
    from datetime import datetime

    datetime.fromisoformat(s.replace("Z", "+00:00"))
    return s


def run_query(sql: str) -> Any:
    cmd = shlex.split(os.environ.get("ORION_ANALYSIS_PSQL", DEFAULT_PSQL))
    full = "set default_transaction_read_only = on;\nbegin read only;\n" + sql + "\ncommit;\n"
    proc = subprocess.run(cmd, input=full, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"psql failed ({proc.returncode}): {proc.stderr.strip()[:500]}")
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("psql returned no rows")
    # json_agg prints one element per line; the whole stdout is one JSON value.
    return json.loads(out)


def _minus_hours(ts: str, h: int) -> str:
    from datetime import datetime, timedelta

    t = datetime.fromisoformat(ts.replace("Z", "+00:00")) - timedelta(hours=h)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(start: str, end: str) -> dict:
    start, end = _validate_ts(start), _validate_ts(end)
    return {
        "window": {"start": start, "end": end},
        "bins15": run_query(_bins_sql(start, end)),
        "energy_hourly": run_query(_energy_sql(start, end)),
        # 6 h earlier so a lagged weather covariate (central AC proxy: 3 h) has values from the start.
        "weather_hourly": run_query(_weather_sql(_minus_hours(start, 6), end)),
        "cooling15": run_query(_cooling_sql(start, end)),
        "house_hourly": run_query(_house_sql(_minus_hours(start, 6), end)),
        "change_events": run_query(CHANGE_EVENTS_SQL),
        "queries": {
            "cooling15": _cooling_sql(start, end).strip(),
            "house_hourly": _house_sql(_minus_hours(start, 6), end).strip(),
            "change_events": CHANGE_EVENTS_SQL.strip(),
            "bins15": _bins_sql(start, end).strip(),
            "energy_hourly": _energy_sql(start, end).strip(),
            "weather_hourly": _weather_sql(start, end).strip(),
        },
    }


def load_or_fetch(start: str, end: str, cache: str | None) -> dict:
    if cache and Path(cache).exists():
        data = json.loads(Path(cache).read_text())
        if data.get("window") == {"start": start, "end": end} and "house_hourly" in data:
            return data
    data = fetch(start, end)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(data))
    return data
