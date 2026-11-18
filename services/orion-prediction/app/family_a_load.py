"""Family A raw-reading loader: grain-bucket means of every configured input, read-only.

One query per source table, bucketed in Postgres (``date_bin``), all keys as bind parameters (the
config's keys are validated against ``[a-z0-9_]`` in app/family_a.py, never formatted into SQL).
Only raw readings are read: ``orion_biometrics_summary.measurements`` (never ``pressures`` /
``headroom`` / ``composites``), ``orion_biometrics_cluster.measurements_by_node``,
``home_cooling_sample.cooling_watts``, and the ``measurements`` object of the office / closet
sample payloads. Stale or offline office / closet / cooling samples are not readings.
A bucket with no reading stays NaN.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

import numpy as np
import pandas as pd

from app.family_a import InputSpec

READ_TIMEOUT_MS = 120_000


def _num(expr: str) -> str:
    """A jsonb value as float only when it is a JSON number (a string or null reads as no reading).
    ``expr`` appears twice, so its bind parameters are passed twice (``_twice``)."""
    return f"CASE WHEN jsonb_typeof({expr}) = 'number' THEN ({expr})::text::float END"


def _twice(params: list) -> list:
    return params + params


def _since(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def build_queries(inputs: Sequence[InputSpec], grain_min: int, start: datetime, end: datetime) -> list[tuple[str, list, list[str]]]:
    """(sql, args, column ids) per source group. Pure, so the SQL shape is testable without a DB."""
    bin_ = f"'{int(grain_min)} minutes'"
    groups: dict[tuple, list[InputSpec]] = {}
    for spec in inputs:
        s = spec.source
        groups.setdefault((s.table, s.node, s.role), []).append(spec)
    out = []
    for (table, node, role), specs in groups.items():
        ids = [s.id for s in specs]
        if table == "biometrics":
            sel = ", ".join(f"avg({_num('measurements->%s')}) AS c{i}" for i in range(len(specs)))
            ts = "timestamp::timestamptz"
            sql = (f"SELECT date_bin({bin_}, {ts}, TIMESTAMPTZ '2000-01-01') AS b, {sel} FROM orion_biometrics_summary "
                   "WHERE node = %s AND timestamp >= %s AND timestamp < %s AND jsonb_typeof(measurements) = 'object' "
                   "GROUP BY 1 ORDER BY 1")
            args = [a for s in specs for a in _twice([s.source.key])] + [node, _since(start), _since(end)]
        elif table == "cluster":
            sel = ", ".join(f"avg({_num('measurements_by_node->%s->%s')}) AS c{i}" for i in range(len(specs)))
            sql = (f"SELECT date_bin({bin_}, observed_at, TIMESTAMPTZ '2000-01-01') AS b, {sel} "
                   "FROM orion_biometrics_cluster WHERE observed_at >= %s AND observed_at < %s GROUP BY 1 ORDER BY 1")
            args = [a for s in specs for a in _twice([node, s.source.key])] + [start, end]
        elif table == "cooling":
            sql = (f"SELECT date_bin({bin_}, ts, TIMESTAMPTZ '2000-01-01') AS b, avg(cooling_watts) AS c0 "
                   "FROM home_cooling_sample WHERE role = %s AND ts >= %s AND ts < %s AND device_online "
                   "AND NOT coalesce(stale, false) GROUP BY 1 ORDER BY 1")
            args = [role, start, end]
            ids = ids[:1]
        elif table in ("office_env", "closet_env"):
            tbl = "office_env_sample" if table == "office_env" else "closet_env_sample"
            expr = _num("(payload_json::jsonb)->'measurements'->%s")
            sel = ", ".join(f"avg({expr}) AS c{i}" for i in range(len(specs)))
            sql = (f"SELECT date_bin({bin_}, ts, TIMESTAMPTZ '2000-01-01') AS b, {sel} FROM {tbl} "
                   "WHERE node = %s AND ts >= %s AND ts < %s AND sensor_online AND NOT stale GROUP BY 1 ORDER BY 1")
            args = [a for s in specs for a in _twice([s.source.key])] + [node, start, end]
        else:   # parse_config already refuses any other table
            raise ValueError(f"unknown table {table}")
        out.append((sql, args, ids))
    return out


def load_frame(conninfo: str, inputs: Sequence[InputSpec], grain_min: int, start: datetime,
               end: datetime) -> tuple[pd.DataFrame, dict[str, datetime | None]]:
    """(frame, last_seen): a full grain grid [start, end) x input ids (NaN = no reading), and the newest
    bucket with a reading per input (the staleness check)."""
    import psycopg

    index = pd.date_range(start, end - timedelta(minutes=grain_min), freq=f"{grain_min}min", tz="UTC")
    frame = pd.DataFrame(index=index, columns=[s.id for s in inputs], dtype=float)
    with psycopg.connect(conninfo, autocommit=False) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute(f"SET LOCAL statement_timeout = {READ_TIMEOUT_MS}")
        for sql, args, ids in build_queries(inputs, grain_min, start, end):
            rows = conn.execute(sql, args).fetchall()
            if not rows:
                continue
            ts = pd.DatetimeIndex([r[0] for r in rows]).tz_convert("UTC")
            for i, iid in enumerate(ids):
                vals = pd.Series([r[1 + i] for r in rows], index=ts, dtype=float)
                frame[iid] = vals.reindex(index)
    last_seen = {}
    for c in frame.columns:
        ok = frame[c].dropna()
        last_seen[c] = ok.index[-1].to_pydatetime() + timedelta(minutes=grain_min) if len(ok) else None
    return frame.astype(float), last_seen


def finite_share(frame: pd.DataFrame) -> dict[str, float]:
    return {c: float(np.isfinite(frame[c].to_numpy(dtype=float)).mean()) for c in frame.columns}
