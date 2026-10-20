"""Reads telemetry, owns the prediction_* tables. Sync psycopg (called via asyncio.to_thread).

orion_biometrics_summary.timestamp is VARCHAR ('YYYY-MM-DD HH:MM:SS.ffffff+00'), indexed with node:
filtered as a string on that prefix, parsed as timestamptz (same approach as orion-hardware-watch).
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from app.series import SeriesSpec

_KEY = re.compile(r"^[a-z0-9_]{1,40}$")
TABLES = {
    "prediction_model_run": ("run_id", "selection_id", "kind", "assignment", "residual_paths"),
    "prediction_forecast": ("forecast_id", "role", "quantiles", "score_status", "in_80"),
    "prediction_experiment": ("experiment_id", "design", "guard", "status"),
    "prediction_intervention": ("intervention_id", "action", "arm", "experiment_id"),
    "prediction_effect_estimate": ("estimate_id", "claim_kind", "effect", "details"),
    "prediction_change_event": ("event_id", "scope", "effect", "started_at", "ended_at"),
}


def _since(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _j(value: Any) -> str:
    return json.dumps(value, default=str, allow_nan=False)


class PostgresStore:
    def __init__(self, conninfo: str):
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(conninfo=conninfo, min_size=1, max_size=3, open=True,
                                   kwargs={"row_factory": dict_row, "autocommit": True,
                                           "application_name": "orion-prediction"})

    def close(self) -> None:
        self.pool.close()

    def _all(self, sql: str, args: Sequence = ()) -> list[dict]:
        with self.pool.connection() as conn:
            cur = conn.execute(sql, args)
            return list(cur.fetchall()) if cur.description else []

    def check_schema(self) -> None:
        """Operator-applied migration (services/orion-sql-db/manual_migration_prediction_v1.sql)."""
        for table, cols in TABLES.items():
            self._all(f"SELECT {', '.join(cols)} FROM {table} LIMIT 0")

    # --- telemetry ------------------------------------------------------------------------
    def points(self, spec: SeriesSpec, since: datetime, until: datetime | None = None) -> list[tuple[datetime, float | None]]:
        src = spec.source
        if src.kind == "biometrics":
            if not _KEY.match(src.key or ""):
                raise ValueError(f"bad measurement key {src.key!r}")
            sql = ("SELECT timestamp::timestamptz AS ts, (measurements->>%s)::float AS v FROM orion_biometrics_summary "
                   "WHERE node = %s AND timestamp >= %s AND measurements ? %s")
            args: list = [src.key, src.node, _since(since), src.key]
            if until is not None:
                sql += " AND timestamp < %s"
                args.append(_since(until))
        elif src.kind == "cooling":
            sql = "SELECT ts, cooling_watts AS v FROM home_cooling_sample WHERE role = %s AND ts >= %s"
            args = [src.role, since]
            if until is not None:
                sql += " AND ts < %s"
                args.append(until)
        else:
            raise ValueError(f"unknown source kind {src.kind}")
        return [(r["ts"], r["v"]) for r in self._all(sql + " ORDER BY 1", args)]

    # --- model runs -----------------------------------------------------------------------
    def insert_runs(self, rows: list[dict]) -> None:
        cols = ("run_id", "selection_id", "series_id", "kind", "model", "fitted_at", "train_start", "train_end",
                "n_obs", "grain_min", "usable_seasonalities", "refused_seasonalities", "n_windows", "metrics",
                "assignment", "served_baseline", "backtest_seconds", "residual_paths", "errors")
        jsonb = {"usable_seasonalities", "refused_seasonalities", "metrics", "assignment", "residual_paths", "errors"}
        sql = (f"INSERT INTO prediction_model_run ({', '.join(cols)}) VALUES "
               f"({', '.join('%s::jsonb' if c in jsonb else '%s' for c in cols)})")
        with self.pool.connection() as conn, conn.transaction():
            for r in rows:
                conn.execute(sql, [(_j(r[c]) if c in jsonb else r[c]) for c in cols])

    def latest_selection(self, series_id: str) -> list[dict]:
        return self._all(
            "SELECT * FROM prediction_model_run WHERE selection_id = ("
            " SELECT selection_id FROM prediction_model_run WHERE series_id = %s ORDER BY fitted_at DESC LIMIT 1)"
            " ORDER BY kind, model", (series_id,))

    # --- forecasts + scores ---------------------------------------------------------------
    def insert_forecasts(self, rows: list[dict]) -> int:
        sql = ("INSERT INTO prediction_forecast (run_id, series_id, model, role, issued_at, origin_ts, target_ts, "
               "horizon_min, point, quantiles) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb) "
               "ON CONFLICT (series_id, role, origin_ts, horizon_min) DO NOTHING")
        n = 0
        with self.pool.connection() as conn, conn.transaction():
            for r in rows:
                cur = conn.execute(sql, (r["run_id"], r["series_id"], r["model"], r["role"], r["issued_at"],
                                         r["origin_ts"], r["target_ts"], r["horizon_min"], r["point"],
                                         _j(r["quantiles"])))
                n += cur.rowcount
        return n

    def last_origin(self, series_id: str) -> datetime | None:
        rows = self._all("SELECT max(origin_ts) AS o FROM prediction_forecast "
                         "WHERE series_id = %s AND role = 'champion'", (series_id,))
        return rows[0]["o"] if rows else None

    def last_issued(self, series_id: str) -> datetime | None:
        rows = self._all("SELECT max(issued_at) AS i FROM prediction_forecast WHERE series_id = %s", (series_id,))
        return rows[0]["i"] if rows else None

    def latest_forecast(self, series_id: str) -> list[dict]:
        return self._all(
            "SELECT * FROM prediction_forecast WHERE series_id = %s AND role = 'champion' AND origin_ts = ("
            " SELECT max(origin_ts) FROM prediction_forecast WHERE series_id = %s AND role = 'champion')"
            " ORDER BY horizon_min", (series_id, series_id))

    def due_for_scoring(self, series_id: str, before: datetime, limit: int = 2000) -> list[dict]:
        return self._all(
            "SELECT forecast_id, target_ts, quantiles, point FROM prediction_forecast "
            "WHERE series_id = %s AND scored_at IS NULL AND target_ts < %s ORDER BY target_ts LIMIT %s",
            (series_id, before, limit))

    def write_scores(self, scores: list[dict]) -> None:
        sql = ("UPDATE prediction_forecast SET scored_at = %s, score_status = %s, actual = %s, abs_error = %s, "
               "in_80 = %s, in_95 = %s WHERE forecast_id = %s AND scored_at IS NULL")
        with self.pool.connection() as conn, conn.transaction():
            for s in scores:
                conn.execute(sql, (s["scored_at"], s["score_status"], s["actual"], s["abs_error"], s["in_80"],
                                   s["in_95"], s["forecast_id"]))

    def live_scores(self, series_id: str, since: datetime) -> list[dict]:
        return self._all(
            "SELECT role, origin_ts, horizon_min, abs_error, in_80, in_95 FROM prediction_forecast "
            "WHERE series_id = %s AND score_status = 'scored' AND target_ts >= %s", (series_id, since))

    # --- interventions, experiments, effects ----------------------------------------------
    def insert_intervention(self, row: dict) -> dict:
        row = {"intervention_id": str(uuid.uuid4()), "arm": None, "experiment_id": None, "notes": "",
               "created_at": datetime.now(timezone.utc), **row}
        self._all(
            "INSERT INTO prediction_intervention (intervention_id, action, started_at, ended_at, arm, experiment_id, "
            "source, notes, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (row["intervention_id"], row["action"], row["started_at"], row["ended_at"], row["arm"],
             row["experiment_id"], row["source"], row["notes"], row["created_at"]))
        return row

    def interventions(self, action: str | None = None, experiment_id: str | None = None, limit: int = 500) -> list[dict]:
        where, args = [], []
        if action:
            where.append("action = %s")
            args.append(action)
        if experiment_id:
            where.append("experiment_id = %s")
            args.append(experiment_id)
        sql = "SELECT * FROM prediction_intervention"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return self._all(sql + " ORDER BY started_at DESC LIMIT %s", (*args, limit))

    def intervention(self, intervention_id: str) -> dict | None:
        rows = self._all("SELECT * FROM prediction_intervention WHERE intervention_id = %s", (intervention_id,))
        return rows[0] if rows else None

    def insert_experiment(self, row: dict) -> None:
        self._all(
            "INSERT INTO prediction_experiment (experiment_id, action, series_id, created_at, burn_in_min, design, "
            "guard, status) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,'designed')",
            (row["experiment_id"], row["action"], row["series_id"], row["created_at"], row["burn_in_min"],
             _j(row["design"]), _j(row["guard"])))

    def experiment(self, experiment_id: str) -> dict | None:
        rows = self._all("SELECT * FROM prediction_experiment WHERE experiment_id = %s", (experiment_id,))
        return rows[0] if rows else None

    def set_experiment_status(self, experiment_id: str, status: str) -> None:
        self._all("UPDATE prediction_experiment SET status = %s WHERE experiment_id = %s", (status, experiment_id))

    def insert_effect(self, row: dict) -> dict:
        row = {"estimate_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc),
               "intervention_id": None, "experiment_id": None, **row}
        self._all(
            "INSERT INTO prediction_effect_estimate (estimate_id, action, series_id, claim_kind, intervention_id, "
            "experiment_id, effect, ci_lo, ci_hi, p_value, n, details, created_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)",
            (row["estimate_id"], row["action"], row["series_id"], row["claim_kind"], row["intervention_id"],
             row["experiment_id"], row["effect"], row["ci_lo"], row["ci_hi"], row["p_value"], row["n"],
             _j(row["details"]), row["created_at"]))
        return row

    # --- change events ----------------------------------------------------------------------
    def change_events(self) -> list[dict]:
        return self._all("SELECT event_id, scope, effect, started_at, ended_at, title, notes, source, created_at "
                         "FROM prediction_change_event ORDER BY started_at")

    def insert_change_event(self, row: dict) -> dict:
        row = {"event_id": str(uuid.uuid4()), "ended_at": None, "notes": "",
               "created_at": datetime.now(timezone.utc), **row}
        self._all("INSERT INTO prediction_change_event (event_id, scope, effect, started_at, ended_at, title, notes, "
                  "source, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                  (row["event_id"], row["scope"], row["effect"], row["started_at"], row["ended_at"], row["title"],
                   row["notes"], row["source"], row["created_at"]))
        return row

    def effects(self, action: str, series_id: str) -> list[dict]:
        return self._all("SELECT * FROM prediction_effect_estimate WHERE action = %s AND series_id = %s "
                         "ORDER BY created_at DESC LIMIT 200", (action, series_id))
