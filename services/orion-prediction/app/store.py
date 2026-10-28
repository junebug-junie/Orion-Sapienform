"""Reads telemetry, owns the prediction_* tables. Sync psycopg (called via asyncio.to_thread).

orion_biometrics_summary.timestamp is VARCHAR ('YYYY-MM-DD HH:MM:SS.ffffff+00'), indexed with node:
filtered as a string on that prefix, parsed as timestamptz (same approach as orion-hardware-watch).
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from app.series import SeriesSpec

# Event tables the event_pair estimator may read (read-only), keyed by selector ``source``.
# Each entry: (table, timestamp column, event-type column, stream column). Add a table here and to
# EventSelector.source in app/main.py; nothing else changes.
EVENT_SOURCES = {
    "vision_events": ("vision_events", "created_at", "event_type", "stream_id"),
}
EVENT_READ_TIMEOUT_MS = 10_000

_KEY = re.compile(r"^[a-z0-9_]{1,40}$")
TABLES = {
    "prediction_model_run": ("run_id", "selection_id", "kind", "assignment", "residual_paths"),
    "prediction_forecast": ("forecast_id", "role", "quantiles", "score_status", "in_80"),
    "prediction_experiment": ("experiment_id", "design", "guard", "status"),
    "prediction_intervention": ("intervention_id", "action", "arm", "experiment_id"),
    "prediction_effect_estimate": ("estimate_id", "claim_kind", "effect", "details"),
    "prediction_change_event": ("event_id", "scope", "effect", "started_at", "ended_at", "retracted_at"),
    "prediction_shadow_forecast": ("shadow_id", "model", "origin_ts", "horizon_min", "point", "abs_error"),
    "prediction_weather_hourly": ("hour_ts", "temp_f", "is_forecast", "source", "fetched_at"),
    "prediction_exceedance": ("exceedance_id", "probability", "window_start", "window_end", "hit_raw", "hit_mean"),
    "prediction_action_bet": ("bet_id", "skill", "context", "place", "p", "p_baseline", "counts", "result"),
}
ACTION_BET_COLS = ("bet_id", "domain", "body", "skill", "context", "place", "action", "model", "p", "p_baseline",
                   "level", "counts", "chosen_rank", "session_id", "placed_at", "settled_at", "result", "outcome",
                   "detail")
_ACTION_BET_JSONB = {"context", "action", "counts", "detail"}
_SETTLED = "result IN ('success', 'failure')"


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
        elif src.kind == "cluster":
            sql = ("SELECT observed_at AS ts, (measurements_by_node->%s->>%s)::float AS v FROM orion_biometrics_cluster "
                   "WHERE observed_at >= %s AND (measurements_by_node->%s->>%s) IS NOT NULL")
            args = [src.node, src.key, since, src.node, src.key]
            if until is not None:
                sql += " AND observed_at < %s"
                args.append(until)
        elif src.kind == "weather":
            # Observed hours only: a stored forecast must never be read as if it had happened (an outage
            # would otherwise feed 48 h of forecasts in as history, invisibly).
            sql = "SELECT hour_ts AS ts, temp_f AS v FROM prediction_weather_hourly WHERE NOT is_forecast AND hour_ts >= %s"
            args = [since]
            if until is not None:
                sql += " AND hour_ts < %s"
                args.append(until)
        else:
            raise ValueError(f"unknown source kind {src.kind}")
        return [(r["ts"], r["v"]) for r in self._all(sql + " ORDER BY 1", args)]

    def first_reading(self, spec: SeriesSpec) -> datetime | None:
        """Earliest non-null reading of a cluster covariate (zero_before_first). Cached once found -- it
        never changes, and the scan costs ~0.7 s on the live fleet table; a None is re-checked hourly."""
        cache = self.__dict__.setdefault("_first_cache", {})
        hit = cache.get(spec.series_id)
        now = datetime.now(timezone.utc)
        if hit is not None and (hit[0] is not None or now - hit[1] < timedelta(hours=1)):
            return hit[0]
        src = spec.source
        if src.kind != "cluster":
            raise ValueError("zero_before_first is supported for cluster sources only")
        rows = self._all("SELECT min(observed_at) AS t FROM orion_biometrics_cluster "
                         "WHERE (measurements_by_node->%s->>%s) IS NOT NULL", (src.node, src.key))
        first = rows[0]["t"] if rows else None
        cache[spec.series_id] = (first, now)
        return first

    # --- weather ----------------------------------------------------------------------------
    def upsert_weather(self, rows: list[tuple[datetime, float | None, bool, str]], fetched_at: datetime) -> int:
        """(hour, temp_f, is_forecast, source). An observed value replaces a forecast; a forecast never
        replaces an observed value."""
        sql = ("INSERT INTO prediction_weather_hourly (hour_ts, temp_f, is_forecast, source, fetched_at) "
               "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (hour_ts) DO UPDATE SET temp_f = EXCLUDED.temp_f, "
               "is_forecast = EXCLUDED.is_forecast, source = EXCLUDED.source, fetched_at = EXCLUDED.fetched_at "
               "WHERE prediction_weather_hourly.is_forecast OR NOT EXCLUDED.is_forecast")
        with self.pool.connection() as conn, conn.transaction():
            for t, v, fc, src in rows:
                conn.execute(sql, (t, v, fc, src, fetched_at))
        return len(rows)

    def weather_span(self) -> tuple[datetime | None, datetime | None]:
        rows = self._all("SELECT min(hour_ts) AS lo, max(hour_ts) FILTER (WHERE NOT is_forecast) AS hi "
                         "FROM prediction_weather_hourly")
        return (rows[0]["lo"], rows[0]["hi"]) if rows else (None, None)

    def missing_weather_hours(self, start: datetime, end: datetime) -> list[datetime]:
        """Hours in [start, end) with no observed row (head of history and interior holes)."""
        rows = self._all("SELECT h AS t FROM generate_series(date_trunc('hour', %s::timestamptz), %s::timestamptz "
                         "- interval '1 hour', interval '1 hour') h WHERE NOT EXISTS (SELECT 1 FROM "
                         "prediction_weather_hourly w WHERE w.hour_ts = h AND NOT w.is_forecast) ORDER BY 1",
                         (start, end))
        return [r["t"] for r in rows]

    def event_times(self, source: str, event_type: str, stream_id: str | None, since: datetime, until: datetime,
                    limit: int) -> list[datetime]:
        """Timestamps of one event selector in [since, until), oldest first. Read-only transaction with a
        statement timeout: this reads a table other services write. Returns up to ``limit + 1`` rows so the
        caller can tell when it hit the cap."""
        if source not in EVENT_SOURCES:
            raise ValueError(f"unknown event source {source!r}")
        table, ts, kind, stream = EVENT_SOURCES[source]
        sql = f"SELECT {ts} AS ts FROM {table} WHERE {kind} = %s AND {ts} >= %s AND {ts} < %s"
        args: list = [event_type, since, until]
        if stream_id is not None:
            sql += f" AND {stream} = %s"
            args.append(stream_id)
        sql += f" ORDER BY {ts} LIMIT %s"
        args.append(limit + 1)
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute(f"SET LOCAL statement_timeout = {int(EVENT_READ_TIMEOUT_MS)}")
            return [r["ts"] for r in conn.execute(sql, args).fetchall()]

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
            "SELECT role, origin_ts, horizon_min, point, abs_error, in_80, in_95 FROM prediction_forecast "
            "WHERE series_id = %s AND score_status = 'scored' AND target_ts >= %s ORDER BY horizon_min, origin_ts",
            (series_id, since))

    # --- shadow forecasts ---------------------------------------------------------------------
    def insert_shadow(self, rows: list[dict]) -> None:
        """Shadow rows in their OWN transaction, after the served issue committed: a shadow failure
        (or a missing table) never costs the served forecast."""
        with self.pool.connection() as conn, conn.transaction():
            for r in rows:
                conn.execute("INSERT INTO prediction_shadow_forecast (series_id, model, issued_at, origin_ts, target_ts, "
                             "horizon_min, point) VALUES (%s,%s,%s,%s,%s,%s,%s) "
                             "ON CONFLICT (series_id, model, origin_ts, horizon_min) DO NOTHING",
                             (r["series_id"], r["model"], r["issued_at"], r["origin_ts"], r["target_ts"],
                              r["horizon_min"], r["point"]))

    def prune_shadow(self, series_id: str, before: datetime) -> int:
        """Retention for this service's own shadow table (verdicts look back at most ~25 days)."""
        rows = self._all("DELETE FROM prediction_shadow_forecast WHERE series_id = %s AND target_ts < %s "
                         "RETURNING 1", (series_id, before))
        return len(rows)

    def due_shadow(self, series_id: str, before: datetime, limit: int = 5000) -> list[dict]:
        return self._all("SELECT shadow_id, target_ts, point FROM prediction_shadow_forecast "
                         "WHERE series_id = %s AND scored_at IS NULL AND target_ts < %s ORDER BY target_ts LIMIT %s",
                         (series_id, before, limit))

    def write_shadow_scores(self, rows: list[dict]) -> None:
        with self.pool.connection() as conn, conn.transaction():
            for r in rows:
                conn.execute("UPDATE prediction_shadow_forecast SET scored_at = %s, actual = %s, abs_error = %s "
                             "WHERE shadow_id = %s AND scored_at IS NULL",
                             (r["scored_at"], r["actual"], r["abs_error"], r["shadow_id"]))

    def shadow_pairs(self, series_id: str, since: datetime) -> list[dict]:
        """Scored shadow forecasts with the baseline forecast for the same origin + horizon."""
        return self._all(
            "SELECT s.model, s.horizon_min, s.origin_ts, s.abs_error AS shadow_err, b.abs_error AS base_err "
            "FROM prediction_shadow_forecast s JOIN prediction_forecast b ON b.series_id = s.series_id "
            "AND b.role = 'baseline' AND b.origin_ts = s.origin_ts AND b.horizon_min = s.horizon_min "
            "WHERE s.series_id = %s AND s.scored_at IS NOT NULL AND s.abs_error IS NOT NULL "
            "AND b.score_status = 'scored' AND s.target_ts >= %s "
            "ORDER BY s.model, s.horizon_min, s.origin_ts", (series_id, since))

    # --- exceedance probabilities + outcomes ----------------------------------------------
    def insert_issue(self, forecast_rows: list[dict], exceedance_rows: list[dict]) -> None:
        """One issue's forecast rows and threshold probabilities in ONE transaction: a failure leaves
        neither, so the next tick re-issues the bucket instead of skipping it half-written."""
        f_sql = ("INSERT INTO prediction_forecast (run_id, series_id, model, role, issued_at, origin_ts, target_ts, "
                 "horizon_min, point, quantiles) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb) "
                 "ON CONFLICT (series_id, role, origin_ts, horizon_min) DO NOTHING")
        e_sql = ("INSERT INTO prediction_exceedance (series_id, origin_ts, issued_at, name, threshold, window_start, "
                 "window_end, probability, n_paths, served_baseline) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                 "ON CONFLICT (series_id, origin_ts, name) DO NOTHING")
        with self.pool.connection() as conn, conn.transaction():
            for r in forecast_rows:
                conn.execute(f_sql, (r["run_id"], r["series_id"], r["model"], r["role"], r["issued_at"],
                                     r["origin_ts"], r["target_ts"], r["horizon_min"], r["point"], _j(r["quantiles"])))
            for r in exceedance_rows:
                conn.execute(e_sql, (r["series_id"], r["origin_ts"], r["issued_at"], r["name"], r["threshold"],
                                     r["window_start"], r["window_end"], r["probability"], r["n_paths"],
                                     r["served_baseline"]))

    def due_exceedances(self, series_id: str, before: datetime, limit: int = 2000) -> list[dict]:
        return self._all("SELECT exceedance_id, threshold, window_start, window_end FROM prediction_exceedance "
                         "WHERE series_id = %s AND scored_at IS NULL AND window_end <= %s ORDER BY window_start LIMIT %s",
                         (series_id, before, limit))

    def write_exceedance_outcomes(self, rows: list[dict]) -> None:
        sql = ("UPDATE prediction_exceedance SET scored_at = %s, hit_raw = %s, hit_mean = %s, max_raw = %s, "
               "n_readings = %s, n_buckets_observed = %s WHERE exceedance_id = %s AND scored_at IS NULL")
        with self.pool.connection() as conn, conn.transaction():
            for r in rows:
                conn.execute(sql, (r["scored_at"], r["hit_raw"], r["hit_mean"], r["max_raw"], r["n_readings"],
                                   r["n_buckets_observed"], r["exceedance_id"]))

    def graded_exceedances(self, series_id: str, since: datetime) -> list[dict]:
        return self._all("SELECT name, threshold, window_start, window_end, probability, hit_raw, hit_mean, "
                         "served_baseline FROM prediction_exceedance "
                         "WHERE series_id = %s AND scored_at IS NOT NULL AND n_readings > 0 "
                         "AND probability IS NOT NULL AND window_start >= %s", (series_id, since))

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
                         "FROM prediction_change_event WHERE retracted_at IS NULL ORDER BY started_at")

    def retract_change_event(self, event_id: str) -> bool:
        rows = self._all("UPDATE prediction_change_event SET retracted_at = now() WHERE event_id = %s "
                         "AND retracted_at IS NULL RETURNING event_id", (event_id,))
        return bool(rows)

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

    # --- action bets (manual_migration_prediction_action_bet_v1.sql) ----------------------------
    @staticmethod
    def _bet(row: dict | None) -> dict | None:
        if row is None:
            return None
        # Only the columns this code knows: a column added by a later migration (applied before this
        # service rebuilds) must not reach ActionBetV1 (extra=forbid) and turn every call into a 500.
        row = {k: v for k, v in dict(row).items() if k in ACTION_BET_COLS}
        row["detail"] = row.get("detail") or {}
        row["context"] = row.get("context") or {}
        row["action"] = row.get("action") or {}
        return row

    def action_bet_counts(self, domain: str | None, skills: Sequence[str] | None, places: Sequence[str] | None,
                          body: str | None = None, combos: bool = False) -> list[dict]:
        """Settled success/failure counts in ONE query: per skill over every domain and body, per
        (domain, skill, context), per (domain, place). ``None`` for skills/places/domain = all.
        ``body`` narrows the situation and place levels only (the skill level pools every body).
        ``combos`` adds each distinct (domain, skill, context, place) that has a place (for /live)."""
        parts, args = [], []
        # Skill and situation levels weigh each place as one unit (1/n per bet), see action_bets.py.
        weighted = ("(SELECT *, CASE WHEN place IS NULL THEN 1.0 ELSE 1.0 / count(*) OVER (PARTITION BY domain, place) "
                    f"END AS w FROM prediction_action_bet WHERE {_SETTLED}) AS weighted")
        wsum = ("COALESCE(sum(w) FILTER (WHERE result = 'success'), 0) AS successes, "
                "COALESCE(sum(w) FILTER (WHERE result = 'failure'), 0) AS failures")
        where = ["TRUE"]
        if skills is not None:
            where.append("skill = ANY(%s)")
            args.append(list(skills))
        parts.append("SELECT 'skill' AS level, skill, NULL::text AS domain, NULL::jsonb AS context, NULL::text AS place, "
                     f"{wsum} FROM {weighted} WHERE {' AND '.join(where)} GROUP BY skill")
        scope, scope_args = [_SETTLED], []
        if domain is not None:
            scope.append("domain = %s")
            scope_args.append(domain)
        if body is not None:
            scope.append("body = %s")
            scope_args.append(body)
        sit = list(scope)
        sit_args = list(scope_args)
        if skills is not None:
            sit.append("skill = ANY(%s)")
            sit_args.append(list(skills))
        parts.append("SELECT 'situation', skill, domain, context, NULL::text, "
                     f"{wsum} FROM {weighted} WHERE {' AND '.join(sit)} GROUP BY skill, domain, context")
        args += sit_args
        if places is None or places:
            pl = list(scope) + ["place IS NOT NULL"]
            pl_args = list(scope_args)
            if places is not None:
                pl.append("place = ANY(%s)")
                pl_args.append(list(places))
            parts.append("SELECT 'place', NULL::text, domain, NULL::jsonb, place, "
                         "(count(*) FILTER (WHERE result = 'success'))::numeric, "
                         "(count(*) FILTER (WHERE result = 'failure'))::numeric "
                         f"FROM prediction_action_bet WHERE {' AND '.join(pl)} GROUP BY domain, place")
            args += pl_args
            if combos:
                parts.append("SELECT 'combo', skill, domain, context, place, 0::numeric, 0::numeric "
                             f"FROM prediction_action_bet WHERE {' AND '.join(pl)} GROUP BY skill, domain, context, place")
                args += pl_args
        return self._all(" UNION ALL ".join(parts), args)

    def get_action_bet(self, bet_id: str) -> dict | None:
        rows = self._all("SELECT * FROM prediction_action_bet WHERE bet_id = %s", (bet_id,))
        return self._bet(rows[0]) if rows else None

    def insert_action_bet(self, row: dict) -> dict | None:
        """Insert once by bet_id; None when a row with that id already exists (the caller reads it)."""
        cols = [c for c in ACTION_BET_COLS if c in row]
        sql = (f"INSERT INTO prediction_action_bet ({', '.join(cols)}) VALUES "
               f"({', '.join('%s::jsonb' if c in _ACTION_BET_JSONB else '%s' for c in cols)}) "
               "ON CONFLICT (bet_id) DO NOTHING RETURNING *")
        rows = self._all(sql, [(_j(row[c]) if c in _ACTION_BET_JSONB else row[c]) for c in cols])
        return self._bet(rows[0]) if rows else None

    def settle_action_bet(self, bet_id: str, result: str, outcome: str, detail: dict, now: datetime) -> dict | None:
        """Open -> settled only; None when the bet is unknown or already settled."""
        rows = self._all("UPDATE prediction_action_bet SET result = %s, outcome = %s, detail = %s::jsonb, "
                         "settled_at = %s WHERE bet_id = %s AND result IS NULL RETURNING *",
                         (result, outcome, _j(detail), now, bet_id))
        return self._bet(rows[0]) if rows else None

    def expire_action_bets(self, placed_before: datetime, now: datetime, outcome: str) -> list[str]:
        rows = self._all("UPDATE prediction_action_bet SET result = 'void', outcome = %s, settled_at = %s "
                         "WHERE result IS NULL AND placed_at < %s RETURNING bet_id", (outcome, now, placed_before))
        return [r["bet_id"] for r in rows]

    def _bet_where(self, domain: str | None, body: str | None, session_id: str | None = None) -> tuple[list, list]:
        where, args = [], []
        for col, val in (("domain", domain), ("body", body), ("session_id", session_id)):
            if val is not None:
                where.append(f"{col} = %s")
                args.append(val)
        return where, args

    def settled_action_bets(self, domain: str | None = None, body: str | None = None, limit: int = 5000) -> list[dict]:
        """Most recent settled success/failure bets (the grading + verdict input)."""
        where, args = self._bet_where(domain, body)
        where.append(_SETTLED)
        return self._all("SELECT bet_id, session_id, placed_at, p, p_baseline, counts, result "
                         f"FROM prediction_action_bet WHERE {' AND '.join(where)} "
                         "ORDER BY settled_at DESC LIMIT %s", (*args, limit))

    def action_bets(self, domain: str | None = None, body: str | None = None, session_id: str | None = None,
                    limit: int = 20) -> list[dict]:
        """A session's bets oldest first; otherwise the newest first."""
        where, args = self._bet_where(domain, body, session_id)
        sql = "SELECT * FROM prediction_action_bet"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY placed_at" + ("" if session_id is not None else " DESC") + ", bet_id LIMIT %s"
        return [self._bet(r) for r in self._all(sql, (*args, limit))]

    def action_bet_status(self, domain: str | None, body: str | None, since: datetime,
                          never_settled: str = "never_settled") -> dict:
        where, args = self._bet_where(domain, body)
        sql = ("SELECT count(*) FILTER (WHERE result IS NULL) AS open, "
               "count(*) FILTER (WHERE result = 'void' AND outcome = %s AND settled_at >= %s) AS never_settled, "
               "count(*) FILTER (WHERE result = 'void' AND settled_at >= %s) AS void, "
               "count(*) FILTER (WHERE result IN ('success', 'failure') AND settled_at >= %s) AS settled "
               "FROM prediction_action_bet")
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self._all(sql, (never_settled, since, since, since, *args))
        r = rows[0] if rows else {}
        return {k: int(r.get(k) or 0) for k in ("open", "never_settled", "void", "settled")}
