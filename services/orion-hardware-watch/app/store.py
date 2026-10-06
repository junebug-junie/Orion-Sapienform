"""Reads telemetry, owns hardware_watch_incident. Sync psycopg (called via asyncio.to_thread);
one small pool, at most 2 connections.

orion_biometrics_summary.timestamp is VARCHAR ('YYYY-MM-DD HH:MM:SS.ffffff+00', uniform since at
least 2026-09-22), indexed with node: filtered as a string on that prefix, parsed as timestamptz.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from orion.hardware_watch.rules import CoolingPoint, TempPoint

INCIDENT_COLUMNS = (
    "incident_id", "rule", "subject", "status", "open_reason", "opened_at", "resolved_at", "resolve_reason",
    "resolved_by", "snooze_until", "shed_requested", "shed_reason", "shed_requested_at", "alert_sent_at",
    "alert_attempts", "alert_error", "urgent_requested_at", "urgent_error", "evidence", "updated_at",
)
_KEY = re.compile(r"^[a-z0-9_]{1,40}$")


def _since(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


class PostgresStore:
    def __init__(self, conninfo: str, cooling_role: str = "cabinet_cooling"):
        self.cooling_role = cooling_role
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(conninfo=conninfo, min_size=1, max_size=2, open=True,
                                   kwargs={"row_factory": dict_row, "autocommit": True,
                                           "application_name": "orion-hardware-watch"})

    def close(self) -> None:
        self.pool.close()

    def _all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self.pool.connection() as conn:
            return list(conn.execute(sql, args).fetchall())

    def check_schema(self) -> None:
        """The migration is operator-applied (services/orion-sql-db/manual_migration_hardware_watch_v1.sql).
        Refuse to start without it rather than alert with no memory of what was already sent."""
        self._all(f"SELECT {', '.join(INCIDENT_COLUMNS)} FROM hardware_watch_incident LIMIT 0")

    # --- telemetry -------------------------------------------------------------------------
    def cooling_points(self, since: datetime) -> list[CoolingPoint]:
        rows = self._all(
            "SELECT ts, cooling_watts, stale, device_online, controller_ready FROM home_cooling_sample "
            "WHERE role = %s AND ts >= %s ORDER BY ts", (self.cooling_role, since))
        return [CoolingPoint(r["ts"], r["cooling_watts"], r["stale"], bool(r["device_online"]),
                             bool(r["controller_ready"])) for r in rows]

    def temp_points(self, node: str, key: str, since: datetime) -> list[TempPoint]:
        if not _KEY.match(key):
            raise ValueError(f"bad measurement key {key!r}")
        rows = self._all(
            "SELECT timestamp::timestamptz AS ts, (measurements->>%s)::float AS v FROM orion_biometrics_summary "
            "WHERE node = %s AND timestamp >= %s AND measurements ? %s ORDER BY timestamp",
            (key, node, _since(since), key))
        return [TempPoint(r["ts"], r["v"]) for r in rows if r["v"] is not None and r["ts"] >= since]

    def gpu_keys(self, node: str, since: datetime) -> list[str]:
        """Per-GPU temperature keys (gpu{N}_temp_c) this node reported since ``since``."""
        rows = self._all(
            "SELECT DISTINCT k FROM orion_biometrics_summary, jsonb_object_keys(measurements) k "
            "WHERE node = %s AND timestamp >= %s AND k ~ '^gpu[0-9]+_temp_c$'", (node, _since(since)))
        return sorted(r["k"] for r in rows)

    # --- incidents -------------------------------------------------------------------------
    def open_incidents(self) -> list[dict]:
        return self._all(f"SELECT {', '.join(INCIDENT_COLUMNS)} FROM hardware_watch_incident "
                         "WHERE status = 'open' ORDER BY opened_at")

    def snoozed_until(self, rule: str, subject: str, open_reasons: tuple[str, ...] | None = None) -> datetime | None:
        """D9: an operator resolve snoozes only the reason(s) it resolved (the resolved row's own
        open_reason), so resolving low_power never silences device_offline (C11). No new column."""
        if open_reasons is None:
            rows = self._all("SELECT max(snooze_until) AS s FROM hardware_watch_incident "
                             "WHERE rule = %s AND subject = %s AND status = 'resolved'", (rule, subject))
        else:
            rows = self._all("SELECT max(snooze_until) AS s FROM hardware_watch_incident WHERE rule = %s "
                             "AND subject = %s AND open_reason = ANY(%s) AND status = 'resolved'",
                             (rule, subject, list(open_reasons)))
        return rows[0]["s"] if rows else None

    def last_side_effect_at(self, column: str, rule: str, subject: str) -> datetime | None:
        """D6: newest ``alert_sent_at`` / ``urgent_requested_at`` for rule+subject across ALL incidents
        (the sliding dedupe window is a query, not a column). A drill (open_reason simulated) never
        suppresses a real alert."""
        if column not in ("alert_sent_at", "urgent_requested_at"):
            raise ValueError(f"bad column {column!r}")
        rows = self._all(f"SELECT max({column}) AS s FROM hardware_watch_incident WHERE rule = %s AND subject = %s "
                         "AND open_reason <> 'simulated'", (rule, subject))
        return rows[0]["s"] if rows else None

    def insert_incident(self, row: dict) -> bool:
        """False when (rule, subject) already has an open incident (the unique partial index)."""
        cols = [c for c in INCIDENT_COLUMNS if c in row]
        vals = [json.dumps(row[c]) if c == "evidence" else row[c] for c in cols]
        rows = self._all(
            f"INSERT INTO hardware_watch_incident ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
            "ON CONFLICT DO NOTHING RETURNING incident_id", tuple(vals))
        return bool(rows)

    def update_incident(self, incident_id: str, **fields: Any) -> None:
        fields["updated_at"] = datetime.now(timezone.utc)
        sets = ", ".join(f"{k} = %s" for k in fields)
        vals = [json.dumps(v) if k == "evidence" else v for k, v in fields.items()]
        self._all(f"UPDATE hardware_watch_incident SET {sets} WHERE incident_id = %s RETURNING incident_id",
                  (*vals, incident_id))

    def resolve_incident(self, incident_id: str, **fields: Any) -> bool:
        """Close an OPEN incident; False when it was already resolved (a racing resolve)."""
        fields["updated_at"] = datetime.now(timezone.utc)
        sets = ", ".join(f"{k} = %s" for k in fields)
        rows = self._all(f"UPDATE hardware_watch_incident SET {sets} WHERE incident_id = %s AND status = 'open' "
                         "RETURNING incident_id", (*fields.values(), incident_id))
        return bool(rows)

    def get_incident(self, incident_id: str) -> dict | None:
        rows = self._all(f"SELECT {', '.join(INCIDENT_COLUMNS)} FROM hardware_watch_incident "
                         "WHERE incident_id = %s", (incident_id,))
        return rows[0] if rows else None

    def list_incidents(self, limit: int = 50) -> list[dict]:
        return self._all(f"SELECT {', '.join(INCIDENT_COLUMNS)} FROM hardware_watch_incident "
                         "ORDER BY opened_at DESC LIMIT %s", (limit,))


class MemoryStore:
    """Same contract, in memory: tests and the replay eval."""

    def __init__(self):
        self.cooling: list[CoolingPoint] = []
        self.temps: dict[tuple[str, str], list[TempPoint]] = {}
        self.incidents: dict[str, dict] = {}

    def close(self) -> None:
        pass

    def check_schema(self) -> None:
        pass

    def cooling_points(self, since: datetime) -> list[CoolingPoint]:
        return [p for p in self.cooling if p.ts >= since]

    def temp_points(self, node: str, key: str, since: datetime) -> list[TempPoint]:
        return [p for p in self.temps.get((node, key), []) if p.ts >= since]

    def gpu_keys(self, node: str, since: datetime) -> list[str]:
        return sorted(k for (n, k), pts in self.temps.items()
                      if n == node and re.match(r"^gpu[0-9]+_temp_c$", k) and any(p.ts >= since for p in pts))

    def open_incidents(self) -> list[dict]:
        return sorted((dict(r) for r in self.incidents.values() if r["status"] == "open"),
                      key=lambda r: r["opened_at"])

    def snoozed_until(self, rule: str, subject: str, open_reasons: tuple[str, ...] | None = None) -> datetime | None:
        vals = [r.get("snooze_until") for r in self.incidents.values()
                if r["rule"] == rule and r["subject"] == subject and r["status"] == "resolved" and r.get("snooze_until")
                and (open_reasons is None or r["open_reason"] in open_reasons)]
        return max(vals) if vals else None

    def last_side_effect_at(self, column: str, rule: str, subject: str) -> datetime | None:
        if column not in ("alert_sent_at", "urgent_requested_at"):
            raise ValueError(f"bad column {column!r}")
        vals = [r.get(column) for r in self.incidents.values()
                if r["rule"] == rule and r["subject"] == subject and r.get(column) and r["open_reason"] != "simulated"]
        return max(vals) if vals else None

    def insert_incident(self, row: dict) -> bool:
        if any(r["status"] == "open" and r["rule"] == row["rule"] and r["subject"] == row["subject"]
               for r in self.incidents.values()):
            return False
        full = {c: None for c in INCIDENT_COLUMNS}
        full.update(alert_attempts=0, shed_requested=False, evidence={})
        full.update(row)
        self.incidents[row["incident_id"]] = full
        return True

    def update_incident(self, incident_id: str, **fields: Any) -> None:
        if incident_id in self.incidents:
            self.incidents[incident_id].update(fields)

    def resolve_incident(self, incident_id: str, **fields: Any) -> bool:
        r = self.incidents.get(incident_id)
        if r is None or r["status"] != "open":
            return False
        r.update(fields)
        return True

    def get_incident(self, incident_id: str) -> dict | None:
        r = self.incidents.get(incident_id)
        return dict(r) if r else None

    def list_incidents(self, limit: int = 50) -> list[dict]:
        return sorted((dict(r) for r in self.incidents.values()), key=lambda r: r["opened_at"], reverse=True)[:limit]
