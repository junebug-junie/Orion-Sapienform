from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_flag(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_postgres_url() -> str | None:
    control = str(os.getenv("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", "")).strip()
    policy = str(os.getenv("SUBSTRATE_POLICY_POSTGRES_URL", "")).strip()
    database = str(os.getenv("DATABASE_URL", "")).strip()
    return control or policy or database or None


def _resolve_sqlite_path() -> str | None:
    explicit = str(os.getenv("SUBSTRATE_MUTATION_CONTROL_SQL_DB_PATH", "")).strip()
    mutation = str(os.getenv("SUBSTRATE_MUTATION_SQL_DB_PATH", "")).strip()
    return explicit or mutation or None


@dataclass
class RuntimeControlSurfaceStore:
    postgres_url: str | None = None
    sql_db_path: str | None = None
    _last_error: str | None = None
    _source_kind: str = "memory"
    _memory: dict[str, dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._memory = {}
        self.postgres_url = self.postgres_url or _resolve_postgres_url()
        self.sql_db_path = self.sql_db_path or _resolve_sqlite_path()
        if self.postgres_url:
            try:
                self._ensure_postgres_schema()
                self._source_kind = "postgres"
                return
            except Exception as exc:
                self._last_error = str(exc)
        if self.sql_db_path:
            try:
                self._ensure_sqlite_schema()
                self._source_kind = "sqlite"
                return
            except Exception as exc:
                self._last_error = str(exc)
        self._source_kind = "memory"

    def source_kind(self) -> str:
        return self._source_kind

    def last_error(self) -> str | None:
        return self._last_error

    def degraded(self) -> bool:
        return self._last_error is not None

    def get(self, key: str) -> dict[str, Any] | None:
        if self._source_kind == "postgres" and self.postgres_url:
            try:
                from sqlalchemy import create_engine, text

                engine = create_engine(self.postgres_url)
                with engine.begin() as conn:
                    row = conn.execute(
                        text("SELECT value_json::text FROM substrate_runtime_control_surface WHERE surface_key=:surface_key"),
                        {"surface_key": key},
                    ).fetchone()
                if not row:
                    return None
                return json.loads(row[0])
            except Exception as exc:
                self._last_error = str(exc)
                return None
        if self._source_kind == "sqlite" and self.sql_db_path:
            try:
                with sqlite3.connect(self.sql_db_path) as conn:
                    row = conn.execute(
                        "SELECT value_json FROM substrate_runtime_control_surface WHERE surface_key=?",
                        (key,),
                    ).fetchone()
                if not row:
                    return None
                return json.loads(row[0])
            except Exception as exc:
                self._last_error = str(exc)
                return None
        return self._memory.get(key)

    def upsert(self, *, key: str, value: dict[str, Any]) -> None:
        value_payload = dict(value)
        value_payload.setdefault("updated_at", _utc_now().isoformat())
        if self._source_kind == "postgres" and self.postgres_url:
            try:
                from sqlalchemy import create_engine, text

                engine = create_engine(self.postgres_url)
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            """
                            INSERT INTO substrate_runtime_control_surface(surface_key, updated_at, value_json)
                            VALUES (:surface_key, :updated_at, CAST(:value_json AS JSONB))
                            ON CONFLICT (surface_key) DO UPDATE SET
                                updated_at=EXCLUDED.updated_at,
                                value_json=EXCLUDED.value_json
                            """
                        ),
                        {
                            "surface_key": key,
                            "updated_at": _utc_now(),
                            "value_json": json.dumps(value_payload, ensure_ascii=False, sort_keys=True),
                        },
                    )
                return
            except Exception as exc:
                self._last_error = str(exc)
        if self._source_kind == "sqlite" and self.sql_db_path:
            try:
                with sqlite3.connect(self.sql_db_path) as conn:
                    conn.execute(
                        """
                        INSERT INTO substrate_runtime_control_surface(surface_key, updated_at, value_json)
                        VALUES (?, ?, ?)
                        ON CONFLICT(surface_key) DO UPDATE SET
                            updated_at=excluded.updated_at,
                            value_json=excluded.value_json
                        """,
                        (key, _utc_now().isoformat(), json.dumps(value_payload, ensure_ascii=False, sort_keys=True)),
                    )
                    conn.commit()
                return
            except Exception as exc:
                self._last_error = str(exc)
        self._memory[key] = value_payload

    def _ensure_sqlite_schema(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS substrate_runtime_control_surface (surface_key TEXT PRIMARY KEY, updated_at TEXT NOT NULL, value_json TEXT NOT NULL)"
            )
            conn.commit()

    def _ensure_postgres_schema(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import create_engine, text

        engine = create_engine(self.postgres_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS substrate_runtime_control_surface (surface_key TEXT PRIMARY KEY, updated_at TIMESTAMPTZ NOT NULL, value_json JSONB NOT NULL)"
                )
            )


_CONTROL_SURFACE_STORE: RuntimeControlSurfaceStore | None = None
_ROUTING_THRESHOLD_KEY = "routing.chat_reflective_lane_threshold"


def control_surface_store() -> RuntimeControlSurfaceStore:
    global _CONTROL_SURFACE_STORE
    if _CONTROL_SURFACE_STORE is None:
        _CONTROL_SURFACE_STORE = RuntimeControlSurfaceStore()
    return _CONTROL_SURFACE_STORE


def get_chat_reflective_lane_threshold(default: float = 0.75) -> float:
    payload = control_surface_store().get(_ROUTING_THRESHOLD_KEY) or {}
    value = payload.get("value")
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        env_value = os.getenv("CHAT_REFLECTIVE_LANE_THRESHOLD")
        if env_value is not None:
            try:
                return max(0.0, min(1.0, float(env_value)))
            except Exception:
                return default
        return default


def set_chat_reflective_lane_threshold(
    *,
    value: float,
    actor: str,
    proposal_id: str | None = None,
    decision_id: str | None = None,
) -> None:
    normalized = max(0.0, min(1.0, float(value)))
    control_surface_store().upsert(
        key=_ROUTING_THRESHOLD_KEY,
        value={
            "surface": _ROUTING_THRESHOLD_KEY,
            "value": normalized,
            "actor": actor,
            "proposal_id": proposal_id,
            "decision_id": decision_id,
            "updated_at": _utc_now().isoformat(),
        },
    )


def inspect_chat_reflective_lane_threshold(default: float = 0.75) -> dict[str, Any]:
    payload = control_surface_store().get(_ROUTING_THRESHOLD_KEY) or {}
    return {
        "surface": _ROUTING_THRESHOLD_KEY,
        "value": get_chat_reflective_lane_threshold(default=default),
        "raw": payload,
        "source_kind": control_surface_store().source_kind(),
        "degraded": control_surface_store().degraded(),
        "error": control_surface_store().last_error(),
    }
