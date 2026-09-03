from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_flag(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _connect_timeout_sec() -> int:
    """Bounded so a database blip degrades the routing path in seconds, not
    minutes. Clamped rather than trusted: this value sits on a hot path."""
    raw = str(os.getenv("SUBSTRATE_CONTROL_SURFACE_CONNECT_TIMEOUT_SEC", "5")).strip()
    try:
        value = int(float(raw))
    except ValueError:
        return 5
    return max(1, min(30, value))


def _resolve_postgres_url() -> str | None:
    control = str(os.getenv("SUBSTRATE_CONTROL_PLANE_POSTGRES_URL", "")).strip()
    policy = str(os.getenv("SUBSTRATE_POLICY_POSTGRES_URL", "")).strip()
    database = str(os.getenv("DATABASE_URL", "")).strip()
    return control or policy or database or None


def _history_max_rows_per_surface() -> int:
    """Bound the history so an append-only table cannot grow without limit.

    Per surface, not overall, so a busy surface can never crowd out the record
    of a quiet one. Generous by default: at roughly one adoption per
    rollback_window_sec this is months of history for the routing surface.
    """
    raw = str(os.getenv("SUBSTRATE_CONTROL_SURFACE_HISTORY_MAX_ROWS", "1000")).strip()
    try:
        value = int(float(raw))
    except ValueError:
        return 1000
    return max(10, min(100000, value))


def _resolve_sqlite_path() -> str | None:
    explicit = str(os.getenv("SUBSTRATE_MUTATION_CONTROL_SQL_DB_PATH", "")).strip()
    mutation = str(os.getenv("SUBSTRATE_MUTATION_SQL_DB_PATH", "")).strip()
    return explicit or mutation or None


class ControlSurfaceWriteError(RuntimeError):
    """A configured durable backend refused a control-surface write.

    Raised rather than swallowed because the fallthrough below is unreachable as
    a recovery path: ``_source_kind`` stays "postgres"/"sqlite" after a failure,
    so ``get()`` never reads the in-memory copy again. Silently keeping going
    meant a caller was told its write succeeded when the value had not moved --
    and ``PatchApplier.apply`` would then mint an adoption recording a change
    that never happened, which is the exact falsehood this history table exists
    to prevent, inverted.
    """


def _decode_json(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return None


def _history_row(surface_key: str, row: Any) -> dict[str, Any]:
    changed_at = row[1]
    return {
        "history_id": row[0],
        "surface_key": surface_key,
        "changed_at": changed_at.isoformat() if hasattr(changed_at, "isoformat") else changed_at,
        "actor": row[2],
        "previous_value": _decode_json(row[3]),
        "new_value": _decode_json(row[4]),
    }


@dataclass
class RuntimeControlSurfaceStore:
    postgres_url: str | None = None
    sql_db_path: str | None = None
    _last_error: str | None = None
    _source_kind: str = "memory"
    _engine_cache: Any = None
    _memory: dict[str, dict[str, Any]] = None  # type: ignore[assignment]
    _memory_history: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._memory = {}
        self._memory_history = []
        # An explicitly-passed sql_db_path is a deliberate isolation request and
        # must win over an ambient env Postgres URL. It did not, and the result is
        # visible in production: `substrate_runtime_control_surface` holds a row
        # written by actor "scheduler_seed" -- a string that exists nowhere but a
        # pytest fixture -- with 4,925 updates on it. Test runs that passed
        # sql_db_path for isolation still resolved DATABASE_URL from the ambient
        # environment and wrote Orion's live routing threshold instead.
        # `and not self.postgres_url` would be dead here: the `or` below already
        # short-circuits when postgres_url was passed explicitly.
        if not self.sql_db_path:
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

    def _engine(self):
        """One pooled engine for the life of the store, with a bounded connect.

        `decision_router.py` calls get_chat_reflective_lane_threshold() on EVERY
        routing decision, so a per-call create_engine() would build a fresh pool
        and a fresh TCP connect + auth handshake on the chat hot path. Worse,
        SQLAlchemy has no default connect timeout: against a host that is
        unreachable-but-not-refusing (restarting, blackholed, partitioned) the
        routing path would block on the OS TCP timeout, which can be minutes.
        """
        if self._engine_cache is None:
            from sqlalchemy import create_engine

            self._engine_cache = create_engine(
                self.postgres_url,
                pool_pre_ping=True,
                connect_args={"connect_timeout": _connect_timeout_sec()},
            )
        return self._engine_cache

    def source_kind(self) -> str:
        return self._source_kind

    def last_error(self) -> str | None:
        return self._last_error

    def degraded(self) -> bool:
        return self._last_error is not None

    def get(self, key: str) -> dict[str, Any] | None:
        if self._source_kind == "postgres" and self.postgres_url:
            try:
                from sqlalchemy import text

                with self._engine().begin() as conn:
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
        """Write a control value, recording what it replaced.

        The current-value table keeps one row per surface, so before this every
        write destroyed the only evidence of what the setting had been. That is
        not a theoretical loss: Orion's first self-modification on 2026-09-02
        moved the routing threshold, and afterwards nothing in the system could
        say what it had moved *from* -- the answer had to be inferred from a
        pytest fixture that had been leaking writes onto the live row.

        The history row is written in the **same transaction** as the value, so
        a surface cannot change without leaving a record. A history failure
        therefore fails the whole write and falls through to the next backend,
        which is deliberate: an unrecorded write is the thing this exists to
        prevent, and silently keeping it would defeat the point.
        """
        value_payload = dict(value)
        value_payload.setdefault("updated_at", _utc_now().isoformat())
        history_row = {
            "history_id": f"control-surface-history-{uuid.uuid4()}",
            "surface_key": key,
            "actor": value_payload.get("actor"),
            "new_value_json": json.dumps(value_payload, ensure_ascii=False, sort_keys=True),
        }
        if self._source_kind == "postgres" and self.postgres_url:
            try:
                from sqlalchemy import text

                with self._engine().begin() as conn:
                    # Read the outgoing value inside the write transaction, and
                    # lock the row. Reading it outside (via self.get(), which
                    # opens its own connection) let a concurrent writer land
                    # between the read and the write, so the history could record
                    # a previous_value that was never live at that moment -- a
                    # lie shaped exactly like a fact, and worse than the missing
                    # row this table replaced.
                    prior = conn.execute(
                        text(
                            "SELECT value_json::text FROM substrate_runtime_control_surface "
                            "WHERE surface_key=:surface_key FOR UPDATE"
                        ),
                        {"surface_key": key},
                    ).fetchone()
                    history_row["previous_value_json"] = prior[0] if prior else None
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
                            "value_json": history_row["new_value_json"],
                        },
                    )
                    conn.execute(
                        text(
                            """
                            INSERT INTO substrate_runtime_control_surface_history(
                                history_id, surface_key, changed_at, actor,
                                previous_value_json, new_value_json
                            )
                            VALUES (
                                :history_id, :surface_key, :changed_at, :actor,
                                CAST(:previous_value_json AS JSONB), CAST(:new_value_json AS JSONB)
                            )
                            """
                        ),
                        {**history_row, "changed_at": _utc_now()},
                    )
                    conn.execute(
                        text(
                            """
                            DELETE FROM substrate_runtime_control_surface_history
                            WHERE surface_key = :surface_key
                              AND history_id NOT IN (
                                  SELECT history_id FROM substrate_runtime_control_surface_history
                                  WHERE surface_key = :surface_key
                                  ORDER BY changed_at DESC
                                  LIMIT :keep
                              )
                            """
                        ),
                        {"surface_key": key, "keep": _history_max_rows_per_surface()},
                    )
                return
            except Exception as exc:
                self._last_error = str(exc)
                raise ControlSurfaceWriteError(
                    f"control surface write failed for {key!r}: {exc}"
                ) from exc
        if self._source_kind == "sqlite" and self.sql_db_path:
            conn = sqlite3.connect(self.sql_db_path)
            try:
                # BEGIN IMMEDIATE takes the write lock up front, so the read
                # below and the two writes are one transaction. sqlite3's
                # implicit BEGIN only fires before the INSERT, which would leave
                # the read outside -- the same non-atomic derivation the
                # postgres branch avoids with FOR UPDATE.
                conn.isolation_level = None
                conn.execute("BEGIN IMMEDIATE")
                prior = conn.execute(
                    "SELECT value_json FROM substrate_runtime_control_surface WHERE surface_key=?",
                    (key,),
                ).fetchone()
                history_row["previous_value_json"] = prior[0] if prior else None
                conn.execute(
                    """
                    INSERT INTO substrate_runtime_control_surface(surface_key, updated_at, value_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(surface_key) DO UPDATE SET
                        updated_at=excluded.updated_at,
                        value_json=excluded.value_json
                    """,
                    (key, _utc_now().isoformat(), history_row["new_value_json"]),
                )
                conn.execute(
                    """
                    INSERT INTO substrate_runtime_control_surface_history(
                        history_id, surface_key, changed_at, actor,
                        previous_value_json, new_value_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        history_row["history_id"],
                        key,
                        _utc_now().isoformat(),
                        history_row["actor"],
                        history_row["previous_value_json"],
                        history_row["new_value_json"],
                    ),
                )
                conn.execute(
                    """
                    DELETE FROM substrate_runtime_control_surface_history
                    WHERE surface_key = ?
                      AND history_id NOT IN (
                          SELECT history_id FROM substrate_runtime_control_surface_history
                          WHERE surface_key = ?
                          ORDER BY changed_at DESC
                          LIMIT ?
                      )
                    """,
                    (key, key, _history_max_rows_per_surface()),
                )
                conn.execute("COMMIT")
                return
            except Exception as exc:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                self._last_error = str(exc)
                raise ControlSurfaceWriteError(
                    f"control surface write failed for {key!r}: {exc}"
                ) from exc
            finally:
                conn.close()
        history_row["previous_value_json"] = (
            json.dumps(self._memory[key], ensure_ascii=False, sort_keys=True) if key in self._memory else None
        )
        self._memory_history.append(
            {
                "history_id": history_row["history_id"],
                "surface_key": key,
                "changed_at": _utc_now().isoformat(),
                "actor": history_row["actor"],
                "previous_value": _decode_json(history_row["previous_value_json"]),
                "new_value": value_payload,
            }
        )
        self._memory[key] = value_payload
        # Per surface, matching the SQL prune. A global cap would let a busy
        # surface evict a quiet surface's history, which is the one thing the
        # bound is not allowed to do.
        cap = _history_max_rows_per_surface()
        for_key = [i for i, e in enumerate(self._memory_history) if e.get("surface_key") == key]
        for index in reversed(for_key[:-cap] if len(for_key) > cap else []):
            del self._memory_history[index]

    def history(self, key: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent changes to one surface, newest first."""
        bounded = max(1, min(500, int(limit)))
        if self._source_kind == "postgres" and self.postgres_url:
            try:
                from sqlalchemy import text

                with self._engine().begin() as conn:
                    rows = conn.execute(
                        text(
                            """
                            SELECT history_id, changed_at, actor,
                                   previous_value_json::text, new_value_json::text
                            FROM substrate_runtime_control_surface_history
                            WHERE surface_key=:surface_key
                            ORDER BY changed_at DESC
                            LIMIT :limit
                            """
                        ),
                        {"surface_key": key, "limit": bounded},
                    ).fetchall()
                return [_history_row(key, row) for row in rows]
            except Exception as exc:
                self._last_error = str(exc)
                return []
        if self._source_kind == "sqlite" and self.sql_db_path:
            try:
                with sqlite3.connect(self.sql_db_path) as conn:
                    rows = conn.execute(
                        """
                        SELECT history_id, changed_at, actor, previous_value_json, new_value_json
                        FROM substrate_runtime_control_surface_history
                        WHERE surface_key=?
                        ORDER BY changed_at DESC
                        LIMIT ?
                        """,
                        (key, bounded),
                    ).fetchall()
                return [_history_row(key, row) for row in rows]
            except Exception as exc:
                self._last_error = str(exc)
                return []
        entries = [e for e in self._memory_history if e.get("surface_key") == key]
        return list(reversed(entries))[:bounded]

    def _ensure_sqlite_schema(self) -> None:
        if not self.sql_db_path:
            return
        with sqlite3.connect(self.sql_db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS substrate_runtime_control_surface (surface_key TEXT PRIMARY KEY, updated_at TEXT NOT NULL, value_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS substrate_runtime_control_surface_history ("
                "history_id TEXT PRIMARY KEY, surface_key TEXT NOT NULL, changed_at TEXT NOT NULL, "
                "actor TEXT, previous_value_json TEXT, new_value_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS substrate_runtime_control_surface_history_key_time "
                "ON substrate_runtime_control_surface_history (surface_key, changed_at DESC)"
            )
            conn.commit()

    def _ensure_postgres_schema(self) -> None:
        if not self.postgres_url:
            return
        from sqlalchemy import text

        with self._engine().begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS substrate_runtime_control_surface (surface_key TEXT PRIMARY KEY, updated_at TIMESTAMPTZ NOT NULL, value_json JSONB NOT NULL)"
                )
            )
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS substrate_runtime_control_surface_history ("
                    "history_id TEXT PRIMARY KEY, surface_key TEXT NOT NULL, changed_at TIMESTAMPTZ NOT NULL, "
                    "actor TEXT, previous_value_json JSONB, new_value_json JSONB NOT NULL)"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS substrate_runtime_control_surface_history_key_time "
                    "ON substrate_runtime_control_surface_history (surface_key, changed_at DESC)"
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
