"""Hub presence — Orion's chat liveness as a self-state observable.

Records chat-turn timestamps in-process and mirrors a small snapshot to the
``substrate_hub_presence`` Postgres row (single-row upsert, presence_id='hub')
so the self-state runtime can hydrate ``SelfStateV1.hub_presence``.

Every path here is best-effort: a presence write must NEVER break a chat turn.
Apply ``services/orion-sql-db/manual_migration_hub_presence_v1.sql`` before
expecting rows; without it (or without POSTGRES_URI) only the in-process
snapshot is available.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any

logger = logging.getLogger("orion-hub.hub_presence")

_TURNS_WINDOW_SEC = 300.0
_ACTIVE_MAX_AGE_SEC = 120.0
_IDLE_MAX_AGE_SEC = 900.0
# At most one Postgres upsert per this interval; turns between writes still
# land in the in-process deque and are reflected in the next snapshot.
_WRITE_MIN_INTERVAL_SEC = 5.0

_turn_timestamps: deque[float] = deque(maxlen=200)
_last_write_at: float = 0.0
_lock = threading.Lock()


def reset() -> None:
    """Test helper: clear in-process presence state."""
    global _last_write_at
    with _lock:
        _turn_timestamps.clear()
        _last_write_at = 0.0


def presence_snapshot(now: float | None = None) -> dict[str, Any] | None:
    """Presence from in-process turn history; None before the first turn."""
    ts = float(now if now is not None else time.time())
    with _lock:
        turns = list(_turn_timestamps)
    if not turns:
        return None
    last_turn_age_sec = max(0.0, ts - turns[-1])
    recent = sum(1 for t in turns if ts - t <= _TURNS_WINDOW_SEC)
    if last_turn_age_sec < _ACTIVE_MAX_AGE_SEC:
        health = "active"
    elif last_turn_age_sec < _IDLE_MAX_AGE_SEC:
        health = "idle"
    else:
        health = "dormant"
    return {
        "last_turn_age_sec": round(last_turn_age_sec, 3),
        "turns_per_minute": round(recent / (_TURNS_WINDOW_SEC / 60.0), 3),
        "connection_health": health,
    }


def _write_snapshot_to_postgres(snapshot: dict[str, Any]) -> None:
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return
    try:
        import json

        from sqlalchemy import create_engine, text

        engine = create_engine(uri, pool_pre_ping=True)
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO substrate_hub_presence (presence_id, generated_at, presence_json, updated_at)
                        VALUES ('hub', now(), CAST(:presence_json AS jsonb), now())
                        ON CONFLICT (presence_id) DO UPDATE SET
                            generated_at = EXCLUDED.generated_at,
                            presence_json = EXCLUDED.presence_json,
                            updated_at = EXCLUDED.updated_at
                        """
                    ),
                    {"presence_json": json.dumps(snapshot)},
                )
        finally:
            engine.dispose()
    except Exception as exc:
        logger.warning("hub_presence_write_failed error=%s", exc)


def record_turn(now: float | None = None) -> None:
    """Record one chat turn; best-effort, never raises, never blocks chat.

    The Postgres mirror runs on a daemon thread and is rate-limited to one
    write per _WRITE_MIN_INTERVAL_SEC.
    """
    global _last_write_at
    try:
        ts = float(now if now is not None else time.time())
        with _lock:
            _turn_timestamps.append(ts)
            due = (ts - _last_write_at) >= _WRITE_MIN_INTERVAL_SEC
            if due:
                _last_write_at = ts
        if not due:
            return
        # Env-first like the substrate debug routes: keeps this hot path free
        # of the full settings import. Mirrored in app.settings for compose.
        flag = os.getenv("HUB_PRESENCE_WRITER_ENABLED", "true").strip().lower()
        if flag in {"0", "false", "no", "off"}:
            return
        snapshot = presence_snapshot(ts)
        if snapshot is None:
            return
        threading.Thread(
            target=_write_snapshot_to_postgres,
            args=(snapshot,),
            name="hub-presence-writer",
            daemon=True,
        ).start()
    except Exception as exc:
        logger.warning("hub_presence_record_failed error=%s", exc)
