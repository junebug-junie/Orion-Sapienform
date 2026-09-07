"""Bounded fail-open fetch of Oríon's own "recent attention" cue.

Mirrors `metacog_trend_reader.py`'s shape (module-level cached engine,
`asyncio.wait_for(asyncio.to_thread(...))`, fail-open, never raises to the
caller) for a different source table: `substrate_attention_schema`, which
already has five producers writing into it (cortex_turn, curiosity, reverie,
substrate_attention, durable_run) via orion-sql-writer, each row carrying a
human-readable `reason_narrative`.

This is an ambient background-awareness cue for `chat_stance_brief.j2`, not a
status report -- see `orion.substrate.recent_attention_cue`'s docstring for
the framing.

Read-only. Never mutates `ctx` itself -- `executor.py`'s MetacogContextService
step is the only writer of `ctx["recent_attention"]`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine, text

from orion.substrate.recent_attention_cue import build_recent_attention_cue

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

# Per-connection statement_timeout (ms), independent of the outer asyncio
# timeout below -- same belt-and-suspenders pattern as
# metacog_trend_reader.py's _METACOG_TREND_QUERY_STATEMENT_TIMEOUT_MS.
_RECENT_ATTENTION_QUERY_STATEMENT_TIMEOUT_MS = 800

_ENGINE = None
_ENGINE_URL: str | None = None


def _flag_enabled() -> bool:
    return os.getenv("ENABLE_RECENT_ATTENTION_CUE", "true").strip().lower() in _TRUTHY


def _dsn() -> str:
    # Same conjourney instance every other cortex-exec Postgres reader in this
    # service targets (substrate_attention_schema lives there too). Reuses the
    # felt-state / endogenous-runtime DSN fallback chain rather than adding a
    # third DB-URL env key for the same database.
    return (
        os.getenv("SUBSTRATE_FELT_STATE_DATABASE_URL", "").strip()
        or os.getenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL", "").strip()
    )


def _timeout_sec() -> float:
    raw = os.getenv("RECENT_ATTENTION_CUE_FETCH_TIMEOUT_SEC", "0.8").strip()
    try:
        return max(0.01, float(raw))
    except ValueError:
        return 0.8


def _limit() -> int:
    raw = os.getenv("RECENT_ATTENTION_CUE_LIMIT", "3").strip()
    try:
        value = int(raw)
    except ValueError:
        return 3
    return value if value > 0 else 3


def _stale_after_sec() -> float:
    # 15 minutes: the fastest lane writing this table, substrate_attention,
    # ticks roughly every 30s, so a 15-minute silence across all five
    # producers (cortex_turn, curiosity, reverie, substrate_attention,
    # durable_run) means the system has actually gone quiet, not that it's
    # merely between ticks.
    raw = os.getenv("RECENT_ATTENTION_CUE_STALE_AFTER_SEC", "900").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 900.0


def _get_engine():
    global _ENGINE, _ENGINE_URL
    url = _dsn()
    if not url:
        return None
    if _ENGINE is None or _ENGINE_URL != url:
        _ENGINE = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={
                "options": f"-c statement_timeout={_RECENT_ATTENTION_QUERY_STATEMENT_TIMEOUT_MS}"
            },
        )
        _ENGINE_URL = url
    return _ENGINE


def _fetch_sync() -> dict[str, Any] | None:
    engine = _get_engine()
    if engine is None:
        raise RuntimeError("recent_attention_cue_dsn_unset")
    limit = _limit()
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    "SELECT process, reason_narrative, generated_at "
                    "FROM substrate_attention_schema "
                    # reason_narrative is NOT NULL DEFAULT '' on the live table
                    # (not absent, just empty) -- filtered here, before LIMIT,
                    # so an empty-narrative burst in the most recent rows can't
                    # silently starve the cue of real narrated rows sitting
                    # just past the window. Confirmed at review 2026-09-07.
                    "WHERE reason_narrative <> '' "
                    "ORDER BY generated_at DESC LIMIT :limit"
                ),
                {"limit": limit},
            )
            .mappings()
            .all()
        )
    return build_recent_attention_cue(
        [dict(row) for row in rows],
        now=datetime.now(timezone.utc),
        limit=limit,
        stale_after_sec=_stale_after_sec(),
    )


async def fetch_recent_attention_cue(correlation_id: str) -> dict[str, Any] | None:
    """Bounded fail-open fetch. Returns `None` on any disabled/unset/timeout/
    error condition -- caller renders that as an empty cue, never a crash."""
    if not _flag_enabled():
        return None
    if not _dsn():
        logger.debug(
            "recent_attention_cue_dsn_unset correlation_id=%s", correlation_id
        )
        return None

    timeout_sec = _timeout_sec()
    t0 = time.perf_counter()
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_fetch_sync), timeout=timeout_sec
        )
    except asyncio.TimeoutError:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(
            "recent_attention_cue_fetch_timeout correlation_id=%s elapsed_ms=%s timeout_sec=%s",
            correlation_id,
            elapsed_ms,
            timeout_sec,
        )
        return None
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(
            "recent_attention_cue_fetch_failed correlation_id=%s elapsed_ms=%s exc_type=%s err=%s",
            correlation_id,
            elapsed_ms,
            type(exc).__name__,
            exc,
        )
        return None


def reset_recent_attention_reader_engine_for_tests() -> None:
    global _ENGINE, _ENGINE_URL
    _ENGINE = None
    _ENGINE_URL = None
