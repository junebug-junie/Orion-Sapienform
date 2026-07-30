"""Bounded fail-open fetch of MetacogContextService's "recent trend signals"
evidence cue.

Tier 1 patch, per `docs/superpowers/specs/2026-07-28-metacog-turn-scoped-trend-
reducer-design.md`'s "2026-07-29/30 update". Wires
`orion.substrate.metacog_trend_signals`'s pure read functions into this
service's own engine-caching / DSN-resolution / bounded-timeout conventions --
mirrors `drive_state_postgres.py`'s shape (module-level cached engine,
`asyncio.wait_for(asyncio.to_thread(...))`, fail-open, never raises to the
caller). Differs from that module in one way: the statement-timeout guard is
set as a per-connection `-c statement_timeout=...` GUC via `connect_args`
rather than a per-transaction `SET LOCAL`, because the two reader functions
below each own a short-lived `engine.connect()` rather than sharing one
transaction (see `_get_engine()`'s docstring comment).

Read-only. Never mutates `ctx` itself -- `executor.py`'s MetacogContextService
step is the only writer of `ctx["recent_trend_signals"]`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import create_engine

from orion.substrate.metacog_trend_signals import (
    PREDICTION_ERROR_NODE_IDS,
    build_recent_trend_signals_cue,
    latest_biometrics_induction_by_node,
    latest_node_prediction_errors,
)

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

# Per-connection statement_timeout (ms), independent of the outer asyncio
# timeout below -- same belt-and-suspenders pattern as
# drive_state_postgres.py's _DRIVE_STATE_QUERY_STATEMENT_TIMEOUT_MS. Set above
# that module's 300ms: 2026-07-30 EXPLAIN ANALYZE against real
# orion_biometrics_induction history (46k rows, no index on node/timestamp yet
# -- see this patch's PR report for the disclosed follow-up) measured ~60-150ms
# for the induction query alone; this cap leaves real headroom rather than
# racing that number.
_METACOG_TREND_QUERY_STATEMENT_TIMEOUT_MS = 800

_ENGINE = None
_ENGINE_URL: str | None = None


def _flag_enabled() -> bool:
    return os.getenv("ENABLE_METACOG_TREND_CUE", "true").strip().lower() in _TRUTHY


def _dsn() -> str:
    # Same conjourney instance every other cortex-exec Postgres reader in this
    # service targets. Reuses the felt-state / endogenous-runtime DSN fallback
    # chain rather than adding a third DB-URL env key for the same database.
    return (
        os.getenv("SUBSTRATE_FELT_STATE_DATABASE_URL", "").strip()
        or os.getenv("ENDOGENOUS_RUNTIME_SQL_DATABASE_URL", "").strip()
    )


def _timeout_sec() -> float:
    raw = os.getenv("METACOG_TREND_CUE_FETCH_TIMEOUT_SEC", "0.8").strip()
    try:
        return max(0.01, float(raw))
    except ValueError:
        return 0.8


def _induction_nodes() -> tuple[str, ...]:
    raw = os.getenv("METACOG_TREND_CUE_NODES", "athena,atlas,circe")
    nodes = tuple(n.strip() for n in raw.split(",") if n.strip())
    return nodes or ("athena", "atlas", "circe")


def _induction_max_age_sec() -> float:
    raw = os.getenv("METACOG_TREND_CUE_INDUCTION_MAX_AGE_SEC", "180").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 180.0


def _get_engine():
    global _ENGINE, _ENGINE_URL
    url = _dsn()
    if not url:
        return None
    if _ENGINE is None or _ENGINE_URL != url:
        # statement_timeout set as a per-connection GUC via connect_args
        # (applies to every statement on every connection this engine hands
        # out), not a per-transaction `SET LOCAL` -- the two reader functions
        # below each open their own short-lived `engine.connect()` (matching
        # `latest_bus_synaptic_prediction_error(engine)`'s own signature/
        # testing convention in orion/substrate/bus_synaptic_surprise.py
        # rather than threading a shared Connection through both), so there
        # is no single transaction to attach a `SET LOCAL` to.
        _ENGINE = create_engine(
            url,
            pool_pre_ping=True,
            connect_args={
                "options": f"-c statement_timeout={_METACOG_TREND_QUERY_STATEMENT_TIMEOUT_MS}"
            },
        )
        _ENGINE_URL = url
    return _ENGINE


def _fetch_sync() -> dict[str, Any] | None:
    engine = _get_engine()
    if engine is None:
        raise RuntimeError("metacog_trend_cue_dsn_unset")
    prediction_errors = latest_node_prediction_errors(engine, PREDICTION_ERROR_NODE_IDS)
    induction_by_node = latest_biometrics_induction_by_node(
        engine, _induction_nodes(), max_age_sec=_induction_max_age_sec()
    )
    return build_recent_trend_signals_cue(
        prediction_errors=prediction_errors,
        induction_by_node=induction_by_node,
        generated_at=datetime.now(timezone.utc),
    )


async def fetch_recent_trend_signals(correlation_id: str) -> dict[str, Any] | None:
    """Bounded fail-open fetch. Returns `None` on any disabled/unset/timeout/
    error condition -- caller renders that as an empty cue, never a crash."""
    if not _flag_enabled():
        return None
    if not _dsn():
        logger.debug(
            "metacog_trend_cue_dsn_unset correlation_id=%s", correlation_id
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
            "metacog_trend_cue_fetch_timeout correlation_id=%s elapsed_ms=%s timeout_sec=%s",
            correlation_id,
            elapsed_ms,
            timeout_sec,
        )
        return None
    except Exception as exc:  # noqa: BLE001 -- fail-open by contract
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(
            "metacog_trend_cue_fetch_failed correlation_id=%s elapsed_ms=%s exc_type=%s err=%s",
            correlation_id,
            elapsed_ms,
            type(exc).__name__,
            exc,
        )
        return None


def reset_metacog_trend_reader_engine_for_tests() -> None:
    global _ENGINE, _ENGINE_URL
    _ENGINE = None
    _ENGINE_URL = None
