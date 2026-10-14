"""Self-readings for stance_react and reverie (self-calibration spec section C, patch 4).

Reads ``substrate_node_prediction_error_history`` (written by orion-substrate-runtime) read-only
into one process-wide ``NodePeHistoryCache`` -- seeded once with 7 days, then topped up
incrementally (the same cache orion-attention-runtime's field contest uses) -- and renders the
block with ``orion.attention.self_readings.project_self_readings``. No LLM, no new table.

Fail-open and bounded: the whole refresh+render runs in a worker thread under
``SELF_READINGS_FETCH_TIMEOUT_SEC``. A timeout, a read error, a refresh already running in
another caller, or a calm body all render nothing (``text == ""``) and log
``self_readings_skipped reason=...`` with the caller's correlation id. A timed-out refresh keeps
running in its thread and seeds the cache for the next caller (stance and reverie share it).

"Tends to" clauses (spec section E, ``SELF_SIGNAL_OUTCOME_ASSOCIATION_ENABLED``): the newest
``self_signal_outcome_association`` row per (signal, outcome), refreshed within 3 h, read at most
every 5 min in a background thread (and once at startup warm) -- never on the turn's path or under
the history lock. Not loaded yet / unreadable -> no clause, readings unchanged. Flag off -> the
block is byte-identical to before.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable

from orion.schemas.self_reading import SelfReadingsBlockV1

from .settings import settings

logger = logging.getLogger("orion-thought.self_readings")

_FETCH_SQL = (
    "SELECT node_id, observed_at, value FROM substrate_node_prediction_error_history "
    "WHERE observed_at >= :since ORDER BY observed_at"
)
# Newest association per (signal, outcome) written by orion-equilibrium-service's hourly rollup
# (self-calibration spec section E). Only rows refreshed within ASSOCIATION_MAX_AGE count: a
# stalled producer stops the clause instead of repeating an old claim forever.
_ASSOC_COLUMNS = (
    "association_id", "computed_at", "signal_id", "outcome", "window_start", "window_end",
    "n_high", "events_high", "n_normal", "events_normal", "rate_high", "rate_normal", "ratio",
    "ci_low", "ci_high", "ci_method", "min_hours", "verdict", "excluded_reason", "clause",
)
_ASSOC_SQL = (
    f"SELECT DISTINCT ON (signal_id, outcome) {', '.join(_ASSOC_COLUMNS)} "
    "FROM self_signal_outcome_association WHERE computed_at >= :since "
    "ORDER BY signal_id, outcome, computed_at DESC LIMIT 200"
)
ASSOCIATION_MAX_AGE = timedelta(hours=3)
# Re-read at most this often (the producer refreshes hourly).
ASSOCIATION_REFRESH_SEC = 300.0
# The 7-day seed is ~62k rows (0.3 s live 2026-10-14); a stuck database must not pin a thread.
_STATEMENT_TIMEOUT_MS = 5000
_CONNECT_TIMEOUT_SEC = 5

# orion.attention.self_readings imports orion.attention.world_first, which imports orion.substrate
# (the graph-engine package, ~0.7 s and 36 modules in this container, 2026-10-14). Imported lazily,
# inside the worker thread, so importing this module (and app.reverie / app.bus_listener) keeps the
# thin-import boundary (tests/test_reverie_thin_import_boundary.py); the cost lands once, off the
# event loop, and main.py's startup warm pays it before the first turn.
_cache = None
_lock = threading.Lock()
_assoc: tuple[float, dict] | None = None  # (monotonic fetched_at, signal_id -> association)
_assoc_lock = threading.Lock()
_assoc_refreshing = False
_engine = None
_engine_url: str | None = None


def _get_engine():
    """Own read engine: statement_timeout is fixed at engine construction, so sharing another
    reader's engine would let whichever caller built it first decide the bound (store.py)."""
    global _engine, _engine_url
    from .store import _database_url

    url = _database_url()
    if _engine is None or _engine_url != url:
        from sqlalchemy import create_engine

        _engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_size=1,
            max_overflow=1,
            connect_args={
                "options": f"-c statement_timeout={_STATEMENT_TIMEOUT_MS}",
                "connect_timeout": _CONNECT_TIMEOUT_SEC,
            },
        )
        _engine_url = url
    return _engine


def fetch_since(since: datetime) -> Iterable[tuple[str, datetime, float]]:
    """Seam for tests. Raises on a read error (the caller renders nothing)."""
    from sqlalchemy import text

    with _get_engine().connect() as conn:
        rows = conn.execute(text(_FETCH_SQL), {"since": since}).fetchall()
    return [(str(r[0]), r[1], float(r[2])) for r in rows]


def fetch_associations(since: datetime) -> list[dict]:
    """Seam for tests. Raises on a read error (the caller renders no clause)."""
    from sqlalchemy import text

    with _get_engine().connect() as conn:
        rows = conn.execute(text(_ASSOC_SQL), {"since": since}).fetchall()
    return [dict(zip(_ASSOC_COLUMNS, r)) for r in rows]


def refresh_associations(now: datetime) -> None:
    """Blocking read of the newest association rows into the in-memory index. Runs off the turn
    path (startup warm, or a background thread started by ``_associations``). Never raises: an
    unreadable table caches an empty index, retried after ASSOCIATION_REFRESH_SEC."""
    global _assoc, _assoc_refreshing
    try:
        from orion.attention.self_signal_outcomes import index_by_signal
        from orion.schemas.self_signal_outcome import SelfSignalOutcomeAssociationV1

        rows = []
        for raw in fetch_associations(now - ASSOCIATION_MAX_AGE):
            try:
                rows.append(SelfSignalOutcomeAssociationV1.model_validate(raw))
            except Exception as exc:  # noqa: BLE001 -- one malformed row hides only itself
                logger.debug("self_signal_association_row_invalid err=%s", exc)
        index = dict(index_by_signal(rows))
    except Exception as exc:  # noqa: BLE001 -- no table yet / read error: no clause, readings unchanged
        logger.debug("self_signal_association_fetch_failed err=%s: %s", type(exc).__name__, exc)
        index = {}
    with _assoc_lock:
        _assoc = (time.monotonic(), index)
        _assoc_refreshing = False


def _associations(now: datetime) -> dict | None:
    """signal_id -> renderable association from the in-memory index, or None (flag off / not
    loaded yet / none fresh). Never reads the database on the caller's thread: a stale index starts
    one background refresh and this call uses what it has, so the self-readings budget and the
    history lock never wait on this query."""
    global _assoc_refreshing
    if not settings.self_signal_outcome_association_enabled:
        return None
    with _assoc_lock:
        cached = _assoc
        stale = cached is None or time.monotonic() - cached[0] >= ASSOCIATION_REFRESH_SEC
        start = stale and not _assoc_refreshing
        if start:
            _assoc_refreshing = True
    if start:
        threading.Thread(
            target=refresh_associations, args=(now,), name="self-signal-associations", daemon=True
        ).start()
    return (cached[1] or None) if cached is not None else None


def _refresh_and_project(now: datetime, lock_wait_sec: float) -> SelfReadingsBlockV1:
    global _cache
    if not _lock.acquire(timeout=max(0.0, lock_wait_sec)):
        return SelfReadingsBlockV1(rendered_at=now, skip_reason="busy")
    try:
        from orion.attention.pe_history_cache import NodePeHistoryCache
        from orion.attention.self_readings import project_self_readings

        if _cache is None:
            _cache = NodePeHistoryCache()
        try:
            _cache.refresh(fetch_since, now=now)
        except Exception as exc:  # noqa: BLE001 -- unread history is "no reading", never calm
            logger.debug("self_readings_fetch_failed err=%s: %s", type(exc).__name__, exc)
            return SelfReadingsBlockV1(rendered_at=now, skip_reason="fetch_failed")
        return project_self_readings(_cache, now=now, associations=_associations(now))
    finally:
        _lock.release()


def warm(now: datetime | None = None) -> None:
    """Import the calibration modules and seed the 7-day cache (startup, in a thread). Never raises."""
    try:
        if settings.self_signal_outcome_association_enabled:
            refresh_associations(now or datetime.now(timezone.utc))
        block = _refresh_and_project(now or datetime.now(timezone.utc), 30.0)
        logger.info("self_readings_warm reason=%s chars=%s", block.skip_reason, len(block.text))
    except Exception as exc:  # noqa: BLE001
        logger.warning("self_readings_warm_failed err=%s: %s", type(exc).__name__, exc)


async def build_self_readings_block(
    correlation_id: str, *, purpose: str, enabled: bool, now: datetime | None = None,
) -> SelfReadingsBlockV1:
    """Always returns a block; ``text`` is "" when nothing may be shown. Never raises."""
    now = now or datetime.now(timezone.utc)
    if not enabled:
        return SelfReadingsBlockV1(rendered_at=now, skip_reason="disabled")
    timeout = settings.self_readings_fetch_timeout_sec
    started = time.perf_counter()
    try:
        block = await asyncio.wait_for(asyncio.to_thread(_refresh_and_project, now, timeout), timeout)
    except asyncio.TimeoutError:
        block = SelfReadingsBlockV1(rendered_at=now, skip_reason="fetch_timeout")
    except Exception as exc:  # noqa: BLE001 -- the block must never fail a turn or a reverie tick
        logger.warning("self_readings_failed corr=%s err=%s: %s", correlation_id, type(exc).__name__, exc)
        block = SelfReadingsBlockV1(rendered_at=now, skip_reason="render_failed")
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
    if block.text:
        logger.info("self_readings corr=%s purpose=%s chars=%s shown=%s dropped=%s ms=%s",
                    correlation_id, purpose, len(block.text), ",".join(block.shown),
                    block.lines_dropped, elapsed_ms)
    else:
        logger.info("self_readings_skipped corr=%s purpose=%s reason=%s ms=%s",
                    correlation_id, purpose, block.skip_reason, elapsed_ms)
    return block
