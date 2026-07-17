from __future__ import annotations

import asyncio
import logging
from typing import Optional, Tuple

logger = logging.getLogger("orion.spark.concept_induction.chat_history_pg")

try:
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    asyncpg = None

_CHAT_HISTORY_TABLE = "chat_history_log"

# This lookup runs inline in ConceptWorker's single sequential bus-consume loop
# (handle_envelope), which also processes every other intake channel (drives,
# tensions, homeostatic signals). A Postgres outage must not stall that loop for
# more than a few seconds per event, so both connect and query are bounded well
# under the worker's other timeouts.
_CONNECT_TIMEOUT_SEC = 3.0
_QUERY_TIMEOUT_SEC = 3.0

_pool: "asyncpg.Pool | None" = None
_pool_dsn: Optional[str] = None
_pool_lock = asyncio.Lock()


async def _get_pool(dsn: str) -> "asyncpg.Pool | None":
    """Lazily create (or rotate, if the DSN changed) the shared connection pool.

    A fresh TCP + auth handshake per chat turn would add a real per-event cost to
    the worker's hot path; pooling amortizes that across the worker's lifetime,
    matching the pattern already used by several other services in this repo
    (e.g. services/orion-substrate-telemetry/app/pg_pool.py).
    """
    global _pool, _pool_dsn
    if asyncpg is None:
        return None
    async with _pool_lock:
        if _pool is not None and _pool_dsn == dsn:
            return _pool
        if _pool is not None:
            await _pool.close()
            _pool = None
        try:
            _pool = await asyncpg.create_pool(
                dsn=dsn,
                min_size=1,
                max_size=4,
                timeout=_CONNECT_TIMEOUT_SEC,
                command_timeout=_QUERY_TIMEOUT_SEC,
            )
            _pool_dsn = dsn
        except Exception:
            logger.debug("chat_history_pg_pool_init_failed", exc_info=True)
            _pool = None
            _pool_dsn = None
        return _pool


async def close_pool() -> None:
    """Close the shared pool. Call from service shutdown for a clean exit."""
    global _pool, _pool_dsn
    async with _pool_lock:
        if _pool is not None:
            await _pool.close()
        _pool = None
        _pool_dsn = None


async def fetch_chat_turn_by_correlation_id(
    correlation_id: str,
    *,
    dsn: str,
    retries: int = 3,
    retry_delay_sec: float = 0.3,
) -> Optional[Tuple[str, str]]:
    """Fetch the canonical (prompt, response) pair for a chat turn from Postgres.

    This is the same text the hub already published on the bus (no separate
    cleaning step happens before the SQL write) — the point is a single canonical
    row per turn rather than trusting whatever shape a given intake channel's
    envelope happens to carry, not stance-artifact-free content.

    Retries cover the row not being committed yet (the SQL writer commits off the
    same bus event this worker consumes, so a fresh turn can race it) as well as
    transient connection errors — both are retried up to ``retries`` times.
    Returns ``None`` on asyncpg being unavailable or the row never appearing, so
    callers fall back to the envelope's own fields; this is a best-effort
    refinement, never a hard dependency on Postgres.
    """
    if asyncpg is None or not correlation_id:
        return None
    query = f"""
        SELECT prompt, response
        FROM {_CHAT_HISTORY_TABLE}
        WHERE correlation_id = $1
        ORDER BY created_at DESC
        LIMIT 1
    """
    for attempt in range(max(1, retries)):
        row = None
        pool = await _get_pool(dsn)
        if pool is not None:
            try:
                row = await pool.fetchrow(query, correlation_id)
            except Exception:
                logger.debug(
                    "chat_history_pg_lookup_failed correlation_id=%s attempt=%s",
                    correlation_id,
                    attempt,
                    exc_info=True,
                )
                row = None
        if row is not None:
            prompt = str(row["prompt"] or "").strip()
            response = str(row["response"] or "").strip()
            if prompt or response:
                logger.debug(
                    "chat_history_pg_lookup_hit correlation_id=%s attempt=%s",
                    correlation_id,
                    attempt,
                )
                return prompt, response
        if attempt < retries - 1:
            await asyncio.sleep(retry_delay_sec)
    logger.debug(
        "chat_history_pg_lookup_miss correlation_id=%s retries=%s",
        correlation_id,
        retries,
    )
    return None
