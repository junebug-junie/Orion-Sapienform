"""Lazily-initialized process-level substrate graph store for orion-recall.

Mirrors the singleton-caching pattern in
``orion/substrate/relational/adapters/concept_induction_ctx.py::_get_store``
-- same never-raise-on-init-failure contract, same env-driven backend
selection (``build_substrate_store_from_env``). A dedicated singleton lives
here rather than importing cortex-exec's or Hub's store instance: each
service builds its own store handle against the same shared FALKORDB_URI
backend (see ``.env_example``), there is no cross-service store object to
share, and importing across service boundaries is against this repo's
service-isolation convention (see CLAUDE.md section 5).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

from orion.substrate import build_substrate_store_from_env
from orion.substrate.store import SubstrateGraphStore

logger = logging.getLogger(__name__)

_STORE: Optional[SubstrateGraphStore] = None
# Serialises construction. The boot warmup thread and a recall's
# concept_region thread can both reach get_substrate_store() while _STORE is
# still None; without this, both would run the slow Falkor hydration and one
# handle would be thrown away. Only taken while _STORE is None.
_STORE_LOCK = threading.Lock()

# Request-path wait for _STORE_LOCK. If the warmup (or another request) is
# mid-hydration, a recall thread gives up after this and concept_region
# returns nothing for that turn, instead of parking a worker thread for as
# long as FalkorDB takes.
REQUEST_LOCK_TIMEOUT_S = 2.0
# Warmup's own wait (lock + build). The hydration thread itself cannot be
# killed; the socket timeouts below are what bound it.
WARMUP_TIMEOUT_S = 30.0

# Redis socket timeouts for this service's substrate store client. Live
# 2026-09-30 from the host: the four hydrate reads (4,588 nodes, 10,000
# edges) took 1.4-1.6s in total, each under 1s; the whole cold
# get_substrate_store() (reads + decode) measured 6.25s in the container.
# 30s per socket read is ~40x the slowest single query; 5s to connect.
FALKOR_SOCKET_TIMEOUT_S = 30.0
FALKOR_SOCKET_CONNECT_TIMEOUT_S = 5.0

# After a failed or empty build, wait before trying again so a down FalkorDB
# is not hammered once per purposeful turn. Doubles per consecutive failure.
RETRY_BACKOFF_BASE_S = 5.0
RETRY_BACKOFF_MAX_S = 300.0
_next_retry_at: float = 0.0
_consecutive_failures: int = 0
_last_failure_reason: Optional[str] = None


def _build_store() -> SubstrateGraphStore:
    return build_substrate_store_from_env(
        falkor_socket_timeout_s=FALKOR_SOCKET_TIMEOUT_S,
        falkor_socket_connect_timeout_s=FALKOR_SOCKET_CONNECT_TIMEOUT_S,
    )


def _hydrate_failure_reason(store: SubstrateGraphStore) -> Optional[str]:
    """None if the store is usable, else why not.

    FalkorSubstrateStore reads only its in-process cache on the recall path
    and never refreshes it there (concept_region avoids snapshot() on
    purpose), so a store whose boot hydrate failed -- FalkorDB not up yet,
    the query raised and was swallowed -- would serve an empty graph for the
    whole process lifetime. Stores without the signal (in-memory, other
    backends) are accepted as-is.
    """
    ok = getattr(store, "last_hydrate_ok", None)
    if ok is False:
        return "hydrate_failed"
    if ok is True and int(getattr(store, "last_hydrate_node_count", 0) or 0) <= 0:
        return "hydrate_empty"
    return None


def _record_failure(reason: str) -> None:
    global _next_retry_at, _consecutive_failures, _last_failure_reason
    _consecutive_failures += 1
    backoff = min(RETRY_BACKOFF_MAX_S, RETRY_BACKOFF_BASE_S * (2 ** (_consecutive_failures - 1)))
    _next_retry_at = time.monotonic() + backoff
    _last_failure_reason = reason
    logger.warning(
        "recall_substrate_store_unavailable reason=%s consecutive_failures=%s retry_in_s=%.0f",
        reason,
        _consecutive_failures,
        backoff,
    )


def last_failure_reason() -> Optional[str]:
    return _last_failure_reason


def get_substrate_store(*, lock_timeout_s: float = REQUEST_LOCK_TIMEOUT_S) -> Optional[SubstrateGraphStore]:
    """Return (or lazily initialise) the process-level substrate store.

    Never raises; returns None when the store is not available right now:
    construction raised, the hydrate failed or came back empty (not cached,
    retried after a backoff), the backoff has not elapsed, or another thread
    is mid-construction and did not finish within ``lock_timeout_s``.
    Collectors reading from a ``None`` store return empty, never raise (see
    ``collectors/concept_region.py``).
    """

    global _STORE, _consecutive_failures, _last_failure_reason
    store = _STORE
    if store is not None:
        return store
    if time.monotonic() < _next_retry_at:
        return None
    if not _STORE_LOCK.acquire(timeout=max(0.0, float(lock_timeout_s))):
        logger.debug("recall_substrate_store_lock_timeout timeout_s=%s", lock_timeout_s)
        return None
    try:
        if _STORE is not None:
            return _STORE
        if time.monotonic() < _next_retry_at:
            return None
        try:
            built = _build_store()
        except Exception as exc:
            _record_failure(f"build_error:{type(exc).__name__}")
            logger.debug("recall_substrate_store_init_failed error=%s", exc)
            return None
        reason = _hydrate_failure_reason(built)
        if reason is not None:
            _record_failure(reason)
            return None
        _STORE = built
        _consecutive_failures = 0
        _last_failure_reason = None
        return built
    finally:
        _STORE_LOCK.release()


async def warm_substrate_store(*, timeout_s: float = WARMUP_TIMEOUT_S) -> bool:
    """Build the substrate store once, off the event loop, so the first
    purposeful recall after a restart does not pay the cold hydration.

    Best-effort: never raises, returns True only if a usable (hydrated,
    non-empty) store is cached afterwards. Logs ``recall_substrate_store_warmed``
    only then; a failed or empty hydrate logs
    ``recall_substrate_store_warmup_failed reason=...`` and leaves the store
    uncached so a later call retries after the backoff.

    A recall that arrives while this is mid-hydration does not wait for it
    past ``REQUEST_LOCK_TIMEOUT_S``: its concept_region thread gives up on the
    lock and returns nothing for that turn, and the recall itself never waits
    past its own deadline either way. On this function's own timeout the
    hydration thread keeps running (threads cannot be cancelled), bounded by
    the Falkor socket timeouts above; if it later succeeds the store is cached.
    """
    started = time.perf_counter()
    try:
        store = await asyncio.wait_for(
            asyncio.to_thread(get_substrate_store, lock_timeout_s=timeout_s), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        logger.warning("recall_substrate_store_warmup_failed reason=timeout timeout_s=%s", timeout_s)
        return False
    except Exception as exc:
        logger.warning("recall_substrate_store_warmup_failed reason=error error=%s", exc)
        return False
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if store is None:
        logger.warning(
            "recall_substrate_store_warmup_failed reason=%s elapsed_ms=%s",
            last_failure_reason() or "unavailable",
            elapsed_ms,
        )
        return False
    logger.info(
        "recall_substrate_store_warmed elapsed_ms=%s nodes=%s",
        elapsed_ms,
        getattr(store, "last_hydrate_node_count", None),
    )
    return True


def _reset_for_tests() -> None:
    global _STORE, _next_retry_at, _consecutive_failures, _last_failure_reason
    _STORE = None
    _next_retry_at = 0.0
    _consecutive_failures = 0
    _last_failure_reason = None


__all__ = ["get_substrate_store", "warm_substrate_store", "last_failure_reason"]
