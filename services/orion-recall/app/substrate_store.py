"""Process-level substrate store handle for orion-recall's concept_region collector.

Recall never loads the substrate graph. With ``SUBSTRATE_STORE_BACKEND=falkor``
this returns a ``FalkorDirectConceptStore`` (``orion/substrate/falkor_direct.py``):
every concept_region read is a bounded ``GRAPH.RO_QUERY`` against FalkorDB and
the reinforcement write is a single ``MERGE ... SET``. There is no hydrate, no
warm-up and no snapshot -- the handle has no ``snapshot()`` at all.

Before 2026-10-06 this module cached a ``FalkorSubstrateStore``, whose reads
come from a complete in-process copy of the graph. Once PR #2500 made that
copy complete it took 17-25s to build, and recall carried a boot warm-up,
lock timeouts and a retry backoff purely to hide it. All of that is gone.

Other backends: ``in_memory`` (or unset) returns an empty in-memory store, as
before. Any other backend (routed/graphdb/sparql) would bring its own cached
graph, so recall refuses it: concept_region returns nothing rather than
loading a graph.

A dedicated handle lives here rather than importing cortex-exec's or Hub's
store: each service builds its own client against the shared FALKORDB_URI,
and importing across service boundaries is against this repo's
service-isolation convention (CLAUDE.md section 5).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from orion.substrate.falkor_direct import build_falkor_direct_concept_store_from_env
from orion.substrate.store import InMemorySubstrateGraphStore

logger = logging.getLogger(__name__)

# Redis socket timeouts for this service's Falkor clients. Every read recall
# makes is bounded; the slowest single query observed live is ~0.3s from the
# host and ~0.7s in-container (a full 500/500 region read, not the hot path).
# 1.5s per socket read caps what a hung FalkorDB can cost one query; the
# breaker below caps what it can cost a run of turns.
FALKOR_SOCKET_TIMEOUT_S = 1.5
FALKOR_SOCKET_CONNECT_TIMEOUT_S = 1.0

# Circuit breaker: after this many consecutive Falkor timeouts, concept_region
# is skipped (store handle reported as unavailable) for the cooldown. After
# the cooldown one turn is let through; a success closes the breaker, another
# timeout reopens it. No fallback of any kind -- a skipped turn simply has no
# concept_region fragments.
BREAKER_TIMEOUT_THRESHOLD = 3
BREAKER_COOLDOWN_S = 60.0


class ConceptRegionBreakerOpen(RuntimeError):
    """Raised by the guarded store while the breaker is open."""


def _is_timeout(exc: BaseException) -> bool:
    try:
        from redis.exceptions import TimeoutError as RedisTimeoutError
    except Exception:  # pragma: no cover - redis is a hard dependency
        RedisTimeoutError = TimeoutError  # type: ignore[misc,assignment]
    return isinstance(exc, (RedisTimeoutError, TimeoutError))


class ConceptRegionBreaker:
    def __init__(
        self,
        *,
        threshold: int = BREAKER_TIMEOUT_THRESHOLD,
        cooldown_s: float = BREAKER_COOLDOWN_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = max(1, int(threshold))
        self._cooldown_s = float(cooldown_s)
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive_timeouts = 0
        self._open_until = 0.0
        self.trips = 0
        self.timeouts = 0
        self.skipped = 0

    def is_open(self) -> bool:
        with self._lock:
            return self._clock() < self._open_until

    def record_success(self) -> None:
        with self._lock:
            if self._consecutive_timeouts:
                logger.info(
                    "recall_concept_region_breaker_closed after_consecutive_timeouts=%s",
                    self._consecutive_timeouts,
                )
            self._consecutive_timeouts = 0

    def record_timeout(self) -> None:
        with self._lock:
            self.timeouts += 1
            self._consecutive_timeouts += 1
            if self._consecutive_timeouts >= self._threshold and self._clock() >= self._open_until:
                self.trips += 1
                self._open_until = self._clock() + self._cooldown_s
                logger.warning(
                    "recall_concept_region_breaker_open consecutive_timeouts=%s cooldown_s=%.0f trips=%s timeouts_total=%s",
                    self._consecutive_timeouts,
                    self._cooldown_s,
                    self.trips,
                    self.timeouts,
                )

    def record_skip(self) -> None:
        with self._lock:
            self.skipped += 1
            remaining = max(0.0, self._open_until - self._clock())
            skipped = self.skipped
        logger.info(
            "recall_concept_region_breaker_skip skipped_total=%s reopen_in_s=%.1f", skipped, remaining
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "open": self._clock() < self._open_until,
                "consecutive_timeouts": self._consecutive_timeouts,
                "timeouts": self.timeouts,
                "trips": self.trips,
                "skipped": self.skipped,
            }


class _BreakerGuardedStore:
    """Delegates to the direct Falkor store; feeds every call's outcome to the
    breaker and refuses calls while it is open, so one turn's remaining reads
    (e.g. per-node reinforcement) do not each wait out a socket timeout."""

    def __init__(self, inner: Any, breaker: ConceptRegionBreaker) -> None:
        self._inner = inner
        self._breaker = breaker

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr
        breaker = self._breaker

        def _guarded(*args: Any, **kwargs: Any) -> Any:
            if breaker.is_open():
                raise ConceptRegionBreakerOpen(name)
            try:
                result = attr(*args, **kwargs)
            except Exception as exc:
                if _is_timeout(exc):
                    breaker.record_timeout()
                raise
            breaker.record_success()
            return result

        return _guarded


_BREAKER = ConceptRegionBreaker()

_STORE: Optional[Any] = None
_BUILT = False
# Construction does no network I/O, so this lock is only held for
# microseconds; it just keeps two first callers from building two handles.
_STORE_LOCK = threading.Lock()

_IN_MEMORY_BACKENDS = {"", "in_memory", "memory", "mem", "local"}
_FALKOR_BACKENDS = {"falkor", "falkordb"}


def _build_store() -> Optional[Any]:
    backend = str(os.getenv("SUBSTRATE_STORE_BACKEND", "")).strip().lower()
    if backend in _FALKOR_BACKENDS:
        direct = build_falkor_direct_concept_store_from_env(
            socket_timeout_s=FALKOR_SOCKET_TIMEOUT_S,
            socket_connect_timeout_s=FALKOR_SOCKET_CONNECT_TIMEOUT_S,
        )
        return _BreakerGuardedStore(direct, _BREAKER) if direct is not None else None
    if backend in _IN_MEMORY_BACKENDS:
        return InMemorySubstrateGraphStore()
    logger.warning(
        "recall_substrate_store_unsupported_backend backend=%s reason=would_load_full_graph",
        backend,
    )
    return None


def get_substrate_store() -> Optional[Any]:
    """Return the process-level store handle, or None if unavailable.

    Never raises and never touches the network. Falkor being down shows up
    later, as a failed read inside the collector, which returns empty. While
    the timeout breaker is open this returns None (logged and counted), so
    concept_region is skipped for the turn.
    """
    global _STORE, _BUILT
    if _BREAKER.is_open():
        _BREAKER.record_skip()
        return None
    if _BUILT:
        return _STORE
    with _STORE_LOCK:
        if _BUILT:
            return _STORE
        try:
            _STORE = _build_store()
        except Exception as exc:  # noqa: BLE001 - never raise into a recall
            logger.warning("recall_substrate_store_init_failed error=%s", exc)
            _STORE = None
        _BUILT = True
        return _STORE


def breaker_stats() -> dict[str, Any]:
    return _BREAKER.stats()


def _reset_for_tests() -> None:
    global _STORE, _BUILT, _BREAKER
    _STORE = None
    _BUILT = False
    _BREAKER = ConceptRegionBreaker()


__all__ = ["get_substrate_store", "breaker_stats", "ConceptRegionBreaker"]
