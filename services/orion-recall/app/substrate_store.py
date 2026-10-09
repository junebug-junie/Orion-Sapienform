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
from orion.substrate.falkor_store import bootstrap_substrate_reader
from orion.substrate.store import InMemorySubstrateGraphStore

logger = logging.getLogger(__name__)

# Redis socket timeouts for this service's Falkor clients. Every read recall
# makes is bounded; the slowest single query observed live is ~0.3s from the
# host and ~0.7s in-container (a full 500/500 region read, not the hot path).
# 1.5s per socket read caps what a hung FalkorDB can cost one query; the
# breaker below caps what it can cost a run of turns.
FALKOR_SOCKET_TIMEOUT_S = 1.5
FALKOR_SOCKET_CONNECT_TIMEOUT_S = 1.0

# Circuit breaker over Falkor timeouts. Three states:
#   closed    -- calls go through; a timeout bumps the consecutive count, a
#                success resets it; BREAKER_TIMEOUT_THRESHOLD in a row opens it.
#   open      -- every call is refused (concept_region skipped, logged and
#                counted) until BREAKER_COOLDOWN_S has passed. Successes from
#                calls already in flight when it opened are ignored.
#   half-open -- after the cooldown exactly one call (the probe) is let
#                through; everyone else keeps skipping. The probe's success
#                closes the breaker; its timeout reopens it immediately for
#                another cooldown; any other error releases the probe slot so
#                the next call probes.
# No fallback of any kind -- a skipped turn simply has no concept_region
# fragments.
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


_CLOSED, _OPEN, _HALF_OPEN = "closed", "open", "half_open"


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
        self._state = _CLOSED
        self._consecutive_timeouts = 0
        self._open_until = 0.0
        self._probe_in_flight = False
        self.trips = 0
        self.timeouts = 0
        self.skipped = 0

    def _open_locked(self, reason: str) -> None:
        self._state = _OPEN
        self._probe_in_flight = False
        self._open_until = self._clock() + self._cooldown_s
        self.trips += 1
        logger.warning(
            "recall_concept_region_breaker_open reason=%s consecutive_timeouts=%s cooldown_s=%.0f trips=%s timeouts_total=%s",
            reason,
            self._consecutive_timeouts,
            self._cooldown_s,
            self.trips,
            self.timeouts,
        )

    def would_skip(self) -> bool:
        """Peek (no state change): True while open and cooling down, or while
        half-open with the probe already taken."""
        with self._lock:
            if self._state == _OPEN:
                return self._clock() < self._open_until
            return self._state == _HALF_OPEN and self._probe_in_flight

    def allow(self) -> bool:
        """Gate one call. Test-and-set under the lock: after the cooldown the
        first caller becomes the probe; concurrent callers are refused."""
        with self._lock:
            if self._state == _CLOSED:
                return True
            if self._state == _OPEN:
                if self._clock() < self._open_until:
                    return False
                self._state = _HALF_OPEN
                self._probe_in_flight = True
                logger.info("recall_concept_region_breaker_half_open probe=1")
                return True
            # half-open
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    def is_open(self) -> bool:
        return self.would_skip()

    def record_success(self) -> None:
        with self._lock:
            if self._state == _CLOSED:
                self._consecutive_timeouts = 0
            elif self._state == _HALF_OPEN:
                logger.info(
                    "recall_concept_region_breaker_closed after_consecutive_timeouts=%s",
                    self._consecutive_timeouts,
                )
                self._state = _CLOSED
                self._probe_in_flight = False
                self._consecutive_timeouts = 0
            # open: a success from a call already in flight does not mask it

    def record_timeout(self) -> None:
        with self._lock:
            self.timeouts += 1
            self._consecutive_timeouts += 1
            if self._state == _CLOSED and self._consecutive_timeouts >= self._threshold:
                self._open_locked("threshold")
            elif self._state == _HALF_OPEN:
                self._open_locked("probe_timeout")

    def record_other_failure(self) -> None:
        """A non-timeout error: Falkor answered, so it says nothing about a
        hang. In half-open it just frees the probe slot."""
        with self._lock:
            if self._state == _HALF_OPEN:
                self._probe_in_flight = False

    def record_skip(self) -> None:
        with self._lock:
            self.skipped += 1
            remaining = max(0.0, self._open_until - self._clock())
            skipped = self.skipped
            state = self._state
        logger.info(
            "recall_concept_region_breaker_skip state=%s skipped_total=%s reopen_in_s=%.1f",
            state,
            skipped,
            remaining,
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "open": self._state != _CLOSED,
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
            if not breaker.allow():
                raise ConceptRegionBreakerOpen(name)
            try:
                result = attr(*args, **kwargs)
            except Exception as exc:
                if _is_timeout(exc):
                    breaker.record_timeout()
                else:
                    breaker.record_other_failure()
                raise
            breaker.record_success()
            return result

        return _guarded


_BREAKER = ConceptRegionBreaker()

_STORE: Optional[Any] = None
_BUILT = False
# Construction does no network I/O, so this lock is only held for
# microseconds; it just keeps two first callers from building two handles.
# The node_id index bootstrap (a network call) runs in a background thread
# started after the build, outside this lock -- never on a turn's path.
_STORE_LOCK = threading.Lock()
_INDEX_THREAD: Optional[threading.Thread] = None

_IN_MEMORY_BACKENDS = {"", "in_memory", "memory", "mem", "local"}
_FALKOR_BACKENDS = {"falkor", "falkordb"}


def _build_store() -> Optional[Any]:
    backend = str(os.getenv("SUBSTRATE_STORE_BACKEND", "")).strip().lower()
    if backend in _FALKOR_BACKENDS:
        direct = build_falkor_direct_concept_store_from_env(
            socket_timeout_s=FALKOR_SOCKET_TIMEOUT_S,
            socket_connect_timeout_s=FALKOR_SOCKET_CONNECT_TIMEOUT_S,
            ensure_indexes=False,
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

    Never raises and never touches the network on the caller's thread: the
    build only creates (lazily-connecting) clients, and the node_id index
    bootstrap is fired once in a background daemon thread after the build,
    outside ``_STORE_LOCK``. Falkor being down shows up later, as a failed
    read inside the collector, which returns empty. While the timeout breaker
    is open (or half-open with its probe already taken) this returns None,
    logged and counted, so concept_region is skipped for the turn.
    """
    global _STORE, _BUILT
    if _BREAKER.would_skip():
        _BREAKER.record_skip()
        return None
    if _BUILT:
        return _STORE
    built_now = False
    with _STORE_LOCK:
        if _BUILT:
            return _STORE
        try:
            _STORE = _build_store()
        except Exception as exc:  # noqa: BLE001 - never raise into a recall
            logger.warning("recall_substrate_store_init_failed error=%s", exc)
            _STORE = None
        _BUILT = True
        built_now = True
        store = _STORE
    if built_now and store is not None and not isinstance(store, InMemorySubstrateGraphStore):
        _start_index_bootstrap()
    return store


def _start_index_bootstrap() -> None:
    """Fire-and-forget: ensure the SubstrateNode(node_id) index exists.
    Bounded by ensure_substrate_indexes' own client timeouts; never raises."""
    global _INDEX_THREAD
    uri = str(os.getenv("FALKORDB_URI", "")).strip()
    graph = str(os.getenv("FALKORDB_SUBSTRATE_GRAPH", "orion_substrate")).strip() or "orion_substrate"
    if not uri:
        return

    def _run() -> None:
        try:
            bootstrap_substrate_reader(uri, graph)
        except Exception as exc:  # noqa: BLE001 - background, best effort
            logger.warning("recall_substrate_index_bootstrap_failed error=%s", exc)

    _INDEX_THREAD = threading.Thread(target=_run, name="recall-substrate-index", daemon=True)
    _INDEX_THREAD.start()


def breaker_stats() -> dict[str, Any]:
    return _BREAKER.stats()


def _reset_for_tests() -> None:
    global _STORE, _BUILT, _BREAKER, _INDEX_THREAD
    _STORE = None
    _BUILT = False
    _BREAKER = ConceptRegionBreaker()
    _INDEX_THREAD = None


__all__ = ["get_substrate_store", "breaker_stats", "ConceptRegionBreaker"]
