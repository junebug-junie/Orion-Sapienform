"""Build Hub's substrate graph stores once, in the background, never on the event loop.

Why this exists (2026-10-20): Hub built its three Falkor-backed concept graph
stores (Orion, Self Atlas, AI Town) at import time in ``api_routes.py``. Each
``FalkorSubstrateStore`` constructor scans its whole graph into memory, so
uvicorn could not bind :8080 or start a single bus listener until every scan
finished (measured live: ~19 s Orion, ~26 s Self, ~113 s AI Town).

Same fix shape as orion-substrate-runtime PR #2784
(``services/orion-substrate-runtime/app/worker.py``
``_get_substrate_graph_store`` / ``_start_substrate_graph_store_warm``):

- each store is built exactly once, under a lock, in a daemon thread;
- import and startup never wait for the build;
- an event-loop-thread caller never builds inline;
- a caller that arrives before the store is ready gets an explicit
  not-ready answer (``SubstrateStoreNotReadyError`` / ``None`` from
  ``get_ready()``), never an empty cache dressed up as a real graph.

The holder is a thin proxy: once ready, attribute access delegates to the real
store, so objects that captured the holder at import time
(``GraphConsolidationEvaluator(store=...)`` etc.) keep working unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Iterable

logger = logging.getLogger("orion-hub.substrate_store_warm")

STATE_NOT_STARTED = "not_started"
STATE_WARMING = "warming"
STATE_READY = "ready"
STATE_FAILED = "failed"

# A failed build is retried by the next caller, but not more often than this.
RETRY_AFTER_FAILURE_SEC = 30.0


def on_event_loop_thread() -> bool:
    """True when called from a thread that is running an asyncio event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class SubstrateStoreNotReadyError(RuntimeError):
    """The store is still loading (or its build failed). Not an empty graph."""

    def __init__(self, name: str, state: str, error: str | None = None) -> None:
        self.store_name = name
        self.state = state
        self.error = error
        self.reason = "substrate_store_build_failed" if state == STATE_FAILED else "substrate_store_warming"
        detail = f"substrate store '{name}' not ready: {self.reason} (state={state})"
        if error:
            detail += f" error={error}"
        super().__init__(detail)

    def as_payload(self) -> dict[str, Any]:
        return {
            "available": False,
            "reason": self.reason,
            "store": self.store_name,
            "state": self.state,
            "error": self.error,
        }


class WarmingSubstrateStore:
    """One substrate store, built once off the event loop."""

    def __init__(self, name: str, builder: Callable[[], Any]) -> None:
        self._warm_name = name
        self._warm_builder = builder
        self._warm_store: Any = None
        self._warm_state = STATE_NOT_STARTED
        self._warm_error: str | None = None
        self._warm_started_mono: float | None = None
        self._warm_failed_mono: float | None = None
        self._warm_build_seconds: float | None = None
        self._warm_build_count = 0
        self._warm_lock = threading.Lock()
        self._warm_ready = threading.Event()

    # ── state ────────────────────────────────────────────────────────────
    @property
    def warm_name(self) -> str:
        return self._warm_name

    def is_ready(self) -> bool:
        return self._warm_store is not None

    def get_ready(self) -> Any:
        """The real store, or None while it is still loading. Never builds."""
        return self._warm_store

    def warm_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "store": self._warm_name,
            "state": self._warm_state,
            "ready": self._warm_store is not None,
            "build_seconds": self._warm_build_seconds,
            "error": self._warm_error,
        }
        if self._warm_store is not None:
            status["store_type"] = type(self._warm_store).__name__
        elif self._warm_started_mono is not None and self._warm_state == STATE_WARMING:
            status["warming_for_sec"] = round(time.monotonic() - self._warm_started_mono, 1)
        return status

    # ── building ─────────────────────────────────────────────────────────
    def _claim_build(self) -> bool:
        """Move to WARMING if nobody is building and nothing is built. Caller holds no lock."""
        with self._warm_lock:
            if self._warm_store is not None or self._warm_state == STATE_WARMING:
                return False
            if (self._warm_state == STATE_FAILED and self._warm_failed_mono is not None
                    and time.monotonic() - self._warm_failed_mono < RETRY_AFTER_FAILURE_SEC):
                return False
            self._warm_state = STATE_WARMING
            self._warm_started_mono = time.monotonic()
            self._warm_ready.clear()
            return True

    def _run_build(self) -> None:
        started = time.monotonic()
        try:
            store = self._warm_builder()
        except Exception as exc:  # builder failure must never take Hub down
            elapsed = time.monotonic() - started
            with self._warm_lock:
                self._warm_state = STATE_FAILED
                self._warm_error = f"{type(exc).__name__}: {exc}"
                self._warm_failed_mono = time.monotonic()
                self._warm_build_seconds = round(elapsed, 2)
            logger.exception("substrate_store_warm_failed store=%s secs=%.1f", self._warm_name, elapsed)
        else:
            elapsed = time.monotonic() - started
            with self._warm_lock:
                self._warm_store = store
                self._warm_state = STATE_READY
                self._warm_error = None
                self._warm_build_seconds = round(elapsed, 2)
                self._warm_build_count += 1
            logger.info(
                "substrate_store_ready store=%s type=%s secs=%.1f",
                self._warm_name, type(store).__name__, elapsed,
            )
        finally:
            self._warm_ready.set()

    def build_blocking(self) -> Any:
        """Build now on this thread if nobody else is. Refuses the event-loop thread."""
        if on_event_loop_thread():
            raise RuntimeError(f"substrate store '{self._warm_name}' must not be built on the event loop")
        if self._claim_build():
            self._run_build()
        return self._warm_store

    def start_warm(self) -> bool:
        """Build in a daemon thread, once. Returns True if this call started it."""
        if not self._claim_build():
            return False
        logger.info("substrate_store_warm_started store=%s", self._warm_name)
        threading.Thread(
            target=self._run_build, name=f"substrate-store-warm-{self._warm_name}", daemon=True,
        ).start()
        return True

    def wait_ready(self, timeout: float | None = None) -> Any:
        """Block until the build finishes (or timeout); return the store or None.

        For background threads only (e.g. the startup concept seed). Refuses the
        event-loop thread.
        """
        if on_event_loop_thread():
            raise RuntimeError(f"wait_ready on substrate store '{self._warm_name}' from the event loop")
        if self._warm_store is not None:
            return self._warm_store
        if self._warm_state == STATE_NOT_STARTED:
            self.start_warm()
        self._warm_ready.wait(timeout)
        return self._warm_store

    def require(self) -> Any:
        """The real store, or raise ``SubstrateStoreNotReadyError``. Never builds inline."""
        store = self._warm_store
        if store is not None:
            return store
        if self._warm_state in (STATE_NOT_STARTED, STATE_FAILED):
            self.start_warm()
        raise SubstrateStoreNotReadyError(self._warm_name, self._warm_state, self._warm_error)

    # ── proxy ────────────────────────────────────────────────────────────
    def __getattr__(self, attr: str) -> Any:
        # Only called for names not found on the holder itself.
        if attr.startswith("_warm"):
            raise AttributeError(attr)
        return getattr(self.require(), attr)

    def __repr__(self) -> str:
        return f"WarmingSubstrateStore(name={self._warm_name!r}, state={self._warm_state!r})"


def resolve_ready_store(store: Any) -> Any:
    """Real store for a holder (None while it loads); any other object unchanged."""
    if isinstance(store, WarmingSubstrateStore):
        return store.get_ready()
    return store


def store_type_name(store: Any) -> str:
    """Type name of the real store behind a holder ("warming" while it loads)."""
    if isinstance(store, WarmingSubstrateStore):
        ready = store.get_ready()
        return type(ready).__name__ if ready is not None else "warming"
    return type(store).__name__


def start_background_warm(holders: Iterable[WarmingSubstrateStore]) -> threading.Thread | None:
    """Build the given holders one after another in one daemon thread.

    Sequential on purpose: the first holder (Orion's own graph) is ready as
    soon as possible, and the three full scans do not compete for FalkorDB
    and the GIL at the same time. A holder already built or warming is skipped.
    """
    pending = [holder for holder in holders if not holder.is_ready()]
    if not pending:
        return None

    def _warm_all() -> None:
        for holder in pending:
            try:
                holder.build_blocking()
            except Exception:  # pragma: no cover - build_blocking already logs builder errors
                logger.exception("substrate_store_warm_unexpected store=%s", holder.warm_name)

    thread = threading.Thread(target=_warm_all, name="substrate-store-warm", daemon=True)
    thread.start()
    logger.info("substrate_store_background_warm_started stores=%s", ",".join(h.warm_name for h in pending))
    return thread


# How long a background consumer (startup seed, topic-foundry startup tick)
# waits for the stores before it gives up and reports not-ready. The slowest
# live scan (AI Town) measured ~113 s on 2026-10-20.
STARTUP_CONSUMER_WAIT_SEC = 900.0


def wait_all_ready(holders: Iterable[Any], timeout: float) -> list[str]:
    """Block (off the event loop) until every holder is ready or the shared
    deadline passes. Returns the names still not ready. Non-holders count as ready."""
    deadline = time.monotonic() + max(0.0, timeout)
    missing: list[str] = []
    for holder in holders:
        if not isinstance(holder, WarmingSubstrateStore):
            continue
        holder.wait_ready(max(0.0, deadline - time.monotonic()))
        if not holder.is_ready():
            missing.append(holder.warm_name)
    return missing


async def not_ready_exception_handler(_request: Any, exc: SubstrateStoreNotReadyError) -> Any:
    """FastAPI handler: a route touched a store that is still loading. Answer
    503 with the reason -- never an empty graph and never a bare 500."""
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=503, content=exc.as_payload(), headers={"Retry-After": "30"})
