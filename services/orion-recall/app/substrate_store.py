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
# Serialises first-call construction. The boot warmup thread and a recall's
# concept_region thread can both reach get_substrate_store() while _STORE is
# still None; without this, both would run the slow Falkor hydration and one
# handle would be thrown away. The lock is only taken while _STORE is None.
_STORE_LOCK = threading.Lock()

# Upper bound on how long the boot warmup waits before logging and giving up
# waiting. The hydration thread itself cannot be killed and keeps going; if it
# later succeeds, _STORE is populated anyway. Measured cold hydration live
# 2026-09-30: 6.25s.
WARMUP_TIMEOUT_S = 30.0


def get_substrate_store() -> Optional[SubstrateGraphStore]:
    """Return (or lazily initialise) the process-level substrate store.

    Never raises: a construction failure is logged and the caller degrades
    to ``None`` (collectors reading from a ``None`` store return empty,
    never raise -- see ``collectors/concept_region.py``).
    """

    global _STORE
    store = _STORE
    if store is not None:
        return store
    with _STORE_LOCK:
        if _STORE is None:
            try:
                _STORE = build_substrate_store_from_env()
            except Exception as exc:
                logger.debug("recall_substrate_store_init_failed error=%s", exc)
                return None
        return _STORE


async def warm_substrate_store(*, timeout_s: float = WARMUP_TIMEOUT_S) -> bool:
    """Build the substrate store once, off the event loop, so the first
    purposeful recall after a restart does not pay the cold hydration.

    Best-effort: never raises, returns True only if a store handle exists
    afterwards. On timeout the hydration thread keeps running in the
    background (threads cannot be cancelled); a recall that races it waits on
    ``_STORE_LOCK`` inside its own thread, under the recall deadline, instead
    of building a second handle.
    """
    started = time.perf_counter()
    try:
        store = await asyncio.wait_for(asyncio.to_thread(get_substrate_store), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.warning("recall_substrate_store_warmup_timeout timeout_s=%s", timeout_s)
        return False
    except Exception as exc:
        logger.warning("recall_substrate_store_warmup_failed error=%s", exc)
        return False
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if store is None:
        logger.warning("recall_substrate_store_warmup_failed elapsed_ms=%s store=None", elapsed_ms)
        return False
    logger.info("recall_substrate_store_warmed elapsed_ms=%s", elapsed_ms)
    return True


__all__ = ["get_substrate_store", "warm_substrate_store"]
