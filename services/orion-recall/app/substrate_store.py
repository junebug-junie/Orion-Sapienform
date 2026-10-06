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
from typing import Any, Optional

from orion.substrate.falkor_direct import build_falkor_direct_concept_store_from_env
from orion.substrate.store import InMemorySubstrateGraphStore

logger = logging.getLogger(__name__)

# Redis socket timeouts for this service's Falkor clients. Live 2026-10-06
# from the host against orion_substrate (868 concepts, 37,772 edges): the
# ranking read takes ~15ms, the edge-cut read ~100-130ms. 5s is ~40x the
# slowest; a hung FalkorDB cannot pin a recall worker thread past it.
FALKOR_SOCKET_TIMEOUT_S = 5.0
FALKOR_SOCKET_CONNECT_TIMEOUT_S = 2.0

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
        return build_falkor_direct_concept_store_from_env(
            socket_timeout_s=FALKOR_SOCKET_TIMEOUT_S,
            socket_connect_timeout_s=FALKOR_SOCKET_CONNECT_TIMEOUT_S,
        )
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
    later, as a failed read inside the collector, which returns empty.
    """
    global _STORE, _BUILT
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


def _reset_for_tests() -> None:
    global _STORE, _BUILT
    _STORE = None
    _BUILT = False


__all__ = ["get_substrate_store"]
