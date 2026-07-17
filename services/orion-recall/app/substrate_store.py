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

import logging
from typing import Optional

from orion.substrate import build_substrate_store_from_env
from orion.substrate.store import SubstrateGraphStore

logger = logging.getLogger(__name__)

_STORE: Optional[SubstrateGraphStore] = None


def get_substrate_store() -> Optional[SubstrateGraphStore]:
    """Return (or lazily initialise) the process-level substrate store.

    Never raises: a construction failure is logged and the caller degrades
    to ``None`` (collectors reading from a ``None`` store return empty,
    never raise -- see ``collectors/concept_region.py``).
    """

    global _STORE
    if _STORE is None:
        try:
            _STORE = build_substrate_store_from_env()
        except Exception as exc:
            logger.debug("recall_substrate_store_init_failed error=%s", exc)
            return None
    return _STORE


__all__ = ["get_substrate_store"]
