"""Lazily-initialized process-level FalkorDB client for the ``orion_recall`` graph.

Mirrors ``substrate_store.py``'s pattern exactly: env-driven backend
selection, never raises on init failure, one singleton per process. A
dedicated module rather than reusing ``substrate_store.py`` because that one
is scoped to the ``orion_substrate`` graph (a different FalkorDB graph on the
same shared instance) -- keeping them separate avoids one caller accidentally
querying the wrong graph.

FALKORDB_URI is read directly from the environment (not via this service's
own Settings class), same convention already used for
FALKORDB_SUBSTRATE_GRAPH in substrate_store.py / .env_example -- this
service's connection-info env vars intentionally bypass pydantic Settings.
FALKORDB_RECALL_GRAPH reuses the exact env var name orion-meta-tags (the
actual producer of this graph) already uses.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from orion.graph.falkor_client import RedisGraphQueryClient

logger = logging.getLogger(__name__)

_CLIENT: Optional[RedisGraphQueryClient] = None


def get_recall_falkor_client() -> Optional[RedisGraphQueryClient]:
    """Return (or lazily initialise) the process-level ``orion_recall`` Falkor client.

    Never raises: a construction failure is logged and the caller degrades
    to ``None`` (fetch_falkor_chatturn_fragments returns [] on a None client).
    Retries construction on every call while uninitialized -- same as
    substrate_store.py's get_substrate_store() -- so a transient failure
    (FalkorDB briefly unreachable at first use) can self-heal on a later call
    rather than staying permanently None for the life of the process.
    """

    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    uri = os.getenv("FALKORDB_URI", "").strip()
    graph_name = os.getenv("FALKORDB_RECALL_GRAPH", "orion_recall").strip()
    if not uri:
        logger.debug("recall_falkor_store_init_skipped reason=no_falkordb_uri")
        return None
    try:
        _CLIENT = RedisGraphQueryClient(uri=uri, graph_name=graph_name)
    except Exception as exc:
        logger.debug("recall_falkor_store_init_failed error=%s", exc)
        return None
    return _CLIENT


__all__ = ["get_recall_falkor_client"]
