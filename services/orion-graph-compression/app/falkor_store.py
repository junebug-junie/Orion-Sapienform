"""Lazily-initialized process-level FalkorDB clients for graph-compression's
federators.

Mirrors ``orion-recall/app/recall_falkor_store.py``'s pattern exactly:
env-driven backend selection, never raises on init failure, one singleton
per graph per process. FALKORDB_URI/FALKORDB_SUBSTRATE_GRAPH/
FALKORDB_RECALL_GRAPH are read directly from the environment (not via this
service's own Settings class), reusing the exact env var names
substrate-runtime and orion-recall/orion-meta-tags already use for the same
shared FalkorDB instance -- one Redis endpoint, three separate named graphs.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from orion.graph.falkor_client import RedisGraphQueryClient

logger = logging.getLogger(__name__)

_SUBSTRATE_CLIENT: Optional[RedisGraphQueryClient] = None
_RECALL_CLIENT: Optional[RedisGraphQueryClient] = None


def _build_client(*, graph_env: str, default_graph: str, log_ctx: str) -> Optional[RedisGraphQueryClient]:
    uri = os.getenv("FALKORDB_URI", "").strip()
    graph_name = os.getenv(graph_env, default_graph).strip()
    if not uri:
        logger.debug("%s_init_skipped reason=no_falkordb_uri", log_ctx)
        return None
    try:
        return RedisGraphQueryClient(uri=uri, graph_name=graph_name)
    except Exception as exc:
        logger.debug("%s_init_failed error=%s", log_ctx, exc)
        return None


def get_substrate_falkor_client() -> Optional[RedisGraphQueryClient]:
    """Return (or lazily initialise) the process-level ``orion_substrate`` Falkor client.

    Retries construction on every call while uninitialized, so a transient
    failure (FalkorDB briefly unreachable at first use) can self-heal on a
    later tick rather than staying permanently None for the life of the
    process.
    """
    global _SUBSTRATE_CLIENT
    if _SUBSTRATE_CLIENT is not None:
        return _SUBSTRATE_CLIENT
    _SUBSTRATE_CLIENT = _build_client(
        graph_env="FALKORDB_SUBSTRATE_GRAPH",
        default_graph="orion_substrate",
        log_ctx="graph_compression_substrate_falkor",
    )
    return _SUBSTRATE_CLIENT


def get_recall_falkor_client() -> Optional[RedisGraphQueryClient]:
    """Return (or lazily initialise) the process-level ``orion_recall`` Falkor client."""
    global _RECALL_CLIENT
    if _RECALL_CLIENT is not None:
        return _RECALL_CLIENT
    _RECALL_CLIENT = _build_client(
        graph_env="FALKORDB_RECALL_GRAPH",
        default_graph="orion_recall",
        log_ctx="graph_compression_recall_falkor",
    )
    return _RECALL_CLIENT


__all__ = ["get_substrate_falkor_client", "get_recall_falkor_client"]
