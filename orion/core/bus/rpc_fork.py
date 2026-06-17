"""Helpers for isolating RPC reply listeners from long-lived bus subscribers."""

from __future__ import annotations

import logging

from .async_service import OrionBusAsync

logger = logging.getLogger("orion.bus.rpc_fork")


async def fork_rpc_client(parent: OrionBusAsync) -> OrionBusAsync:
    """
    Fork a dedicated bus client with an RPC worker for outbound request/reply traffic.

    Use when the parent bus instance also runs long-lived subscribe()/listen() loops
    (Hunter, trace caches, etc.) so reply messages are not stolen or dropped.
    """
    child = await parent.fork(start_rpc_worker=True)
    logger.info("[rpc-fork] fork_rpc_client ready enabled=%s url=%s", child.enabled, child.url)
    return child
