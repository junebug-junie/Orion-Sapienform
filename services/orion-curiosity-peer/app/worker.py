from __future__ import annotations

import logging
from typing import Any

from .settings import Settings

logger = logging.getLogger("orion-curiosity-peer.worker")


async def handle_help_request(settings: Settings, raw: Any) -> None:
    """Process one HelpRequestV1. Full path lands in Tasks 12–13."""
    _ = (settings, raw)
    logger.debug("curiosity_peer_handle_noop scaffold")


async def run_consumer(settings: Settings, bus: Any, stop: Any) -> None:
    """Subscribe to help requests; no-op handler until invoker is wired."""
    channel = settings.CHANNEL_HELP_REQUEST
    async with bus.subscribe(channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            if stop.is_set():
                break
            try:
                data = msg.get("data")
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                # Scaffold: acknowledge traffic without invoking peers.
                logger.info(
                    "curiosity_peer_help_seen channel=%s bytes=%s (scaffold noop)",
                    channel,
                    len(data) if isinstance(data, str) else 0,
                )
                await handle_help_request(settings, data)
            except Exception:
                logger.exception("curiosity_peer_handle_failed")
