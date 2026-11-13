# app/bus.py

import asyncio
import logging

from orion.core.bus.resilience import run_subscription

from .service import PsuService
from .settings import settings

logger = logging.getLogger("gpu-cluster-power.bus")


async def _command_listener_async(service: PsuService) -> None:
    """
    Async bus listener that runs as a background task on the main loop.
    """
    if not service.bus_enabled or not service.bus:
        logger.info("PSU command listener not started: bus disabled or not initialized.")
        return

    channel = settings.bus_channel_psu_command
    logger.info("Starting PSU command listener on channel: %s", channel)

    async def handle(msg: dict) -> None:
        try:
            # Validate message type
            if msg.get("type") != "message":
                return

            raw_data = msg.get("data")
            if not raw_data:
                return

            # Decode using the bus codec (handles envelopes/JSON)
            decoded = service.bus.codec.decode(raw_data)
            if not decoded.ok:
                logger.warning("Failed to decode PSU command: %s", decoded.error)
                return

            payload = decoded.envelope.payload
            # Ensure payload is a dict before accessing fields
            if not isinstance(payload, dict):
                logger.warning("PSU command payload is not a dict: %r", payload)
                return

            action = payload.get("action")
            mode = payload.get("mode")  # optional
        except Exception:
            logger.warning("Invalid PSU command payload structure", exc_info=True)
            return

        if action not in ("on", "off", "cycle"):
            logger.warning("Unknown PSU action: %s", action)
            return

        logger.info("PSU bus command received: %s (mode=%s)", action, mode)

        # Awaited inline: one PSU action at a time, in arrival order.
        try:
            await service.handle_command(action, mode=mode)
        except Exception as e:
            logger.error("Error executing PSU action %s: %s", action, e)

    # Resubscribes with backoff after a Redis drop instead of ending the listener.
    try:
        await run_subscription(
            service.bus, channel, handle, stop_event=None, name="gpu-cluster-power-psu-command"
        )
    except asyncio.CancelledError:
        logger.info("PSU command listener task cancelled.")


def start_command_listener_background(service: PsuService) -> None:
    """
    Schedule the async listener on the running loop.
    """
    if not service.bus_enabled:
        logger.info("Bus not enabled; not starting command listener.")
        return

    # Simply create a background task on the current loop
    asyncio.create_task(_command_listener_async(service))
    logger.info("PSU command listener task scheduled.")
