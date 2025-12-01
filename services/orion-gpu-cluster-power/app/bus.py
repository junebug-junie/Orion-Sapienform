# app/bus.py

import asyncio
import logging
import threading

from .service import PsuService
from .settings import settings

logger = logging.getLogger("gpu-cluster-power.bus")


def _command_listener_sync(service: PsuService, loop: asyncio.AbstractEventLoop) -> None:
    """
    Blocking bus listener that runs in a background thread.
    Uses OrionBus.subscribe(...) (sync generator) and forwards actions
    into the asyncio loop as coroutines.
    """
    if not service.bus_enabled or not service.bus:
        logger.info("PSU command listener not started: bus disabled or not initialized.")
        return

    channel = settings.bus_channel_psu_command
    logger.info("Starting PSU command listener on channel: %s", channel)

    # OrionBus.subscribe(*channels) → sync generator of message dicts
    for msg in service.bus.subscribe(channel):
        try:
            data = msg.get("data", {}) or {}
            action = data.get("action")
            mode = data.get("mode")  # optional, may be None
        except Exception:
            logger.warning("Invalid PSU command payload: %r", msg)
            continue

        if action not in ("on", "off", "cycle"):
            logger.warning("Unknown PSU action: %s", action)
            continue

        logger.info("PSU bus command received: %s (mode=%s)", action, mode)

        fut = asyncio.run_coroutine_threadsafe(
            service.handle_command(action, mode=mode),
            loop,
        )
        try:
            fut.result()  # or drop this if you want fire-and-forget
        except Exception as e:
            logger.error("Error executing PSU action %s: %s", action, e)


def start_command_listener_background(service: PsuService) -> None:
    """
    Capture the current asyncio loop and start the blocking listener
    in a daemon thread.
    """
    if not service.bus_enabled:
        logger.info("Bus not enabled; not starting command listener.")
        return

    loop = asyncio.get_running_loop()

    t = threading.Thread(
        target=_command_listener_sync,
        args=(service, loop),
        daemon=True,
    )
    t.start()
    logger.info("PSU command listener thread started.")
