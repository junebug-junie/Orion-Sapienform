from __future__ import annotations

import asyncio
import logging
import signal
import sys

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly

from .settings import Settings, get_settings
from .worker import run_consumer

logger = logging.getLogger("orion-curiosity-peer")


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[CURIOSITY_PEER] %(asctime)s %(levelname)s - %(name)s - %(message)s"
        )
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def build_heartbeat_chassis(settings: Settings) -> HeartbeatOnly:
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            node_name=settings.CURIOSITY_PEER_NODE_NAME,
            bus_url=settings.ORION_BUS_URL,
            bus_enabled=settings.ORION_BUS_ENABLED,
            heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        )
    )


async def main_async() -> None:
    setup_logging()
    settings = get_settings()
    stop = asyncio.Event()

    def _handle_signal() -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass

    if not settings.CURIOSITY_PEER_ENABLED:
        logger.warning(
            "curiosity_peer_disabled idle "
            "(set CURIOSITY_PEER_ENABLED=true; Hub HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED "
            "must also be true at enqueue time)"
        )
        await stop.wait()
        return

    heartbeat = build_heartbeat_chassis(settings)
    if settings.ORION_BUS_ENABLED:
        await heartbeat.start_background()

    bus = OrionBusAsync(
        settings.ORION_BUS_URL,
        enabled=settings.ORION_BUS_ENABLED,
        enforce_catalog=settings.ORION_BUS_ENFORCE_CATALOG,
    )
    try:
        if settings.ORION_BUS_ENABLED:
            await bus.connect()
            logger.info(
                "curiosity_peer_listening channel=%s model=%s",
                settings.CHANNEL_HELP_REQUEST,
                settings.CURIOSITY_PEER_MODEL,
            )
            await run_consumer(settings, bus, stop)
        else:
            logger.warning("orion_bus_disabled_idle")
            await stop.wait()
    finally:
        if settings.ORION_BUS_ENABLED:
            await heartbeat.stop()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
