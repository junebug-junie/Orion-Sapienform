from __future__ import annotations

import asyncio

from loguru import logger

from app.bus_observer import run_bus_observer_loop
from app.settings import settings


def main() -> None:
    if not settings.bus_observer_enabled:
        logger.info("bus-observer disabled (BUS_OBSERVER_ENABLED=false); exiting")
        return
    asyncio.run(run_bus_observer_loop())


if __name__ == "__main__":
    main()
