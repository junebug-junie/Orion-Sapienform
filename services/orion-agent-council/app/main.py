# services/orion-agent-council/app/main.py
from __future__ import annotations

import logging

from orion.core.bus.service import OrionBus

from .settings import settings
from .council import run_council_loop

logging.basicConfig(
    level=logging.INFO,
    format="[AGENT-COUNCIL] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("agent-council.main")


def main() -> None:
    logger.info(
        "Starting Agent Council worker (service=%s v=%s)",
        settings.service_name,
        settings.service_version,
    )

    bus = OrionBus(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )

    if not bus.enabled:
        logger.error("Orion bus is disabled; council loop will not start.")
        return

    logger.info("Connecting to Orion bus at %s", settings.orion_bus_url)
    try:
        # Blocking loop – stays alive until container is stopped
        run_council_loop(bus)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down Agent Council.")
    except Exception as e:
        logger.exception("Agent Council crashed with an unexpected error: %s", e)


if __name__ == "__main__":
    main()
