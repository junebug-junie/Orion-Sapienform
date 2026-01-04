# services/orion-agent-council/app/main.py
from __future__ import annotations

import asyncio
import logging
import sys

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope

from .settings import settings
from .council import CouncilService

# Configure logging to stdout for Docker/K8s
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agent-council.main")


class CouncilHunter(Hunter):
    """
    Custom Hunter that initializes the CouncilService logic
    once the bus is ready.
    """
    def __init__(self, config: ChassisConfig):
        super().__init__(
            cfg=config,
            handler=self.handle_message,
            # We listen on the specific intake channel defined in settings
            pattern=settings.channel_intake
        )
        self.logic: CouncilService | None = None

    async def handle_message(self, env: BaseEnvelope) -> None:
        # Lazy initialization ensures 'self.bus' is connected and ready
        if not self.logic:
            self.logic = CouncilService(self.bus)
        
        await self.logic.handle_envelope(env)


async def main_async():
    logger.info("Initializing Agent Council (Titanium Chassis)...")

    config = ChassisConfig(
        service_name=settings.service_name,
        service_version=settings.service_version,
        node_name=settings.node_name,
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
    )

    chassis = CouncilHunter(config)
    await chassis.start()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Fatal error in Agent Council")
        sys.exit(1)


if __name__ == "__main__":
    main()
