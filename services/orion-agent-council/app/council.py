from __future__ import annotations

import logging
from typing import Dict

from orion.core.bus.service import OrionBus

from .deliberation import DeliberationEngine
from .settings import settings

logger = logging.getLogger("agent-council.council")


class CouncilRouter:
    """
    Very thin wrapper:
      - owns DeliberationEngine
      - decides which events are 'council' events
      - delegates to engine.run()
    """

    def __init__(self, bus: OrionBus) -> None:
        self.bus = bus
        self.engine = DeliberationEngine(bus)

    def handle_message(self, data: Dict) -> None:
        event = data.get("event")
        if event != "council_deliberation":
            logger.warning("Ignoring non-deliberation event on council: %r", event)
            return

        self.engine.run(data)


def run_council_loop(bus: OrionBus) -> None:
    router = CouncilRouter(bus)

    logger.info(
        "Starting council loop on %s (service=%s v%s)",
        settings.channel_intake,
        settings.service_name,
        settings.service_version,
    )

    for msg in bus.subscribe(settings.channel_intake):
        if msg.get("type") != "message":
            continue

        data = msg.get("data")
        if not isinstance(data, dict):
            logger.warning("Non-dict message on %s: %r", msg.get("channel"), data)
            continue

        try:
            router.handle_message(data)
        except Exception as e:
            logger.error("Error handling council message: %s data=%r", e, data, exc_info=True)
