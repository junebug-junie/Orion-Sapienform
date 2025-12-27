from __future__ import annotations

import asyncio
import logging

from orion.core.bus.bus_schemas import ExecutionEnvelopeV1
from orion.core.bus.bus_service_chassis import ChassisConfig, Rabbit

from .handlers import handle_llm_request
from .settings import settings


async def amain() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[LLM-GW] %(levelname)s - %(name)s - %(message)s",
    )

    cfg = ChassisConfig(
        service_name=settings.llm_service_name,
        service_version=getattr(settings, "service_version", "0.1.0"),
        bus_url=settings.orion_bus_url,
        bus_enabled=settings.orion_bus_enabled,
    )

    svc = Rabbit[
        ExecutionEnvelopeV1,
        dict,
    ](
        cfg,
        intake_channel=settings.channel_llm_intake,
        request_model=ExecutionEnvelopeV1,
        handler=handle_llm_request,
    )

    await svc.run()


if __name__ == "__main__":
    asyncio.run(amain())
