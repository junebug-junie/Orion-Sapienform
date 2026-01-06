from __future__ import annotations

import asyncio
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.spark.concept_induction.settings import ConceptSettings


async def main():
    settings = ConceptSettings()
    bus = OrionBusAsync(settings.orion_bus_url, enabled=settings.orion_bus_enabled)
    await bus.connect()

    corr = uuid4()
    env = BaseEnvelope(
        kind="chat.message",
        correlation_id=corr,
        source=ServiceRef(name="test-client", version="0.0.0", node=settings.node_name),
        payload={
            "content": "Juniper and Orion talked about self-reflection and planning a hike near the lake.",
            "user": "juniper",
            "role": "user",
        },
    )
    await bus.publish(settings.intake_channels[0], env)
    print(f"Published test chat event correlation_id={corr}")

    async with bus.subscribe(settings.profile_channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            decoded = bus.codec.decode(msg.get("data"))
            if not decoded.ok:
                continue
            env = decoded.envelope
            if env.correlation_id != corr:
                continue
            print("Received profile event:", env.payload)
            break

    await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
