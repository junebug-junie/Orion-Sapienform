from __future__ import annotations

import uuid
from typing import Any

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.harness_finalize import HarnessRunStepV1


def _envelope_correlation_id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        return uuid.uuid4()


async def publish_harness_run_step(
    bus: Any,
    *,
    correlation_id: str,
    step_index: int,
    step: dict[str, Any],
    channel: str,
    source_name: str = "orion-harness-governor",
) -> None:
    payload = HarnessRunStepV1(
        correlation_id=correlation_id,
        step_index=step_index,
        step=step,
    )
    envelope = BaseEnvelope(
        kind="harness.run.step.v1",
        source=ServiceRef(name=source_name),
        correlation_id=_envelope_correlation_id(correlation_id),
        payload=payload.model_dump(mode="json"),
    )
    await bus.publish(channel, envelope)
