"""Best-effort Pub/Sub facts, emitted only after the corresponding durable write."""
import logging

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.reading import ReadingLifecycleV1

REQUESTED_CHANNEL = "orion:reading:requested"
LIFECYCLE_CHANNEL = "orion:reading:lifecycle"
TOOL_CHANNEL = "orion:reading:tool:request"
TOOL_RESULT_PREFIX = "orion:reading:tool:result:"
logger = logging.getLogger(__name__)


async def _publish(bus, channel, kind, payload, source):
    if bus is None:
        return
    try:
        await bus.publish(channel, BaseEnvelope(
            kind=kind, source=source or ServiceRef(name="orion-hub"),
            correlation_id=payload.request_id if hasattr(payload, "request_id") else payload.request.request_id,
            payload=payload.model_dump(mode="json"),
        ))
    except Exception:
        logger.warning("reading_event_publish_failed channel=%s", channel, exc_info=True)


async def publish_accepted(bus, request, *, source=None):
    await _publish(bus, REQUESTED_CHANNEL, "reading.requested.v1", request, source)


async def publish_lifecycle(bus, seed, stage, *, source=None, trace_id=None, error=None):
    from orion.world_pulse_read.queue import request_for_seed
    await _publish(bus, LIFECYCLE_CHANNEL, "reading.lifecycle.v1", ReadingLifecycleV1(
        request=request_for_seed(seed), seed_id=seed.seed_id, stage=stage,
        trace_id=trace_id, error=error,
    ), source)
