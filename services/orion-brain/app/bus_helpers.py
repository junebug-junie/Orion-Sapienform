from typing import Dict
from orion.core.bus import OrionBus
from app.config import REDIS_URL, EVENTS_ENABLE, BUS_OUT_ENABLE

# Global bus instance
bus = OrionBus(url=REDIS_URL)

async def emit_event(kind: str, fields: Dict):
    """Publish telemetry event to the Orion event bus."""
    if not EVENTS_ENABLE or not bus.enabled:
        return
    try:
        bus.publish(f"events.{kind}", fields)
    except Exception:
        pass

async def emit_bus(topic: str, content: Dict):
    """Publish structured model response or message to the bus output stream."""
    if not BUS_OUT_ENABLE or not bus.enabled:
        return
    try:
        bus.publish(topic, content)
    except Exception:
        pass
