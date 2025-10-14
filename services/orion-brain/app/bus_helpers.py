from typing import Dict, Optional
from orion.core.bus.service import OrionBus
from app.config import (
    ORION_BUS_URL, ORION_BUS_ENABLED,
    CHANNEL_BRAIN_OUT, CHANNEL_BRAIN_STATUS, CHANNEL_BRAIN_STREAM,
)

_bus: Optional[OrionBus] = None

def get_bus() -> OrionBus:
    """Lazy-load OrionBus instance using current config."""
    global _bus
    if _bus is None:
        _bus = OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)
    return _bus

async def emit_brain_event(event_type: str, data: Dict):
    """Publish a Brain status or internal event."""
    bus = get_bus()
    if not bus.enabled:
        return
    try:
        topic = CHANNEL_BRAIN_STATUS if event_type == "status" else CHANNEL_BRAIN_STREAM
        bus.publish(topic, data)
    except Exception as e:
        print(f"[bus] emit_brain_event failed: {e}")

async def emit_brain_output(data: Dict):
    """Publish a completed LLM inference result."""
    bus = get_bus()
    if not bus.enabled:
        return
    try:
        bus.publish(CHANNEL_BRAIN_OUT, data)
    except Exception as e:
        print(f"[bus] emit_brain_output failed: {e}")
