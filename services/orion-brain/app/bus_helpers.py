from typing import Dict
from orion.core.bus.service import OrionBus
from app.config import (
    ORION_BUS_URL, ORION_BUS_ENABLED,
    CHANNEL_BRAIN_OUT, CHANNEL_BRAIN_STATUS, CHANNEL_BRAIN_STREAM,
)

# NOTE: No global _bus variable. We create it on-demand for thread-safety.

def emit_brain_event(event_type: str, data: Dict):
    """Publish a Brain status or internal event."""
    bus = OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)
    if not bus.enabled:
        return
    try:
        topic = CHANNEL_BRAIN_STATUS if event_type == "status" else CHANNEL_BRAIN_STREAM
        bus.publish(topic, data)
    except Exception as e:
        print(f"[bus] emit_brain_event failed: {e}")

def emit_brain_output(data: Dict):
    """Publish a completed LLM inference result."""
    bus = OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)
    if not bus.enabled:
        return
    try:
        bus.publish(CHANNEL_BRAIN_OUT, data)
    except Exception as e:
        print(f"[bus] emit_brain_output failed: {e}")
