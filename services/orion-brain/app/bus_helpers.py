from typing import Dict, Optional
from orion.core.bus.service import OrionBus
from app.config import REDIS_URL, EVENTS_ENABLE, BUS_OUT_ENABLE

_bus: Optional[OrionBus] = None

def get_bus() -> OrionBus:
    """
    Lazy-load OrionBus instance. Ensures we use the current REDIS_URL
    from environment (after all .env overlays are loaded).
    """
    global _bus
    if _bus is None:
        _bus = OrionBus(url=REDIS_URL)
    return _bus

async def emit_event(kind: str, fields: Dict):
    """Publish telemetry event to the Orion event bus."""
    if not EVENTS_ENABLE:
        return
    bus = get_bus()
    if not bus.enabled:
        return
    try:
        bus.publish(f"events.{kind}", fields)
    except Exception as e:
        print(f"[bus] emit_event failed: {e}")

async def emit_bus(topic: str, content: Dict):
    """Publish structured model response or message to the bus output stream."""
    if not BUS_OUT_ENABLE:
        return
    bus = get_bus()
    if not bus.enabled:
        return
    try:
        bus.publish(topic, content)
    except Exception as e:
        print(f"[bus] emit_bus failed: {e}")
