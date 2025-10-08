# orion/core/bus/telemetry.py

import time
import json
from datetime import datetime
from orion.core.bus.service import OrionBus

def start_telemetry_loop(
    publish_channel: str,
    get_payload_func,
    bus_url: str,
    interval: int = 30,
    label: str = "telemetry"
):
    """
    Periodically invokes `get_payload_func()` and publishes the result
    to a Redis bus using OrionBus on `publish_channel`.

    Args:
        publish_channel: Redis channel name to publish to
        get_payload_func: Callable that returns a serializable dict
        bus_url: Redis connection URI
        interval: Seconds between broadcasts
        label: Optional name for log output clarity
    """
    print(f"🛰️  Telemetry loop [{label}] → {publish_channel} every {interval}s")
    bus = OrionBus(url=bus_url)

    while True:
        try:
            payload = get_payload_func()
            bus.publish(publish_channel, json.dumps(payload))
            print(f"[{datetime.utcnow().isoformat()}] 📡 {label}: published telemetry to {publish_channel}")
        except Exception as e:
            print(f"❌ Telemetry loop [{label}] error: {e}")
        time.sleep(interval)
