# orion/core/bus/telemetry.py

import asyncio
from datetime import datetime

from orion.core.bus.async_service import OrionBusAsync

def start_telemetry_loop(
    publish_channel: str,
    get_payload_func,
    bus_url: str,
    interval: int = 30,
    label: str = "telemetry"
):
    """
    Periodically invokes `get_payload_func()` and publishes the result
    to a Redis bus using OrionBusAsync on `publish_channel`.

    Args:
        publish_channel: Redis channel name to publish to
        get_payload_func: Callable that returns a serializable dict
        bus_url: Redis connection URI
        interval: Seconds between broadcasts
        label: Optional name for log output clarity
    """
    print(f"🛰️  Telemetry loop [{label}] → {publish_channel} every {interval}s")

    async def _loop() -> None:
        bus = OrionBusAsync(url=bus_url)
        await bus.connect()
        try:
            while True:
                try:
                    payload = get_payload_func()
                    await bus.publish(publish_channel, payload)
                    print(f"[{datetime.utcnow().isoformat()}] 📡 {label}: published telemetry to {publish_channel}")
                except Exception as e:
                    print(f"❌ Telemetry loop [{label}] error: {e}")
                await asyncio.sleep(interval)
        finally:
            await bus.close()

    asyncio.run(_loop())
