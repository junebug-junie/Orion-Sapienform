import asyncio
import os
import sys
import threading
from typing import Callable, Dict, Any, Optional, Coroutine

from orion.core.bus.async_service import OrionBusAsync

from .settings import Settings


SERVICE_NAME = os.getenv("ORION_SERVICE_NAME", "orion-voip-endpoint")
NODE_NAME = os.getenv("ORION_NODE_NAME", "athena")


def _run_async(coro: Coroutine[Any, Any, Any]) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    loop.create_task(coro)


def init_bus(settings: Settings) -> Optional[OrionBusAsync]:
    """
    Initialize OrionBus from settings.bus_redis_url (VOIP_BUS_REDIS_URL).
    """
    if not settings.bus_redis_url:
        print("[VOIP] No bus_redis_url configured; bus disabled", flush=True)
        return None

    bus = OrionBusAsync(url=str(settings.bus_redis_url))
    _run_async(bus.connect())
    return bus


def make_bus_publish(bus: Optional[OrionBusAsync], settings: Settings) -> Callable[[str], None]:
    """
    Return a small helper that publishes structured events to VOIP_BUS_STATUS_CHANNEL.
    """

    def bus_publish(event: str, **extra: Any) -> None:
        if not bus or not bus.enabled:
            return
        payload = {
            "service": SERVICE_NAME,
            "node": NODE_NAME,
            "event": event,
            "sip_ext": settings.sip_ext,
            "lan_host_ip": str(settings.lan_host_ip),
            "tailscale_host_ip": str(settings.tailscale_host_ip),
            **extra,
        }
        async def _publish() -> None:
            try:
                await bus.publish(settings.bus_status_channel, payload)
            except Exception as e:
                print(f"[VOIP] Bus publish error: {e}", flush=True)

        _run_async(_publish())

    return bus_publish


async def bus_listener_loop(
    bus: OrionBusAsync,
    settings: Settings,
    bus_publish: Callable[[str], None],
    action_handlers: Dict[str, Callable[[], Any]],
) -> None:
    """
    Blocking loop: listens for commands on VOIP_BUS_COMMAND_CHANNEL and dispatches actions.
    """
    channel = settings.bus_command_channel
    if not channel:
        print("[VOIP] No bus_command_channel configured; listener not started", flush=True)
        return

    print(
        f"[VOIP] Starting bus listener on {channel} (url={settings.bus_redis_url})",
        flush=True,
    )

    try:
        async with bus.subscribe(channel) as pubsub:
            async for msg in bus.iter_messages(pubsub):
                data = msg.get("data") or {}
                if isinstance(data, (bytes, bytearray)):
                    try:
                        data = data.decode("utf-8", "ignore")
                    except Exception:
                        data = {}
                if isinstance(data, str):
                    try:
                        import json

                        data = json.loads(data)
                    except Exception:
                        data = {}
                action = data.get("action")
                print(f"[VOIP] Bus command received: {action} payload={data}", flush=True)

                try:
                    handler = action_handlers.get(action)
                    if handler is None:
                        bus_publish("unknown_command", action=action, raw=data)
                        continue
                    handler()
                except Exception as e:
                    bus_publish("command_error", action=action, error=str(e))
    except Exception as e:
        print(f"[VOIP] Bus listener crashed: {e}", flush=True)
        bus_publish("listener_crashed", error=str(e))


def start_bus_listener_thread(
    bus: Optional[OrionBusAsync],
    settings: Settings,
    bus_publish: Callable[[str], None],
    action_handlers: Dict[str, Callable[[], Any]],
) -> None:
    """
    Start the bus listener in a daemon thread if the bus is enabled.
    """
    if not bus or not bus.enabled:
        print("[VOIP] Bus disabled; listener not started", flush=True)
        return

    t = threading.Thread(
        target=lambda: asyncio.run(bus_listener_loop(bus, settings, bus_publish, action_handlers)),
        daemon=True,
    )
    t.start()
    print("[VOIP] Bus listener thread started", flush=True)
