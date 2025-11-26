import os
import sys
import threading
from typing import Callable, Dict, Any, Optional

from orion.core.bus.service import OrionBus

from .settings import Settings


SERVICE_NAME = os.getenv("ORION_SERVICE_NAME", "orion-voip-endpoint")
NODE_NAME = os.getenv("ORION_NODE_NAME", "athena")


def init_bus(settings: Settings) -> Optional[OrionBus]:
    """
    Initialize OrionBus from settings.bus_redis_url (VOIP_BUS_REDIS_URL).
    """
    if not settings.bus_redis_url:
        print("[VOIP] No bus_redis_url configured; bus disabled", flush=True)
        return None

    bus = OrionBus(url=str(settings.bus_redis_url))
    if not bus.enabled:
        print("[VOIP] OrionBus disabled after init", flush=True)
        return None

    return bus


def make_bus_publish(bus: Optional[OrionBus], settings: Settings) -> Callable[[str], None]:
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
        try:
            bus.publish(settings.bus_status_channel, payload)
        except Exception as e:
            print(f"[VOIP] Bus publish error: {e}", flush=True)

    return bus_publish


def bus_listener_loop(
    bus: OrionBus,
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
        for msg in bus.raw_subscribe(channel):
            data = msg.get("data") or {}
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
    bus: Optional[OrionBus],
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
        target=bus_listener_loop,
        args=(bus, settings, bus_publish, action_handlers),
        daemon=True,
    )
    t.start()
    print("[VOIP] Bus listener thread started", flush=True)
