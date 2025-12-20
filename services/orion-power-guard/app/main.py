from __future__ import annotations

import asyncio
import logging
import sys
import time
from typing import Optional

from orion.core.bus.service import OrionBus

from .models import PowerEvent
from .settings import get_settings
# [CHANGED] Use the NIS client (USB) instead of SNMP
from .ups_nis_client import NISUPSClient

logger = logging.getLogger("orion-power-guard")


# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────

def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[POWER_GUARD] %(asctime)s %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


# ─────────────────────────────────────────────
# Core monitor loop
# ─────────────────────────────────────────────

async def monitor_ups() -> None:
    settings = get_settings()

    logger.info(
        "Starting Orion Power Guard (USB Mode) — service=%s version=%s node=%s ups=%s host=%s",
        settings.SERVICE_NAME,
        settings.SERVICE_VERSION,
        settings.POWER_GUARD_NODE_NAME,
        settings.POWER_GUARD_UPS_NAME,
        settings.POWER_GUARD_UPS_HOST,
    )

    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    
    # [CHANGED] Initialize NIS Client
    # We use port 3551 (apcupsd default)
    ups = NISUPSClient(
        host=settings.POWER_GUARD_UPS_HOST,
        port=3551
    )

    poll_interval = settings.POWER_GUARD_POLL_INTERVAL_SEC
    grace_sec = settings.POWER_GUARD_ONBATTERY_GRACE_SEC

    on_battery_started_monotonic: Optional[float] = None
    last_on_battery: bool = False
    grace_event_sent: bool = False

    while True:
        try:
            status = await ups.get_status()
        except Exception:
            logger.exception("Failed to read UPS status via NIS; will retry.")
            await asyncio.sleep(poll_interval)
            continue

        now_mono = time.monotonic()

        # Visible poll line each cycle so you can confirm readings
        logger.info(
            "UPS poll: raw=%s on_battery=%s charge=%s%% volts=%s",
            status.raw_status,
            status.on_battery,
            status.battery_charge_pct,
            status.line_voltage
        )

        # Transition ONLINE -> ONBATT
        if status.on_battery and not last_on_battery:
            on_battery_started_monotonic = now_mono
            grace_event_sent = False

            event = PowerEvent(
                kind="power.guard.on_battery",
                node=settings.POWER_GUARD_NODE_NAME,
                ups_name=settings.POWER_GUARD_UPS_NAME,
                status=status,
                details={"message": "UPS switched to battery"},
            )
            _publish_event(bus, settings.CHANNEL_POWER_EVENTS, event)
            logger.warning(
                "UPS switched to battery for node=%s; starting grace timer (%.1fs).",
                settings.POWER_GUARD_NODE_NAME,
                grace_sec,
            )

        # While ONBATT, check if grace period elapsed
        if status.on_battery and on_battery_started_monotonic is not None:
            elapsed = now_mono - on_battery_started_monotonic
            if elapsed >= grace_sec and not grace_event_sent:
                grace_event_sent = True

                event = PowerEvent(
                    kind="power.guard.grace_elapsed",
                    node=settings.POWER_GUARD_NODE_NAME,
                    ups_name=settings.POWER_GUARD_UPS_NAME,
                    status=status,
                    details={
                        "message": (
                            "UPS on battery beyond grace period; "
                            "consider graceful shutdown"
                        ),
                        "elapsed_sec": elapsed,
                    },
                )
                _publish_event(bus, settings.CHANNEL_POWER_EVENTS, event)

                logger.error(
                    "UPS on battery beyond grace period (elapsed=%.1fs >= %.1fs).",
                    elapsed,
                    grace_sec,
                )

                if settings.POWER_GUARD_ENABLE_SHUTDOWN:
                    _run_shutdown(settings.POWER_GUARD_SHUTDOWN_CMD)

        # Transition ONBATT -> ONLINE
        if not status.on_battery and last_on_battery:
            on_battery_started_monotonic = None
            grace_event_sent = False

            event = PowerEvent(
                kind="power.guard.restored",
                node=settings.POWER_GUARD_NODE_NAME,
                ups_name=settings.POWER_GUARD_UPS_NAME,
                status=status,
                details={"message": "Utility power restored"},
            )
            _publish_event(bus, settings.CHANNEL_POWER_EVENTS, event)
            logger.info(
                "Utility power restored for node=%s.", settings.POWER_GUARD_NODE_NAME
            )

        last_on_battery = status.on_battery

        await asyncio.sleep(poll_interval)


def _publish_event(bus: OrionBus, channel: str, event: PowerEvent) -> None:
    if not getattr(bus, "enabled", False):
        logger.info("Bus disabled; skipping publish for event kind=%s", event.kind)
        return

    # Use model_dump(mode='json') to serialize datetime objects safely
    payload = event.model_dump(mode='json')
    try:
        bus.publish(channel, payload)
        logger.info("Published power event kind=%s channel=%s", event.kind, channel)
    except Exception:
        logger.exception(
            "Failed to publish power event kind=%s channel=%s", event.kind, channel
        )


def _run_shutdown(cmd: str) -> None:
    import subprocess

    logger.warning("Executing shutdown command: %s", cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.warning("Shutdown command executed successfully.")
    except subprocess.CalledProcessError as exc:
        logger.error("Shutdown command failed: %s", exc)
    except Exception:
        logger.exception("Unexpected error while running shutdown command.")


async def _main_async() -> None:
    await monitor_ups()


def main() -> None:
    setup_logging()
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        logger.info("Power Guard service interrupted; exiting.")


if __name__ == "__main__":
    main()
