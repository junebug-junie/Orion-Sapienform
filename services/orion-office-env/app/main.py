from __future__ import annotations

import asyncio
import logging
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.core.bus.codec import OrionCodec
from orion.schemas.telemetry.office_env import SAMPLE_KIND_V1, OfficeEnvSampleV1

from .sample import build_sample
from .serial_reader import FrameStore, run_reader
from .settings import Settings, get_settings

logger = logging.getLogger("orion-office-env")

_STORE = FrameStore()
_STATUS: dict[str, Any] = {
    "office_sensor": "unknown",
    "sample_age_sec": None,
    "serial_open": False,
    "frames_ok": 0,
    "lines_rejected": 0,
}


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[ORION_OFFICE_ENV] %(asctime)s %(levelname)s - %(name)s - %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def office_status() -> dict[str, Any]:
    """Heartbeat details: is the office reading real right now?"""
    return dict(_STATUS)


def make_sample(
    store: FrameStore, settings: Settings, now_fn: Callable[[], datetime] = _utcnow
) -> OfficeEnvSampleV1:
    frame, received_at = store.latest()
    sample = build_sample(
        frame=frame,
        received_at=received_at,
        now=now_fn(),
        stale_after_sec=settings.OFFICE_STALE_AFTER_SEC,
        node=settings.INSTANCE_ID,
        sensor_id=settings.OFFICE_SENSOR_ID,
    )
    _STATUS.update(
        office_sensor="stale" if sample.state.stale else "fresh",
        sample_age_sec=sample.provenance.sample_age_sec,
        serial_open=store.serial_open,
        frames_ok=store.frames_ok,
        lines_rejected=store.lines_rejected,
    )
    return sample


async def publish_sample(bus: OrionBusAsync, settings: Settings, sample: OfficeEnvSampleV1) -> None:
    if not getattr(bus, "enabled", False):
        logger.info("Bus disabled; skipping office sample publish")
        return
    envelope = BaseEnvelope(
        kind=SAMPLE_KIND_V1,
        source=ServiceRef(
            name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.INSTANCE_ID
        ),
        payload=sample.model_dump(mode="json", by_alias=True),
    )
    try:
        await bus.publish(settings.OFFICE_ENV_SAMPLE_CHANNEL, envelope)
        logger.info(
            "Published office sample temp_c=%s humidity_pct=%s stale=%s channel=%s",
            sample.measurements.office_temp_c,
            sample.measurements.office_humidity_pct,
            sample.state.stale,
            settings.OFFICE_ENV_SAMPLE_CHANNEL,
        )
    except Exception:
        logger.exception("Failed to publish office sample channel=%s", settings.OFFICE_ENV_SAMPLE_CHANNEL)


async def publish_loop() -> None:
    settings = get_settings()
    logger.info(
        "Starting orion-office-env service=%s port=%s interval_sec=%s",
        settings.SERVICE_NAME,
        settings.OFFICE_SERIAL_PORT,
        settings.OFFICE_PUBLISH_INTERVAL_SEC,
    )
    bus = OrionBusAsync(
        url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED, codec=OrionCodec()
    )
    if bus.enabled:
        await bus.connect()

    stop = threading.Event()
    reader = threading.Thread(
        target=run_reader,
        kwargs=dict(
            store=_STORE,
            port=settings.OFFICE_SERIAL_PORT,
            baud=settings.OFFICE_SERIAL_BAUD,
            stop=stop,
        ),
        name="office-serial-reader",
        daemon=True,
    )
    reader.start()
    try:
        while True:
            try:
                await publish_sample(bus, settings, make_sample(_STORE, settings))
            except Exception:
                logger.exception("Office publish cycle failed; will retry")
            await asyncio.sleep(settings.OFFICE_PUBLISH_INTERVAL_SEC)
    finally:
        stop.set()


def build_heartbeat_chassis(settings: Optional[Settings] = None) -> HeartbeatOnly:
    s = settings if settings is not None else get_settings()
    return HeartbeatOnly(
        ChassisConfig(
            service_name=s.SERVICE_NAME,
            service_version=s.SERVICE_VERSION,
            node_name=s.INSTANCE_ID,
            bus_url=s.ORION_BUS_URL,
            bus_enabled=s.ORION_BUS_ENABLED,
            heartbeat_interval_sec=s.HEARTBEAT_INTERVAL_SEC,
            health_channel=s.ORION_HEALTH_CHANNEL,
        ),
        heartbeat_details=office_status,
    )


async def _main_async() -> None:
    settings = get_settings()
    heartbeat_chassis: Optional[HeartbeatOnly] = None
    try:
        heartbeat_chassis = build_heartbeat_chassis(settings)
        await heartbeat_chassis.start_background()
    except Exception:
        logger.exception("system_health_heartbeat_start_failed")
        heartbeat_chassis = None
    try:
        await publish_loop()
    finally:
        if heartbeat_chassis is not None:
            try:
                await heartbeat_chassis.stop()
            except Exception:
                logger.exception("system_health_heartbeat_stop_error")


def main() -> None:
    setup_logging()
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        logger.info("orion-office-env interrupted; exiting.")


if __name__ == "__main__":
    main()
