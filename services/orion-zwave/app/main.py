from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.core.bus.codec import OrionCodec
from orion.schemas.telemetry.home_cooling import (
    CoolingControllerV1,
    CoolingDeviceV1,
    CoolingMeasurementsV1,
    CoolingObservedStateV1,
    CoolingProvenanceV1,
    HomeCoolingSampleV1,
)

from .settings import Settings, get_settings
from .zwave_client import ZWaveJSClient, extract_meter_watts, extract_switch_on

logger = logging.getLogger("orion-zwave")


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[ORION_ZWAVE] %(asctime)s %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def build_cooling_sample(
    *,
    node_id: int,
    controller_ready: bool,
    device_online: bool,
    watts: Optional[float],
    volts: Optional[float],
    amps: Optional[float],
    switch_on: Optional[bool],
    device_path: Optional[str],
    product: Optional[str],
    now: datetime,
    device_id: str = "shelly-wave-plug-ac",
    device_name: str = "portable_ac",
    instance_id: str = "athena",
) -> HomeCoolingSampleV1:
    measurements = CoolingMeasurementsV1(
        cooling_watts=watts,
        cooling_volts=volts,
        cooling_amps=amps,
    )
    state = CoolingObservedStateV1(switch_on=switch_on) if switch_on is not None else CoolingObservedStateV1()

    return HomeCoolingSampleV1(
        ts=now,
        node=instance_id,
        controller=CoolingControllerV1(
            ready=controller_ready,
            device_path=device_path,
        ),
        device=CoolingDeviceV1(
            id=device_id,
            name=device_name,
            product=product,
            online=device_online,
        ),
        measurements=measurements,
        state=state,
        provenance=CoolingProvenanceV1(zwave_node_id=node_id),
    )


def _source(settings: Settings) -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        node=settings.INSTANCE_ID,
    )


async def _publish_sample(
    bus: OrionBusAsync,
    settings: Settings,
    sample: HomeCoolingSampleV1,
) -> None:
    if not getattr(bus, "enabled", False):
        logger.info("Bus disabled; skipping cooling sample publish")
        return

    envelope = BaseEnvelope(
        kind="home.cooling.sample.v1",
        source=_source(settings),
        payload=sample.model_dump(mode="json", by_alias=True),
    )
    try:
        await bus.publish(settings.COOLING_SAMPLE_CHANNEL, envelope)
        logger.info(
            "Published cooling sample watts=%s switch_on=%s channel=%s",
            sample.measurements.cooling_watts,
            sample.state.switch_on,
            settings.COOLING_SAMPLE_CHANNEL,
        )
    except Exception:
        logger.exception(
            "Failed to publish cooling sample channel=%s",
            settings.COOLING_SAMPLE_CHANNEL,
        )


async def poll_cooling_loop() -> None:
    settings = get_settings()

    logger.info(
        "Starting Orion Z-Wave — service=%s enabled=%s node_id=%s ws=%s",
        settings.SERVICE_NAME,
        settings.ORION_ZWAVE_ENABLED,
        settings.ZWAVE_NODE_ID,
        settings.ZWAVE_JS_WS_URL,
    )

    bus = OrionBusAsync(
        url=settings.ORION_BUS_URL,
        enabled=settings.ORION_BUS_ENABLED,
        codec=OrionCodec(),
    )
    if bus.enabled:
        await bus.connect()

    if not settings.ORION_ZWAVE_ENABLED:
        logger.info(
            "ORION_ZWAVE_ENABLED=false; heartbeat-only mode until Shelly is paired"
        )
        while True:
            await asyncio.sleep(settings.COOLING_POLL_INTERVAL_SEC)
        return

    client = ZWaveJSClient(settings.ZWAVE_JS_WS_URL, settings.ZWAVE_NODE_ID)
    try:
        await client.connect()
    except Exception:
        logger.exception("Failed to connect to zwave-js-server; retrying in poll interval")
        while True:
            await asyncio.sleep(settings.COOLING_POLL_INTERVAL_SEC)
            try:
                await client.connect()
                break
            except Exception:
                logger.exception("Z-Wave reconnect attempt failed")

    try:
        while True:
            try:
                values = client.get_values(settings.ZWAVE_NODE_ID)
                watts = extract_meter_watts(values)
                switch_on = extract_switch_on(values)
                sample = build_cooling_sample(
                    node_id=settings.ZWAVE_NODE_ID,
                    controller_ready=client.controller_ready,
                    device_online=client.device_online(settings.ZWAVE_NODE_ID),
                    watts=watts,
                    volts=None,
                    amps=None,
                    switch_on=switch_on,
                    device_path="/dev/zwave",
                    product=client.product_name(settings.ZWAVE_NODE_ID),
                    now=datetime.now(timezone.utc),
                    device_id=settings.ZWAVE_DEVICE_ID,
                    device_name=settings.ZWAVE_DEVICE_NAME,
                    instance_id=settings.INSTANCE_ID,
                )
                await _publish_sample(bus, settings, sample)
            except Exception:
                logger.exception("Cooling poll cycle failed; will retry")

            await asyncio.sleep(settings.COOLING_POLL_INTERVAL_SEC)
    finally:
        await client.close()


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
        )
    )


async def _main_async() -> None:
    settings = get_settings()
    heartbeat_chassis: Optional[HeartbeatOnly] = None
    try:
        heartbeat_chassis = build_heartbeat_chassis(settings)
        await heartbeat_chassis.start_background()
        logger.info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.SERVICE_NAME,
            settings.HEARTBEAT_INTERVAL_SEC,
        )
    except Exception:
        logger.exception("system_health_heartbeat_start_failed")
        heartbeat_chassis = None
    try:
        await poll_cooling_loop()
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
        logger.info("Orion Z-Wave service interrupted; exiting.")


if __name__ == "__main__":
    main()
