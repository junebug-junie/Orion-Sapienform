from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Optional

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
from .zwave_client import (
    ZWaveJSClient,
    extract_meter_amps,
    extract_meter_volts,
    extract_meter_watts,
    extract_switch_on,
)

logger = logging.getLogger("orion-zwave")

# Well under the worst case of connect (10s open) + bootstrap (10s read + 30s per RPC).
RECONNECT_TIMEOUT_SEC = 20.0


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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_COOLING_STATUS: dict[str, Any] = {
    "cooling_sensor": "unknown",
    "cooling_sample_age_sec": None,
    "zwave_connected": False,
    "consecutive_poll_failures": 0,
}


def cooling_status() -> dict[str, Any]:
    """Heartbeat details: is the AC reading real right now?"""
    return dict(_COOLING_STATUS)


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
    last_fresh_at: Optional[datetime],
    stale_after_sec: float,
    device_id: str = "shelly-wave-plug-ac",
    device_name: str = "portable_ac",
    instance_id: str = "athena",
) -> HomeCoolingSampleV1:
    age = max(0.0, (now - last_fresh_at).total_seconds()) if last_fresh_at is not None else None
    # No fresh wattage is stale even when the last real answer is recent (e.g. right after a
    # reconnect, when only snapshot values are cached).
    stale = age is None or age > stale_after_sec or watts is None
    if stale:
        measurements = CoolingMeasurementsV1()
        state = CoolingObservedStateV1(stale=True)
    else:
        measurements = CoolingMeasurementsV1(
            cooling_watts=watts,
            cooling_volts=volts,
            cooling_amps=amps,
        )
        state = CoolingObservedStateV1(switch_on=switch_on, stale=False)

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
        provenance=CoolingProvenanceV1(zwave_node_id=node_id, sample_age_sec=age),
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


async def poll_once(
    client: ZWaveJSClient,
    settings: Settings,
    now_fn: Callable[[], datetime] = _utcnow,
) -> HomeCoolingSampleV1:
    node_id = settings.ZWAVE_NODE_ID
    try:
        await client.refresh_meter_watts(node_id)
        now = now_fn()
        values = client.get_values(node_id)
        # Only a meter value the plug itself answered or pushed within the stale window may be
        # published — never a cached one under a fresh flag.
        fresh = client.fresh_values(node_id, now=now, max_age_sec=settings.COOLING_STALE_AFTER_SEC)
        sample = build_cooling_sample(
            node_id=node_id,
            controller_ready=client.controller_ready,
            device_online=client.device_online(node_id),
            watts=extract_meter_watts(fresh),
            volts=extract_meter_volts(fresh),
            amps=extract_meter_amps(fresh),
            switch_on=extract_switch_on(values),
            device_path="/dev/zwave",
            product=client.product_name(node_id),
            now=now,
            last_fresh_at=client.last_fresh_at(node_id),
            stale_after_sec=settings.COOLING_STALE_AFTER_SEC,
            device_id=settings.ZWAVE_DEVICE_ID,
            device_name=settings.ZWAVE_DEVICE_NAME,
            instance_id=settings.INSTANCE_ID,
        )
    except Exception:
        _COOLING_STATUS.update(
            cooling_sensor="error",
            cooling_sample_age_sec=None,
            zwave_connected=client.connected,
            consecutive_poll_failures=client.consecutive_poll_failures,
        )
        raise
    _COOLING_STATUS.update(
        cooling_sensor="stale" if sample.state.stale else "fresh",
        cooling_sample_age_sec=sample.provenance.sample_age_sec,
        zwave_connected=client.connected,
        consecutive_poll_failures=client.consecutive_poll_failures,
    )
    if sample.state.stale:
        logger.warning(
            "cooling_sample_stale age_sec=%s consecutive_poll_failures=%s connected=%s",
            sample.provenance.sample_age_sec,
            client.consecutive_poll_failures,
            client.connected,
        )
    return sample


async def _ensure_connected(client: ZWaveJSClient, settings: Settings) -> None:
    """A dead socket must not leave us publishing cache forever: reconnect when needed.

    Bounded so a half-alive server cannot hold back this cycle's stale sample."""
    if client.connected:
        return
    try:
        await asyncio.wait_for(_reconnect(client), timeout=RECONNECT_TIMEOUT_SEC)
        logger.info("zwave_js_connected ws=%s", settings.ZWAVE_JS_WS_URL)
    except Exception as exc:
        if isinstance(exc, TimeoutError):
            # Cancellation skips connect()'s own cleanup; don't leave a half-open socket behind.
            try:
                await client.close()
            except Exception:
                logger.exception("zwave_js_close_after_timeout_failed")
        logger.warning(
            "zwave_js_connect_failed ws=%s error=%s: %s",
            settings.ZWAVE_JS_WS_URL,
            type(exc).__name__,
            exc,
        )


async def _reconnect(client: ZWaveJSClient) -> None:
    await client.close()
    await client.connect()


async def run_poll_cycle(client: ZWaveJSClient, bus: OrionBusAsync, settings: Settings) -> None:
    """One cycle: reconnect if needed, then always poll and publish — stale while disconnected,
    so silence stays visible downstream."""
    await _ensure_connected(client, settings)
    try:
        sample = await poll_once(client, settings)
        await _publish_sample(bus, settings, sample)
    except Exception:
        logger.exception("Cooling poll cycle failed; will retry")


async def poll_cooling_loop() -> None:
    settings = get_settings()

    logger.info(
        "Starting Orion Z-Wave — service=%s enabled=%s node_id=%s ws=%s",
        settings.SERVICE_NAME,
        settings.ORION_ZWAVE_ENABLED,
        settings.ZWAVE_NODE_ID,
        settings.ZWAVE_JS_WS_URL,
    )

    if not settings.ORION_ZWAVE_ENABLED:
        logger.info(
            "ORION_ZWAVE_ENABLED=false; heartbeat-only mode until Shelly is paired"
        )
        while True:
            await asyncio.sleep(settings.COOLING_POLL_INTERVAL_SEC)
        return

    bus = OrionBusAsync(
        url=settings.ORION_BUS_URL,
        enabled=settings.ORION_BUS_ENABLED,
        codec=OrionCodec(),
    )
    if bus.enabled:
        await bus.connect()

    client = ZWaveJSClient(settings.ZWAVE_JS_WS_URL, settings.ZWAVE_NODE_ID)
    try:
        while True:
            await run_poll_cycle(client, bus, settings)
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
        ),
        heartbeat_details=cooling_status,
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
