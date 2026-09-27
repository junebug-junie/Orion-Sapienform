from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.core.bus.codec import OrionCodec
from orion.energy.ledger import UsageLedger
from orion.energy.tariff import load_tariff
from orion.schemas.power import PowerIntentSettledV1

from .bills import load_processed_bills, scan_bills
from .inbox import latest_processed_at, load_processed, scan_inbox
from .pipeline import EnergyChannels, EnergyPipeline, Outbound, StakesConfig
from .portal_status import read_portal_status
from .settings import Settings, get_settings

logger = logging.getLogger("orion-energy")


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[ORION_ENERGY] %(asctime)s %(levelname)s - %(name)s - %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def build_pipeline(settings: Settings) -> EnergyPipeline:
    ledger = UsageLedger(
        load_tariff(settings.ENERGY_TARIFF_PATH),
        tz=ZoneInfo(settings.ENERGY_TIMEZONE),
        cycle_start_day=settings.ENERGY_BILLING_CYCLE_START_DAY,
    )
    pipeline = EnergyPipeline(
        ledger=ledger,
        channels=EnergyChannels(
            usage=settings.ENERGY_USAGE_CHANNEL,
            accrued=settings.ENERGY_ACCRUED_CHANNEL,
            run_cost=settings.ENERGY_RUN_COST_CHANNEL,
            bill_actual=settings.ENERGY_BILL_ACTUAL_CHANNEL,
            bill_forecast=settings.ENERGY_BILL_FORECAST_CHANNEL,
            reconcile=settings.ENERGY_RECONCILE_CHANNEL,
            stakes=settings.ENERGY_STAKES_CHANNEL,
            importer_status=settings.ENERGY_IMPORTER_STATUS_CHANNEL,
        ),
        usage_point_id=settings.ENERGY_USAGE_POINT_ID or None,
        pending_hours=settings.ENERGY_RUN_COST_PENDING_HOURS,
        stakes=StakesConfig(
            near_ratio=settings.ENERGY_STAKES_NEAR_RATIO,
            over_ratio=settings.ENERGY_STAKES_OVER_RATIO,
            stale_after_hours=settings.ENERGY_STALE_AFTER_HOURS,
            portal_enabled=settings.ENERGY_PORTAL_ENABLED,
            portal_interval_hours=settings.ENERGY_PORTAL_INTERVAL_HOURS,
        ),
    )
    replayed = load_processed(Path(settings.ENERGY_PROCESSED_DIR))
    pipeline.replay(replayed)
    bills = load_processed_bills(Path(settings.ENERGY_BILL_PROCESSED_DIR))
    pipeline.replay_bills(bills)
    logger.info(
        "energy_ledger_replayed intervals=%d bills=%d usage_point=%s",
        len(replayed), len(bills), pipeline.usage_point(),
    )
    return pipeline


def _source(settings: Settings) -> ServiceRef:
    return ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.INSTANCE_ID)


async def publish_all(bus: OrionBusAsync, settings: Settings, outbound: Iterable[Outbound]) -> None:
    # A two-year backfill is ~35k messages; pub/sub drops what a slow sql-writer can't buffer.
    pause = 1.0 / settings.ENERGY_PUBLISH_MAX_PER_SEC if settings.ENERGY_PUBLISH_MAX_PER_SEC > 0 else 0.0
    for item in outbound:
        envelope = BaseEnvelope(
            kind=item.kind, source=_source(settings), payload=item.payload.model_dump(mode="json")
        )
        try:
            await bus.publish(item.channel, envelope)
        except Exception:
            logger.exception("energy_publish_failed channel=%s kind=%s", item.channel, item.kind)
        if pause:
            await asyncio.sleep(pause)


async def inbox_loop(bus: OrionBusAsync, settings: Settings, pipeline: EnergyPipeline, lock: asyncio.Lock) -> None:
    inbox, processed = Path(settings.ENERGY_INBOX_DIR), Path(settings.ENERGY_PROCESSED_DIR)
    bill_inbox, bill_processed = Path(settings.ENERGY_BILL_INBOX_DIR), Path(settings.ENERGY_BILL_PROCESSED_DIR)
    while True:
        # Usage and bills run in separate try blocks: scanning moves files to processed/,
        # so a failure in one must not drop rows the other already moved.
        try:
            rows = await asyncio.to_thread(scan_inbox, inbox, processed, now=datetime.now(timezone.utc))
            if rows:
                async with lock:
                    outbound = pipeline.ingest_intervals(rows, now=datetime.now(timezone.utc))
                await publish_all(bus, settings, outbound)
                logger.info(
                    "energy_ingested intervals=%d published=%d pending=%d",
                    len(rows), len(outbound), pipeline.pending_count(),
                )
        except Exception:
            logger.exception("energy_inbox_cycle_failed kind=usage")
        try:
            bills = await asyncio.to_thread(scan_bills, bill_inbox, bill_processed, now=datetime.now(timezone.utc))
            if bills:
                async with lock:
                    outbound = pipeline.ingest_bills(bills, now=datetime.now(timezone.utc))
                await publish_all(bus, settings, outbound)
                logger.info("energy_ingested bills=%d published=%d", len(bills), len(outbound))
        except Exception:
            logger.exception("energy_inbox_cycle_failed kind=bills")
        await asyncio.sleep(settings.ENERGY_SCAN_INTERVAL_SEC)


async def status_loop(bus: OrionBusAsync, settings: Settings, pipeline: EnergyPipeline, lock: asyncio.Lock) -> None:
    processed = Path(settings.ENERGY_PROCESSED_DIR)
    status_path = Path(settings.ENERGY_PORTAL_STATUS_PATH)
    while True:
        try:
            portal = await asyncio.to_thread(read_portal_status, status_path) if settings.ENERGY_PORTAL_ENABLED else None
            last_file_at = await asyncio.to_thread(latest_processed_at, processed, ".xml")
            async with lock:
                outbound = pipeline.status_tick(now=datetime.now(timezone.utc), portal=portal, last_file_at=last_file_at)
            await publish_all(bus, settings, outbound)
            importer, stakes = outbound[0].payload, outbound[1].payload
            logger.info(
                "energy_status state=%s reason=%s pressure=%s pressure_reason=%s",
                importer.state, importer.reason, stakes.pressure, stakes.pressure_reason,
            )
        except Exception:
            logger.exception("energy_status_cycle_failed")
        await asyncio.sleep(settings.ENERGY_STATUS_INTERVAL_SEC)


async def settlement_loop(bus: OrionBusAsync, settings: Settings, pipeline: EnergyPipeline, lock: asyncio.Lock) -> None:
    codec = OrionCodec()
    async with bus.subscribe(settings.POWER_SETTLED_CHANNEL) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            try:
                decoded = codec.decode(msg.get("data"))
                if not decoded.ok:
                    logger.warning("energy_settlement_decode_failed error=%s", decoded.error)
                    continue
                settled = PowerIntentSettledV1.model_validate(decoded.envelope.payload)
                async with lock:
                    outbound = pipeline.on_settlement(settled, now=datetime.now(timezone.utc))
                await publish_all(bus, settings, outbound)
                est = outbound[0].payload
                logger.info(
                    "energy_run_cost intent_id=%s usd=%s gap=%s house_usd=%s house_gap=%s",
                    settled.intent_id,
                    est.estimated_run_cost_usd,
                    est.run_cost_gap,
                    est.house_share_cost_usd,
                    est.house_share_gap,
                )
            except Exception:
                logger.exception("energy_settlement_handle_failed")


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
    heartbeat: Optional[HeartbeatOnly] = None
    try:
        heartbeat = build_heartbeat_chassis(settings)
        await heartbeat.start_background()
    except Exception:
        logger.exception("system_health_heartbeat_start_failed")
        heartbeat = None
    bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED, codec=OrionCodec())
    try:
        if not bus.enabled:
            logger.warning("ORION_BUS_ENABLED=false; orion-energy has nothing to do")
            while True:
                await asyncio.sleep(3600)
        await bus.connect()
        pipeline = build_pipeline(settings)
        lock = asyncio.Lock()
        await asyncio.gather(
            inbox_loop(bus, settings, pipeline, lock),
            settlement_loop(bus, settings, pipeline, lock),
            status_loop(bus, settings, pipeline, lock),
        )
    finally:
        if heartbeat is not None:
            try:
                await heartbeat.stop()
            except Exception:
                logger.exception("system_health_heartbeat_stop_error")


def main() -> None:
    setup_logging()
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        logger.info("orion-energy interrupted; exiting.")


if __name__ == "__main__":
    main()
