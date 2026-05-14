from __future__ import annotations

import asyncio
import logging

import asyncpg
from orion.core.bus.bus_service_chassis import BaseChassis, ChassisConfig
from orion.core.bus.codec import OrionCodec
from orion.schemas.substrate_telemetry import SubstrateTierOutcomesPayloadV1

from . import db
from .settings import settings

logger = logging.getLogger("orion.substrate.telemetry")


class SubstrateTelemetryService(BaseChassis):
    def __init__(self) -> None:
        super().__init__(
            ChassisConfig(
                service_name=settings.service_name,
                service_version=settings.service_version,
                node_name=settings.node_name,
                bus_url=settings.orion_bus_url,
                bus_enabled=settings.orion_bus_enabled,
                heartbeat_interval_sec=settings.heartbeat_interval_sec,
            )
        )
        self.codec = OrionCodec()
        self.bus.codec = self.codec

    async def _retention_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn = await asyncpg.connect(dsn=settings.postgres_uri)
                try:
                    await db.ensure_schema(conn)
                    await db.global_retention_sweep(conn, max_age_days=settings.retention_days)
                finally:
                    await conn.close()
            except Exception as exc:
                logger.warning("substrate_telemetry_retention_sweep_failed err=%s", exc)
            try:
                await asyncio.sleep(float(settings.retention_scan_interval_sec))
            except asyncio.CancelledError:
                break

    async def _handle_message(self, data: bytes | None) -> None:
        if not data:
            return
        decoded = self.codec.decode(data)
        if not decoded.ok or decoded.envelope is None:
            logger.debug("substrate_telemetry_decode_skip")
            return
        env = decoded.envelope
        if env.kind != "substrate.tier_outcomes.v1":
            return
        try:
            payload = SubstrateTierOutcomesPayloadV1.model_validate(env.payload)
        except Exception as exc:
            logger.warning("substrate_telemetry_payload_invalid err=%s", exc)
            return
        src = env.source
        conn = await asyncpg.connect(dsn=settings.postgres_uri)
        try:
            await db.ensure_schema(conn)
            await db.insert_event(
                conn,
                correlation_id=env.correlation_id,
                envelope_kind=env.kind,
                generated_at=payload.generated_at,
                cold_anchors=list(payload.cold_anchors),
                tier_outcomes=dict(payload.tier_outcomes),
                degraded_producers=list(payload.degraded_producers),
                source_service=src.name,
                source_node=src.node,
            )
            await db.prune_correlation(
                conn,
                correlation_id=env.correlation_id,
                keep_newest=settings.per_correlation_row_cap,
            )
        finally:
            await conn.close()

    async def _run(self) -> None:
        ret_task = asyncio.create_task(self._retention_loop(), name="substrate-telemetry-retention")
        try:
            async with self.bus.subscribe(settings.channel_substrate_tier_outcomes) as pubsub:
                async for msg in self.bus.iter_messages(pubsub):
                    if self._stop.is_set():
                        break
                    try:
                        raw = msg.get("data")
                        await self._handle_message(raw if isinstance(raw, (bytes, bytearray)) else None)
                    except Exception as exc:
                        logger.warning("substrate_telemetry_message_handler_err err=%s", exc)
        finally:
            ret_task.cancel()
            try:
                await ret_task
            except Exception:
                pass
