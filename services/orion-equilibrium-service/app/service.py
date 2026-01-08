from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from orion.core.bus.bus_service_chassis import BaseChassis, ChassisConfig
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.core.verbs.models import VerbRequestV1
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2, CollapseMirrorStateSnapshot
from orion.schemas.telemetry.system_health import EquilibriumServiceState, EquilibriumSnapshotV1, SystemHealthV1
from orion.schemas.telemetry.spark_signal import SparkSignalV1
from orion.core.bus.codec import OrionCodec

from .settings import settings

logger = logging.getLogger("orion-equilibrium")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EquilibriumService(BaseChassis):
    def __init__(self) -> None:
        super().__init__(
            ChassisConfig(
                service_name=settings.service_name,
                service_version=settings.service_version,
                node_name=settings.node_name or "unknown",
                instance_id=settings.instance_id,
                bus_url=settings.orion_bus_url,
                bus_enabled=settings.orion_bus_enabled,
                heartbeat_interval_sec=settings.heartbeat_interval_sec,
                health_channel=settings.health_channel,
            )
        )
        self.codec = OrionCodec()
        self._state: Dict[str, Dict[str, Any]] = {}
        self.expected_services = settings.expected_services()

    async def _load_state(self) -> None:
        try:
            raw = await self.bus.redis.hgetall(settings.redis_state_key)
            for key, blob in raw.items():
                try:
                    data = json.loads(blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else blob)
                    if isinstance(data, dict):
                        if "last_seen_ts" in data and isinstance(data["last_seen_ts"], str):
                            data["last_seen_ts"] = datetime.fromisoformat(data["last_seen_ts"])
                        self._state[key] = data
                except Exception:
                    continue
        except Exception as e:
            logger.warning("Failed to load persisted equilibrium state: %s", e)

    async def _persist_state(self, key: str, data: Dict[str, Any]) -> None:
        try:
            serializable = dict(data)
            ts = serializable.get("last_seen_ts")
            if isinstance(ts, datetime):
                serializable["last_seen_ts"] = ts.isoformat()
            await self.bus.redis.hset(settings.redis_state_key, key, json.dumps(serializable))
        except Exception as e:
            logger.warning("Failed to persist equilibrium state for %s: %s", key, e)

    def _service_key(self, payload: SystemHealthV1) -> str:
        node = payload.node or "unknown"
        return f"{payload.service}@{node}"

    def _evaluate_state(self, payload: SystemHealthV1) -> None:
        key = self._service_key(payload)
        record = {
            "service": payload.service,
            "node": payload.node,
            "version": payload.version,
            "instance": payload.instance,
            "boot_id": payload.boot_id,
            "status": payload.status,
            "last_seen_ts": payload.last_seen_ts,
            "heartbeat_interval_sec": float(payload.heartbeat_interval_sec or 10.0),
            "details": payload.details or {},
        }
        self._state[key] = record
        asyncio.create_task(self._persist_state(key, record))

    def _compute_uptime(self, last_seen: datetime, interval: float, now: datetime, window_sec: int) -> float:
        grace = interval * settings.grace_multiplier
        delta_ms = (now - last_seen).total_seconds() * 1000.0
        if delta_ms <= grace * 1000.0:
            return 1.0
        down_ms = delta_ms - grace * 1000.0
        return max(0.0, min(1.0, 1.0 - (down_ms / (window_sec * 1000.0))))

    def _build_service_state(self, record: Dict[str, Any], now: datetime) -> EquilibriumServiceState:
        last_seen = record.get("last_seen_ts") or now
        if not isinstance(last_seen, datetime):
            try:
                last_seen = datetime.fromisoformat(str(last_seen))
            except Exception:
                last_seen = now
        interval = float(record.get("heartbeat_interval_sec", 10.0))
        grace = interval * settings.grace_multiplier
        delta_ms = (now - last_seen).total_seconds() * 1000.0
        status = record.get("status", "ok")
        if delta_ms > grace * 1000.0:
            status = "down"
        uptime_pct = {str(w): self._compute_uptime(last_seen, interval, now, w) for w in settings.windows_sec}
        down_for_ms = max(0, int(delta_ms - grace * 1000.0))

        return EquilibriumServiceState(
            service=str(record.get("service")),
            node=record.get("node"),
            status=status,
            last_seen_ts=last_seen,
            heartbeat_interval_sec=interval,
            down_for_ms=down_for_ms,
            uptime_pct=uptime_pct,
            boot_id=record.get("boot_id"),
            version=record.get("version"),
            instance=record.get("instance"),
            details=record.get("details") or {},
        )

    async def _publish_snapshot(self) -> None:
        now = _utcnow()
        states: List[EquilibriumServiceState] = []
        for key, rec in list(self._state.items()):
            try:
                states.append(self._build_service_state(rec, now))
            except Exception as e:
                logger.warning("Failed to build state for %s: %s", key, e)

        # Ensure expected services are present even if never seen
        for svc in self.expected_services:
            if not any(s.service == svc for s in states):
                states.append(
                    EquilibriumServiceState(
                        service=svc,
                        node=None,
                        status="down",
                        last_seen_ts=now,
                        heartbeat_interval_sec=float(settings.heartbeat_interval_sec),
                        down_for_ms=int(settings.grace_multiplier * settings.heartbeat_interval_sec * 1000),
                        uptime_pct={str(w): 0.0 for w in settings.windows_sec},
                    )
                )

        distress_components = [1.0 - s.uptime_pct.get(str(min(settings.windows_sec)), 1.0) for s in states] or [0.0]
        distress_score = float(sum(distress_components) / len(distress_components))
        zen_score = max(0.0, 1.0 - distress_score)

        snapshot = EquilibriumSnapshotV1(
            source_service=settings.service_name,
            source_node=settings.node_name,
            producer_boot_id=self.boot_id,
            generated_at=now,
            grace_multiplier=settings.grace_multiplier,
            windows_sec=settings.windows_sec,
            expected_services=self.expected_services,
            services=states,
            distress_score=distress_score,
            zen_score=zen_score,
        )

        env = BaseEnvelope(
            kind="equilibrium.snapshot.v1",
            source=self._source(),
            payload=snapshot.model_dump(mode="json"),
        )

        signal = SparkSignalV1(
            signal_type="equilibrium",
            intensity=distress_score,
            valence_delta=-distress_score * 0.2,
            coherence_delta=-distress_score * 0.1,
            as_of_ts=now,
            ttl_ms=int(settings.publish_interval_sec * 2000),
            source_service=settings.service_name,
            source_node=settings.node_name,
        )
        signal_env = BaseEnvelope(
            kind="spark.signal.v1",
            source=self._source(),
            payload=signal.model_dump(mode="json"),
        )

        try:
            await self.bus.publish(settings.channel_equilibrium_snapshot, env)
            await self.bus.publish(settings.channel_spark_signal, signal_env)
            logger.info("Published equilibrium snapshot distress=%.3f zen=%.3f", distress_score, zen_score)
        except Exception as e:
            logger.error("Failed to publish equilibrium snapshot: %s", e)

    async def _publish_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._publish_snapshot()
            except Exception as e:
                logger.error("Publish loop error: %s", e)
            await asyncio.sleep(float(settings.publish_interval_sec))

    async def _publish_collapse_mirror(self) -> None:
        if not self.bus.enabled:
            return

        snapshot = CollapseMirrorStateSnapshot(
            observer_state=["monitoring"],
            telemetry={
                "equilibrium": {
                    "distress_score": None,
                    "zen_score": None,
                    "services_tracked": len(self._state),
                }
            },
        )

        entry = CollapseMirrorEntryV2(
            observer="Oríon",
            trigger="equilibrium.metacognition_tick",
            observer_state=["monitoring"],
            field_resonance="ambient",
            type="system-change",
            emergent_entity="Oríon",
            summary="Periodic metacognition snapshot emitted by equilibrium monitor.",
            mantra="steady presence",
            snapshot_kind="baseline",
            state_snapshot=snapshot,
        )

        request = VerbRequestV1(
            trigger="orion.collapse.log",
            schema_id="CollapseMirrorEntryV2",
            payload=entry.model_dump(mode="json"),
            request_id=str(uuid4()),
            caller=settings.service_name,
            meta={"origin": "equilibrium-service"},
        )

        envelope = BaseEnvelope(
            kind="verb.request",
            source=self._source(),
            correlation_id=request.request_id,
            payload=request.model_dump(mode="json"),
        )

        await self.bus.publish("orion:verb:request", envelope)
        logger.info("Requested collapse mirror log from equilibrium tick event_id=%s", entry.event_id)

    async def _collapse_loop(self) -> None:
        interval = float(settings.collapse_mirror_interval_sec)
        while not self._stop.is_set():
            try:
                await self._publish_collapse_mirror()
            except Exception as e:
                logger.warning("Collapse mirror loop error: %s", e)
            await asyncio.sleep(interval)

    async def _run(self) -> None:
        await self._load_state()
        publisher = asyncio.create_task(self._publish_loop())
        collapse_task = asyncio.create_task(self._collapse_loop())

        async with self.bus.subscribe(settings.health_channel) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._stop.is_set():
                    break
                decoded = self.codec.decode(msg.get("data"))
                if not decoded.ok:
                    continue
                env = decoded.envelope
                payload_dict = env.payload if isinstance(env.payload, dict) else {}

                try:
                    if env.kind == "system.health.v1":
                        heartbeat = SystemHealthV1.model_validate(payload_dict)
                    elif env.kind == "system.health":
                        heartbeat = SystemHealthV1(
                            service=payload_dict.get("service"),
                            node=payload_dict.get("node"),
                            version=payload_dict.get("version"),
                            instance=None,
                            boot_id=str(payload_dict.get("boot_id") or uuid4()),
                            status=payload_dict.get("status") or "ok",
                            last_seen_ts=_utcnow(),
                            heartbeat_interval_sec=float(payload_dict.get("heartbeat_interval_sec") or settings.heartbeat_interval_sec),
                            details=payload_dict.get("details") or {},
                        )
                    else:
                        continue
                    self._evaluate_state(heartbeat)
                except Exception as e:
                    logger.warning("Failed to handle heartbeat: %s", e)

        publisher.cancel()
        collapse_task.cancel()
        try:
            await publisher
        except Exception:
            pass
        try:
            await collapse_task
        except Exception:
            pass
