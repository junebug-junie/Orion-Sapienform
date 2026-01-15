from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from uuid import UUID, uuid4

from orion.core.bus.bus_service_chassis import BaseChassis, ChassisConfig
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.telemetry.metacognition import MetacognitionTickV1
from orion.schemas.telemetry.metacog_trigger import MetacogTriggerV1
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
        self._last_metacog_trigger_ts: float = 0.0

    def _trace_meta(
        self,
        *,
        trace_id: str,
        event_id: str,
        parent_event_id: str | None = None,
        created_at: datetime | None = None,
    ) -> Dict[str, Any]:
        return {
            "trace_id": trace_id,
            "event_id": event_id,
            "parent_event_id": parent_event_id,
            "source_service": settings.service_name,
            "created_at": (created_at or _utcnow()).isoformat(),
        }

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

    def _calculate_metrics(self) -> Tuple[float, float, List[EquilibriumServiceState]]:
        """Shared logic to calculate current distress/zen and build state list."""
        now = _utcnow()
        states: List[EquilibriumServiceState] = []

        retention = float(settings.state_retention_sec)
        keys_to_purge = []

        # 1. Build states from observed heartbeats
        for key, rec in list(self._state.items()):
            try:
                # Check for staleness
                last_seen = rec.get("last_seen_ts")
                if not isinstance(last_seen, datetime):
                     try:
                         last_seen = datetime.fromisoformat(str(last_seen))
                     except Exception:
                         last_seen = now

                delta_sec = (now - last_seen).total_seconds()
                if delta_sec > retention:
                    keys_to_purge.append(key)
                    continue

                states.append(self._build_service_state(rec, now))
            except Exception:
                continue

        # Prune ghosts
        if keys_to_purge:
            for k in keys_to_purge:
                self._state.pop(k, None)
            # Async prune from Redis (fire and forget)
            asyncio.create_task(self.bus.redis.hdel(settings.redis_state_key, *keys_to_purge))
            logger.info("Pruned %d stale services from equilibrium state", len(keys_to_purge))

        # 2. Force expected services if missing
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

        # 3. Calculate Scores
        # Use the smallest window (usually 60s) for immediate distress
        smallest_window = str(min(settings.windows_sec)) if settings.windows_sec else "60"

        distress_components = [1.0 - s.uptime_pct.get(smallest_window, 1.0) for s in states] or [0.0]
        distress_score = float(sum(distress_components) / len(distress_components)) if distress_components else 0.0
        zen_score = max(0.0, 1.0 - distress_score)

        return distress_score, zen_score, states

    async def _publish_snapshot(self) -> None:
        now = _utcnow()

        # Use shared calculation
        distress_score, zen_score, states = self._calculate_metrics()

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

    async def _publish_metacognition_tick(self) -> None:
        if not self.bus.enabled:
            return

        now = _utcnow()
 
        # Use shared calculation (ignore the detailed states list here)
        distress_score, zen_score, _ = self._calculate_metrics()
        services_tracked = len(self._state)

        tick = MetacognitionTickV1(
            generated_at=now,
            source_service=settings.service_name,
            source_node=settings.node_name,
            distress_score=distress_score,
            zen_score=zen_score,
            services_tracked=services_tracked,
            snapshot={
                "equilibrium": {
                    "services_tracked": services_tracked,
                }
            },
        )

        # Populate correlation_id in payload for persistence
        tick.correlation_id = tick.tick_id

        try:
            tick_uuid = UUID(str(tick.tick_id))
        except ValueError:
            tick_uuid = uuid4()

        trace_meta = self._trace_meta(
            trace_id=str(tick_uuid),
            event_id=str(tick_uuid),
            created_at=now,
        )

        env = BaseEnvelope(
            kind="metacognition.tick.v1",
            source=self._source(),
            correlation_id=tick_uuid,
            id=tick_uuid,
            trace=trace_meta,
            payload=tick.model_dump(mode="json"),
        )

        await self.bus.publish(settings.channel_metacognition_tick, env)
        logger.info(
            "Published metacognition tick "
            f"tick_id={tick.tick_id} trace_id={trace_meta['trace_id']} "
            f"distress={distress_score:.3f} channel={settings.channel_metacognition_tick}"
        )

    async def _publish_metacog_trigger(self, trigger: MetacogTriggerV1) -> None:
        now_ts = datetime.now().timestamp()

        # Simple cooldown check
        if (now_ts - self._last_metacog_trigger_ts) < settings.metacog_cooldown_sec:
            logger.info("Metacog trigger skipped due to cooldown (%s)", trigger.trigger_kind)
            return

        self._last_metacog_trigger_ts = now_ts

        # 1. Publish Trigger Event (for observability)
        trace_id = uuid4()
        event_id = uuid4()
        trace_meta = self._trace_meta(
            trace_id=str(trace_id),
            event_id=str(event_id),
            created_at=_utcnow(),
        )
        env = BaseEnvelope(
            kind="orion.metacog.trigger.v1",
            source=self._source(),
            correlation_id=trace_id,
            id=event_id,
            trace=trace_meta,
            payload=trigger.model_dump(mode="json"),
        )
        try:
            await self.bus.publish(settings.channel_metacog_trigger, env)
            logger.info(
                "Published metacog trigger "
                f"kind={trigger.trigger_kind} trace_id={trace_meta['trace_id']} "
                f"channel={settings.channel_metacog_trigger}"
            )
        except Exception as e:
            logger.error(f"Failed to publish metacog trigger: {e}")
            return

        if settings.metacog_publish_verb_request:
            # Legacy path intentionally disabled; must route through cortex-orch.
            logger.error(
                "Metacog legacy verb request is disabled (bypasses cortex-orch). "
                "Set EQUILIBRIUM_METACOG_PUBLISH_VERB_REQUEST=false and rely on "
                f"orion:equilibrium:metacog:trigger routing. trace_id={trace_meta['trace_id']}"
            )

    async def _publish_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._publish_snapshot()
            except Exception as e:
                logger.error(f"Publish loop error: {e}")
            await asyncio.sleep(float(settings.publish_interval_sec))

    async def _collapse_loop(self) -> None:
        interval = float(settings.collapse_mirror_interval_sec)
        while not self._stop.is_set():
            try:
                await self._publish_metacognition_tick()
            except Exception as e:
                logger.warning(f"Metacognition tick loop error: {e}")
            await asyncio.sleep(interval)

    async def _metacog_baseline_loop(self) -> None:
        if not settings.metacog_enable:
            return

        interval = float(settings.metacog_baseline_interval_sec)
        while not self._stop.is_set():
            try:
                await asyncio.sleep(interval)
                distress, zen, _ = self._calculate_metrics()
                trigger = MetacogTriggerV1(
                    trigger_kind="baseline",
                    reason="scheduled_check",
                    zen_state="zen" if zen > 0.5 else "not_zen",
                    pressure=distress,
                    recall_enabled=settings.metacog_recall_enabled,
                )
                await self._publish_metacog_trigger(trigger)
            except Exception as e:
                logger.error(f"Metacog baseline loop error: {e}")

    async def _run(self) -> None:
        await self._load_state()
        publisher = asyncio.create_task(self._publish_loop())
        collapse_task = asyncio.create_task(self._collapse_loop())
        metacog_task = asyncio.create_task(self._metacog_baseline_loop())

        # Build list of channels to subscribe to
        channels = [settings.health_channel]
        if settings.metacog_enable:
            channels.append(settings.channel_collapse_mirror_user_event)
            channels.append(settings.channel_pad_signal)

        async with self.bus.subscribe(*channels) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                if self._stop.is_set():
                    break

                channel = msg.get("channel")
                # aioredis returns channel as bytes or str depending on decoding
                if hasattr(channel, "decode"):
                    channel = channel.decode("utf-8")

                decoded = self.codec.decode(msg.get("data"))
                if not decoded.ok:
                    logger.warning(f"Equilibrium decode failed channel={channel} error={decoded.error}")
                    continue
                env = decoded.envelope
                payload_dict = env.payload if isinstance(env.payload, dict) else {}

                try:
                    # Health Heartbeats
                    if channel == settings.health_channel:
                        if env.kind == "system.health.v1":
                            heartbeat = SystemHealthV1.model_validate(payload_dict)
                            self._evaluate_state(heartbeat)

                    # Metacog Triggers (only if enabled)
                    elif settings.metacog_enable:
                        distress, zen, _ = self._calculate_metrics()

                        if channel == settings.channel_pad_signal:
                            # Landing Pad Signal
                            # We look for "salience" or similar in generic payload
                            salience = 0.0
                            if isinstance(payload_dict, dict):
                                salience = float(payload_dict.get("salience", 0.0))

                            if salience >= settings.metacog_pad_pulse_threshold:
                                trigger = MetacogTriggerV1(
                                    trigger_kind="pulse",
                                    reason=f"pad_signal_high:{salience:.2f}",
                                    zen_state="zen" if zen > 0.5 else "not_zen",
                                    pressure=distress,
                                    signal_refs=[str(env.correlation_id or "unknown")],
                                    upstream=payload_dict,
                                    recall_enabled=settings.metacog_recall_enabled,
                                )
                                await self._publish_metacog_trigger(trigger)

                        elif channel == settings.channel_collapse_mirror_user_event:
                            # User manually triggered collapse
                            # This is a "dense" event

                            # CRITICAL: Prevent infinite feedback loops
                            observer = str(payload_dict.get("observer") or "").lower()
                            if observer == "orion":
                                continue

                            trigger = MetacogTriggerV1(
                                trigger_kind="manual",
                                reason="user_collapse_event",
                                zen_state="zen" if zen > 0.5 else "not_zen",
                                pressure=distress,
                                upstream={"event_id": payload_dict.get("event_id")},
                                recall_enabled=settings.metacog_recall_enabled,
                            )
                            await self._publish_metacog_trigger(trigger)

                except Exception as e:
                    logger.warning("Failed to process message on %s: %s", channel, e)

        publisher.cancel()
        collapse_task.cancel()
        metacog_task.cancel()

        await asyncio.gather(publisher, collapse_task, metacog_task, return_exceptions=True)
