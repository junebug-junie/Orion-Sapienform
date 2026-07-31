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
from .substrate_metacog_gate import build_substrate_metacog_trigger
from .repair_pressure_metacog_gate import build_repair_pressure_metacog_trigger
from .telemetry_anomaly_metacog_gate import build_telemetry_anomaly_metacog_trigger
from .chat_turn_metacog_gate import (
    ChatTurnCorrelator,
    build_chat_turn_metacog_trigger,
    evaluate_chat_turn_gate_conditions,
    is_chat_turn_evidence_terminal,
)
from .transport_metacog_gate import (
    build_transport_metacog_trigger_from_bus_synaptic,
    build_transport_metacog_trigger_from_grammar_atom,
    build_transport_metacog_trigger_from_snapshot,
)
from .insight_metacog_gate import build_insight_metacog_trigger
from .flow_metacog_gate import build_flow_metacog_trigger
from .repair_pressure_trend_gate import (
    evaluate_repair_pressure_trend,
    state_from_dict,
    state_to_dict,
)
from orion.metacog.trend_reducer import MetacogTrendStateV1
from .attention_self_model_reader import AttentionSelfModelReader
from orion.substrate.metacog_trigger_signals import (
    ConfidenceSample,
    detect_confidence_recovery,
    detect_flow_regime,
)
from .downtime_transition_tracker import DowntimeTransitionTracker
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload
from orion.schemas.telemetry.system_health import EquilibriumServiceState, EquilibriumSnapshotV1, SystemHealthV1
from orion.schemas.telemetry.spark_signal import SparkSignalV1
from orion.core.bus.codec import OrionCodec

from .settings import settings

logger = logging.getLogger("orion-equilibrium")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Re-log the generative gates' stale-window warning every Nth consecutive stale
# poll. 60 polls at the default 30s cadence is ~30 minutes -- frequent enough
# that a stopped writer stays visible in the logs, rare enough not to bury the
# first real gate fire.
_STALE_WINDOW_RELOG_EVERY: int = 60


def _node_age_sec(observed_at: str | None) -> float | None:
    """Seconds since a substrate node's `observed_at` ISO timestamp.

    Returns None when absent or unparseable -- the gate treats None as "age
    unknown" and does NOT suppress on it. Deliberate: a parsing change upstream
    must not silently switch this evidence source off, which would be the same
    "detector quietly stops detecting" failure this whole arc has been chasing.
    A frozen node is the case we can actually detect, and that is the one
    guarded.
    """
    if not observed_at:
        return None
    try:
        parsed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed).total_seconds()


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
        # Per-kind cooldown lanes. Kinds not listed here share the global lane above
        # (self._last_metacog_trigger_ts / settings.metacog_cooldown_sec) -- see
        # _cooldown_sec_for_kind(). chat_turn was the first kind to need its own lane
        # (2026-07-23, a shared-lane bug: a burst of chat_turn fires silently starved
        # baseline/manual/pulse/relational/telemetry_anomaly's own fires); transport
        # (2026-07-24) got one from day one instead of repeating that bug.
        self._last_trigger_ts_by_kind: Dict[str, float] = {}
        self._last_baseline_scores: Tuple[float, float] = (-1.0, -1.0)
        self._bus_synaptic_falkor_client: Any = None
        self._baseline_skip_count: int = 0
        self._chat_turn_correlator: ChatTurnCorrelator | None = None
        self._downtime_tracker = DowntimeTransitionTracker()
        self._attention_self_model_reader: AttentionSelfModelReader | None = None
        # De-dupe keys for the two generative gates. These two are NOT equally
        # strong, on purpose -- stated plainly rather than implied:
        #
        # insight: keyed on `low_at`, the tick that armed the recovery. That is
        #   real episode identity: a genuinely new recovery requires a new low
        #   crossing, so one real recovery publishes exactly once.
        #   NOT keyed on `high_at`, which looks stable but isn't -- review
        #   finding 2026-07-30, reproduced against the real detector: when a
        #   high run breaks on a single sub-threshold tick and re-forms,
        #   `high_at` re-anchors to the new run and the same recovery fired
        #   twice, 390s apart, clearing the 300s cooldown. `low_at` was
        #   identical across both fires.
        #
        # flow: there is no equivalent stable anchor. `ended_at` is the newest
        #   tick in a trailing window, so it advances every tick while the
        #   plateau continues; this key therefore only suppresses re-processing
        #   the *same* newest row (real, since the 30s poll and the ~30s
        #   write tick drift against each other). Flow is treated as an ongoing
        #   *state* re-announced no more often than its own cooldown lane
        #   (EQUILIBRIUM_METACOG_FLOW_COOLDOWN_SEC, default 1800s), not as a
        #   once-per-episode event. Anchoring it to the true start of the
        #   contiguous run would require fetching further back than the
        #   evaluation window, which is not worth the extra query today.
        #   Measured over 21h of real history: 71 condition-true windows reduce
        #   to 7 actual publishes once that cooldown is applied.
        #
        # Both keys are recorded only after an *actual* publish, never on a
        # cooldown-suppressed one -- otherwise the event would be marked seen
        # while never having been emitted, and (since insight's key is stable)
        # never retried.
        self._last_insight_low_at: str | None = None
        self._last_flow_ended_at: str | None = None
        # Consecutive stale-window polls, for rate-limiting that warning the same
        # way AttentionSelfModelReader._log_failure rate-limits its own.
        self._stale_window_polls: int = 0
        # Checkpointed EWMA state for the repair_pressure_trend gate (hop 0 of
        # the stream-of-consciousness hop-chain design). Cold state until
        # _load_state() restores a real persisted checkpoint, if one exists --
        # a restart legitimately re-enters cold-start rather than fabricating
        # history, same discipline as every other persisted-state field here.
        self._repair_pressure_trend_state: MetacogTrendStateV1 = MetacogTrendStateV1()
        # Rising-edge state for the bus_synaptic transport branch. False at boot
        # means an anomaly already in progress at startup fires once on the
        # first poll -- correct ("this is news to this process") and bounded,
        # versus the pre-2026-07-30 behavior of firing every 30s forever.
        self._bus_synaptic_above_threshold: bool = False

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
                    if isinstance(key, (bytes, bytearray)):
                        key = key.decode("utf-8")
                    else:
                        key = str(key)
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

    async def _load_repair_pressure_trend_state(self) -> None:
        try:
            raw = await self.bus.redis.get(settings.metacog_repair_pressure_trend_state_key)
            if raw is None:
                return
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            self._repair_pressure_trend_state = state_from_dict(json.loads(raw))
        except Exception as e:
            logger.warning("Failed to load repair_pressure_trend state: %s", e)

    async def _persist_repair_pressure_trend_state(self) -> None:
        try:
            await self.bus.redis.set(
                settings.metacog_repair_pressure_trend_state_key,
                json.dumps(state_to_dict(self._repair_pressure_trend_state)),
            )
        except Exception as e:
            logger.warning("Failed to persist repair_pressure_trend state: %s", e)

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

    async def _publish_service_transitions(self, states: List[EquilibriumServiceState], now: datetime) -> None:
        """Detect and publish real status transitions for this tick's states.

        Called exactly once per publish tick (from `_publish_snapshot`, on
        `EQUILIBRIUM_PUBLISH_INTERVAL_SEC`) -- see `DowntimeTransitionTracker`'s
        own docstring for why it must not also be driven from the other,
        higher-frequency `_calculate_metrics()` call sites in this class.
        """
        if not settings.equilibrium_transition_publish_enable:
            return

        try:
            transitions = self._downtime_tracker.detect(
                states,
                now=now,
                source_service=settings.service_name,
                source_node=settings.node_name,
                producer_boot_id=self.boot_id,
            )
        except Exception as e:
            logger.error("Downtime transition detection failed: %s", e)
            return

        for transition in transitions:
            env = BaseEnvelope(
                kind="equilibrium.service.transition.v1",
                source=self._source(),
                payload=transition.model_dump(mode="json"),
            )
            try:
                await self.bus.publish(settings.channel_equilibrium_transition, env)
                logger.info(
                    "Published equilibrium service transition service=%s node=%s "
                    "from=%s to=%s down_duration_ms=%s",
                    transition.service,
                    transition.node,
                    transition.from_status,
                    transition.to_status,
                    transition.down_duration_ms,
                )
            except Exception as e:
                logger.error("Failed to publish equilibrium service transition: %s", e)

    async def _publish_snapshot(self) -> None:
        now = _utcnow()

        # Use shared calculation
        distress_score, zen_score, states = self._calculate_metrics()

        await self._publish_service_transitions(states, now)

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

    # Per-kind cooldown settings, checked in order. A kind not listed here shares the
    # global lane (settings.metacog_cooldown_sec / self._last_metacog_trigger_ts) --
    # baseline/manual/pulse/relational/telemetry_anomaly all still do, unchanged.
    # Each entry here is a kind designed to fire on a cadence fundamentally different
    # from that shared periodic/rare pattern; sharing the global lane would let a burst
    # of one kind silently starve the others (chat_turn's own bug, fixed 2026-07-23 --
    # see this service's README.md "chat_turn metacog trigger" section, "Operational
    # note"). transport (2026-07-24) got its own lane from day one instead of
    # repeating that bug.
    # insight/flow (2026-07-30) likewise get their own lanes from day one: they
    # are the first non-rupture generative kinds, fire on slow-moving regimes
    # rather than discrete incidents, and must not be able to starve any
    # rupture-shaped kind's fires (or each other's).
    _PER_KIND_COOLDOWN_SETTINGS_ATTR = {
        "chat_turn": "metacog_chat_turn_cooldown_sec",
        "transport": "metacog_transport_cooldown_sec",
        "insight": "metacog_insight_cooldown_sec",
        "flow": "metacog_flow_cooldown_sec",
        # Own lane from day one -- code review (2026-07-30) caught that this
        # trigger is evaluated on the same message, same branch, as the
        # pre-existing `relational` trigger; without its own lane, any
        # appraisal firing both would have relational silently starve this
        # one via the shared global lane every time. See
        # metacog_repair_pressure_trend_cooldown_sec's own comment.
        "repair_pressure_trend": "metacog_repair_pressure_trend_cooldown_sec",
    }

    def _cooldown_sec_for_kind(self, trigger_kind: str) -> float:
        attr = self._PER_KIND_COOLDOWN_SETTINGS_ATTR.get(trigger_kind)
        if attr is not None:
            return getattr(settings, attr)
        return settings.metacog_cooldown_sec

    async def _publish_metacog_trigger(self, trigger: MetacogTriggerV1) -> bool:
        """Returns True only if the trigger was really published to the bus.

        Callers that keep their own event-identity de-dupe state (the generative
        gates) must record that state only on a True return -- recording it on a
        cooldown-suppressed fire would mark the event as seen while never having
        emitted it. Every pre-existing caller ignores the return value, which is
        the unchanged behavior for them.
        """
        now_ts = datetime.now().timestamp()

        cooldown_sec = self._cooldown_sec_for_kind(trigger.trigger_kind)
        has_own_lane = trigger.trigger_kind in self._PER_KIND_COOLDOWN_SETTINGS_ATTR
        last_ts = (
            self._last_trigger_ts_by_kind.get(trigger.trigger_kind, 0.0)
            if has_own_lane
            else self._last_metacog_trigger_ts
        )

        if (now_ts - last_ts) < cooldown_sec:
            logger.info("Metacog trigger skipped due to cooldown (%s)", trigger.trigger_kind)
            return False

        if has_own_lane:
            self._last_trigger_ts_by_kind[trigger.trigger_kind] = now_ts
        else:
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
            return False

        if settings.metacog_publish_verb_request:
            # Legacy path intentionally disabled; must route through cortex-orch.
            logger.error(
                "Metacog legacy verb request is disabled (bypasses cortex-orch). "
                "Set EQUILIBRIUM_METACOG_PUBLISH_VERB_REQUEST=false and rely on "
                f"orion:equilibrium:metacog:trigger routing. trace_id={trace_meta['trace_id']}"
            )

        return True

    async def _handle_chat_turn_evidence(
        self,
        *,
        distress: float,
        zen: float,
        correlation_id: str,
        thought_event: Dict[str, Any] | None = None,
        run_artifact: Dict[str, Any] | None = None,
        timed_out: bool = False,
        timeout_reason: str | None = None,
    ) -> None:
        if not correlation_id or self._chat_turn_correlator is None:
            return

        merged_thought, merged_run, merged_timed_out, merged_timeout_reason = (
            await self._chat_turn_correlator.accumulate(
                correlation_id=correlation_id,
                thought_event=thought_event,
                run_artifact=run_artifact,
                timed_out=timed_out,
                timeout_reason=timeout_reason,
            )
        )

        if not is_chat_turn_evidence_terminal(
            thought_event=merged_thought, run_artifact=merged_run, timed_out=merged_timed_out
        ):
            return

        # Real, always-on evidence of what the gate actually saw and decided --
        # previously the only info-level output on this path was "Published
        # metacog trigger" on an actual fire, so a no-fire terminal evaluation
        # (the common case) was invisible: confirming "the gate correctly saw
        # this turn and found nothing" vs. "the gate silently never evaluated
        # it" required reverse-engineering grammar_atoms/harness-governor logs
        # by hand, live-verified as a real gap post-deploy.
        fired_conditions = evaluate_chat_turn_gate_conditions(
            thought_event=merged_thought,
            run_artifact=merged_run,
            timed_out=merged_timed_out,
            timeout_reason=merged_timeout_reason,
            surprise_threshold=settings.metacog_chat_turn_surprise_threshold,
        )
        reflection = (merged_run or {}).get("reflection") or {}
        substrate_appraisal = (merged_run or {}).get("substrate_appraisal") or {}
        logger.info(
            "chat_turn_gate_evaluated corr_id=%s fired=%s fired_conditions=%s "
            "disposition=%s boundary_register=%s alignment_verdict=%s strain_unresolved=%s "
            "surprise_level=%s compliance_verdict=%s exit_code=%s finalize_degraded_reason=%s "
            "timed_out=%s timeout_reason=%s",
            correlation_id,
            bool(fired_conditions),
            fired_conditions,
            (merged_thought or {}).get("disposition"),
            (merged_thought or {}).get("boundary_register"),
            reflection.get("alignment_verdict"),
            reflection.get("strain_unresolved"),
            substrate_appraisal.get("surprise_level"),
            (merged_run or {}).get("compliance_verdict"),
            (merged_run or {}).get("exit_code"),
            (merged_run or {}).get("finalize_degraded_reason"),
            merged_timed_out,
            merged_timeout_reason,
        )

        trigger = build_chat_turn_metacog_trigger(
            correlation_id=correlation_id,
            thought_event=merged_thought,
            run_artifact=merged_run,
            timed_out=merged_timed_out,
            timeout_reason=merged_timeout_reason,
            zen_state="zen" if zen > 0.5 else "not_zen",
            pressure=distress,
            recall_enabled=settings.metacog_recall_enabled,
            surprise_threshold=settings.metacog_chat_turn_surprise_threshold,
        )
        if trigger is not None:
            await self._publish_metacog_trigger(trigger)

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

    async def _maybe_emit_baseline_metacog_trigger(self) -> bool:
        """Evaluate distress/zen and publish a baseline metacog trigger when due."""
        distress, zen, _ = self._calculate_metrics()
        last_d, last_z = self._last_baseline_scores
        unchanged = abs(distress - last_d) < 0.01 and abs(zen - last_z) < 0.01
        max_skips = max(0, int(settings.metacog_baseline_max_skips))

        if unchanged and self._baseline_skip_count < max_skips:
            self._baseline_skip_count += 1
            logger.info(
                "Skipping baseline trigger (no change). distress=%.3f zen=%.3f skip=%d max_skips=%d",
                distress,
                zen,
                self._baseline_skip_count,
                max_skips,
            )
            return False

        if unchanged and max_skips > 0:
            logger.info(
                "Forcing baseline trigger after unchanged scores. distress=%.3f zen=%.3f skip=%d",
                distress,
                zen,
                self._baseline_skip_count,
            )

        self._baseline_skip_count = 0
        self._last_baseline_scores = (distress, zen)

        if settings.metacog_substrate_trigger_enable:
            substrate_trigger = build_substrate_metacog_trigger(
                zen_state="zen" if zen > 0.5 else "not_zen",
                pressure=distress,
                recall_enabled=settings.metacog_recall_enabled,
                dense_threshold=float(settings.metacog_substrate_dense_threshold),
                pulse_threshold=float(settings.metacog_substrate_pulse_threshold),
            )
            if substrate_trigger is not None:
                await self._publish_metacog_trigger(substrate_trigger)
                return True

        trigger = MetacogTriggerV1(
            trigger_kind="baseline",
            reason="scheduled_check",
            zen_state="zen" if zen > 0.5 else "not_zen",
            pressure=distress,
            recall_enabled=settings.metacog_recall_enabled,
        )
        await self._publish_metacog_trigger(trigger)
        return True

    async def _metacog_baseline_loop(self) -> None:
        if not settings.metacog_enable:
            return

        interval = float(settings.metacog_baseline_interval_sec)
        while not self._stop.is_set():
            try:
                await self._maybe_emit_baseline_metacog_trigger()
            except Exception as e:
                logger.error(f"Metacog baseline loop error: {e}")
            await asyncio.sleep(interval)

    async def _spark_heartbeat_loop(self) -> None:
        if not self.bus.enabled:
            return

        interval = float(settings.equilibrium_spark_heartbeat_interval_sec)
        while not self._stop.is_set():
            try:
                distress, zen, _ = self._calculate_metrics()
                now = _utcnow()
                trace_id = uuid4()
                trace = CognitionTracePayload(
                    correlation_id=str(trace_id),
                    mode="heartbeat",
                    verb="equilibrium_heartbeat",
                    timestamp=now.timestamp(),
                    source_service=settings.service_name,
                    source_node=settings.node_name,
                    metadata={
                        "heartbeat": True,
                        "distress": float(distress),
                        "zen": float(zen),
                    },
                )
                env = BaseEnvelope(
                    kind="cognition.trace",
                    source=self._source(),
                    correlation_id=trace_id,
                    id=trace_id,
                    payload=trace.model_dump(mode="json"),
                )
                await self.bus.publish(settings.channel_cognition_trace_pub, env)
                logger.info(
                    "Published equilibrium heartbeat trace "
                    f"trace_id={trace_id} channel={settings.channel_cognition_trace_pub}"
                )
            except Exception as e:
                logger.warning(f"Equilibrium heartbeat loop error: {e}")

            await asyncio.sleep(interval)

    def _get_bus_synaptic_falkor_client(self):
        """Cached read-only FalkorDB client for the durable substrate graph
        (orion_substrate, written by orion-substrate-runtime's
        _write_prediction_error_node -- a *different* graph from bus-mirror's
        own orion_bus_synapse). Fail-open: returns None on init error."""
        try:
            client = self._bus_synaptic_falkor_client
            if client is None:
                from orion.graph.falkor_client import RedisGraphQueryClient

                client = RedisGraphQueryClient(
                    uri=settings.falkordb_uri,
                    graph_name=settings.falkordb_substrate_graph,
                )
                self._bus_synaptic_falkor_client = client
            return client
        except Exception:
            logger.exception("bus_synaptic_falkor_client_init_failed")
            return None

    async def _bus_synaptic_poll_loop(self) -> None:
        """Third transport evidence source, polling node:substrate.bus_synaptic's
        prediction_error directly from FalkorDB -- not message-driven like
        Options A/C, so this needs its own timer, mirroring
        _spark_heartbeat_loop's shape."""
        if not (
            settings.metacog_transport_trigger_enable
            and settings.metacog_transport_bus_synaptic_poll_enable
        ):
            return

        interval = float(settings.metacog_transport_bus_synaptic_poll_interval_sec)
        while not self._stop.is_set():
            try:
                client = self._get_bus_synaptic_falkor_client()
                if client is not None:
                    # asyncio.to_thread: client.graph_query is a synchronous
                    # blocking Redis call -- running it inline would block
                    # this event loop (starving publish_loop/heartbeat/the
                    # bus-message consumer) for the round-trip. Mirrors how
                    # the sibling call in orion-substrate-runtime's
                    # _bus_synaptic_tick is dispatched (worker.py's own tick
                    # is offloaded via asyncio.to_thread at its call site).
                    # observed_at is fetched alongside the value so the gate
                    # can refuse a frozen node -- see its staleness guard.
                    rows = await asyncio.to_thread(
                        client.graph_query,
                        "MATCH (n:SubstrateNode) WHERE n.node_id = $node_id "
                        "RETURN n.prediction_error AS error, n.observed_at AS observed_at",
                        {"node_id": "node:substrate.bus_synaptic"},
                    )
                    error = None
                    observed_at = None
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        value = row.get("error")
                        if isinstance(value, (int, float)):
                            error = float(value)
                        raw_observed = row.get("observed_at")
                        if isinstance(raw_observed, str) and raw_observed:
                            observed_at = raw_observed
                    if error is not None:
                        distress, zen, _ = self._calculate_metrics()
                        trigger = build_transport_metacog_trigger_from_bus_synaptic(
                            error,
                            zen_state="zen" if zen > 0.5 else "not_zen",
                            pressure=distress,
                            recall_enabled=settings.metacog_recall_enabled,
                            error_threshold=settings.metacog_transport_bus_synaptic_error_threshold,
                            previously_above=self._bus_synaptic_above_threshold,
                            node_age_sec=_node_age_sec(observed_at),
                        )
                        # Edge state is updated from the RAW level, independently
                        # of whether a trigger was built: a fire suppressed by
                        # staleness or by the cooldown lane must still count as
                        # "we are now above", or the next poll would read as a
                        # fresh rising edge and re-fire forever -- reintroducing
                        # exactly the per-tick spam this change removes.
                        threshold = (
                            settings.metacog_transport_bus_synaptic_error_threshold
                        )
                        clear_at = (
                            threshold
                            * settings.metacog_transport_bus_synaptic_clear_ratio
                        )
                        if error >= threshold:
                            self._bus_synaptic_above_threshold = True
                        elif error < clear_at:
                            self._bus_synaptic_above_threshold = False
                        # Between clear_at and threshold the previous state is
                        # HELD -- the hysteresis band. Without it a reading
                        # oscillating across the threshold re-arms and re-fires
                        # on every crossing, which for a bimodal metric is most
                        # of them.
                        if trigger is not None:
                            await self._publish_metacog_trigger(trigger)
            except Exception:
                logger.exception("bus_synaptic_poll_loop_failed")

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def _get_attention_self_model_reader(self) -> AttentionSelfModelReader:
        """Cached read-only Postgres reader for `substrate_attention_self_model`
        (written by orion-substrate-runtime's _attention_self_model_tick, PR
        #1459).

        Unlike _get_bus_synaptic_falkor_client this cannot fail here and so does
        not return None: construction only stores a DSN, and the connection is
        opened lazily inside the reader, which is where the real fail-open
        (returning no samples) lives.
        """
        reader = self._attention_self_model_reader
        if reader is None:
            reader = AttentionSelfModelReader(
                dsn=settings.metacog_generative_postgres_uri
            )
            self._attention_self_model_reader = reader
        return reader

    def _generative_fetch_limit(self) -> int:
        """Rows to fetch per poll.

        Deliberately not just `window_ticks`: an operator raising
        EQUILIBRIUM_METACOG_FLOW_MIN_TICKS above it would otherwise turn the flow
        gate into a silent permanent no-op (the detector returns None whenever
        it receives fewer than min_ticks samples). Taking the max makes the
        documented "must cover the widest window either detector needs"
        invariant true by construction instead of by operator discipline.
        """
        return max(
            int(settings.metacog_generative_window_ticks),
            int(settings.metacog_flow_min_ticks),
            int(settings.metacog_insight_max_ticks_to_cross)
            + int(settings.metacog_insight_confirm_ticks),
        )

    def _generative_samples_are_fresh(self, samples: List[ConfidenceSample]) -> bool:
        """Reject a window whose newest row is too old to describe the present.

        The tick that writes these rows is itself flag-gated, so it can stop
        while this loop keeps polling -- and a frozen window keeps satisfying
        both gate conditions indefinitely (reproduced pre-fix: a window of rows
        3 days old fired the flow gate). Without this, the gates would be the
        "reducers alive but cursors stale" failure CLAUDE.md §0A calls out.
        """
        if not samples:
            return False
        max_age = float(settings.metacog_generative_max_age_sec)
        age_sec = (
            datetime.now(timezone.utc) - samples[-1].generated_at
        ).total_seconds()
        if age_sec > max_age:
            # Rate-limited the same way AttentionSelfModelReader._log_failure is:
            # at a 30s poll this would otherwise emit ~2880 identical warnings a
            # day while the writer is down. Re-logged periodically rather than
            # suppressed outright, because "the writer stopped" is exactly the
            # kind of condition that should stay visible until it's fixed.
            self._stale_window_polls += 1
            if (
                self._stale_window_polls == 1
                or self._stale_window_polls % _STALE_WINDOW_RELOG_EVERY == 0
            ):
                logger.warning(
                    "generative_metacog_window_stale age_sec=%.1f max_age_sec=%.1f "
                    "newest_row=%s consecutive_stale_polls=%d -- is "
                    "SUBSTRATE_ATTENTION_SELF_MODEL_TICK_ENABLED still writing?",
                    age_sec,
                    max_age,
                    samples[-1].generated_at.isoformat(),
                    self._stale_window_polls,
                )
            return False
        if self._stale_window_polls:
            logger.info(
                "generative_metacog_window_fresh_again after %d stale poll(s)",
                self._stale_window_polls,
            )
            self._stale_window_polls = 0
        return True

    async def _generative_metacog_poll_loop(self) -> None:
        """Evaluates the two generative (non-rupture) gates -- insight and flow.

        One loop, one query per tick, two conditions: both read the same
        trailing window of `prediction_error_confidence` rows, so polling the
        same table twice would be pure waste. Not message-driven (nothing
        publishes this table to the bus), so it needs its own timer -- same
        shape as _bus_synaptic_poll_loop.

        See docs/superpowers/specs/2026-07-28-collapse-mirror-generative-
        triggers-design.md. Both gates ship disabled; this returns immediately
        unless one is explicitly enabled.
        """
        if not settings.metacog_enable:
            return
        if not (
            settings.metacog_insight_trigger_enable
            or settings.metacog_flow_trigger_enable
        ):
            return

        interval = float(settings.metacog_generative_poll_interval_sec)
        limit = self._generative_fetch_limit()
        while not self._stop.is_set():
            try:
                reader = self._get_attention_self_model_reader()
                # asyncio.to_thread: psycopg2 is a synchronous blocking driver --
                # running the round-trip inline would stall the event loop
                # (starving publish_loop/heartbeat/the bus consumer). Same
                # reasoning as _bus_synaptic_poll_loop's own to_thread hop.
                samples = await asyncio.to_thread(
                    reader.fetch_recent_samples, limit=limit
                )
                if self._generative_samples_are_fresh(samples):
                    distress, zen, _ = self._calculate_metrics()
                    zen_state = "zen" if zen > 0.5 else "not_zen"

                    if settings.metacog_insight_trigger_enable:
                        await self._evaluate_insight_gate(
                            samples, zen_state=zen_state, pressure=distress
                        )
                    if settings.metacog_flow_trigger_enable:
                        await self._evaluate_flow_gate(
                            samples, zen_state=zen_state, pressure=distress
                        )
            except Exception:
                logger.exception("generative_metacog_poll_loop_failed")

            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _evaluate_insight_gate(
        self, samples: List[ConfidenceSample], *, zen_state: str, pressure: float
    ) -> None:
        max_ticks = int(settings.metacog_insight_max_ticks_to_cross)
        recovery = detect_confidence_recovery(
            samples,
            low_threshold=settings.metacog_insight_low_threshold,
            high_threshold=settings.metacog_insight_high_threshold,
            max_ticks_to_cross=max_ticks,
            confirm_ticks=int(settings.metacog_insight_confirm_ticks),
            # Tick count converted to a real wall-clock bound, because dropped
            # rows make row-distance and time-distance different things.
            max_cross_span_sec=(
                max_ticks
                * float(settings.metacog_generative_expected_tick_sec)
                * float(settings.metacog_generative_span_tolerance)
            ),
        )
        if recovery is None:
            return

        # Episode identity keyed on the arming low, not the high run -- see the
        # note in __init__ for the double-fire this avoids.
        low_at = recovery.low_at.isoformat()
        if low_at == self._last_insight_low_at:
            return

        trigger = build_insight_metacog_trigger(
            recovery,
            zen_state=zen_state,
            pressure=pressure,
            recall_enabled=settings.metacog_recall_enabled,
            low_threshold=settings.metacog_insight_low_threshold,
            high_threshold=settings.metacog_insight_high_threshold,
        )
        if trigger is None:
            return

        logger.info(
            "insight_gate_fired low=%.3f high=%.3f ticks_to_cross=%d span_sec=%.1f",
            recovery.low_value,
            recovery.high_value,
            recovery.ticks_to_cross,
            recovery.cross_span_sec,
        )
        # Only mark the episode seen once it was really emitted -- a
        # cooldown-suppressed fire must stay retryable on the next poll.
        if await self._publish_metacog_trigger(trigger):
            self._last_insight_low_at = low_at

    async def _evaluate_flow_gate(
        self, samples: List[ConfidenceSample], *, zen_state: str, pressure: float
    ) -> None:
        min_ticks = int(settings.metacog_flow_min_ticks)
        regime = detect_flow_regime(
            samples,
            floor=settings.metacog_flow_floor,
            max_stdev=settings.metacog_flow_max_stdev,
            min_ticks=min_ticks,
            # A window of N rows should cover about (N-1) tick intervals; anything
            # much longer means rows are missing and "sustained" would be a lie.
            max_span_sec=(
                max(min_ticks - 1, 1)
                * float(settings.metacog_generative_expected_tick_sec)
                * float(settings.metacog_generative_span_tolerance)
            ),
        )
        if regime is None:
            return

        ended_at = regime.ended_at.isoformat()
        if ended_at == self._last_flow_ended_at:
            return

        trigger = build_flow_metacog_trigger(
            regime,
            zen_state=zen_state,
            pressure=pressure,
            recall_enabled=settings.metacog_recall_enabled,
            floor=settings.metacog_flow_floor,
            max_stdev=settings.metacog_flow_max_stdev,
        )
        if trigger is None:
            return

        logger.info(
            "flow_gate_fired min=%.3f mean=%.3f stdev=%.4f ticks=%d span_sec=%.1f",
            regime.min_value,
            regime.mean_value,
            regime.stdev_value,
            regime.tick_count,
            regime.span_sec,
        )
        if await self._publish_metacog_trigger(trigger):
            self._last_flow_ended_at = ended_at

    async def _run(self) -> None:
        await self._load_state()
        await self._load_repair_pressure_trend_state()
        publisher = asyncio.create_task(self._publish_loop())
        collapse_task = asyncio.create_task(self._collapse_loop())
        metacog_task = asyncio.create_task(self._metacog_baseline_loop())
        heartbeat_task = None
        if settings.equilibrium_spark_heartbeat_enable:
            heartbeat_task = asyncio.create_task(self._spark_heartbeat_loop())
        bus_synaptic_poll_task = asyncio.create_task(self._bus_synaptic_poll_loop())
        generative_poll_task = asyncio.create_task(self._generative_metacog_poll_loop())

        # Build list of channels to subscribe to
        channels = [settings.health_channel]
        if settings.metacog_enable:
            channels.append(settings.channel_collapse_mirror_user_event)
            if settings.metacog_relational_trigger_enable:
                channels.append(settings.channel_repair_pressure_appraisal)
            if settings.metacog_telemetry_anomaly_trigger_enable:
                channels.append(settings.channel_field_channel_anomaly_score)
            if settings.metacog_chat_turn_trigger_enable:
                self._chat_turn_correlator = ChatTurnCorrelator(
                    self.bus.redis, ttl_seconds=settings.metacog_chat_turn_correlator_ttl_sec
                )
                channels.append(settings.channel_thought_artifact)
                channels.append(settings.channel_harness_run_artifact)
                channels.append(settings.channel_grammar_event)
            if settings.metacog_transport_trigger_enable:
                channels.append(settings.channel_rpc_health_snapshot)
                # channel_grammar_event may already be subscribed above for
                # chat_turn's own exec_turn_timeout/stance_timeout filtering --
                # avoid a duplicate pubsub subscription to the same channel.
                if settings.channel_grammar_event not in channels:
                    channels.append(settings.channel_grammar_event)

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

                        if channel == settings.channel_collapse_mirror_user_event:
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

                        elif (
                            channel == settings.channel_repair_pressure_appraisal
                            and settings.metacog_relational_trigger_enable
                        ):
                            # Real repair_pressure_v2 appraisal, published by
                            # orion-hub's pre_turn_appraisal_wiring.py whenever the
                            # repair_pressure paradigm actually ran -- replaces the
                            # retired turn_change_classify SHIFT gate as the
                            # relational trigger's evidence source.
                            trigger = build_repair_pressure_metacog_trigger(
                                correlation_id=str(payload_dict.get("correlation_id") or ""),
                                appraisal=payload_dict,
                                zen_state="zen" if zen > 0.5 else "not_zen",
                                pressure=distress,
                                recall_enabled=settings.metacog_recall_enabled,
                                level_floor=settings.metacog_relational_level_threshold,
                                confidence_floor=settings.metacog_relational_confidence_threshold,
                            )
                            if trigger is not None:
                                await self._publish_metacog_trigger(trigger)

                            # Hop 0 (stream-of-consciousness hop-chain design): fold this
                            # same appraisal into a persisted EWMA trend baseline,
                            # independent of whether the relational gate above fired.
                            # Distinct signal, distinct question ("has this kept
                            # happening" vs. "did this one turn cross a floor").
                            if settings.metacog_repair_pressure_trend_trigger_enable:
                                new_state, trend_trigger = evaluate_repair_pressure_trend(
                                    self._repair_pressure_trend_state,
                                    payload_dict,
                                    zen_state="zen" if zen > 0.5 else "not_zen",
                                    pressure=distress,
                                    recall_enabled=settings.metacog_recall_enabled,
                                    confidence_floor=settings.metacog_repair_pressure_trend_confidence_floor,
                                    min_samples=settings.metacog_repair_pressure_trend_min_samples,
                                    elevated_zscore=settings.metacog_repair_pressure_trend_elevated_zscore,
                                    sustained_hits=settings.metacog_repair_pressure_trend_sustained_hits,
                                )
                                self._repair_pressure_trend_state = new_state
                                asyncio.create_task(self._persist_repair_pressure_trend_state())
                                if trend_trigger is not None:
                                    await self._publish_metacog_trigger(trend_trigger)

                        elif (
                            channel == settings.channel_field_channel_anomaly_score
                            and settings.metacog_telemetry_anomaly_trigger_enable
                        ):
                            # Real field_channel_corpus.v1 anomaly score,
                            # published by orion-field-digester's periodic
                            # anomaly-scoring loop against a trained
                            # orion/mood_arc/fit_encoder.py encoder.
                            trigger = build_telemetry_anomaly_metacog_trigger(
                                correlation_id=str(payload_dict.get("correlation_id") or ""),
                                score=payload_dict,
                                zen_state="zen" if zen > 0.5 else "not_zen",
                                pressure=distress,
                                recall_enabled=settings.metacog_recall_enabled,
                                threshold_multiplier=settings.metacog_telemetry_anomaly_threshold_multiplier,
                            )
                            if trigger is not None:
                                await self._publish_metacog_trigger(trigger)

                        elif (
                            channel == settings.channel_thought_artifact
                            and settings.metacog_chat_turn_trigger_enable
                        ):
                            # Real ThoughtEventV1, published by orion-thought
                            # (services/orion-thought/app/bus_listener.py) for
                            # every chat turn.
                            correlation_id = str(payload_dict.get("correlation_id") or "")
                            await self._handle_chat_turn_evidence(
                                distress=distress,
                                zen=zen,
                                correlation_id=correlation_id,
                                thought_event=payload_dict,
                            )

                        elif (
                            channel == settings.channel_harness_run_artifact
                            and settings.metacog_chat_turn_trigger_enable
                        ):
                            # Real HarnessRunV1, published by orion-harness-governor
                            # (services/orion-harness-governor/app/bus_listener.py)
                            # on every real handle_harness_run_request exit path.
                            correlation_id = str(payload_dict.get("correlation_id") or "")
                            await self._handle_chat_turn_evidence(
                                distress=distress,
                                zen=zen,
                                correlation_id=correlation_id,
                                run_artifact=payload_dict,
                            )

                        elif channel == settings.channel_grammar_event:
                            # orion:grammar:event is the canonical sql-writer
                            # ingress channel and carries many unrelated event
                            # kinds -- only act on real timeout signals this
                            # service's enabled trigger kinds care about, everything
                            # else is ignored here:
                            #   exec_turn_timeout: the harness-governor RPC never
                            #     returned (services/orion-hub/scripts/grammar_emit.py::
                            #     build_turn_timeout_grammar_events, Patch B / PR #1287).
                            #   stance_disposition + text_value=="stance_timeout":
                            #     the *earlier* ThoughtClient.react() RPC never
                            #     returned (orion/hub/turn_orchestrator.py's
                            #     `if thought is None:` branch, same
                            #     _publish_unified_turn_chat_grammar() call as every
                            #     other stance_disposition atom). Both mean Hub gave
                            #     up before ever reaching the harness governor, so
                            #     run_artifact will never arrive for either.
                            #   rpc_transport_timeout: ANY rpc_request() timeout,
                            #     bus-wide across all 37+ real call sites sharing
                            #     OrionBusAsync (orion/core/bus/async_service.py::
                            #     _emit_rpc_timeout_grammar). Generalizes the two
                            #     above beyond harness/thought specifically -- feeds
                            #     the transport trigger, not chat_turn.
                            atom = payload_dict.get("atom") if isinstance(payload_dict, dict) else None
                            semantic_role = atom.get("semantic_role") if isinstance(atom, dict) else None

                            if settings.metacog_chat_turn_trigger_enable:
                                timeout_reason: str | None = None
                                if semantic_role == "exec_turn_timeout":
                                    timeout_reason = "exec_turn_timeout"
                                elif semantic_role == "stance_disposition" and isinstance(
                                    atom, dict
                                ) and atom.get("text_value") == "stance_timeout":
                                    timeout_reason = "stance_react_timeout"

                                if timeout_reason is not None:
                                    correlation_id = str(payload_dict.get("correlation_id") or "")
                                    await self._handle_chat_turn_evidence(
                                        distress=distress,
                                        zen=zen,
                                        correlation_id=correlation_id,
                                        timed_out=True,
                                        timeout_reason=timeout_reason,
                                    )

                            if settings.metacog_transport_trigger_enable and semantic_role == "rpc_transport_timeout":
                                correlation_id = str(payload_dict.get("correlation_id") or "")
                                trigger = build_transport_metacog_trigger_from_grammar_atom(
                                    atom,
                                    correlation_id=correlation_id,
                                    zen_state="zen" if zen > 0.5 else "not_zen",
                                    pressure=distress,
                                    recall_enabled=settings.metacog_recall_enabled,
                                )
                                if trigger is not None:
                                    await self._publish_metacog_trigger(trigger)

                        elif (
                            channel == settings.channel_rpc_health_snapshot
                            and settings.metacog_transport_trigger_enable
                        ):
                            # Real RpcHealthSnapshotV1, published every
                            # RPC_HEALTH_PUBLISH_INTERVAL_SEC by orion-cortex-exec /
                            # orion-cortex-orch (orion/core/bus/rpc_health_publish.py,
                            # PR #1313/#1315, live-verified).
                            trigger = build_transport_metacog_trigger_from_snapshot(
                                payload_dict,
                                zen_state="zen" if zen > 0.5 else "not_zen",
                                pressure=distress,
                                recall_enabled=settings.metacog_recall_enabled,
                                latency_p95_threshold_ms=settings.metacog_transport_latency_p95_threshold_ms,
                            )
                            if trigger is not None:
                                await self._publish_metacog_trigger(trigger)

                except Exception as e:
                    logger.warning("Failed to process message on %s: %s", channel, e)

        publisher.cancel()
        collapse_task.cancel()
        metacog_task.cancel()
        if heartbeat_task:
            heartbeat_task.cancel()
        bus_synaptic_poll_task.cancel()
        generative_poll_task.cancel()

        await asyncio.gather(
            publisher,
            collapse_task,
            metacog_task,
            *( [heartbeat_task] if heartbeat_task else [] ),
            bus_synaptic_poll_task,
            generative_poll_task,
            return_exceptions=True,
        )

        # Release the Postgres connection the generative gates cached, if any --
        # the FalkorDB client above is Redis-backed and pooled, this one holds a
        # real long-lived psycopg2 socket.
        if self._attention_self_model_reader is not None:
            self._attention_self_model_reader.close()
