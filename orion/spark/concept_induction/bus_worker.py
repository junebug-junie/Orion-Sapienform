from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Dict, List, Optional, Set
from uuid import uuid4

from orion.autonomy.deviation_gate import DeviationGate
from orion.autonomy.signal_drive_map import load_signal_drive_map
from orion.autonomy.signal_tension import SignalTensionSource
from orion.autonomy.tension_ratelimit import TensionRateLimiter
from orion.autonomy.endogenous_origination import OriginationConfig, OriginationEngine
from orion.signals.models import OrionSignalV1
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.work_queue import RedisStreamWorkQueue
from orion.core.schemas.concept_induction import ConceptProfile, ConceptProfileDelta
from orion.core.schemas.drives import ArtifactProvenance, GraphReadyArtifact, TensionEventV1, TurnDossierV1
from orion.schemas.feedback_frame import FeedbackFrameV1
from orion.schemas.self_state import SelfStateV1
from orion.schemas.vector.schemas import VectorWriteRequest

from .audit import build_drive_audit
from .drive_attribution import (
    compute_tick_attribution,
    dominant_drive_from_attribution,
    select_lead_tension,
)
from .drives import DriveEngine, DriveMathConfig, drive_state_from_values
from .dossier import build_evidence_items, build_source_event_ref, build_turn_dossier, extract_trace_id, extract_turn_id
from orion.autonomy.curiosity_reuse import outcome_from_followup, select_reusable_followup
from orion.autonomy.episode_journal import dispatch_autonomy_episode_journal
from orion.autonomy.fetch_backend_resolve import resolve_fetch_backend
from orion.autonomy.goal_archive import maybe_archive_after_goal_publish
from orion.autonomy.models import ActionOutcomeEmitV1
from orion.autonomy.policy_act import (
    maybe_execute_substrate_act_after_metabolism,
    resolve_episode_intent,
)
from orion.autonomy.substrate_metabolism import metabolize_substrate_signals, metabolism_enabled
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.schemas.world_pulse import WorldPulseRunResultV1

from .goals import GoalProposalEngine
from .identity import (
    build_identity_snapshot,
    entity_id_for_subject,
    model_layer_for_subject,
    resolve_subject_identity,
)
from .inducer import ConceptInducer, WindowEvent
from .settings import ConceptSettings
from .store import LocalProfileStore
from .summarizer import Summarizer
from .tensions import (
    derive_pressure_competition_tensions,
    extract_tensions,
    extract_tensions_from_action_outcome,
    extract_tensions_from_feedback,
    extract_tensions_from_self_state,
)
from .rdf_materialization import build_concept_profile_rdf_request
from .falkor_materialization import (
    build_falkor_substrate_store,
    materialize_concept_profile_to_falkor,
)
from .wp_stream_consumer import WorldPulseStreamConsumer, utcnow

logger = logging.getLogger("orion.spark.concept.worker")


@dataclass
class ConceptInductionTrigger:
    source_kind: str
    source_event_id: str
    correlation_id: str | None
    subjects: List[str]
    trigger_reason: str
    event_timestamp: datetime
    salience: float | None = None


@dataclass
class WorkerLivenessState:
    worker_initialized: bool = False
    worker_task_created: bool = False
    worker_task_done: bool = False
    worker_task_cancelled: bool = False
    worker_last_exception: str | None = None
    bus_consumer_started: bool = False
    subscribed_channels: List[str] | None = None
    last_event_received_at: str | None = None
    total_events_seen: int = 0
    total_events_accepted: int = 0
    total_events_rejected: int = 0
    stop_reason: str | None = None


def _service_ref(cfg: ConceptSettings) -> ServiceRef:
    return ServiceRef(
        name=cfg.service_name,
        version=cfg.service_version,
        node=cfg.node_name,
    )


class ConceptWorker:
    """Manages windowed intake and triggers induction."""

    def __init__(
        self,
        cfg: ConceptSettings,
        *,
        fetch_backend: Callable[..., Awaitable[dict]] | None = None,
        substrate_store: object | None = None,
    ) -> None:
        self.cfg = cfg
        self._fetch_backend = fetch_backend or resolve_fetch_backend()
        self._substrate_store = substrate_store
        self.bus = OrionBusAsync(
            cfg.orion_bus_url,
            enabled=cfg.orion_bus_enabled,
            enforce_catalog=cfg.orion_bus_enforce_catalog,
        )
        self.store = LocalProfileStore(cfg.store_path)
        self.drive_engine = DriveEngine(
            DriveMathConfig(
                decay_tau_sec=cfg.drive_decay_tau_sec,
                saturation_gain=cfg.drive_saturation_gain,
                activate_threshold=cfg.drive_activation_on,
                deactivate_threshold=cfg.drive_activation_off,
                leaky_math_enabled=cfg.drive_leaky_math_enabled,
            )
        )
        # Homeostatic tension source: deviation-gated OrionSignal/failure/health
        # -> TensionEventV1. Gated on cfg.homeostatic_drives_enabled at the call
        # site; constructed always so state persists across ticks.
        # Endogenous origination (default-off): a bounded ring of SelfStateV1
        # from which a spontaneous want can arise in exogenous silence. State
        # persists across ticks, so it is constructed once here.
        self.origination_engine = OriginationEngine(
            OriginationConfig(
                window=cfg.origination_window,
                threshold=cfg.origination_threshold,
                cooldown_sec=cfg.origination_cooldown_sec,
                mag_cap=cfg.endogenous_mag_cap,
                w_drift=cfg.origination_w_drift,
                w_dwell=cfg.origination_w_dwell,
                w_agency=cfg.origination_w_agency,
                exogenous_floor=cfg.origination_exogenous_floor,
            )
        )
        self.signal_tension_source = SignalTensionSource(
            gate=DeviationGate(
                alpha=cfg.deviation_ewma_alpha,
                z_threshold=cfg.deviation_z_threshold,
                sigma_floor=cfg.deviation_sigma_floor,
                impulse_k=cfg.signal_tension_impulse_k,
            ),
            sdm=load_signal_drive_map(),
            ratelimiter=TensionRateLimiter(
                cap=cfg.signal_tension_cap_per_window,
                window_sec=float(cfg.signal_tension_window_sec),
            ),
        )
        self.goal_engine = GoalProposalEngine(cfg.goal_proposal_cooldown_minutes)
        self.last_run: Dict[str, datetime] = {}
        service_ref = _service_ref(cfg)
        self.summarizer = Summarizer(
            use_cortex=cfg.use_cortex_orch,
            verb_name=cfg.cortex_orch_verb,
            bus=self.bus,
            request_channel=cfg.cortex_request_channel,
            result_prefix=cfg.cortex_result_prefix,
            timeout_sec=cfg.cortex_timeout_sec,
            service_ref=service_ref,
        )
        self.inducer = ConceptInducer(
            cfg,
            summarizer=self.summarizer if cfg.use_cortex_orch else None,
            store_loader=self.store.load,
            store_saver=self.store.save,
            service_ref=service_ref,
        )
        self.window: Dict[str, List[WindowEvent]] = {s: [] for s in cfg.subjects}
        self.inflight_subjects: Set[str] = set()
        self.recent_event_seen: Dict[str, datetime] = {}
        self.trigger_decisions: List[dict] = []
        self._previous_self_state: Optional[SelfStateV1] = None
        self._stopped = False
        self._liveness = WorkerLivenessState(
            worker_initialized=True,
            subscribed_channels=list(cfg.intake_channels),
        )
        self._service_ref = service_ref
        self._wp_stream_queue: RedisStreamWorkQueue | None = None
        self._wp_stream_consumer: WorldPulseStreamConsumer | None = None
        self._wp_stream_task: Optional[asyncio.Task] = None
        if cfg.wp_run_result_stream_enabled:
            self._wp_stream_queue = RedisStreamWorkQueue(cfg.orion_bus_url)
            self._wp_stream_consumer = WorldPulseStreamConsumer(
                queue=self._wp_stream_queue,
                stream=cfg.wp_run_result_stream_key,
                group=cfg.wp_run_result_stream_group,
                consumer=f"{cfg.node_name}:{os.getpid()}",
                dlq_stream=cfg.wp_run_result_dlq_key,
                handler=self._handle_wp_stream_envelope,
                is_processed=self.store.is_episode_run_processed,
                mark_processed=lambda run_id: self.store.mark_episode_run_processed(
                    run_id, processed_at=utcnow()
                ),
                max_attempts=cfg.wp_run_result_max_attempts,
                block_ms=cfg.wp_run_result_block_ms,
                autoclaim_idle_ms=cfg.wp_run_result_autoclaim_idle_ms,
            )

    async def _handle_wp_stream_envelope(self, envelope: BaseEnvelope) -> None:
        """Handler for durable world-pulse run-result stream messages.

        Routes through the same ``handle_envelope`` path the pub/sub loop uses; the
        stream key stands in as the intake-channel label (the world_pulse branch keys
        on ``env.kind``, not the channel).
        """
        await self.handle_envelope(envelope, self.cfg.wp_run_result_stream_key)

    def _pubsub_patterns(self) -> List[str]:
        """Pub/sub intake channels, minus any that are consumed via a durable stream.

        When the run-result stream is enabled, drop ``orion:world_pulse:run:result``
        from pub/sub so the event is processed exactly once (via the stream), not twice.
        """
        patterns = list(self.cfg.intake_channels)
        if self.cfg.wp_run_result_stream_enabled:
            patterns = [p for p in patterns if p != "orion:world_pulse:run:result"]
        if self.cfg.homeostatic_drives_enabled:
            # Specific organ/failure/health channels only (never the signals
            # wildcard). Deduped against intake so we never subscribe twice.
            for extra in self._homeostatic_channels():
                if extra not in patterns:
                    patterns.append(extra)
        return patterns

    def _homeostatic_channels(self) -> List[str]:
        return [
            *self.cfg.homeostatic_signal_channels,
            *self.cfg.homeostatic_failure_channels,
        ]

    def _homeostatic_source(self, intake_channel: str) -> Optional[str]:
        """Classify a channel as a homeostatic source: 'signal' | 'failure', or
        None if it is a normal concept-induction intake. Health degradation
        arrives as a mesh_health OrionSignal on the 'signal' rail, so there is no
        separate equilibrium branch."""
        if intake_channel in self.cfg.homeostatic_signal_channels:
            return "signal"
        if intake_channel in self.cfg.homeostatic_failure_channels:
            return "failure"
        return None

    async def _dispatch_autonomy_episode_journal(
        self,
        parent: BaseEnvelope,
        *,
        goal_artifact_id: str,
        spawned_correlation_id: str,
        narrative_seed: str,
    ) -> dict:
        return await dispatch_autonomy_episode_journal(
            bus=self.bus,
            parent=parent,
            source=self._service_ref,
            goal_artifact_id=goal_artifact_id,
            spawned_correlation_id=spawned_correlation_id,
            narrative_seed=narrative_seed,
            cortex_request_channel=self.cfg.cortex_request_channel,
            cortex_result_prefix=self.cfg.cortex_result_prefix,
            journal_write_channel=self.cfg.journal_write_channel,
            timeout_sec=self.cfg.autonomy_episode_journal_timeout_sec,
            session_id=self.cfg.journal_session_id,
            user_id=self.cfg.journal_user_id,
            author=self.cfg.journal_author,
        )

    def _accepted_source_mapping(self) -> dict:
        return {
            "chat_turn": [
                "channel contains chat:history",
                "channel contains chat:social",
                "channel contains chat:gpt",
            ],
            "journal_write": [
                "kind=journal.entry.created.v1",
                "kind=journal.entry.write.v1",
                "kind startswith journal",
            ],
            "dream_result": [
                "kind=dream.result.v1",
                "kind startswith dream.",
            ],
            "self_review_result": [
                "kind startswith self.review",
                "kind startswith self_review",
            ],
            "metacog_tick": [
                "channel contains metacognition:tick",
                "kind startswith metacognition.tick",
            ],
            "cognition_trace": ["channel contains cognition:trace"],
            "collapse_event": ["channel contains collapse"],
            "self_state_update": ["kind=substrate.self_state.v1"],
            "feedback_outcome": ["kind=feedback.frame.v1"],
            "generic_activity": ["fallback"],
        }

    def _prune_window(self, subject: str) -> None:
        window = self.window.get(subject, [])
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.cfg.window_max_minutes)
        self.window[subject] = [
            w for w in window if w.timestamp >= cutoff
        ][-self.cfg.window_max_events :]

    def _detect_subject(self, env: BaseEnvelope, intake_channel: str = "") -> str:
        return resolve_subject_identity(env, intake_channel)

    def _model_layer(self, subject: str, intake_channel: str) -> str:
        del intake_channel
        return model_layer_for_subject(subject)

    def _entity_id(self, subject: str, model_layer: str) -> str:
        return entity_id_for_subject(subject, model_layer)

    def _extract_text(self, env: BaseEnvelope) -> Optional[str]:
        payload = env.payload
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        if not isinstance(payload, dict):
            if isinstance(payload, str):
                return payload
            return None
        for key in ("content", "text", "message", "summary"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val
        prompt = payload.get("prompt")
        response = payload.get("response")
        if isinstance(prompt, str) and isinstance(response, str):
            merged = f"{prompt}\n{response}".strip()
            if merged:
                return merged
        if isinstance(response, str) and response.strip():
            return response
        return None

    def _source_kind(self, env: BaseEnvelope, intake_channel: str) -> str:
        lowered_kind = (env.kind or "").lower()
        lowered_channel = (intake_channel or "").lower()
        if (
            "chat:history" in lowered_channel
            or "chat:social" in lowered_channel
            or "chat:gpt" in lowered_channel
        ):
            return "chat_turn"
        if lowered_kind in {"journal.entry.created.v1", "journal.entry.write.v1"} or "journal" in lowered_kind:
            return "journal_write"
        if lowered_kind == "dream.result.v1" or lowered_kind.startswith("dream."):
            return "dream_result"
        if lowered_kind.startswith("self.review") or lowered_kind.startswith("self_review"):
            return "self_review_result"
        if "metacognition:tick" in lowered_channel or lowered_kind.startswith("metacognition.tick"):
            return "metacog_tick"
        if "cognition:trace" in lowered_channel:
            return "cognition_trace"
        if "collapse" in lowered_channel:
            return "collapse_event"
        if lowered_kind == "substrate.self_state.v1":
            return "self_state_update"
        if lowered_kind == "feedback.frame.v1":
            return "feedback_outcome"
        return "generic_activity"

    def _select_trigger_subjects(
        self,
        env: BaseEnvelope,
        intake_channel: str,
        source_kind: str,
        text: Optional[str],
    ) -> List[str]:
        payload = env.payload.model_dump() if hasattr(env.payload, "model_dump") else env.payload
        payload = payload if isinstance(payload, dict) else {}
        selected: Set[str] = set()
        detected = self._detect_subject(env, intake_channel)
        if detected in {"orion", "juniper", "relationship"}:
            selected.add(detected)
        role = str(payload.get("role") or "").strip().lower()
        user = str(payload.get("user") or payload.get("speaker") or "").strip().lower()
        lowered_text = (text or "").lower()

        if source_kind in {"metacog_tick", "self_review_result", "cognition_trace"}:
            selected.add("orion")
        if source_kind in {"chat_turn", "journal_write", "dream_result"}:
            if role == "assistant" or "orion" in lowered_text:
                selected.add("orion")
            if role == "user" or user.startswith("juniper") or "juniper" in lowered_text:
                selected.add("juniper")
                selected.add("relationship")
        if source_kind == "collapse_event":
            selected.add("relationship")
        ordered = [subject for subject in self.cfg.subjects if subject in selected]
        return ordered

    def _record_trigger_decision(self, record: dict) -> None:
        self.trigger_decisions.append(record)
        max_records = max(1, int(self.cfg.concept_trigger_recent_decisions))
        self.trigger_decisions = self.trigger_decisions[-max_records:]

    def _should_skip_due_to_dedupe(self, trigger: ConceptInductionTrigger) -> bool:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.cfg.concept_trigger_dedupe_sec)
        self.recent_event_seen = {
            key: seen_at for key, seen_at in self.recent_event_seen.items() if seen_at >= cutoff
        }
        if trigger.source_event_id in self.recent_event_seen:
            return True
        self.recent_event_seen[trigger.source_event_id] = now
        return False

    def _build_trigger(
        self,
        env: BaseEnvelope,
        intake_channel: str,
        text: Optional[str],
    ) -> ConceptInductionTrigger | None:
        source_kind = self._source_kind(env, intake_channel)
        subjects = self._select_trigger_subjects(env, intake_channel, source_kind, text)
        if not subjects:
            return None
        source_event_id = str(getattr(env, "id", None) or env.correlation_id or uuid4())
        corr_id = str(env.correlation_id) if env.correlation_id else None
        return ConceptInductionTrigger(
            source_kind=source_kind,
            source_event_id=source_event_id,
            correlation_id=corr_id,
            subjects=subjects,
            trigger_reason=f"{source_kind}:subject-selection",
            event_timestamp=env.created_at,
        )

    async def _publish_profile(self, profile: ConceptProfile, corr_id) -> None:
        env = BaseEnvelope(
            kind="memory.concepts.profile.v1",
            source=_service_ref(self.cfg),
            correlation_id=corr_id,
            payload=profile.model_dump(mode="json"),
        )
        await self.bus.publish(self.cfg.profile_channel, env)

    async def _publish_delta(self, delta: ConceptProfileDelta, corr_id) -> None:
        env = BaseEnvelope(
            kind="memory.concepts.delta.v1",
            source=_service_ref(self.cfg),
            correlation_id=corr_id,
            payload=delta.model_dump(mode="json"),
        )
        await self.bus.publish(self.cfg.delta_channel, env)

    async def _forward_vector(self, profile: ConceptProfile, corr_id) -> None:
        for concept in profile.concepts:
            req = VectorWriteRequest(
                id=concept.concept_id,
                kind="memory.concept",
                content=concept.label,
                metadata={
                    "subject": profile.subject,
                    "revision": profile.revision,
                },
                vector=None,
            )
            env = BaseEnvelope(
                kind="vector.write",
                source=_service_ref(self.cfg),
                correlation_id=corr_id,
                payload=req.model_dump(mode="json"),
            )
            await self.bus.publish(self.cfg.forward_vector_channel, env)

    def _graph_materialization_destination(self) -> str:
        backend = str(self.cfg.concept_profile_graph_backend or "disabled").strip().lower()
        if backend == "falkor":
            return f"falkor:{self.cfg.falkordb_substrate_graph}"
        if backend == "rdf":
            return self.cfg.forward_rdf_channel
        return backend

    def _get_substrate_store(self):
        if self._substrate_store is not None:
            return self._substrate_store
        self._substrate_store = build_falkor_substrate_store(
            uri=self.cfg.falkordb_uri,
            graph_name=self.cfg.falkordb_substrate_graph,
            hydrate=False,
        )
        return self._substrate_store

    async def _materialize_profile_graph(self, profile: ConceptProfile, corr_id) -> bool:
        concept_count = len(profile.concepts)
        cluster_count = len(profile.clusters)
        graph_write_attempted = False
        graph_write_succeeded = False
        error_kind: str | None = None
        backend = str(self.cfg.concept_profile_graph_backend or "disabled").strip().lower()
        destination = backend
        schema_kind = "spark.concept_profile.graph.v1"

        if backend == "disabled":
            logger.info(
                "concept_profile_graph_materialization_skipped backend=disabled subject=%s "
                "revision=%s concept_count=%d cluster_count=%d",
                profile.subject,
                profile.revision,
                concept_count,
                cluster_count,
            )
            return True

        try:
            if backend == "falkor":
                destination = f"falkor:{self.cfg.falkordb_substrate_graph}"
                schema_kind = "substrate.concept_atlas.falkor.v1"
                graph_write_attempted = True
                # Falkor client is a sync redis client; keep GRAPH.QUERY off the
                # event loop so induction ticks don't stall bus intake.
                store = self._get_substrate_store()
                await asyncio.to_thread(
                    materialize_concept_profile_to_falkor,
                    profile=profile,
                    store=store,
                )
                graph_write_succeeded = True
            elif backend == "rdf":
                destination = self.cfg.forward_rdf_channel
                graph_request = build_concept_profile_rdf_request(
                    profile=profile,
                    correlation_id=str(corr_id) if corr_id else None,
                    writer_service=self.cfg.service_name,
                    writer_version=self.cfg.service_version,
                )
                graph_write_attempted = True
                env = BaseEnvelope(
                    kind="rdf.write.request",
                    source=_service_ref(self.cfg),
                    correlation_id=corr_id,
                    payload=graph_request.model_dump(mode="json"),
                )
                await self.bus.publish(self.cfg.forward_rdf_channel, env)
                graph_write_succeeded = True
            else:
                # Validator should prevent this; fail closed.
                logger.warning(
                    "concept_profile_graph_materialization_unknown_backend backend=%s subject=%s",
                    backend,
                    profile.subject,
                )
                return False
        except Exception as exc:  # noqa: BLE001
            error_kind = type(exc).__name__
            logger.warning(
                "concept_profile_postsave stage=graph_materialization_failed subject=%s revision=%s "
                "correlation_id=%s destination=%s error=%s",
                profile.subject,
                profile.revision,
                corr_id,
                destination,
                error_kind,
            )
            logger.warning(
                "concept_profile_graph_materialization subject=%s revision=%s concept_count=%d "
                "cluster_count=%d graph_write_attempted=%s graph_write_succeeded=%s "
                "destination=%s schema_kind=%s error_kind=%s",
                profile.subject,
                profile.revision,
                concept_count,
                cluster_count,
                graph_write_attempted,
                graph_write_succeeded,
                destination,
                schema_kind,
                error_kind,
            )
            return False

        logger.info(
            "concept_profile_graph_materialization subject=%s revision=%s concept_count=%d "
            "cluster_count=%d graph_write_attempted=%s graph_write_succeeded=%s destination=%s "
            "schema_kind=%s error_kind=%s",
            profile.subject,
            profile.revision,
            concept_count,
            cluster_count,
            graph_write_attempted,
            graph_write_succeeded,
            destination,
            schema_kind,
            error_kind,
        )
        return True

    def _log_drive_pressure_probe(self, subject: str, pressures: dict) -> None:
        """Measurement-only (Phase 4, 2026-07-12): log `DriveEngine`'s pressure
        vector right after every `save_drive_state` so it can be compared
        offline against `AutonomyStateV2`'s independently-computed pressures
        (logged from `chat_stance._run_autonomy_reducer`) by grepping both
        services' logs and correlating on `subject` + nearby timestamp. Never
        raises: a logging failure here must not break the drive-update rail,
        which runs on live bus traffic.
        """
        try:
            logger.info(
                "drive_engine_pressure_probe subject=%s pressures=%s",
                subject,
                {k: round(v, 4) for k, v in dict(pressures or {}).items()},
            )
        except Exception:
            logger.warning("drive_engine_pressure_probe_failed subject=%s", subject, exc_info=True)

    async def _publish_drive_state(self, payload, corr_id) -> None:
        env = BaseEnvelope(
            kind="memory.drives.state.v1",
            source=_service_ref(self.cfg),
            correlation_id=corr_id,
            payload=payload.model_dump(mode="json"),
        )
        await self.bus.publish(self.cfg.drive_state_channel, env)

    async def _publish_tension_event(self, event, corr_id) -> None:
        env = BaseEnvelope(
            kind="memory.tension.event.v1",
            source=_service_ref(self.cfg),
            correlation_id=corr_id,
            payload=event.model_dump(mode="json"),
        )
        await self.bus.publish(self.cfg.tension_event_channel, env)

    async def _publish_artifact(self, artifact: GraphReadyArtifact, channel: str, corr_id) -> None:
        env = BaseEnvelope(
            kind=artifact.kind,
            source=_service_ref(self.cfg),
            correlation_id=corr_id,
            payload=artifact.model_dump(mode="json"),
        )
        await self.bus.publish(channel, env)

    async def _publish_dossier(self, dossier: TurnDossierV1, corr_id) -> None:
        env = BaseEnvelope(
            kind=dossier.kind,
            source=_service_ref(self.cfg),
            correlation_id=corr_id,
            payload=dossier.model_dump(mode="json"),
        )
        await self.bus.publish(self.cfg.turn_dossier_channel, env)

    async def _publish_action_outcome(
        self, emit: ActionOutcomeEmitV1, corr_id
    ) -> None:
        env = BaseEnvelope(
            kind="action.outcome.emit.v1",
            source=_service_ref(self.cfg),
            correlation_id=corr_id,
            payload=emit.model_dump(mode="json"),
        )
        await self.bus.publish(self.cfg.action_outcome_channel, env)

    @staticmethod
    def _payload_dict(env: BaseEnvelope) -> dict:
        payload = env.payload
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        return payload if isinstance(payload, dict) else {}

    def _tensions_from_self_state(self, env: BaseEnvelope, intake_channel: str) -> List[TensionEventV1]:
        try:
            self_state = SelfStateV1.model_validate(self._payload_dict(env))
        except Exception:
            logger.warning("substrate_self_state_parse_failed kind=%s", env.kind)
            return []
        tensions = extract_tensions_from_self_state(
            envelope=env,
            intake_channel=intake_channel,
            self_state=self_state,
            previous_self_state=self._previous_self_state,
        )
        self._previous_self_state = self_state

        # Endogenous origination (default-off, proposal mode): observe every
        # self-state into the ring, and — only in exogenous silence — let a want
        # arise from Orion's own internal dynamics. The endogenous tension is an
        # ordinary TensionEventV1 (origin=endogenous) merged here, so it flows
        # through the unchanged publish -> DriveEngine -> goals path. The exogenous
        # tension count for this tick is the count already produced above, so a
        # tick that carried real turn-effect deltas suppresses endogeny.
        if self.cfg.endogenous_origination_enabled:
            try:
                self.origination_engine.observe(self_state)
                endogenous = self.origination_engine.maybe_originate(
                    exogenous_tension_count=len(tensions),
                    now=datetime.now(timezone.utc),
                )
                if endogenous is not None:
                    logger.info(
                        "endogenous_origination_fired drive_impacts=%s signal=%s",
                        endogenous.drive_impacts,
                        self.origination_engine.last_signal,
                    )
                    tensions = list(tensions) + [endogenous]
            except Exception:
                logger.warning("endogenous_origination_failed", exc_info=True)

        return tensions

    def _tensions_from_feedback(self, env: BaseEnvelope, intake_channel: str) -> List[TensionEventV1]:
        try:
            frame = FeedbackFrameV1.model_validate(self._payload_dict(env))
        except Exception:
            logger.warning("feedback_frame_parse_failed kind=%s", env.kind)
            return []
        return extract_tensions_from_feedback(
            envelope=env,
            intake_channel=intake_channel,
            feedback_frame=frame,
        )

    def _tensions_from_action_outcome(self, env: BaseEnvelope, intake_channel: str) -> List[TensionEventV1]:
        # Self-publish guard: this service is itself already a producer on
        # orion:autonomy:action:outcome (curiosity-fetch outcomes, via
        # _publish_action_outcome below) with no self-filter anywhere in
        # this file today. Redis pub/sub delivers to all subscribers
        # including the publisher, so without this check a curiosity-fetch
        # outcome would loop back through this NEW path too, on top of its
        # own dedicated in-process handling -- double-processing the same
        # event. Only outcomes from OTHER services (today: Layer 9 /
        # orion-execution-dispatch-runtime) reach the extractor below.
        source_name = getattr(getattr(env, "source", None), "name", None)
        if source_name == self.cfg.service_name:
            return []
        try:
            outcome = ActionOutcomeEmitV1.model_validate(self._payload_dict(env))
        except Exception:
            logger.warning("action_outcome_parse_failed kind=%s", env.kind)
            return []
        logger.info(
            "action_outcome_received source=%s kind=%s action_id=%s success=%s",
            source_name, outcome.kind, outcome.action_id, outcome.success,
        )
        result = extract_tensions_from_action_outcome(
            envelope=env,
            intake_channel=intake_channel,
            outcome=outcome,
        )
        logger.info(
            "action_outcome_tensions_extracted count=%d drive_impacts=%s",
            len(result), [t.drive_impacts for t in result],
        )
        return result

    def _parse_signal(self, env: BaseEnvelope) -> Optional[OrionSignalV1]:
        try:
            return OrionSignalV1.model_validate(self._payload_dict(env))
        except Exception:
            return None

    async def _handle_signal_drive_tick(
        self, env: BaseEnvelope, intake_channel: str, source: str
    ) -> None:
        """Thin drive-only rail for homeostatic sources: mint deviation tensions,
        update + publish drive state/audit, and return. No concept induction, no
        goals, no identity snapshot. Degrades to a no-op (never raises)."""
        subject = "orion"
        model_layer = model_layer_for_subject(subject)
        entity_id = entity_id_for_subject(subject, model_layer)
        now = datetime.now(timezone.utc)
        now_mono = now.timestamp()

        tensions: List[TensionEventV1] = []
        try:
            if source == "signal":
                sig = self._parse_signal(env)
                if sig is not None:
                    tensions = self.signal_tension_source.from_signal(
                        sig, now=now_mono, channel=intake_channel
                    )
            elif source == "failure":
                tensions = self.signal_tension_source.from_failure(
                    severity=self.cfg.homeostatic_failure_severity,
                    now=now_mono,
                    channel=intake_channel,
                    correlation_id=str(env.correlation_id),
                    summary=env.kind or "failure_event",
                )
        except Exception:
            logger.warning(
                "homeostatic_signal_tick_failed channel=%s source=%s",
                intake_channel, source, exc_info=True,
            )
            return

        if not tensions:
            # No deviation this tick. The leaky integrator decays on wall-clock
            # elapsed at the next update, so skipping here loses no decay.
            return

        # The whole drive-update/publish section is guarded: this rail runs at
        # ~1/s and the bus loop re-raises, so a single bad prior state (e.g. a
        # tz-naive updated_at) must degrade to a no-op, never tear down the worker.
        try:
            for tension in tensions:
                await self._publish_tension_event(tension, env.correlation_id)

            prior = self.store.load_drive_state(subject)
            previous_ts: Optional[datetime] = None
            if isinstance(prior.get("updated_at"), str):
                try:
                    previous_ts = datetime.fromisoformat(prior["updated_at"])
                    if previous_ts.tzinfo is None:
                        previous_ts = previous_ts.replace(tzinfo=timezone.utc)
                except ValueError:
                    previous_ts = None

            pressures, activations = self.drive_engine.update(
                previous_pressures=prior.get("pressures"),
                previous_activations=prior.get("activations"),
                tensions=tensions,
                now=now,
                previous_ts=previous_ts,
            )
            self.store.save_drive_state(
                subject, pressures=pressures, activations=activations, updated_at=now
            )
            self._log_drive_pressure_probe(subject, pressures)

            trace_id = extract_trace_id(env)
            turn_id = extract_turn_id(env)
            drive_state = drive_state_from_values(
                subject=subject,
                model_layer=model_layer,
                entity_id=entity_id,
                ts=now,
                pressures=pressures,
                activations=activations,
                confidence=0.72,
                correlation_id=str(env.correlation_id),
                trace_id=trace_id,
                turn_id=turn_id,
                provenance=ArtifactProvenance(
                    intake_channel=intake_channel,
                    correlation_id=str(env.correlation_id),
                    trace_id=trace_id,
                    turn_id=turn_id,
                    tension_refs=[t.artifact_id for t in tensions],
                ),
                related_nodes=[f"subject:{subject}"] + [t.artifact_id for t in tensions],
            )
            await self._publish_drive_state(drive_state, env.correlation_id)

            tick_attribution = compute_tick_attribution(tensions)
            lead_tension = select_lead_tension(tensions)
            dominant_drive = dominant_drive_from_attribution(
                tick_attribution, lead_tension=lead_tension
            )
            drive_audit = build_drive_audit(
                env=env,
                intake_channel=intake_channel,
                drive_state=drive_state,
                tensions=tensions,
                tick_attribution=tick_attribution,
                dominant_drive=dominant_drive,
            )
            await self._publish_artifact(drive_audit, self.cfg.drive_audit_channel, env.correlation_id)
        except Exception:
            logger.warning(
                "homeostatic_drive_update_failed channel=%s source=%s",
                intake_channel, source, exc_info=True,
            )
            return

    async def handle_envelope(self, env: BaseEnvelope, intake_channel: str) -> None:
        # Homeostatic sources ride a thin drive-only rail: they update drives and
        # return BEFORE concept induction, so a 1/s biometrics tick never triggers
        # windowing/induction. Gated on the flag; unmapped content mints nothing.
        if self.cfg.homeostatic_drives_enabled:
            source = self._homeostatic_source(intake_channel)
            if source is not None:
                await self._handle_signal_drive_tick(env, intake_channel, source)
                return

        text = self._extract_text(env)

        if env.kind == "substrate.self_state.v1":
            subject = "orion"
            model_layer = model_layer_for_subject(subject)
            entity_id = entity_id_for_subject(subject, model_layer)
            spark_tensions = self._tensions_from_self_state(env, intake_channel)
        elif env.kind == "feedback.frame.v1":
            subject = "orion"
            model_layer = model_layer_for_subject(subject)
            entity_id = entity_id_for_subject(subject, model_layer)
            spark_tensions = self._tensions_from_feedback(env, intake_channel)
        elif env.kind == "action.outcome.emit.v1":
            subject = "orion"
            model_layer = model_layer_for_subject(subject)
            entity_id = entity_id_for_subject(subject, model_layer)
            spark_tensions = self._tensions_from_action_outcome(env, intake_channel)
        else:
            subject = self._detect_subject(env, intake_channel)
            model_layer = self._model_layer(subject, intake_channel)
            entity_id = self._entity_id(subject, model_layer)
            spark_tensions = extract_tensions(
                envelope=env,
                intake_channel=intake_channel,
                subject=subject,
                model_layer=model_layer,
                entity_id=entity_id,
            )
        published_artifacts: List[GraphReadyArtifact] = []
        for tension in spark_tensions:
            await self._publish_tension_event(tension, env.correlation_id)
            published_artifacts.append(tension)

        metabolism_tensions: List[TensionEventV1] = []
        metabolism_drive_deltas: dict[str, float] = {}
        metabolism_curiosity_signals: List[FrontierInvocationSignalV1] = []
        metabolism_curiosity_notes: List[str] = []
        spawned_correlation_id: str | None = None
        wp_result: WorldPulseRunResultV1 | None = None
        if env.kind == "world.pulse.run.result.v1":
            try:
                wp_result = WorldPulseRunResultV1.model_validate(self._payload_dict(env))
                spawned_correlation_id = wp_result.run.run_id
                if metabolism_enabled():
                    metabolism = metabolize_substrate_signals(world_pulse_result=wp_result)
                    metabolism_tensions = list(metabolism.tensions)
                    metabolism_drive_deltas = dict(metabolism.drive_deltas)
                    metabolism_curiosity_signals = list(metabolism.curiosity_signals)
                    metabolism_curiosity_notes = [
                        str(sig.evidence_summary or "").strip()
                        for sig in metabolism_curiosity_signals
                        if str(sig.evidence_summary or "").strip()
                    ]
                    for tension in metabolism_tensions:
                        await self._publish_tension_event(tension, env.correlation_id)
                        published_artifacts.append(tension)
            except Exception:
                logger.warning("substrate_metabolism_failed kind=%s", env.kind, exc_info=True)

        all_spark_tensions = spark_tensions + metabolism_tensions

        prior_drive_state = self.store.load_drive_state(subject)
        previous_ts = None
        if isinstance(prior_drive_state.get("updated_at"), str):
            try:
                previous_ts = datetime.fromisoformat(prior_drive_state["updated_at"])
            except ValueError:
                previous_ts = None
        now = datetime.now(timezone.utc)
        pressures, activations = self.drive_engine.update(
            previous_pressures=prior_drive_state.get("pressures"),
            previous_activations=prior_drive_state.get("activations"),
            tensions=all_spark_tensions,
            now=now,
            previous_ts=previous_ts,
        )
        self.store.save_drive_state(subject, pressures=pressures, activations=activations, updated_at=now)
        self._log_drive_pressure_probe(subject, pressures)

        pressure_tensions = derive_pressure_competition_tensions(
            envelope=env,
            intake_channel=intake_channel,
            subject=subject,
            model_layer=model_layer,
            entity_id=entity_id,
            pressures=pressures,
        )
        for tension in pressure_tensions:
            await self._publish_tension_event(tension, env.correlation_id)
            published_artifacts.append(tension)

        all_tensions = all_spark_tensions + pressure_tensions

        trace_id = extract_trace_id(env)
        turn_id = extract_turn_id(env)
        source_event_ref = build_source_event_ref(env, intake_channel)
        evidence_items = build_evidence_items(env, intake_channel, all_tensions[0].provenance.evidence_text if all_tensions else None)
        drive_state = drive_state_from_values(
            subject=subject,
            model_layer=model_layer,
            entity_id=entity_id,
            ts=now,
            pressures=pressures,
            activations=activations,
            confidence=0.72,
            correlation_id=str(env.correlation_id),
            trace_id=trace_id,
            turn_id=turn_id,
            provenance=ArtifactProvenance(
                intake_channel=intake_channel,
                correlation_id=str(env.correlation_id),
                trace_id=trace_id,
                turn_id=turn_id,
                evidence_text=(all_tensions[0].provenance.evidence_text if all_tensions else None),
                evidence_summary=(evidence_items[0].summary if evidence_items else None),
                source_event_refs=[source_event_ref],
                evidence_items=evidence_items,
                tension_refs=[tension.artifact_id for tension in all_tensions],
            ),
            related_nodes=[f"subject:{subject}"] + [tension.artifact_id for tension in all_tensions],
        )
        await self._publish_drive_state(drive_state, env.correlation_id)

        tick_attribution = compute_tick_attribution(
            all_tensions,
            metabolism_deltas=metabolism_drive_deltas or None,
        )
        lead_tension = select_lead_tension(all_tensions)
        dominant_drive = dominant_drive_from_attribution(
            tick_attribution,
            lead_tension=lead_tension,
        )
        drive_audit = build_drive_audit(
            env=env,
            intake_channel=intake_channel,
            drive_state=drive_state,
            tensions=all_tensions,
            tick_attribution=tick_attribution,
            dominant_drive=dominant_drive,
        )
        await self._publish_artifact(drive_audit, self.cfg.drive_audit_channel, env.correlation_id)
        published_artifacts.append(drive_audit)

        identity_snapshot = build_identity_snapshot(
            drive_state=drive_state,
            source_event_ref=source_event_ref,
            evidence_items=evidence_items,
            tensions=all_tensions,
        )
        await self._publish_artifact(identity_snapshot, self.cfg.identity_snapshot_channel, env.correlation_id)
        published_artifacts.append(identity_snapshot)

        window_summary = evidence_items[0].summary if evidence_items else drive_state.provenance.evidence_summary
        if metabolism_curiosity_notes:
            gap_summary = "; ".join(metabolism_curiosity_notes)
            window_summary = (
                f"{window_summary}; {gap_summary}" if window_summary else gap_summary
            )
        goal_decision = self.goal_engine.propose(
            env=env,
            intake_channel=intake_channel,
            drive_state=drive_state,
            tensions=all_tensions,
            store=self.store,
            dominant_drive=drive_audit.dominant_drive,
            window_summary=window_summary,
            spawned_correlation_id=spawned_correlation_id,
        )
        suppressed_signatures: List[str] = []
        if goal_decision.proposal is not None:
            await self._publish_artifact(goal_decision.proposal, self.cfg.goal_proposal_channel, env.correlation_id)
            published_artifacts.append(goal_decision.proposal)
            maybe_archive_after_goal_publish(subject=subject)
        elif goal_decision.suppressed_signature:
            suppressed_signatures.append(goal_decision.suppressed_signature)

        # Idempotency backstop for the episode path (flag-gated so the flag-off path is
        # byte-identical: nothing is ever marked, so is_episode_run_processed is always
        # False). Guards the expensive substrate act (Firecrawl fetch + journal RPC)
        # against at-least-once stream redelivery AND any accidental double-delivery via
        # the pub/sub path (e.g. a glob intake pattern that _pubsub_patterns can't strip).
        episode_dedup_enabled = self.cfg.wp_run_result_stream_enabled
        episode_already_processed = (
            episode_dedup_enabled
            and bool(spawned_correlation_id)
            and self.store.is_episode_run_processed(spawned_correlation_id)
        )
        if (
            env.kind == "world.pulse.run.result.v1"
            and metabolism_enabled()
            and metabolism_curiosity_signals
            and spawned_correlation_id
            and episode_already_processed
        ):
            logger.info(
                "wp_run_result_episode_skip_duplicate run_id=%s (already processed)",
                spawned_correlation_id,
            )
        elif (
            env.kind == "world.pulse.run.result.v1"
            and metabolism_enabled()
            and metabolism_curiosity_signals
            and spawned_correlation_id
        ):
            policy_budget: dict[str, int] = {}
            try:

                async def _journal_dispatch(**kwargs):
                    return await self._dispatch_autonomy_episode_journal(env, **kwargs)

                prefetched_outcome = None
                if (
                    wp_result is not None
                    and wp_result.digest is not None
                    and wp_result.digest.curiosity_followups
                ):
                    reusable = select_reusable_followup(
                        wp_result.digest.curiosity_followups,
                        metabolism_curiosity_signals,
                    )
                    if reusable is not None:
                        prefetched_outcome = outcome_from_followup(
                            reusable, run_id=spawned_correlation_id
                        )
                        logger.info(
                            "wp_curiosity_followup_reused run_id=%s section=%s action_id=%s articles=%s",
                            spawned_correlation_id,
                            reusable.section,
                            prefetched_outcome.action_id,
                            len(prefetched_outcome.articles),
                        )

                act_result = await maybe_execute_substrate_act_after_metabolism(
                    episode_intent=resolve_episode_intent(
                        store=self.store,
                        subject=subject,
                        run_id=spawned_correlation_id,
                    ),
                    drive_state=drive_state,
                    curiosity_signals=metabolism_curiosity_signals,
                    spawned_correlation_id=spawned_correlation_id,
                    fetch_backend=self._fetch_backend,
                    journal_dispatch=_journal_dispatch,
                    budget_used=policy_budget,
                    episode_journal_enabled=self.cfg.autonomy_episode_journal_enabled,
                    prefetched_outcome=prefetched_outcome,
                    # Without these, maybe_execute_readonly_recall_after_goal's
                    # own `if bus is None: return decision, None` guard degrades
                    # silently before any RPC is attempted -- the recall-first
                    # capability would evaluate policy, log "allowed", and never
                    # actually fire. Reuse this worker's own bus connection and
                    # identity, same as every other bus call this class makes.
                    recall_bus=self.bus,
                    recall_source=_service_ref(self.cfg),
                )
                if act_result.fetch_outcome is not None:
                    try:
                        # correlation_id traces back to the triggering world-pulse
                        # envelope; the spawned autonomy run id is embedded in action_id.
                        await self._publish_action_outcome(
                            ActionOutcomeEmitV1.from_outcome(
                                subject=subject,
                                outcome=act_result.fetch_outcome,
                            ),
                            env.correlation_id,
                        )
                        logger.info(
                            "action_outcome_emitted subject=%s action_id=%s success=%s spawned=%s",
                            subject,
                            act_result.fetch_outcome.action_id,
                            act_result.fetch_outcome.success,
                            spawned_correlation_id,
                        )
                    except Exception:
                        logger.warning(
                            "action_outcome_emit_failed subject=%s spawned=%s",
                            subject,
                            spawned_correlation_id,
                            exc_info=True,
                        )
                if act_result.recall_outcome is not None:
                    try:
                        await self._publish_action_outcome(
                            ActionOutcomeEmitV1.from_outcome(
                                subject=subject,
                                outcome=act_result.recall_outcome,
                            ),
                            env.correlation_id,
                        )
                        logger.info(
                            "action_outcome_emitted subject=%s action_id=%s success=%s spawned=%s",
                            subject,
                            act_result.recall_outcome.action_id,
                            act_result.recall_outcome.success,
                            spawned_correlation_id,
                        )
                    except Exception:
                        logger.warning(
                            "action_outcome_emit_failed subject=%s spawned=%s",
                            subject,
                            spawned_correlation_id,
                            exc_info=True,
                        )
            except Exception:
                logger.warning(
                    "substrate_act_after_metabolism_failed subject=%s spawned=%s",
                    subject,
                    spawned_correlation_id,
                    exc_info=True,
                )
            if episode_dedup_enabled and spawned_correlation_id:
                # Mark after the attempt so a later redelivery does not re-run the fetch
                # + journal RPC. The block swallows its own errors, so "attempted" is the
                # dedup boundary; a mid-handler crash (before this line) leaves it unmarked
                # and is safely re-attempted on the next stream redelivery.
                self.store.mark_episode_run_processed(
                    spawned_correlation_id, processed_at=utcnow()
                )

        dossier = build_turn_dossier(
            env=env,
            intake_channel=intake_channel,
            subject=subject,
            model_layer=model_layer,
            entity_id=entity_id,
            published=published_artifacts,
            suppressed_goal_signatures=suppressed_signatures,
        )
        await self._publish_dossier(dossier, env.correlation_id)

        # Structured substrate/feedback signals carry no text: skip concept induction.
        if env.kind in {"substrate.self_state.v1", "feedback.frame.v1", "action.outcome.emit.v1"}:
            return

        source_kind = self._source_kind(env, intake_channel)
        subjects = self._select_trigger_subjects(env, intake_channel, source_kind, text)
        source_event_id = str(getattr(env, "id", None) or env.correlation_id or uuid4())
        corr_id = str(env.correlation_id) if env.correlation_id else None
        logger.info(
            "concept_induction_trigger_received source_kind=%s source_event_id=%s correlation_id=%s subjects=%s",
            source_kind,
            source_event_id,
            corr_id,
            ",".join(subjects),
        )
        if not self.cfg.concept_autonomous_trigger_enabled:
            logger.info(
                "concept_induction_trigger_decision decision=disabled source_kind=%s source_event_id=%s correlation_id=%s",
                source_kind,
                source_event_id,
                corr_id,
            )
            self._record_trigger_decision({
                "decision": "disabled",
                "reason": "autonomous_trigger_disabled",
                "source_kind": source_kind,
                "source_event_id": source_event_id,
                "correlation_id": corr_id,
                "subjects": subjects,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            return

        if not subjects:
            logger.info(
                "concept_induction_trigger_rejected reason=no_subject_match source_kind=%s kind=%s channel=%s correlation_id=%s",
                source_kind,
                env.kind,
                intake_channel,
                env.correlation_id,
            )
            self._record_trigger_decision({
                "decision": "rejected",
                "reason": "no_subject_match",
                "source_kind": source_kind,
                "source_event_id": source_event_id,
                "correlation_id": corr_id,
                "subjects": [],
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            return
        trigger = ConceptInductionTrigger(
            source_kind=source_kind,
            source_event_id=source_event_id,
            correlation_id=corr_id,
            subjects=subjects,
            trigger_reason=f"{source_kind}:subject-selection",
            event_timestamp=env.created_at,
        )
        if self._should_skip_due_to_dedupe(trigger):
            logger.info(
                "concept_induction_trigger_decision decision=coalesced source_kind=%s source_event_id=%s subjects=%s",
                trigger.source_kind,
                trigger.source_event_id,
                ",".join(trigger.subjects),
            )
            self._record_trigger_decision({
                "decision": "coalesced",
                "reason": "dedupe_window",
                "trigger": asdict(trigger),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            return

        logger.info(
            "concept_induction_subject_selected source_kind=%s source_event_id=%s subjects=%s",
            trigger.source_kind,
            trigger.source_event_id,
            ",".join(trigger.subjects),
        )
        for selected_subject in trigger.subjects:
            if text:
                event = WindowEvent(text=text, timestamp=env.created_at, envelope=env, intake_channel=intake_channel)
                self.window.setdefault(selected_subject, []).append(event)
                self._prune_window(selected_subject)
            if selected_subject in self.inflight_subjects:
                logger.info(
                    "concept_induction_generation_skipped decision=queued source_kind=%s subject=%s correlation_id=%s reason=inflight",
                    trigger.source_kind,
                    selected_subject,
                    trigger.correlation_id,
                )
                self._record_trigger_decision({
                    "decision": "queued",
                    "reason": "subject_inflight",
                    "subject": selected_subject,
                    "trigger": asdict(trigger),
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                })
                continue
            last = self.last_run.get(selected_subject)
            cooldown = timedelta(seconds=self.cfg.concept_trigger_cooldown_sec)
            if last and (now - last) < cooldown:
                remaining = int((cooldown - (now - last)).total_seconds())
                logger.info(
                    "concept_induction_generation_skipped decision=skipped_due_to_cooldown source_kind=%s subject=%s correlation_id=%s cooldown_remaining_sec=%d",
                    trigger.source_kind,
                    selected_subject,
                    trigger.correlation_id,
                    remaining,
                )
                self._record_trigger_decision({
                    "decision": "skipped_due_to_cooldown",
                    "reason": f"cooldown_remaining_sec={remaining}",
                    "subject": selected_subject,
                    "trigger": asdict(trigger),
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                })
                continue
            if not self.window.get(selected_subject):
                logger.info(
                    "concept_induction_generation_skipped decision=skipped_no_window source_kind=%s subject=%s correlation_id=%s",
                    trigger.source_kind,
                    selected_subject,
                    trigger.correlation_id,
                )
                self._record_trigger_decision({
                    "decision": "skipped_no_window",
                    "reason": "empty_window",
                    "subject": selected_subject,
                    "trigger": asdict(trigger),
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                })
                continue
            logger.info(
                "concept_induction_generation_enqueued source_kind=%s subject=%s correlation_id=%s source_event_id=%s",
                trigger.source_kind,
                selected_subject,
                trigger.correlation_id,
                trigger.source_event_id,
            )
            self._record_trigger_decision({
                "decision": "triggered",
                "subject": selected_subject,
                "trigger": asdict(trigger),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            self.inflight_subjects.add(selected_subject)
            try:
                await self.run_for_subject(selected_subject, corr_id=env.correlation_id)
            finally:
                self.inflight_subjects.discard(selected_subject)

    async def run_for_subject(self, subject: str, corr_id=None) -> None:
        corr_id = corr_id or uuid4()
        window = list(self.window.get(subject, []))
        if not window:
            logger.info(
                "concept_induction_run_for_subject_skipped subject=%s correlation_id=%s reason=empty_window",
                subject,
                corr_id,
            )
            return
        logger.info(
            "concept_induction_run_for_subject_invoked subject=%s correlation_id=%s window_events=%d",
            subject,
            corr_id,
            len(window),
        )
        result = await self.inducer.run(subject=subject, window=window)
        logger.info(
            "concept_profile_postsave stage=profile_saved subject=%s revision=%s correlation_id=%s",
            result.profile.subject,
            result.profile.revision,
            corr_id,
        )
        await self._publish_profile(result.profile, corr_id)
        logger.info(
            "concept_profile_postsave stage=profile_published subject=%s revision=%s correlation_id=%s",
            result.profile.subject,
            result.profile.revision,
            corr_id,
        )
        graph_backend = str(self.cfg.concept_profile_graph_backend or "disabled").strip().lower()
        if graph_backend == "disabled":
            logger.info(
                "concept_profile_postsave stage=graph_materialization_skipped subject=%s revision=%s correlation_id=%s destination=disabled",
                result.profile.subject,
                result.profile.revision,
                corr_id,
            )
            await self._materialize_profile_graph(result.profile, corr_id)
        else:
            logger.info(
                "concept_profile_postsave stage=graph_materialization_attempted subject=%s revision=%s correlation_id=%s destination=%s",
                result.profile.subject,
                result.profile.revision,
                corr_id,
                self._graph_materialization_destination(),
            )
            materialized = await self._materialize_profile_graph(result.profile, corr_id)
            if materialized:
                logger.info(
                    "concept_profile_postsave stage=graph_materialization_published subject=%s revision=%s correlation_id=%s destination=%s",
                    result.profile.subject,
                    result.profile.revision,
                    corr_id,
                    self._graph_materialization_destination(),
                )
        if result.delta:
            await self._publish_delta(result.delta, corr_id)
            logger.info(
                "concept_profile_postsave stage=delta_published subject=%s revision=%s correlation_id=%s",
                result.profile.subject,
                result.profile.revision,
                corr_id,
            )
        await self._forward_vector(result.profile, corr_id)
        logger.info(
            "concept_profile_postsave stage=vector_forwarded subject=%s revision=%s correlation_id=%s",
            result.profile.subject,
            result.profile.revision,
            corr_id,
        )
        self.last_run[subject] = datetime.now(timezone.utc)

    def trigger_status(self) -> dict:
        return {
            "intake_channels": list(self.cfg.intake_channels),
            "autonomous_trigger_enabled": self.cfg.concept_autonomous_trigger_enabled,
            "source_kind_mapping": self._accepted_source_mapping(),
            "subjects": list(self.cfg.subjects),
            "cooldown_sec": self.cfg.concept_trigger_cooldown_sec,
            "dedupe_sec": self.cfg.concept_trigger_dedupe_sec,
            "last_induced_at": {
                subject: timestamp.isoformat() for subject, timestamp in self.last_run.items()
            },
            "inflight_subjects": sorted(self.inflight_subjects),
            "recent_decisions": list(self.trigger_decisions),
        }

    def worker_liveness_status(self) -> dict:
        return asdict(self._liveness)

    def mark_task_created(self) -> None:
        self._liveness.worker_task_created = True
        self._liveness.worker_task_done = False
        self._liveness.worker_task_cancelled = False
        self._liveness.worker_last_exception = None
        self._liveness.stop_reason = None

    def record_task_exit(self, task: asyncio.Task) -> None:
        self._liveness.worker_task_done = task.done()
        self._liveness.worker_task_cancelled = task.cancelled()
        if task.cancelled():
            self._liveness.stop_reason = "cancelled"
            logger.info("concept_induction_worker_task_stopped reason=cancelled")
            return
        try:
            exc = task.exception()
        except Exception as callback_exc:  # noqa: BLE001
            self._liveness.worker_last_exception = repr(callback_exc)
            self._liveness.stop_reason = "done_callback_error"
            logger.exception("concept_induction_worker_task_failed error=%s", callback_exc)
            return
        if exc is not None:
            self._liveness.worker_last_exception = repr(exc)
            self._liveness.stop_reason = "failed"
            logger.exception("concept_induction_worker_task_failed error=%s", exc)
        else:
            self._liveness.stop_reason = "completed"
            logger.info("concept_induction_worker_task_stopped reason=completed")

    async def start(self) -> None:
        logger.info("concept_induction_worker_starting")
        await self.bus.connect()
        if self._wp_stream_consumer is not None and self._wp_stream_queue is not None:
            await self._wp_stream_queue.connect()
            # Concurrency note: this runs the stream consumer as a SECOND task alongside
            # the pub/sub loop in the same event loop. Both call into self.store
            # (LocalProfileStore), whose load/save are synchronous with no await between a
            # read and its paired write, so read-modify-write bursts (e.g. drive-state
            # load->save) never interleave. Do NOT introduce an await inside a store
            # read-modify-write without adding a lock, or the two tasks can lose updates.
            self._wp_stream_task = asyncio.create_task(
                self._wp_stream_consumer.run_forever(),
                name="wp-run-result-stream-consumer",
            )
            logger.info(
                "wp_run_result_stream_consumer_task_created stream=%s group=%s",
                self.cfg.wp_run_result_stream_key,
                self.cfg.wp_run_result_stream_group,
            )
        patterns = self._pubsub_patterns()
        uses_glob = any(any(ch in pattern for ch in "*?[") for pattern in patterns)
        self._liveness.subscribed_channels = list(patterns)
        self._liveness.bus_consumer_started = False
        self._liveness.worker_task_done = False
        self._liveness.worker_task_cancelled = False
        self._liveness.worker_last_exception = None
        self._liveness.stop_reason = None
        logger.info(
            "concept_induction_worker_started autonomous_trigger_loop=%s bus_enabled=%s bus_enforce_catalog=%s "
            "intake_channels=%s source_kind_mapping=%s trigger_subjects=%s cooldown_sec=%d dedupe_sec=%d "
            "window_max_events=%d window_max_minutes=%d recent_decisions=%d",
            self.cfg.concept_autonomous_trigger_enabled,
            self.cfg.orion_bus_enabled,
            self.cfg.orion_bus_enforce_catalog,
            patterns,
            self._accepted_source_mapping(),
            self.cfg.subjects,
            self.cfg.concept_trigger_cooldown_sec,
            self.cfg.concept_trigger_dedupe_sec,
            self.cfg.window_max_events,
            self.cfg.window_max_minutes,
            self.cfg.concept_trigger_recent_decisions,
        )
        try:
            async with self.bus.subscribe(*patterns, patterns=uses_glob) as pubsub:
                self._liveness.bus_consumer_started = True
                logger.info("concept_induction_worker_subscribed channels=%s", patterns)
                async for msg in self.bus.iter_messages(pubsub):
                    self._liveness.total_events_seen += 1
                    self._liveness.last_event_received_at = datetime.now(timezone.utc).isoformat()
                    if not isinstance(msg, dict):
                        self._liveness.total_events_rejected += 1
                        continue
                    data = msg.get("data")
                    if data is None:
                        self._liveness.total_events_rejected += 1
                        continue
                    channel = msg.get("channel")
                    if hasattr(channel, "decode"):
                        channel = channel.decode("utf-8")
                    decoded = self.bus.codec.decode(data)
                    if not decoded.ok or decoded.envelope is None:
                        self._liveness.total_events_rejected += 1
                        logger.warning("Concept intake decode failed channel=%s error=%s", channel, decoded.error)
                        continue
                    logger.info(
                        "concept_induction_worker_event_received channel=%s kind=%s",
                        str(channel or "unknown"),
                        decoded.envelope.kind,
                    )
                    self._liveness.total_events_accepted += 1
                    await self.handle_envelope(decoded.envelope, str(channel or "unknown"))
        except asyncio.CancelledError:
            self._liveness.stop_reason = "cancelled"
            logger.info("concept_induction_worker_task_stopped reason=cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            self._liveness.worker_last_exception = repr(exc)
            self._liveness.stop_reason = "failed"
            logger.exception("concept_induction_worker_task_failed error=%s", exc)
            raise
        finally:
            if self._liveness.stop_reason is None:
                self._liveness.stop_reason = "stopped"
                logger.info("concept_induction_worker_task_stopped reason=stopped")
            await self.stop()

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._wp_stream_consumer is not None:
            self._wp_stream_consumer.stop()
        if self._wp_stream_task is not None:
            self._wp_stream_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._wp_stream_task
        if self._wp_stream_queue is not None:
            with suppress(Exception):
                await self._wp_stream_queue.close()
        await self.bus.close()
