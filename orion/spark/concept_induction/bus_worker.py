from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set
from uuid import uuid4

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.concept_induction import ConceptProfile, ConceptProfileDelta
from orion.core.schemas.drives import ArtifactProvenance, GraphReadyArtifact, TurnDossierV1
from orion.schemas.vector.schemas import VectorWriteRequest

from .audit import build_drive_audit
from .drives import DriveEngine, DriveMathConfig, drive_state_from_values
from .dossier import build_evidence_items, build_source_event_ref, build_turn_dossier, extract_trace_id, extract_turn_id
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
from .tensions import extract_tensions
from .rdf_materialization import build_concept_profile_rdf_request

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


def _service_ref(cfg: ConceptSettings) -> ServiceRef:
    return ServiceRef(
        name=cfg.service_name,
        version=cfg.service_version,
        node=cfg.node_name,
    )


class ConceptWorker:
    """Manages windowed intake and triggers induction."""

    def __init__(self, cfg: ConceptSettings) -> None:
        self.cfg = cfg
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
            )
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
        self._stopped = False

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

    async def _materialize_profile_graph(self, profile: ConceptProfile, corr_id) -> None:
        concept_count = len(profile.concepts)
        cluster_count = len(profile.clusters)
        graph_write_attempted = False
        graph_write_succeeded = False
        error_kind: str | None = None
        try:
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
        except Exception as exc:  # noqa: BLE001
            error_kind = type(exc).__name__
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
                self.cfg.forward_rdf_channel,
                "spark.concept_profile.graph.v1",
                error_kind,
            )
            return

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
            self.cfg.forward_rdf_channel,
            "spark.concept_profile.graph.v1",
            error_kind,
        )

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

    async def handle_envelope(self, env: BaseEnvelope, intake_channel: str) -> None:
        text = self._extract_text(env)
        subject = self._detect_subject(env, intake_channel)
        model_layer = self._model_layer(subject, intake_channel)
        entity_id = self._entity_id(subject, model_layer)

        tensions = extract_tensions(
            envelope=env,
            intake_channel=intake_channel,
            subject=subject,
            model_layer=model_layer,
            entity_id=entity_id,
        )
        published_artifacts: List[GraphReadyArtifact] = []
        for tension in tensions:
            await self._publish_tension_event(tension, env.correlation_id)
            published_artifacts.append(tension)

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
            tensions=tensions,
            now=now,
            previous_ts=previous_ts,
        )
        self.store.save_drive_state(subject, pressures=pressures, activations=activations, updated_at=now)

        trace_id = extract_trace_id(env)
        turn_id = extract_turn_id(env)
        source_event_ref = build_source_event_ref(env, intake_channel)
        evidence_items = build_evidence_items(env, intake_channel, tensions[0].provenance.evidence_text if tensions else None)
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
                evidence_text=(tensions[0].provenance.evidence_text if tensions else None),
                evidence_summary=(evidence_items[0].summary if evidence_items else None),
                source_event_refs=[source_event_ref],
                evidence_items=evidence_items,
                tension_refs=[tension.artifact_id for tension in tensions],
            ),
            related_nodes=[f"subject:{subject}"] + [tension.artifact_id for tension in tensions],
        )
        await self._publish_drive_state(drive_state, env.correlation_id)

        drive_audit = build_drive_audit(env=env, intake_channel=intake_channel, drive_state=drive_state, tensions=tensions)
        await self._publish_artifact(drive_audit, self.cfg.drive_audit_channel, env.correlation_id)
        published_artifacts.append(drive_audit)

        identity_snapshot = build_identity_snapshot(
            drive_state=drive_state,
            source_event_ref=source_event_ref,
            evidence_items=evidence_items,
            tensions=tensions,
        )
        await self._publish_artifact(identity_snapshot, self.cfg.identity_snapshot_channel, env.correlation_id)
        published_artifacts.append(identity_snapshot)

        goal_decision = self.goal_engine.propose(
            env=env,
            intake_channel=intake_channel,
            drive_state=drive_state,
            tensions=tensions,
            store=self.store,
        )
        suppressed_signatures: List[str] = []
        if goal_decision.proposal is not None:
            await self._publish_artifact(goal_decision.proposal, self.cfg.goal_proposal_channel, env.correlation_id)
            published_artifacts.append(goal_decision.proposal)
        elif goal_decision.suppressed_signature:
            suppressed_signatures.append(goal_decision.suppressed_signature)

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
        await self._publish_profile(result.profile, corr_id)
        await self._materialize_profile_graph(result.profile, corr_id)
        if result.delta:
            await self._publish_delta(result.delta, corr_id)
        await self._forward_vector(result.profile, corr_id)
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

    async def start(self) -> None:
        await self.bus.connect()
        patterns = list(self.cfg.intake_channels)
        uses_glob = any(any(ch in pattern for ch in "*?[") for pattern in patterns)
        logger.info(
            "concept_induction_worker_startup autonomous_trigger_loop=%s bus_enabled=%s bus_enforce_catalog=%s "
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
                async for msg in self.bus.iter_messages(pubsub):
                    if not isinstance(msg, dict):
                        continue
                    data = msg.get("data")
                    if data is None:
                        continue
                    channel = msg.get("channel")
                    if hasattr(channel, "decode"):
                        channel = channel.decode("utf-8")
                    decoded = self.bus.codec.decode(data)
                    if not decoded.ok or decoded.envelope is None:
                        logger.warning("Concept intake decode failed channel=%s error=%s", channel, decoded.error)
                        continue
                    await self.handle_envelope(decoded.envelope, str(channel or "unknown"))
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        await self.bus.close()
