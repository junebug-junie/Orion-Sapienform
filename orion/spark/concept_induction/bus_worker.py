from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
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

logger = logging.getLogger("orion.spark.concept.worker")


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
        for key in ("content", "text", "message", "summary", "prompt"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return None

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

        if text:
            event = WindowEvent(text=text, timestamp=env.created_at, envelope=env, intake_channel=intake_channel)
            self.window.setdefault(subject, []).append(event)
            self._prune_window(subject)
            last = self.last_run.get(subject)
            if (
                last is None
                or len(self.window[subject]) >= self.cfg.window_max_events
                or (last and (now - last) > timedelta(minutes=self.cfg.window_max_minutes))
            ):
                await self.run_for_subject(subject, corr_id=env.correlation_id)

    async def run_for_subject(self, subject: str, corr_id=None) -> None:
        corr_id = corr_id or uuid4()
        window = list(self.window.get(subject, []))
        if not window:
            return
        result = await self.inducer.run(subject=subject, window=window)
        await self._publish_profile(result.profile, corr_id)
        if result.delta:
            await self._publish_delta(result.delta, corr_id)
        await self._forward_vector(result.profile, corr_id)
        self.last_run[subject] = datetime.now(timezone.utc)

    async def start(self) -> None:
        await self.bus.connect()
        patterns = list(self.cfg.intake_channels)
        uses_glob = any(any(ch in pattern for ch in "*?[") for pattern in patterns)
        logger.info("Concept worker started listening on %s", patterns)
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
