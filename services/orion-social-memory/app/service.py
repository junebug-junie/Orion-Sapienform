from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Dict

from orion.inspection.social import build_social_inspection_snapshot
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.social_chat import SocialRoomTurnStoredV1
from orion.schemas.social_calibration import SocialCalibrationSignalV1, SocialPeerCalibrationV1, SocialTrustBoundaryV1
from orion.schemas.social_context import (
    SocialContextCandidateV1,
    SocialContextSelectionDecisionV1,
    SocialContextWindowV1,
    SocialEpisodeSnapshotV1,
    SocialReentryAnchorV1,
)
from orion.schemas.social_inspection import SocialInspectionSnapshotV1
from orion.schemas.social_freshness import SocialDecaySignalV1, SocialMemoryFreshnessV1, SocialRegroundingDecisionV1
from orion.schemas.social_artifact import SocialArtifactConfirmationV1, SocialArtifactProposalV1, SocialArtifactRevisionV1
from orion.schemas.social_commitment import SocialCommitmentResolutionV1, SocialCommitmentV1
from orion.schemas.social_claim import (
    SocialClaimAttributionV1,
    SocialClaimRevisionV1,
    SocialClaimStanceV1,
    SocialClaimV1,
    SocialConsensusStateV1,
    SocialDivergenceSignalV1,
)
from orion.schemas.social_deliberation import (
    SocialBridgeSummaryV1,
    SocialClarifyingQuestionV1,
    SocialDeliberationDecisionV1,
)
from orion.schemas.social_floor import (
    SocialClosureSignalV1,
    SocialFloorDecisionV1,
    SocialTurnHandoffV1,
)
from orion.schemas.social_gif import SocialGifUsageStateV1
from orion.schemas.social_memory import (
    SocialParticipantContinuityV1,
    SocialRelationalMemoryUpdateV1,
    SocialRoomContinuityV1,
    SocialStanceSnapshotV1,
)
from orion.schemas.social_style import SocialPeerStyleHintV1, SocialRoomRitualSummaryV1
from orion.schemas.social_thread import SocialHandoffSignalV1, SocialThreadStateV1

from .db import Base, engine, get_session, remove_session
from .models import (
    SocialParticipantContinuitySQL,
    SocialPeerStyleHintSQL,
    SocialRoomContinuitySQL,
    SocialRoomRitualSummarySQL,
    SocialStanceSnapshotSQL,
)
from .settings import Settings
from .synthesizer import (
    ClaimTrackingResult,
    artifact_dialogue_records,
    build_deliberation_result,
    build_floor_result,
    build_open_thread,
    build_social_episode_snapshot,
    build_social_context_window,
    build_social_reentry_anchor,
    classify_shared_artifact_decision,
    extract_topics,
    update_room_claim_tracking,
    update_commitments,
    synthesize_social_calibration,
    synthesize_social_memory_hygiene,
    update_participant_continuity,
    update_peer_style_hint,
    update_room_continuity,
    update_room_ritual_summary,
    update_social_gif_usage_state,
    update_social_stance,
)

logger = logging.getLogger("orion-social-memory")


def _row_to_participant(row: SocialParticipantContinuitySQL | None) -> SocialParticipantContinuityV1 | None:
    if row is None:
        return None
    return SocialParticipantContinuityV1(
        peer_key=row.peer_key,
        platform=row.platform,
        room_id=row.room_id,
        participant_id=row.participant_id,
        participant_name=row.participant_name,
        aliases=list(row.aliases or []),
        participant_kind=row.participant_kind,
        recent_shared_topics=list(row.recent_shared_topics or []),
        interaction_tone_summary=row.interaction_tone_summary or "",
        safe_continuity_summary=row.safe_continuity_summary or "",
        evidence_refs=list(row.evidence_refs or []),
        evidence_count=int(row.evidence_count or 0),
        last_seen_at=row.last_seen_at,
        confidence=float(row.confidence or 0.0),
        trust_tier=row.trust_tier or "new",
        shared_artifact_scope=row.shared_artifact_scope or "peer_local",
        shared_artifact_status=row.shared_artifact_status or "unknown",
        shared_artifact_summary=row.shared_artifact_summary or "",
        shared_artifact_reason=row.shared_artifact_reason or "",
        shared_artifact_proposal=SocialArtifactProposalV1.model_validate(row.shared_artifact_proposal) if row.shared_artifact_proposal else None,
        shared_artifact_revision=SocialArtifactRevisionV1.model_validate(row.shared_artifact_revision) if row.shared_artifact_revision else None,
        shared_artifact_confirmation=SocialArtifactConfirmationV1.model_validate(row.shared_artifact_confirmation) if row.shared_artifact_confirmation else None,
        calibration_signals=[SocialCalibrationSignalV1.model_validate(item) for item in (row.calibration_signals or []) if isinstance(item, dict)],
        peer_calibration=SocialPeerCalibrationV1.model_validate(row.peer_calibration) if row.peer_calibration else None,
        trust_boundary=SocialTrustBoundaryV1.model_validate(row.trust_boundary) if row.trust_boundary else None,
        memory_freshness=[SocialMemoryFreshnessV1.model_validate(item) for item in (row.memory_freshness or []) if isinstance(item, dict)],
        decay_signals=[SocialDecaySignalV1.model_validate(item) for item in (row.decay_signals or []) if isinstance(item, dict)],
        regrounding_decisions=[SocialRegroundingDecisionV1.model_validate(item) for item in (row.regrounding_decisions or []) if isinstance(item, dict)],
    )


def _row_to_room(row: SocialRoomContinuitySQL | None) -> SocialRoomContinuityV1 | None:
    if row is None:
        return None
    return SocialRoomContinuityV1(
        room_key=row.room_key,
        platform=row.platform,
        room_id=row.room_id,
        recurring_topics=list(row.recurring_topics or []),
        active_participants=list(row.active_participants or []),
        recent_thread_summary=row.recent_thread_summary or "",
        room_tone_summary=row.room_tone_summary or "",
        open_threads=list(row.open_threads or []),
        evidence_refs=list(row.evidence_refs or []),
        evidence_count=int(row.evidence_count or 0),
        last_updated_at=row.last_updated_at,
        shared_artifact_scope=row.shared_artifact_scope or "room_local",
        shared_artifact_status=row.shared_artifact_status or "unknown",
        shared_artifact_summary=row.shared_artifact_summary or "",
        shared_artifact_reason=row.shared_artifact_reason or "",
        shared_artifact_proposal=SocialArtifactProposalV1.model_validate(row.shared_artifact_proposal) if row.shared_artifact_proposal else None,
        shared_artifact_revision=SocialArtifactRevisionV1.model_validate(row.shared_artifact_revision) if row.shared_artifact_revision else None,
        shared_artifact_confirmation=SocialArtifactConfirmationV1.model_validate(row.shared_artifact_confirmation) if row.shared_artifact_confirmation else None,
        active_threads=[SocialThreadStateV1.model_validate(item) for item in (row.active_threads or []) if isinstance(item, dict)],
        current_thread_key=row.current_thread_key,
        current_thread_summary=row.current_thread_summary or "",
        handoff_signal=SocialHandoffSignalV1.model_validate(row.handoff_signal) if row.handoff_signal else None,
        active_claims=[SocialClaimStanceV1.model_validate(item) for item in (row.active_claims or []) if isinstance(item, dict)],
        recent_claim_revisions=[SocialClaimRevisionV1.model_validate(item) for item in (row.recent_claim_revisions or []) if isinstance(item, dict)],
        claim_attributions=[SocialClaimAttributionV1.model_validate(item) for item in (row.claim_attributions or []) if isinstance(item, dict)],
        claim_consensus_states=[SocialConsensusStateV1.model_validate(item) for item in (row.claim_consensus_states or []) if isinstance(item, dict)],
        claim_divergence_signals=[SocialDivergenceSignalV1.model_validate(item) for item in (row.claim_divergence_signals or []) if isinstance(item, dict)],
        bridge_summary=SocialBridgeSummaryV1.model_validate(row.bridge_summary) if row.bridge_summary else None,
        clarifying_question=SocialClarifyingQuestionV1.model_validate(row.clarifying_question) if row.clarifying_question else None,
        deliberation_decision=SocialDeliberationDecisionV1.model_validate(row.deliberation_decision) if row.deliberation_decision else None,
        turn_handoff=SocialTurnHandoffV1.model_validate(row.turn_handoff) if row.turn_handoff else None,
        closure_signal=SocialClosureSignalV1.model_validate(row.closure_signal) if row.closure_signal else None,
        floor_decision=SocialFloorDecisionV1.model_validate(row.floor_decision) if row.floor_decision else None,
        gif_usage_state=SocialGifUsageStateV1.model_validate(row.gif_usage_state) if row.gif_usage_state else None,
        active_commitments=[SocialCommitmentV1.model_validate(item) for item in (row.active_commitments or []) if isinstance(item, dict)],
        calibration_signals=[SocialCalibrationSignalV1.model_validate(item) for item in (row.calibration_signals or []) if isinstance(item, dict)],
        peer_calibrations=[SocialPeerCalibrationV1.model_validate(item) for item in (row.peer_calibrations or []) if isinstance(item, dict)],
        trust_boundaries=[SocialTrustBoundaryV1.model_validate(item) for item in (row.trust_boundaries or []) if isinstance(item, dict)],
        memory_freshness=[SocialMemoryFreshnessV1.model_validate(item) for item in (row.memory_freshness or []) if isinstance(item, dict)],
        decay_signals=[SocialDecaySignalV1.model_validate(item) for item in (row.decay_signals or []) if isinstance(item, dict)],
        regrounding_decisions=[SocialRegroundingDecisionV1.model_validate(item) for item in (row.regrounding_decisions or []) if isinstance(item, dict)],
    )


def _row_to_stance(row: SocialStanceSnapshotSQL | None) -> SocialStanceSnapshotV1 | None:
    if row is None:
        return None
    return SocialStanceSnapshotV1(
        stance_id=row.stance_id,
        curiosity=float(row.curiosity),
        warmth=float(row.warmth),
        directness=float(row.directness),
        playfulness=float(row.playfulness),
        caution=float(row.caution),
        depth_preference=float(row.depth_preference),
        recent_social_orientation_summary=row.recent_social_orientation_summary or "",
        evidence_refs=list(row.evidence_refs or []),
        evidence_count=int(row.evidence_count or 0),
        last_updated_at=row.last_updated_at,
    )


def _row_to_peer_style(row: SocialPeerStyleHintSQL | None) -> SocialPeerStyleHintV1 | None:
    if row is None:
        return None
    return SocialPeerStyleHintV1(
        peer_style_key=row.peer_style_key,
        platform=row.platform,
        room_id=row.room_id,
        participant_id=row.participant_id,
        participant_name=row.participant_name,
        style_hints_summary=row.style_hints_summary or "",
        preferred_directness=float(row.preferred_directness or 0.5),
        preferred_depth=float(row.preferred_depth or 0.5),
        question_appetite=float(row.question_appetite or 0.5),
        playfulness_tendency=float(row.playfulness_tendency or 0.3),
        formality_tendency=float(row.formality_tendency or 0.5),
        summarization_preference=float(row.summarization_preference or 0.3),
        evidence_count=int(row.evidence_count or 0),
        confidence=float(row.confidence or 0.0),
        last_updated_at=row.last_updated_at,
    )


def _row_to_room_ritual(row: SocialRoomRitualSummarySQL | None) -> SocialRoomRitualSummaryV1 | None:
    if row is None:
        return None
    return SocialRoomRitualSummaryV1(
        ritual_key=row.ritual_key,
        platform=row.platform,
        room_id=row.room_id,
        greeting_style=row.greeting_style,
        reentry_style=row.reentry_style,
        thread_revival_style=row.thread_revival_style,
        pause_handoff_style=row.pause_handoff_style,
        summary_cadence_preference=float(row.summary_cadence_preference or 0.3),
        room_tone_summary=row.room_tone_summary or "",
        culture_summary=row.culture_summary or "",
        evidence_count=int(row.evidence_count or 0),
        confidence=float(row.confidence or 0.0),
        last_updated_at=row.last_updated_at,
    )


class SocialMemoryService:
    def __init__(self, *, settings: Settings, bus: OrionBusAsync | None = None) -> None:
        self.settings = settings
        self.bus = bus
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        Base.metadata.create_all(bind=engine)
        if self.bus is None and self.settings.orion_bus_enabled:
            self.bus = OrionBusAsync(
                self.settings.orion_bus_url,
                enabled=self.settings.orion_bus_enabled,
                enforce_catalog=self.settings.orion_bus_enforce_catalog,
            )
            await self.bus.connect()
        if self.bus is not None:
            self._task = asyncio.create_task(self._consume_social_turns())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self.bus is not None:
            await self.bus.close()
            self.bus = None

    async def _consume_social_turns(self) -> None:
        assert self.bus is not None
        async with self.bus.subscribe(self.settings.social_memory_input_channel) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                decoded = self.bus.codec.decode(msg.get("data"))
                if not decoded.ok:
                    continue
                env = decoded.envelope
                if env.kind != "social.turn.stored.v1":
                    continue
                await self.process_social_turn(env.payload)

    async def process_social_turn(self, payload: Dict[str, Any]) -> SocialRelationalMemoryUpdateV1:
        turn = SocialRoomTurnStoredV1.model_validate(payload)
        social_meta = dict(turn.client_meta or {})
        external_room = dict(social_meta.get("external_room") or {})
        external_participant = dict(social_meta.get("external_participant") or {})
        platform = str(external_room.get("platform") or "").strip() or None
        room_id = str(external_room.get("room_id") or "").strip() or None
        participant_id = str(external_participant.get("participant_id") or "").strip() or None
        participant_name = str(external_participant.get("participant_name") or "").strip() or None
        participant_kind = str(external_participant.get("participant_kind") or "peer_ai").strip() or "peer_ai"
        thread_id = str(external_room.get("thread_id") or "").strip() or None

        sess = get_session()
        try:
            participant_updated = False
            room_updated = False
            peer_style_summary = None
            room_ritual_summary = None
            peer_style_existing = None
            room_ritual_existing = None
            created_commitments: list[SocialCommitmentV1] = []
            commitment_resolutions: list[SocialCommitmentResolutionV1] = []
            claim_tracking = ClaimTrackingResult()
            deliberation = None
            floor_result = None
            if platform and room_id and participant_id:
                topics = extract_topics(turn, limit=self.settings.social_memory_topic_max)
                artifact_dialogue_active = any(
                    isinstance(social_meta.get(key), dict)
                    for key in (
                        "social_artifact_proposal",
                        "social_artifact_revision",
                        "social_artifact_confirmation",
                    )
                )
                peer_artifact_decision = classify_shared_artifact_decision(
                    turn,
                    scope="peer_local",
                    topics=topics,
                )
                peer_artifact_proposal, peer_artifact_revision, peer_artifact_confirmation = artifact_dialogue_records(
                    turn,
                    scope="peer_local",
                )
                room_artifact_decision = classify_shared_artifact_decision(
                    turn,
                    scope="room_local",
                    topics=topics,
                )
                room_artifact_proposal, room_artifact_revision, room_artifact_confirmation = artifact_dialogue_records(
                    turn,
                    scope="room_local",
                )
                participant_existing = _row_to_participant(
                    sess.get(SocialParticipantContinuitySQL, f"{platform}:{room_id}:{participant_id}")
                )
                participant_summary = update_participant_continuity(
                    participant_existing,
                    turn,
                    platform=platform,
                    room_id=room_id,
                    participant_id=participant_id,
                    participant_name=participant_name,
                    participant_kind=participant_kind,
                    topic_limit=self.settings.social_memory_topic_max,
                    evidence_limit=self.settings.social_memory_evidence_max,
                    shared_artifact_decision=peer_artifact_decision,
                    dialogue_active=artifact_dialogue_active,
                    artifact_proposal=peer_artifact_proposal,
                    artifact_revision=peer_artifact_revision,
                    artifact_confirmation=peer_artifact_confirmation,
                )
                participant_updated = True
                peer_style_existing = _row_to_peer_style(
                    sess.get(SocialPeerStyleHintSQL, f"{platform}:{room_id}:{participant_id}")
                )
                if (
                    self.settings.social_memory_style_adaptation_enabled
                    and not artifact_dialogue_active
                    and peer_artifact_decision.status not in {"declined", "deferred"}
                    and peer_artifact_proposal is None
                    and peer_artifact_revision is None
                    and peer_artifact_confirmation is None
                ):
                    peer_style_summary = update_peer_style_hint(
                        peer_style_existing,
                        turn,
                        platform=platform,
                        room_id=room_id,
                        participant_id=participant_id,
                        participant_name=participant_name,
                        confidence_floor=self.settings.social_memory_style_confidence_floor,
                    )

                room_existing = _row_to_room(sess.get(SocialRoomContinuitySQL, f"{platform}:{room_id}"))
                room_ritual_existing = _row_to_room_ritual(sess.get(SocialRoomRitualSummarySQL, f"{platform}:{room_id}"))
                active_commitments, created_commitments, commitment_resolutions = update_commitments(
                    room_existing.active_commitments if room_existing else [],
                    turn,
                    platform=platform,
                    room_id=room_id,
                    topics=topics,
                    thread_summary=(room_existing.current_thread_summary if room_existing else "") or ", ".join(topics[:2]) or "the active room thread",
                    artifact_dialogue_active=artifact_dialogue_active,
                    artifact_confirmation=room_artifact_confirmation,
                    ttl_minutes=self.settings.social_memory_commitment_ttl_minutes,
                    max_open=self.settings.social_memory_commitment_max_open,
                )
                room_summary = update_room_continuity(
                    room_existing,
                    turn,
                    platform=platform,
                    room_id=room_id,
                    participant_label=participant_name or participant_id,
                    thread_id=thread_id,
                    topic_limit=self.settings.social_memory_topic_max,
                    participant_limit=self.settings.social_memory_room_participant_max,
                    evidence_limit=self.settings.social_memory_evidence_max,
                    shared_artifact_decision=room_artifact_decision,
                    dialogue_active=artifact_dialogue_active,
                    artifact_proposal=room_artifact_proposal,
                    artifact_revision=room_artifact_revision,
                    artifact_confirmation=room_artifact_confirmation,
                    thread_ttl_hours=self.settings.social_memory_open_thread_ttl_hours,
                    active_commitments=active_commitments,
                )
                claim_tracking = update_room_claim_tracking(
                    existing=room_existing,
                    turn=turn,
                    platform=platform,
                    room_id=room_id,
                    thread_key=room_summary.current_thread_key,
                    participant_id=participant_id,
                    participant_name=participant_name,
                    artifact_dialogue_active=artifact_dialogue_active,
                    shared_artifact_statuses=[
                        peer_artifact_decision.status,
                        room_artifact_decision.status,
                        room_summary.shared_artifact_status,
                    ],
                    repair_signal=social_meta.get("social_repair_signal"),
                    repair_decision=social_meta.get("social_repair_decision"),
                    epistemic_signal=social_meta.get("social_epistemic_signal"),
                )
                room_summary = room_summary.model_copy(
                    update={
                        "active_claims": claim_tracking.stances or list(room_existing.active_claims if room_existing else []),
                        "recent_claim_revisions": claim_tracking.revisions or list(room_existing.recent_claim_revisions if room_existing else []),
                        "claim_attributions": claim_tracking.attributions or list(room_existing.claim_attributions if room_existing else []),
                        "claim_consensus_states": claim_tracking.consensus_states or list(room_existing.claim_consensus_states if room_existing else []),
                        "claim_divergence_signals": claim_tracking.divergence_signals or list(room_existing.claim_divergence_signals if room_existing else []),
                    }
                )
                calibration = synthesize_social_calibration(
                    existing_participant=participant_existing,
                    existing_room=room_existing,
                    turn=turn,
                    platform=platform,
                    room_id=room_id,
                    participant_id=participant_id,
                    participant_name=participant_name,
                    thread_key=room_summary.current_thread_key,
                    topics=topics,
                    claim_tracking=claim_tracking,
                    artifact_dialogue_active=artifact_dialogue_active,
                )
                participant_summary = participant_summary.model_copy(
                    update={
                        "calibration_signals": [item for item in calibration.signals if item.participant_id == participant_id][:3],
                        "peer_calibration": calibration.peer_calibration or (participant_existing.peer_calibration if participant_existing else None),
                        "trust_boundary": next((item for item in calibration.trust_boundaries if item.participant_id == participant_id), None) or (participant_existing.trust_boundary if participant_existing else None),
                    }
                )
                room_summary = room_summary.model_copy(
                    update={
                        "calibration_signals": calibration.signals,
                        "peer_calibrations": [calibration.peer_calibration] if calibration.peer_calibration else list(room_existing.peer_calibrations if room_existing else []),
                        "trust_boundaries": calibration.trust_boundaries or list(room_existing.trust_boundaries if room_existing else []),
                    }
                )
                deliberation = build_deliberation_result(
                    turn=turn,
                    room=room_summary,
                    claim_tracking=claim_tracking,
                    artifact_dialogue_active=artifact_dialogue_active,
                )
                room_summary = room_summary.model_copy(
                    update={
                        "bridge_summary": deliberation.bridge_summary,
                        "clarifying_question": deliberation.clarifying_question,
                        "deliberation_decision": deliberation.decision,
                    }
                )
                floor_result = build_floor_result(
                    turn=turn,
                    room=room_summary,
                    claim_tracking=claim_tracking,
                    active_commitments=active_commitments,
                    artifact_dialogue_active=artifact_dialogue_active,
                )
                room_summary = room_summary.model_copy(
                    update={
                        "turn_handoff": floor_result.turn_handoff,
                        "closure_signal": floor_result.closure_signal,
                        "floor_decision": floor_result.decision,
                    }
                )
                room_summary = room_summary.model_copy(
                    update={
                        "gif_usage_state": update_social_gif_usage_state(
                            room_existing.gif_usage_state if room_existing else None,
                            turn,
                            platform=platform,
                            room_id=room_id,
                            thread_key=room_summary.current_thread_key,
                            participant_id=participant_id,
                            participant_name=participant_name,
                        )
                    }
                )
                memory_hygiene = synthesize_social_memory_hygiene(
                    existing_participant=participant_existing,
                    existing_room=room_existing,
                    existing_peer_style=peer_style_existing,
                    existing_room_ritual=room_ritual_existing,
                    participant=participant_summary,
                    room=room_summary,
                    peer_style=peer_style_summary or peer_style_existing,
                    room_ritual=room_ritual_summary or room_ritual_existing,
                    turn=turn,
                    platform=platform,
                    room_id=room_id,
                    participant_id=participant_id,
                    thread_key=room_summary.current_thread_key,
                    claim_tracking=claim_tracking,
                    calibration=calibration,
                    commitment_resolutions=commitment_resolutions,
                    artifact_dialogue_active=artifact_dialogue_active,
                    shared_artifact_statuses=[
                        peer_artifact_decision.status,
                        room_artifact_decision.status,
                        room_summary.shared_artifact_status,
                    ],
                )
                participant_freshness = [item for item in memory_hygiene.memory_freshness if item.participant_id == participant_id]
                participant_decay = [item for item in memory_hygiene.decay_signals if item.participant_id == participant_id]
                participant_reground = [item for item in memory_hygiene.regrounding_decisions if item.participant_id == participant_id]
                room_freshness = [item for item in memory_hygiene.memory_freshness if item.participant_id is None]
                room_decay = [item for item in memory_hygiene.decay_signals if item.participant_id is None]
                room_reground = [item for item in memory_hygiene.regrounding_decisions if item.participant_id is None]
                participant_summary = (memory_hygiene.participant or participant_summary).model_copy(update={
                    "memory_freshness": participant_freshness[:4],
                    "decay_signals": participant_decay[:4],
                    "regrounding_decisions": participant_reground[:4],
                })
                room_summary = (memory_hygiene.room or room_summary).model_copy(update={
                    "memory_freshness": room_freshness[:6],
                    "decay_signals": room_decay[:6],
                    "regrounding_decisions": room_reground[:6],
                })
                peer_style_summary = memory_hygiene.peer_style
                room_ritual_summary = memory_hygiene.room_ritual
                if room_summary.gif_usage_state is not None:
                    logger.info(
                        "social_gif_usage_state_updated room_id=%s participant_id=%s turns_since_last=%s density=%.2f consecutive=%s",
                        room_id,
                        participant_id,
                        room_summary.gif_usage_state.turns_since_last_orion_gif,
                        room_summary.gif_usage_state.recent_gif_density,
                        room_summary.gif_usage_state.consecutive_gif_turns,
                    )
                sess.merge(SocialParticipantContinuitySQL(**participant_summary.model_dump(mode="json")))
                for signal in calibration.detected_signals:
                    logger.info(
                        "social_calibration_signal_detected room_id=%s participant_id=%s thread_key=%s kind=%s confidence=%.2f evidence=%s",
                        room_id,
                        signal.participant_id or "room",
                        signal.thread_key,
                        signal.calibration_kind,
                        signal.confidence,
                        signal.evidence_count,
                    )
                if calibration.peer_calibration is not None:
                    logger.info(
                        "social_calibration_updated room_id=%s participant_id=%s kind=%s confidence=%.2f rationale=%s",
                        room_id,
                        calibration.peer_calibration.participant_id,
                        calibration.peer_calibration.calibration_kind,
                        calibration.peer_calibration.confidence,
                        calibration.peer_calibration.rationale[:120],
                    )
                for signal in calibration.decayed_signals:
                    logger.info(
                        "social_calibration_decayed room_id=%s participant_id=%s thread_key=%s kind=%s confidence=%.2f evidence=%s",
                        room_id,
                        signal.participant_id or "room",
                        signal.thread_key,
                        signal.calibration_kind,
                        signal.confidence,
                        signal.evidence_count,
                    )
                for ignored in calibration.ignored_reasons:
                    logger.info(
                        "social_calibration_ignored room_id=%s reason=%s",
                        room_id,
                        ignored[:180],
                    )
                for signal in memory_hygiene.decay_signals:
                    logger.info(
                        "social_decay_signal_detected room_id=%s participant_id=%s thread_key=%s artifact=%s freshness=%s decay=%s confidence=%.2f evidence=%s",
                        room_id,
                        signal.participant_id or "room",
                        signal.thread_key,
                        signal.artifact_kind,
                        signal.freshness_state,
                        signal.decay_level,
                        signal.confidence,
                        signal.evidence_count,
                    )
                for decision in memory_hygiene.regrounding_decisions:
                    event_name = {
                        "soften": "social_state_softened",
                        "reopen": "social_state_reopened",
                        "expire": "social_state_expired",
                        "refresh_needed": "social_refresh_needed_flagged",
                    }.get(decision.decision, "social_state_kept")
                    logger.info(
                        "%s room_id=%s participant_id=%s thread_key=%s artifact=%s freshness=%s decay=%s confidence=%.2f reasons=%s",
                        event_name,
                        room_id,
                        decision.participant_id or "room",
                        decision.thread_key,
                        decision.artifact_kind,
                        decision.freshness_state,
                        decision.decay_level,
                        decision.confidence,
                        ",".join(decision.reasons[:4]),
                    )
                for ignored in memory_hygiene.ignored_reasons:
                    logger.info(
                        "social_decay_ignored room_id=%s reason=%s",
                        room_id,
                        ignored[:180],
                    )
                sess.merge(SocialRoomContinuitySQL(**room_summary.model_dump(mode="json")))
                room_updated = True
                if room_summary.current_thread_key:
                    logger.info(
                        "social_thread_updated room_id=%s thread_key=%s audience_scope=%s summary=%s",
                        room_id,
                        room_summary.current_thread_key,
                        room_summary.active_threads[0].audience_scope if room_summary.active_threads else "none",
                        room_summary.current_thread_summary[:120],
                    )
                if room_summary.handoff_signal and room_summary.handoff_signal.detected:
                    logger.info(
                        "social_handoff_detected room_id=%s thread_key=%s kind=%s rationale=%s",
                        room_id,
                        room_summary.handoff_signal.thread_key,
                        room_summary.handoff_signal.handoff_kind,
                        room_summary.handoff_signal.rationale,
                    )
                for commitment in created_commitments:
                    logger.info(
                        "social_commitment_created room_id=%s thread_key=%s commitment_id=%s type=%s summary=%s",
                        room_id,
                        commitment.thread_key,
                        commitment.commitment_id,
                        commitment.commitment_type,
                        commitment.summary[:120],
                    )
                for resolution in commitment_resolutions:
                    logger.info(
                        "social_commitment_resolved room_id=%s thread_key=%s commitment_id=%s state=%s reason=%s",
                        room_id,
                        resolution.thread_key,
                        resolution.commitment_id,
                        resolution.state,
                        resolution.resolution_reason[:120],
                    )
                for claim in claim_tracking.claims:
                    logger.info(
                        "social_claim_extracted room_id=%s thread_key=%s claim_id=%s stance=%s kind=%s summary=%s",
                        room_id,
                        claim.thread_key,
                        claim.claim_id,
                        claim.stance,
                        claim.claim_kind,
                        claim.normalized_summary[:120],
                    )
                for revision in claim_tracking.revisions:
                    logger.info(
                        "social_claim_revised room_id=%s thread_key=%s claim_id=%s revision=%s new_stance=%s summary=%s",
                        room_id,
                        revision.thread_key,
                        revision.claim_id,
                        revision.revision_type,
                        revision.new_stance,
                        revision.revised_summary[:120],
                    )
                for attribution in claim_tracking.attributions:
                    logger.info(
                        "social_claim_attributed room_id=%s claim_id=%s participants=%s orion_stance=%s",
                        room_id,
                        attribution.claim_id,
                        ",".join(attribution.attributed_participant_ids),
                        attribution.orion_stance,
                    )
                for consensus in claim_tracking.consensus_states:
                    logger.info(
                        "social_claim_consensus_updated room_id=%s claim_id=%s consensus_state=%s support_count=%s",
                        room_id,
                        consensus.claim_id,
                        consensus.consensus_state,
                        consensus.supporting_evidence_count,
                    )
                for divergence in claim_tracking.divergence_signals:
                    logger.info(
                        "social_claim_divergence_detected room_id=%s claim_id=%s consensus_state=%s",
                        room_id,
                        divergence.claim_id,
                        divergence.consensus_state,
                    )
                for ignored in claim_tracking.ignored_reasons:
                    logger.info(
                        "social_claim_ignored room_id=%s reason=%s",
                        room_id,
                        ignored[:180],
                    )
                if deliberation and deliberation.bridge_summary is not None:
                    logger.info(
                        "social_bridge_summary_built room_id=%s thread_key=%s trigger=%s shared_core=%s",
                        room_id,
                        deliberation.bridge_summary.thread_key,
                        deliberation.bridge_summary.trigger,
                        deliberation.bridge_summary.shared_core[:120],
                    )
                if deliberation and deliberation.clarifying_question is not None:
                    logger.info(
                        "social_clarifying_question_built room_id=%s thread_key=%s focus=%s question=%s",
                        room_id,
                        deliberation.clarifying_question.thread_key,
                        deliberation.clarifying_question.question_focus,
                        deliberation.clarifying_question.question_text[:120],
                    )
                if deliberation and deliberation.decision is not None:
                    logger.info(
                        "social_deliberation_decision room_id=%s thread_key=%s decision=%s",
                        room_id,
                        deliberation.decision.thread_key,
                        deliberation.decision.decision_kind,
                    )
                if deliberation:
                    for ignored in deliberation.ignored_reasons:
                        logger.info(
                            "social_deliberation_ignored room_id=%s reason=%s",
                            room_id,
                            ignored[:180],
                        )
                if floor_result and floor_result.turn_handoff is not None:
                    logger.info(
                        "social_turn_handoff_built room_id=%s thread_key=%s decision=%s target=%s",
                        room_id,
                        floor_result.turn_handoff.thread_key,
                        floor_result.turn_handoff.decision_kind,
                        floor_result.turn_handoff.target_participant_name or floor_result.turn_handoff.target_participant_id or floor_result.turn_handoff.audience_scope,
                    )
                if floor_result and floor_result.closure_signal is not None:
                    logger.info(
                        "social_closure_signal_built room_id=%s thread_key=%s kind=%s",
                        room_id,
                        floor_result.closure_signal.thread_key,
                        floor_result.closure_signal.closure_kind,
                    )
                if floor_result and floor_result.decision is not None:
                    logger.info(
                        "social_floor_decision room_id=%s thread_key=%s decision=%s",
                        room_id,
                        floor_result.decision.thread_key,
                        floor_result.decision.decision_kind,
                    )
                if floor_result:
                    for ignored in floor_result.ignored_reasons:
                        logger.info(
                            "social_floor_ignored room_id=%s reason=%s",
                            room_id,
                            ignored[:180],
                        )
                if (
                    self.settings.social_memory_style_adaptation_enabled
                    and not artifact_dialogue_active
                    and room_artifact_decision.status not in {"declined", "deferred"}
                    and room_artifact_proposal is None
                    and room_artifact_revision is None
                    and room_artifact_confirmation is None
                ):
                    room_ritual_summary = update_room_ritual_summary(
                        room_ritual_summary or room_ritual_existing,
                        turn,
                        platform=platform,
                        room_id=room_id,
                        room_summary=room_summary,
                        confidence_floor=self.settings.social_memory_style_confidence_floor,
                    )
                if peer_style_summary is not None:
                    sess.merge(SocialPeerStyleHintSQL(**peer_style_summary.model_dump(mode="json")))
                if room_ritual_summary is not None:
                    sess.merge(SocialRoomRitualSummarySQL(**room_ritual_summary.model_dump(mode="json")))
                if any(
                    item is not None
                    for item in (
                        peer_artifact_proposal,
                        peer_artifact_revision,
                        peer_artifact_confirmation,
                        room_artifact_proposal,
                        room_artifact_revision,
                        room_artifact_confirmation,
                    )
                ) and (
                    room_artifact_confirmation is None or room_artifact_confirmation.decision_state != "accepted"
                ):
                    open_thread = None
                else:
                    open_thread = build_open_thread(
                        turn,
                        platform=platform,
                        room_id=room_id,
                        room_summary=room_summary,
                        participant_label=participant_name or participant_id,
                        thread_id=thread_id,
                        ttl_hours=self.settings.social_memory_open_thread_ttl_hours,
                    )
            else:
                participant_summary = None
                room_summary = None
                open_thread = None

            stance_existing = _row_to_stance(sess.get(SocialStanceSnapshotSQL, "orion-social-room"))
            stance_summary = update_social_stance(
                stance_existing,
                turn,
                evidence_limit=self.settings.social_memory_evidence_max,
            )
            sess.merge(SocialStanceSnapshotSQL(**stance_summary.model_dump(mode="json")))
            sess.commit()

            update = SocialRelationalMemoryUpdateV1(
                turn_id=turn.turn_id,
                correlation_id=turn.correlation_id,
                platform=platform,
                room_id=room_id,
                peer_key=participant_summary.peer_key if participant_summary else None,
                participant_updated=participant_updated,
                room_updated=room_updated,
                stance_updated=True,
                claim_count=len(claim_tracking.claims),
                claim_revision_count=len(claim_tracking.revisions),
                evidence_count=stance_summary.evidence_count,
            )
        finally:
            try:
                sess.close()
            finally:
                remove_session()

        if participant_summary is not None:
            await self._publish(
                self.settings.social_memory_participant_channel,
                "social.participant.continuity.v1",
                participant_summary,
            )
        if room_summary is not None:
            await self._publish(
                self.settings.social_memory_room_channel,
                "social.room.continuity.v1",
                room_summary,
            )
        for commitment in created_commitments:
            await self._publish(
                self.settings.social_memory_commitment_channel,
                "social.commitment.v1",
                commitment,
            )
        for resolution in commitment_resolutions:
            await self._publish(
                self.settings.social_memory_commitment_resolution_channel,
                "social.commitment.resolution.v1",
                resolution,
            )
        for claim in claim_tracking.claims:
            await self._publish(
                self.settings.social_memory_claim_channel,
                "social.claim.v1",
                claim,
            )
        for revision in claim_tracking.revisions:
            await self._publish(
                self.settings.social_memory_claim_revision_channel,
                "social.claim.revision.v1",
                revision,
            )
        for stance in claim_tracking.stances:
            await self._publish(
                self.settings.social_memory_claim_stance_channel,
                "social.claim.stance.v1",
                stance,
            )
        for attribution in claim_tracking.attributions:
            await self._publish(
                self.settings.social_memory_claim_attribution_channel,
                "social.claim.attribution.v1",
                attribution,
            )
        for consensus in claim_tracking.consensus_states:
            await self._publish(
                self.settings.social_memory_claim_consensus_channel,
                "social.claim.consensus.v1",
                consensus,
            )
        for divergence in claim_tracking.divergence_signals:
            await self._publish(
                self.settings.social_memory_claim_divergence_channel,
                "social.claim.divergence.v1",
                divergence,
            )
        if room_summary is not None and room_summary.bridge_summary is not None:
            await self._publish(
                self.settings.social_memory_bridge_summary_channel,
                "social.bridge-summary.v1",
                room_summary.bridge_summary,
            )
        if room_summary is not None and room_summary.clarifying_question is not None:
            await self._publish(
                self.settings.social_memory_clarifying_question_channel,
                "social.clarifying-question.v1",
                room_summary.clarifying_question,
            )
        if room_summary is not None and room_summary.deliberation_decision is not None:
            await self._publish(
                self.settings.social_memory_deliberation_decision_channel,
                "social.deliberation.decision.v1",
                room_summary.deliberation_decision,
            )
        if room_summary is not None and room_summary.turn_handoff is not None:
            await self._publish(
                self.settings.social_memory_turn_handoff_channel,
                "social.turn-handoff.v1",
                room_summary.turn_handoff,
            )
        if room_summary is not None and room_summary.closure_signal is not None:
            await self._publish(
                self.settings.social_memory_closure_signal_channel,
                "social.closure-signal.v1",
                room_summary.closure_signal,
            )
        if room_summary is not None and room_summary.floor_decision is not None:
            await self._publish(
                self.settings.social_memory_floor_decision_channel,
                "social.floor.decision.v1",
                room_summary.floor_decision,
            )
        if peer_style_summary is not None:
            await self._publish(
                self.settings.social_memory_peer_style_channel,
                "social.peer-style.v1",
                peer_style_summary,
            )
        if room_ritual_summary is not None:
            await self._publish(
                self.settings.social_memory_room_ritual_channel,
                "social.room-ritual.v1",
                room_ritual_summary,
            )
        if open_thread is not None:
            await self._publish(
                self.settings.social_memory_open_thread_channel,
                "social.open-thread.v1",
                open_thread,
            )
        await self._publish(
            self.settings.social_memory_stance_channel,
            "social.stance.snapshot.v1",
            stance_summary,
        )
        await self._publish(
            self.settings.social_memory_update_channel,
            "social.relational.update.v1",
            update,
        )
        return update

    async def get_summary(self, *, platform: str, room_id: str, participant_id: str | None) -> Dict[str, Any]:
        sess = get_session()
        try:
            participant = None
            if participant_id:
                participant = _row_to_participant(
                    sess.get(SocialParticipantContinuitySQL, f"{platform}:{room_id}:{participant_id}")
                )
            room = _row_to_room(sess.get(SocialRoomContinuitySQL, f"{platform}:{room_id}"))
            stance = _row_to_stance(sess.get(SocialStanceSnapshotSQL, "orion-social-room"))
            peer_style = _row_to_peer_style(
                sess.get(SocialPeerStyleHintSQL, f"{platform}:{room_id}:{participant_id}")
            ) if participant_id else None
            room_ritual = _row_to_room_ritual(sess.get(SocialRoomRitualSummarySQL, f"{platform}:{room_id}"))
            context_window, context_selection_decision, context_candidates = build_social_context_window(
                platform=platform,
                room_id=room_id,
                participant=participant,
                room=room,
                peer_style=peer_style,
                room_ritual=room_ritual,
            )
            episode_snapshot = build_social_episode_snapshot(
                platform=platform,
                room_id=room_id,
                participant=participant,
                room=room,
            )
            reentry_anchor = build_social_reentry_anchor(
                platform=platform,
                room_id=room_id,
                participant=participant,
                room=room,
                room_ritual=room_ritual,
                episode_snapshot=episode_snapshot,
            )
            for candidate in context_candidates:
                logger.info(
                    "social_context_candidate_considered room_id=%s participant_id=%s kind=%s decision=%s freshness=%s relevance=%.2f",
                    room_id,
                    participant_id or "room",
                    candidate.candidate_kind,
                    candidate.inclusion_decision,
                    candidate.freshness_band,
                    candidate.relevance_score,
                )
                if candidate.inclusion_decision != "include":
                    logger.info(
                        "social_context_candidate_%s room_id=%s participant_id=%s kind=%s freshness=%s reason=%s",
                        "softened" if candidate.inclusion_decision == "soften" else "excluded",
                        room_id,
                        participant_id or "room",
                        candidate.candidate_kind,
                        candidate.freshness_band,
                        candidate.rationale[:160],
                    )
                    if candidate.freshness_band in {"stale", "refresh_needed", "expired"}:
                        logger.info(
                            "social_stale_state_excluded room_id=%s participant_id=%s kind=%s freshness=%s",
                            room_id,
                            participant_id or "room",
                            candidate.candidate_kind,
                            candidate.freshness_band,
                        )
            if context_selection_decision and "local_thread_state_preferred" in context_selection_decision.reasons:
                logger.info(
                    "social_local_thread_preferred room_id=%s participant_id=%s thread_key=%s",
                    room_id,
                    participant_id or "room",
                    context_selection_decision.thread_key,
                )
            if context_window is not None:
                logger.info(
                    "social_context_window_assembled room_id=%s participant_id=%s thread_key=%s selected=%s budget=%s considered=%s",
                    room_id,
                    participant_id or "room",
                    context_window.thread_key,
                    len(context_window.selected_candidates),
                    context_window.budget_max,
                    context_window.total_candidates_considered,
                )
            return {
                "participant": participant.model_dump(mode="json") if participant else None,
                "room": room.model_dump(mode="json") if room else None,
                "stance": stance.model_dump(mode="json") if stance else None,
                "peer_style": peer_style.model_dump(mode="json") if peer_style else None,
                "room_ritual": room_ritual.model_dump(mode="json") if room_ritual else None,
                "episode_snapshot": SocialEpisodeSnapshotV1.model_validate(episode_snapshot).model_dump(mode="json") if episode_snapshot else None,
                "reentry_anchor": SocialReentryAnchorV1.model_validate(reentry_anchor).model_dump(mode="json") if reentry_anchor else None,
                "context_window": context_window.model_dump(mode="json") if context_window else None,
                "context_selection_decision": context_selection_decision.model_dump(mode="json") if context_selection_decision else None,
                "context_candidates": [item.model_dump(mode="json") for item in context_candidates[:12]],
            }
        finally:
            try:
                sess.close()
            finally:
                remove_session()

    async def get_inspection(self, *, platform: str, room_id: str, participant_id: str | None) -> Dict[str, Any]:
        summary = await self.get_summary(platform=platform, room_id=room_id, participant_id=participant_id)
        surfaces = {
            "social_peer_continuity": summary.get("participant") or {},
            "social_room_continuity": summary.get("room") or {},
            "social_context_window": summary.get("context_window") or {},
            "social_context_selection_decision": summary.get("context_selection_decision") or {},
            "social_context_candidates": summary.get("context_candidates") or [],
            "social_episode_snapshot": summary.get("episode_snapshot") or {},
            "social_reentry_anchor": summary.get("reentry_anchor") or {},
        }
        room = dict(summary.get("room") or {})
        inspection = build_social_inspection_snapshot(
            platform=platform,
            room_id=room_id,
            participant_id=participant_id,
            thread_key=(summary.get("context_window") or {}).get("thread_key") or room.get("current_thread_key"),
            surfaces=surfaces,
            source_surface="social-memory-summary",
            source_service=self.settings.service_name,
        )
        logger.info(
            "social_inspection_snapshot_built room_id=%s participant_id=%s sections=%s traces=%s source=%s",
            room_id,
            participant_id or "room",
            len(inspection.sections),
            len(inspection.decision_traces),
            "social-memory-summary",
        )
        for section in inspection.sections:
            logger.info(
                "social_inspection_section_included room_id=%s participant_id=%s kind=%s included=%s traces=%s",
                room_id,
                participant_id or "room",
                section.section_kind,
                len(section.included_artifact_summaries),
                len(section.decision_traces),
            )
        if int(inspection.metadata.get("safety_omissions") or 0) > 0:
            logger.info(
                "social_inspection_safety_omission room_id=%s participant_id=%s omitted=%s",
                room_id,
                participant_id or "room",
                inspection.metadata.get("safety_omissions"),
            )
        return SocialInspectionSnapshotV1.model_validate(inspection).model_dump(mode="json")

    async def _publish(self, channel: str, kind: str, payload: Any) -> None:
        if self.bus is None or not getattr(self.bus, "enabled", False):
            return
        env = BaseEnvelope(
            kind=kind,
            source=ServiceRef(name=self.settings.service_name, version=self.settings.service_version, node=self.settings.node_name),
            payload=payload.model_dump(mode="json"),
        )
        await self.bus.publish(channel, env)
