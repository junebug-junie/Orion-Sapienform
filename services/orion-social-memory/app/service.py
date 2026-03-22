from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Dict

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.social_chat import SocialRoomTurnStoredV1
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
    classify_shared_artifact_decision,
    extract_topics,
    update_room_claim_tracking,
    update_commitments,
    update_participant_continuity,
    update_peer_style_hint,
    update_room_continuity,
    update_room_ritual_summary,
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
        active_commitments=[SocialCommitmentV1.model_validate(item) for item in (row.active_commitments or []) if isinstance(item, dict)],
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
                sess.merge(SocialParticipantContinuitySQL(**participant_summary.model_dump(mode="json")))
                participant_updated = True
                if (
                    self.settings.social_memory_style_adaptation_enabled
                    and not artifact_dialogue_active
                    and peer_artifact_decision.status not in {"declined", "deferred"}
                    and peer_artifact_proposal is None
                    and peer_artifact_revision is None
                    and peer_artifact_confirmation is None
                ):
                    peer_style_existing = _row_to_peer_style(
                        sess.get(SocialPeerStyleHintSQL, f"{platform}:{room_id}:{participant_id}")
                    )
                    peer_style_summary = update_peer_style_hint(
                        peer_style_existing,
                        turn,
                        platform=platform,
                        room_id=room_id,
                        participant_id=participant_id,
                        participant_name=participant_name,
                        confidence_floor=self.settings.social_memory_style_confidence_floor,
                    )
                    sess.merge(SocialPeerStyleHintSQL(**peer_style_summary.model_dump(mode="json")))

                room_existing = _row_to_room(sess.get(SocialRoomContinuitySQL, f"{platform}:{room_id}"))
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
                        "active_claims": claim_tracking.stances,
                        "recent_claim_revisions": claim_tracking.revisions,
                        "claim_attributions": claim_tracking.attributions,
                        "claim_consensus_states": claim_tracking.consensus_states,
                        "claim_divergence_signals": claim_tracking.divergence_signals,
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
                    room_ritual_existing = _row_to_room_ritual(sess.get(SocialRoomRitualSummarySQL, f"{platform}:{room_id}"))
                    room_ritual_summary = update_room_ritual_summary(
                        room_ritual_existing,
                        turn,
                        platform=platform,
                        room_id=room_id,
                        room_summary=room_summary,
                        confidence_floor=self.settings.social_memory_style_confidence_floor,
                    )
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
            return {
                "participant": participant.model_dump(mode="json") if participant else None,
                "room": room.model_dump(mode="json") if room else None,
                "stance": stance.model_dump(mode="json") if stance else None,
                "peer_style": peer_style.model_dump(mode="json") if peer_style else None,
                "room_ritual": room_ritual.model_dump(mode="json") if room_ritual else None,
            }
        finally:
            try:
                sess.close()
            finally:
                remove_session()

    async def _publish(self, channel: str, kind: str, payload: Any) -> None:
        if self.bus is None or not getattr(self.bus, "enabled", False):
            return
        env = BaseEnvelope(
            kind=kind,
            source=ServiceRef(name=self.settings.service_name, version=self.settings.service_version, node=self.settings.node_name),
            payload=payload.model_dump(mode="json"),
        )
        await self.bus.publish(channel, env)
