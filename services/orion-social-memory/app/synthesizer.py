from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from orion.schemas.social_autonomy import SocialOpenThreadV1
from orion.schemas.social_artifact import SocialArtifactConfirmationV1, SocialArtifactProposalV1, SocialArtifactRevisionV1
from orion.schemas.social_chat import SocialRoomTurnStoredV1
from orion.schemas.social_claim import (
    SocialClaimAttributionV1,
    SocialClaimRevisionV1,
    SocialClaimStanceV1,
    SocialClaimV1,
    SocialConsensusStateV1,
    SocialDivergenceSignalV1,
)
from orion.schemas.social_commitment import SocialCommitmentResolutionV1, SocialCommitmentV1
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
    SocialRoomContinuityV1,
    SocialStanceSnapshotV1,
)
from orion.schemas.social_style import SocialPeerStyleHintV1, SocialRoomRitualSummaryV1
from orion.schemas.social_thread import SocialHandoffSignalV1, SocialThreadRoutingDecisionV1, SocialThreadStateV1


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s().]{7,}\d)")
_LONG_DIGIT_RE = re.compile(r"\b\d{8,}\b")
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]{2,}")
_SEALED_RE = re.compile(r"\b(private|sealed|secret|mirror|journal|off[- ]record)\b", re.IGNORECASE)
_STOPWORDS = {
    "with", "from", "that", "this", "have", "about", "what", "when", "where", "would",
    "there", "their", "they", "them", "your", "you", "into", "while", "staying", "still",
    "here", "been", "more", "than", "just", "make", "like", "room", "thread", "conversation",
    "social", "orion", "juniper", "peer", "continue", "ongoing", "really", "want", "need",
}

_ACCEPT_HINTS = (
    "can we keep",
    "let's keep",
    "lets keep",
    "keep this",
    "remember this",
    "carry this",
    "works for me",
    "that fits",
    "good to keep",
    "okay to keep",
    "okay to remember",
    "room norm",
    "shared cue",
)
_DECLINE_HINTS = (
    "don't keep",
    "do not keep",
    "don't remember",
    "do not remember",
    "don't retain",
    "do not retain",
    "don't make this a thing",
    "do not make this a thing",
    "not a room norm",
    "keep this private",
    "off the record",
)
_DEFER_HINTS = (
    "not yet",
    "maybe later",
    "later maybe",
    "let's revisit",
    "lets revisit",
    "revisit later",
    "not sure yet",
    "hold off",
    "for now",
)
_ROOM_SCOPE_HINTS = ("room", "here", "thread", "channel", "group")
_PEER_SCOPE_HINTS = ("between us", "with me", "for me", "about me", "my side")
_ROOM_SUMMARY_HINTS = ("summary", "summarize", "summarizing", "recap", "catch us up", "for the room", "everyone")
_REVIVE_HINTS = ("back to", "return to", "revive", "pick this back up", "again")
_HANDOFF_TO_ORION_HINTS = ("oríon,", "orion,", "what do you think", "over to you", "your take")
_YIELD_HINTS = ("curious what", "someone else", "anyone else", "let them", "your turn")
_COMMITMENT_WEAK_HINTS = ("maybe", "might", "could", "if needed", "probably", "perhaps")
_COMMITMENT_SUMMARY_HINTS = ("i'll summarize", "i will summarize", "let me summarize", "i'll recap", "i will recap", "quick summary in a sec")
_COMMITMENT_RETURN_HINTS = (
    "i'll come back",
    "i will come back",
    "i'll return to",
    "i will return to",
    "come back to",
    "return to",
    "back to that in a sec",
)
_COMMITMENT_ANSWER_HINTS = (
    "let me answer",
    "i'll answer",
    "i will answer",
    "answer the",
    "i'll respond after",
    "i will respond after",
)
_COMMITMENT_YIELD_HINTS = ("i'm yielding", "i am yielding", "i'll yield", "i will yield", "i'll let", "i will let")
_SUMMARY_DELIVERY_HINTS = ("summary:", "quick summary", "short summary", "brief recap", "here's where we are")
_REENTRY_HINTS = ("back with you", "picking this back up", "coming back to", "returning to")
_CLAIM_ASSERTIVE_HINTS = (
    " is ",
    " are ",
    " was ",
    " were ",
    " means ",
    " feels like ",
    " looks like ",
    " works better ",
    " needs to ",
    " should ",
)
_CLAIM_ACCEPT_HINTS = ("we agreed", "we're aligned", "we are aligned", "confirmed", "settled", "yes, that's right")
_CLAIM_CORRECTION_HINTS = (
    "actually",
    "to correct",
    "i was wrong",
    "that was wrong",
    "not ",
    "instead",
    "rather than",
)
_CLAIM_DISPUTE_HINTS = (
    "i don't think",
    "i do not think",
    "i disagree",
    "that's not right",
    "that isn't right",
    "not sure that's right",
)
_CLAIM_WITHDRAW_HINTS = ("never mind", "withdraw that", "scratch that", "i take that back")
_CLAIM_TRACK_LIMIT = 2
_CLAIM_REVISION_LIMIT = 3
_DELIBERATION_REQUEST_HINTS = (
    "where are we actually landing",
    "where are we landing",
    "where are we on this",
    "what are we agreeing on",
    "what are we disagreeing on",
    "what's the split",
    "where do we actually differ",
)
_DELIBERATION_CLARIFY_HINTS = (
    "same thing",
    "same meaning",
    "room-level conclusion",
    "room level conclusion",
    "local view",
    "which part",
    "which thread",
    "which of these",
)
_DELIBERATION_META_STOPWORDS = {
    "room", "thread", "threads", "claim", "claims", "view", "views", "issue", "issues", "thing", "things",
    "people", "person", "everyone", "anyone", "someone", "peer", "peers", "orion", "oríon", "actually", "really",
    "landing", "agreeing", "disagreeing", "shared", "summary", "question", "clarify", "clarifying",
}
_ALIGNMENT_HINTS = (
    "aligned enough",
    "that sounds aligned",
    "sounds aligned enough",
    "we're aligned",
    "we are aligned",
    "that lands for me",
    "works for me",
    "that fits",
    "settled enough",
    "good for now",
)
_LEAVE_OPEN_HINTS = (
    "leave it there",
    "unless someone wants to reopen",
    "happy to leave that open",
    "open edge",
)


@dataclass(frozen=True)
class SharedArtifactDecision:
    status: str = "unknown"
    scope: str = "peer_local"
    summary: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ClaimTrackingResult:
    claims: list[SocialClaimV1] = field(default_factory=list)
    revisions: list[SocialClaimRevisionV1] = field(default_factory=list)
    stances: list[SocialClaimStanceV1] = field(default_factory=list)
    attributions: list[SocialClaimAttributionV1] = field(default_factory=list)
    consensus_states: list[SocialConsensusStateV1] = field(default_factory=list)
    divergence_signals: list[SocialDivergenceSignalV1] = field(default_factory=list)
    ignored_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DeliberationResult:
    bridge_summary: SocialBridgeSummaryV1 | None = None
    clarifying_question: SocialClarifyingQuestionV1 | None = None
    decision: SocialDeliberationDecisionV1 | None = None
    ignored_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FloorResult:
    turn_handoff: SocialTurnHandoffV1 | None = None
    closure_signal: SocialClosureSignalV1 | None = None
    decision: SocialFloorDecisionV1 | None = None
    ignored_reasons: list[str] = field(default_factory=list)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_text(value: str | None) -> str:
    text = str(value or "")
    text = _EMAIL_RE.sub("[redacted-email]", text)
    text = _PHONE_RE.sub("[redacted-phone]", text)
    text = _LONG_DIGIT_RE.sub("[redacted-number]", text)
    return text.strip()


def _claim_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _WORD_RE.findall(text.lower())
        if token.lower() not in _STOPWORDS and len(token) >= 4
    }


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a = set(left)
    b = set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _normalize_claim_text(text: str, *, limit: int = 160) -> str:
    normalized = sanitize_text(text)
    normalized = re.sub(r"^(actually|to correct|quick summary:|summary:|my read is|from what i remember),?\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .,:;!?")
    return normalized[:limit]


def _looks_trackworthy_claim(text: str) -> bool:
    lowered = text.lower()
    if len(text.split()) < 4:
        return False
    if "?" in text and not any(hint in lowered for hint in _CLAIM_CORRECTION_HINTS + _CLAIM_DISPUTE_HINTS):
        return False
    if any(hint in lowered for hint in _CLAIM_CORRECTION_HINTS + _CLAIM_DISPUTE_HINTS + _CLAIM_WITHDRAW_HINTS):
        return True
    if any(hint in lowered for hint in _SUMMARY_DELIVERY_HINTS):
        return True
    return any(hint in lowered for hint in _CLAIM_ASSERTIVE_HINTS)


def _claim_basis(
    *,
    repair_signal: dict[str, Any] | None,
    epistemic_signal: dict[str, Any] | None,
) -> str:
    if isinstance(repair_signal, dict) and repair_signal:
        return "repair_context"
    if isinstance(epistemic_signal, dict) and epistemic_signal:
        return "epistemic_context"
    return "recent_turns"


def _source_participant_for_claim(
    turn: SocialRoomTurnStoredV1,
    *,
    from_response: bool,
    participant_id: str | None,
    participant_name: str | None,
) -> tuple[str | None, str | None]:
    if from_response:
        return "orion", "Oríon"
    return participant_id, participant_name


def _claim_kind_for_text(text: str, *, from_response: bool) -> str:
    lowered = text.lower()
    if any(hint in lowered for hint in _SUMMARY_DELIVERY_HINTS):
        return "shared_summary"
    if any(hint in lowered for hint in ("my read is", "it seems like", "it sounds like")):
        return "inferred_claim"
    return "orion_claim" if from_response else "peer_claim"


def _initial_claim_stance(text: str) -> str:
    lowered = text.lower()
    if any(hint in lowered for hint in _CLAIM_WITHDRAW_HINTS):
        return "withdrawn"
    if any(hint in lowered for hint in _CLAIM_DISPUTE_HINTS):
        return "disputed"
    if any(hint in lowered for hint in _CLAIM_ACCEPT_HINTS):
        return "accepted"
    return "provisional"


def _claim_confidence(text: str, *, stance: str, from_response: bool) -> float:
    lowered = text.lower()
    confidence = 0.48 if from_response else 0.42
    if stance == "accepted":
        confidence += 0.18
    if stance in {"disputed", "withdrawn"}:
        confidence = min(confidence, 0.38)
    if any(hint in lowered for hint in ("maybe", "might", "could", "guess", "probably")):
        confidence = min(confidence, 0.36)
    return max(0.0, min(confidence, 0.85))


def _revision_type(text: str) -> str | None:
    lowered = text.lower()
    if any(hint in lowered for hint in _CLAIM_WITHDRAW_HINTS):
        return "withdrawn"
    if any(hint in lowered for hint in _CLAIM_DISPUTE_HINTS):
        return "disputed"
    if any(hint in lowered for hint in _CLAIM_CORRECTION_HINTS):
        return "corrected"
    return None


def _revised_stance_for(text: str, *, current_stance: str) -> str:
    revision_type = _revision_type(text)
    if revision_type == "withdrawn":
        return "withdrawn"
    if revision_type == "disputed":
        return "disputed"
    if revision_type == "corrected":
        return "corrected" if current_stance != "withdrawn" else "withdrawn"
    return current_stance


def _claim_matches(existing: SocialClaimStanceV1, summary: str) -> float:
    direct = 1.0 if existing.normalized_summary.lower() == summary.lower() else 0.0
    overlap = _jaccard(_claim_tokens(existing.normalized_summary), _claim_tokens(summary))
    return max(direct, overlap)


def _claim_stance_from_claim(claim: SocialClaimV1) -> SocialClaimStanceV1:
    return SocialClaimStanceV1(
        claim_id=claim.claim_id,
        platform=claim.platform,
        room_id=claim.room_id,
        thread_key=claim.thread_key,
        source_participant_id=claim.source_participant_id,
        source_participant_name=claim.source_participant_name,
        claim_kind=claim.claim_kind,
        normalized_summary=claim.normalized_summary,
        current_stance=claim.stance,
        confidence=claim.confidence,
        source_basis=claim.source_basis,
        related_claim_ids=list(claim.related_claim_ids),
        reasons=list(claim.reasons),
        created_at=claim.created_at,
        updated_at=claim.updated_at,
        metadata=dict(claim.metadata),
    )


def _participant_claim_position(claim: SocialClaimV1) -> str:
    if claim.stance == "withdrawn":
        return "withdraw"
    if claim.stance in {"corrected", "revised"}:
        return "correct"
    if claim.stance == "disputed":
        return "dispute"
    return "support"


def _new_attribution_for_claim(claim: SocialClaimV1) -> SocialClaimAttributionV1:
    participant_ids: list[str] = []
    participant_names: dict[str, str] = {}
    participant_stances: dict[str, str] = {}
    orion_stance = "unknown"
    if claim.source_participant_id == "orion":
        orion_stance = _participant_claim_position(claim)
    elif claim.source_participant_id:
        participant_ids = [claim.source_participant_id]
        participant_stances[claim.source_participant_id] = _participant_claim_position(claim)
        if claim.source_participant_name:
            participant_names[claim.source_participant_id] = claim.source_participant_name
    return SocialClaimAttributionV1(
        claim_id=claim.claim_id,
        platform=claim.platform,
        room_id=claim.room_id,
        thread_key=claim.thread_key,
        normalized_claim_key=claim.normalized_summary,
        attributed_participant_ids=participant_ids,
        attributed_participant_names=participant_names,
        participant_stances=participant_stances,  # type: ignore[arg-type]
        orion_stance=orion_stance,  # type: ignore[arg-type]
        confidence=claim.confidence,
        supporting_evidence_count=1,
        updated_at=claim.updated_at,
        reasons=["initial_attribution"],
        metadata={"source": "social-memory"},
    )


def _apply_position_to_attribution(
    attribution: SocialClaimAttributionV1,
    *,
    participant_id: str | None,
    participant_name: str | None,
    position: str,
    confidence: float,
    updated_at: str,
    reason: str,
) -> SocialClaimAttributionV1:
    participant_ids = list(attribution.attributed_participant_ids)
    participant_names = dict(attribution.attributed_participant_names or {})
    participant_stances = dict(attribution.participant_stances or {})
    orion_stance = attribution.orion_stance
    if participant_id == "orion":
        orion_stance = position
    elif participant_id:
        if participant_id not in participant_ids:
            participant_ids.append(participant_id)
        participant_stances[participant_id] = position
        if participant_name:
            participant_names[participant_id] = participant_name
    return attribution.model_copy(
        update={
            "attributed_participant_ids": participant_ids,
            "attributed_participant_names": participant_names,
            "participant_stances": participant_stances,
            "orion_stance": orion_stance,
            "confidence": max(attribution.confidence, confidence),
            "supporting_evidence_count": max(
                attribution.supporting_evidence_count,
                len([value for value in participant_stances.values() if value == "support"]) + (1 if orion_stance == "support" else 0),
            ),
            "updated_at": updated_at,
            "reasons": merge_unique(attribution.reasons, [reason], limit=4),
        }
    )


def _consensus_from_attribution(
    attribution: SocialClaimAttributionV1,
    *,
    latest_revision: SocialClaimRevisionV1 | None = None,
) -> SocialConsensusStateV1:
    participant_stances = dict(attribution.participant_stances or {})
    supporting_ids = [pid for pid, stance in participant_stances.items() if stance == "support"]
    disputing_ids = [pid for pid, stance in participant_stances.items() if stance in {"dispute", "withdraw"}]
    questioning_ids = [pid for pid, stance in participant_stances.items() if stance == "question"]
    support_count = len(supporting_ids) + (1 if attribution.orion_stance == "support" else 0)
    corrected_present = any(stance == "correct" for stance in participant_stances.values()) or attribution.orion_stance == "correct"
    if corrected_present:
        consensus_state = "corrected"
    elif disputing_ids or attribution.orion_stance in {"dispute", "withdraw"}:
        consensus_state = "contested"
    elif support_count >= 3:
        consensus_state = "consensus"
    elif support_count == 2:
        consensus_state = "emerging" if attribution.orion_stance == "support" else "partial"
    else:
        consensus_state = "none"
    reasons = [f"support_count={support_count}", f"disputes={len(disputing_ids)}"]
    if latest_revision is not None:
        reasons.append(f"latest_revision={latest_revision.revision_type}")
    return SocialConsensusStateV1(
        claim_id=attribution.claim_id,
        platform=attribution.platform,
        room_id=attribution.room_id,
        thread_key=attribution.thread_key,
        normalized_claim_key=attribution.normalized_claim_key,
        consensus_state=consensus_state,  # type: ignore[arg-type]
        supporting_participant_ids=supporting_ids,
        disputing_participant_ids=disputing_ids,
        questioning_participant_ids=questioning_ids,
        orion_stance=attribution.orion_stance,
        confidence=min(0.9, max(attribution.confidence, 0.35 + support_count * 0.1)),
        supporting_evidence_count=support_count,
        updated_at=attribution.updated_at,
        reasons=reasons,
        metadata={"source": "social-memory"},
    )


def _divergence_from_attribution(
    attribution: SocialClaimAttributionV1,
    consensus_state: SocialConsensusStateV1,
) -> SocialDivergenceSignalV1 | None:
    if consensus_state.consensus_state not in {"partial", "contested", "corrected"}:
        return None
    return SocialDivergenceSignalV1(
        claim_id=attribution.claim_id,
        platform=attribution.platform,
        room_id=attribution.room_id,
        thread_key=attribution.thread_key,
        normalized_claim_key=attribution.normalized_claim_key,
        divergence_detected=True,
        consensus_state=consensus_state.consensus_state,
        participant_stances=dict(attribution.participant_stances or {}),
        orion_stance=attribution.orion_stance,
        confidence=max(consensus_state.confidence, attribution.confidence),
        supporting_evidence_count=consensus_state.supporting_evidence_count,
        updated_at=consensus_state.updated_at,
        reasons=list(consensus_state.reasons) + ["divergence_requires_attribution"],
        metadata={"source": "social-memory"},
    )


def _claim_candidate(
    text: str,
    *,
    turn: SocialRoomTurnStoredV1,
    platform: str,
    room_id: str,
    thread_key: str | None,
    participant_id: str | None,
    participant_name: str | None,
    from_response: bool,
    repair_signal: dict[str, Any] | None,
    epistemic_signal: dict[str, Any] | None,
) -> SocialClaimV1 | None:
    normalized = _normalize_claim_text(text)
    if not normalized or len(normalized.split()) < 4:
        return None
    if _SEALED_RE.search(normalized):
        return None
    if not _looks_trackworthy_claim(normalized):
        return None
    stance = _initial_claim_stance(normalized)
    source_id, source_name = _source_participant_for_claim(
        turn,
        from_response=from_response,
        participant_id=participant_id,
        participant_name=participant_name,
    )
    return SocialClaimV1(
        platform=platform,
        room_id=room_id,
        thread_key=thread_key,
        source_participant_id=source_id,
        source_participant_name=source_name,
        claim_text=normalized,
        normalized_summary=normalized,
        claim_kind=_claim_kind_for_text(normalized, from_response=from_response),  # type: ignore[arg-type]
        stance=stance,  # type: ignore[arg-type]
        confidence=_claim_confidence(normalized, stance=stance, from_response=from_response),
        source_basis=_claim_basis(repair_signal=repair_signal, epistemic_signal=epistemic_signal),  # type: ignore[arg-type]
        reasons=["assertive_or_corrective_turn"],
        created_at=turn.created_at,
        updated_at=turn.created_at,
        metadata={"source": "social-memory"},
    )


def update_room_claim_tracking(
    existing: SocialRoomContinuityV1 | None,
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    thread_key: str | None,
    participant_id: str | None,
    participant_name: str | None,
    artifact_dialogue_active: bool,
    shared_artifact_statuses: list[str] | None,
    repair_signal: dict[str, Any] | None,
    repair_decision: dict[str, Any] | None,
    epistemic_signal: dict[str, Any] | None,
) -> ClaimTrackingResult:
    statuses = {str(item or "") for item in (shared_artifact_statuses or [])}
    if artifact_dialogue_active or statuses & {"declined", "deferred"}:
        return ClaimTrackingResult(ignored_reasons=["claim tracking skipped for pending_or_non_active_artifact_state"])

    prompt_text = sanitize_text(turn.prompt)
    response_text = sanitize_text(turn.response)
    if _SEALED_RE.search(prompt_text) or _SEALED_RE.search(response_text):
        return ClaimTrackingResult(ignored_reasons=["claim tracking skipped for blocked/private/sealed material"])

    existing_stances = [
        SocialClaimStanceV1.model_validate(item if isinstance(item, dict) else item.model_dump(mode="json"))
        for item in (existing.active_claims if existing else [])
    ]
    stances = list(existing_stances)
    attributions = [
        SocialClaimAttributionV1.model_validate(item if isinstance(item, dict) else item.model_dump(mode="json"))
        for item in (existing.claim_attributions if existing else [])
    ]
    revisions = [
        SocialClaimRevisionV1.model_validate(item if isinstance(item, dict) else item.model_dump(mode="json"))
        for item in (existing.recent_claim_revisions if existing else [])
    ][: _CLAIM_REVISION_LIMIT - 1]
    claims: list[SocialClaimV1] = []
    ignored: list[str] = []

    candidates = [
        _claim_candidate(
            prompt_text,
            turn=turn,
            platform=platform,
            room_id=room_id,
            thread_key=thread_key,
            participant_id=participant_id,
            participant_name=participant_name,
            from_response=False,
            repair_signal=repair_signal,
            epistemic_signal=epistemic_signal,
        ),
        _claim_candidate(
            response_text,
            turn=turn,
            platform=platform,
            room_id=room_id,
            thread_key=thread_key,
            participant_id=participant_id,
            participant_name=participant_name,
            from_response=True,
            repair_signal=repair_signal,
            epistemic_signal=epistemic_signal,
        ),
    ]

    for candidate in [item for item in candidates if item is not None]:
        if candidate.confidence < 0.4 and candidate.stance not in {"disputed", "corrected", "revised", "withdrawn"}:
            ignored.append(f"low-confidence claim ignored: {candidate.normalized_summary[:120]}")
            continue
        revision_type = _revision_type(candidate.normalized_summary)
        matched_idx = -1
        matched_score = 0.0
        for idx, existing_claim in enumerate(stances):
            score = _claim_matches(existing_claim, candidate.normalized_summary)
            if score > matched_score:
                matched_score = score
                matched_idx = idx
        if revision_type is not None and matched_idx >= 0 and matched_score >= 0.10:
            existing_claim = stances[matched_idx]
            new_stance = _revised_stance_for(candidate.normalized_summary, current_stance=existing_claim.current_stance)
            revision = SocialClaimRevisionV1(
                platform=platform,
                room_id=room_id,
                thread_key=thread_key,
                claim_id=existing_claim.claim_id,
                revision_type=revision_type,  # type: ignore[arg-type]
                prior_stance=existing_claim.current_stance,
                new_stance=new_stance,  # type: ignore[arg-type]
                source_participant_id=candidate.source_participant_id,
                source_participant_name=candidate.source_participant_name,
                revised_summary=candidate.normalized_summary,
                confidence=max(existing_claim.confidence, candidate.confidence),
                source_basis=candidate.source_basis,
                related_claim_ids=[existing_claim.claim_id],
                reasons=list(candidate.reasons) + [f"matched_score={matched_score:.2f}"],
                created_at=turn.created_at,
                metadata={"source": "social-memory"},
            )
            revisions.insert(0, revision)
            stances[matched_idx] = existing_claim.model_copy(
                update={
                    "current_stance": new_stance,
                    "normalized_summary": candidate.normalized_summary,
                    "confidence": max(existing_claim.confidence, candidate.confidence),
                    "related_claim_ids": merge_unique(existing_claim.related_claim_ids, [existing_claim.claim_id], limit=3),
                    "reasons": merge_unique(existing_claim.reasons, [f"revised:{revision_type}"], limit=4),
                    "updated_at": turn.created_at,
                }
            )
            claims.append(
                candidate.model_copy(
                    update={
                        "claim_id": existing_claim.claim_id,
                        "stance": new_stance,
                        "related_claim_ids": [existing_claim.claim_id],
                    }
                )
            )
            attribution_idx = next((idx for idx, item in enumerate(attributions) if item.claim_id == existing_claim.claim_id), -1)
            if attribution_idx >= 0:
                attributions[attribution_idx] = _apply_position_to_attribution(
                    attributions[attribution_idx],
                    participant_id=candidate.source_participant_id,
                    participant_name=candidate.source_participant_name,
                    position=_participant_claim_position(candidate.model_copy(update={"stance": new_stance})),
                    confidence=candidate.confidence,
                    updated_at=turn.created_at,
                    reason=f"revision:{revision_type}",
                )
            continue

        support_idx = next(
            (idx for idx, existing_claim in enumerate(stances) if _claim_matches(existing_claim, candidate.normalized_summary) >= 0.75),
            -1,
        )
        if support_idx >= 0:
            existing_claim = stances[support_idx]
            attribution_idx = next((idx for idx, item in enumerate(attributions) if item.claim_id == existing_claim.claim_id), -1)
            if attribution_idx >= 0:
                attributions[attribution_idx] = _apply_position_to_attribution(
                    attributions[attribution_idx],
                    participant_id=candidate.source_participant_id,
                    participant_name=candidate.source_participant_name,
                    position=_participant_claim_position(candidate),
                    confidence=candidate.confidence,
                    updated_at=turn.created_at,
                    reason="additional_support_or_echo",
                )
                stances[support_idx] = existing_claim.model_copy(
                    update={
                        "confidence": max(existing_claim.confidence, candidate.confidence),
                        "updated_at": turn.created_at,
                        "reasons": merge_unique(existing_claim.reasons, ["additional_support_or_echo"], limit=4),
                    }
                )
            else:
                attributions.insert(0, _new_attribution_for_claim(candidate.model_copy(update={"claim_id": existing_claim.claim_id})))
            continue

        if isinstance(repair_decision, dict) and str(repair_decision.get("decision") or "") in {"repair", "clarify"}:
            candidate = candidate.model_copy(
                update={
                    "stance": "provisional",
                    "source_basis": "repair_context",
                    "reasons": list(candidate.reasons) + ["repair-aware provisional claim"],
                }
            )
        claims.append(candidate)
        stances.insert(0, _claim_stance_from_claim(candidate))
        attributions.insert(0, _new_attribution_for_claim(candidate))

    consensus_states: list[SocialConsensusStateV1] = []
    divergence_signals: list[SocialDivergenceSignalV1] = []
    for attribution in attributions[:_CLAIM_TRACK_LIMIT]:
        latest_revision = next((item for item in revisions if item.claim_id == attribution.claim_id), None)
        consensus = _consensus_from_attribution(attribution, latest_revision=latest_revision)
        consensus_states.append(consensus)
        divergence = _divergence_from_attribution(attribution, consensus)
        if divergence is not None:
            divergence_signals.append(divergence)

    return ClaimTrackingResult(
        claims=claims[:_CLAIM_TRACK_LIMIT],
        revisions=revisions[:_CLAIM_REVISION_LIMIT],
        stances=stances[:_CLAIM_TRACK_LIMIT],
        attributions=attributions[:_CLAIM_TRACK_LIMIT],
        consensus_states=consensus_states[:_CLAIM_TRACK_LIMIT],
        divergence_signals=divergence_signals[:_CLAIM_TRACK_LIMIT],
        ignored_reasons=ignored,
    )



def _deliberation_focus_tokens(*texts: str) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in _claim_tokens(str(text or "")):
            if token in _DELIBERATION_META_STOPWORDS:
                continue
            counter[token] += 1
    return [token for token, _ in counter.most_common(4)]


def _shared_core_phrase(
    *,
    room: SocialRoomContinuityV1 | None,
    claim_tracking: ClaimTrackingResult,
) -> str:
    texts: list[str] = []
    texts.extend(item.normalized_summary for item in claim_tracking.stances[:2])
    texts.extend(item.revised_summary for item in claim_tracking.revisions[:2])
    texts.extend((room.recurring_topics if room else [])[:3])
    if room and room.current_thread_summary:
        texts.append(room.current_thread_summary)
    focus = _deliberation_focus_tokens(*texts)
    if len(focus) >= 2:
        return f"the shared core seems to be {focus[0]} / {focus[1]}"
    if len(focus) == 1:
        return f"the shared core seems to be {focus[0]}"
    if room and room.current_thread_summary:
        return sanitize_text(room.current_thread_summary)[:120]
    return "the room is circling the same core topic from different angles"


def _attributed_view_lines(
    *,
    attributions: list[SocialClaimAttributionV1],
    stances: list[SocialClaimStanceV1],
    revisions: list[SocialClaimRevisionV1],
) -> tuple[list[str], dict[str, str]]:
    claim_lookup = {item.claim_id: item for item in stances}
    latest_revision_by_claim = {item.claim_id: item for item in revisions}
    lines: list[str] = []
    participants: dict[str, str] = {}
    for attribution in attributions[:2]:
        claim = claim_lookup.get(attribution.claim_id)
        revision = latest_revision_by_claim.get(attribution.claim_id)
        view_text = revision.revised_summary if revision is not None else (claim.normalized_summary if claim is not None else attribution.normalized_claim_key)
        for participant_id in attribution.attributed_participant_ids:
            participant_name = (attribution.attributed_participant_names or {}).get(participant_id) or participant_id
            participants[participant_id] = participant_name
            stance = (attribution.participant_stances or {}).get(participant_id, "support")
            lines.append(f"{participant_name}: {view_text} [{stance}]")
        if attribution.orion_stance and attribution.orion_stance != "unknown":
            lines.append(f"Oríon: {view_text} [{attribution.orion_stance}]")
            participants["orion"] = "Oríon"
    deduped: list[str] = []
    for line in lines:
        if line not in deduped:
            deduped.append(line)
    return deduped[:4], participants


def _disagreement_edge(
    *,
    claim_tracking: ClaimTrackingResult,
) -> str:
    if claim_tracking.revisions:
        revision = claim_tracking.revisions[0]
        return f"the live disagreement edge is whether the earlier framing still holds or the corrected framing ({revision.revised_summary[:110]}) is the better read"
    if claim_tracking.divergence_signals:
        divergence = claim_tracking.divergence_signals[0]
        claim_key = sanitize_text(divergence.normalized_claim_key)
        return f"the disagreement edge is how strongly to endorse '{claim_key[:100]}' as the room-level read"
    if len(claim_tracking.stances) >= 2:
        return f"the disagreement edge is between '{claim_tracking.stances[0].normalized_summary[:80]}' and '{claim_tracking.stances[1].normalized_summary[:80]}'"
    if claim_tracking.stances:
        return f"the disagreement edge is whether '{claim_tracking.stances[0].normalized_summary[:100]}' is the right room-level framing"
    return "the disagreement edge is still unresolved"


def _bridge_summary_text(shared_core: str, disagreement_edge: str, views: list[str]) -> str:
    view_text = "; ".join(views[:3])
    if view_text:
        return f"Shared core: {shared_core}. Views: {view_text}. Disagreement edge: {disagreement_edge}."[:420]
    return f"Shared core: {shared_core}. Disagreement edge: {disagreement_edge}."[:320]


def build_deliberation_result(
    *,
    turn: SocialRoomTurnStoredV1,
    room: SocialRoomContinuityV1 | None,
    claim_tracking: ClaimTrackingResult,
    artifact_dialogue_active: bool,
) -> DeliberationResult:
    prompt = sanitize_text(turn.prompt)
    lowered = prompt.lower()
    if artifact_dialogue_active:
        return DeliberationResult(ignored_reasons=["deliberation skipped for pending artifact dialogue"])
    if _SEALED_RE.search(prompt) or _SEALED_RE.search(sanitize_text(turn.response)):
        return DeliberationResult(ignored_reasons=["deliberation skipped for blocked/private/sealed material"])

    consensus_states = claim_tracking.consensus_states
    contested = [item for item in consensus_states if item.consensus_state in {"contested", "corrected"}]
    partial = [item for item in consensus_states if item.consensus_state == "partial"]
    repeated_cross_talk = bool(room and len(room.active_threads) >= 2 and any(thread.open_question for thread in room.active_threads[:2]))
    explicit_landing_request = any(hint in lowered for hint in _DELIBERATION_REQUEST_HINTS)
    ambiguity_markers = any(hint in lowered for hint in _DELIBERATION_CLARIFY_HINTS)
    shared_core = _shared_core_phrase(room=room, claim_tracking=claim_tracking)
    views, participants = _attributed_view_lines(
        attributions=claim_tracking.attributions,
        stances=claim_tracking.stances,
        revisions=claim_tracking.revisions,
    )
    disagreement_edge = _disagreement_edge(claim_tracking=claim_tracking)

    routing_audience = "peer"
    if room and room.active_threads:
        routing_audience = room.active_threads[0].audience_scope

    if ambiguity_markers and not (partial or contested or explicit_landing_request):
        question = SocialClarifyingQuestionV1(
            platform=room.platform if room else str((turn.client_meta or {}).get("external_room", {}).get("platform") or "unknown"),
            room_id=room.room_id if room else str((turn.client_meta or {}).get("external_room", {}).get("room_id") or "unknown"),
            thread_key=room.current_thread_key if room else None,
            active_claim_ids=[item.claim_id for item in claim_tracking.stances[:2]],
            active_claim_keys=[item.normalized_summary for item in claim_tracking.stances[:2]],
            trigger="ambiguity",
            question_focus="scope" if "room" in lowered or "local" in lowered else "shared_core",
            question_text=(
                "Are you asking for the room-level landing, or just the local thread read?"
                if ("room" in lowered or "local" in lowered)
                else "Which part should I bridge first — the shared core or the disagreement edge?"
            ),
            attributed_participants=participants,
            confidence=0.74,
            ambiguity_level="high",
            reasons=["ambiguity_markers_present", "question_safer_than_assertion"],
            metadata={"source": "social-memory"},
        )
        decision = SocialDeliberationDecisionV1(
            platform=question.platform,
            room_id=question.room_id,
            thread_key=question.thread_key,
            active_claim_ids=list(question.active_claim_ids),
            active_claim_keys=list(question.active_claim_keys),
            decision_kind="ask_clarifying_question",
            trigger=question.trigger,
            clarifying_question_id=question.clarifying_question_id,
            confidence=question.confidence,
            ambiguity_level=question.ambiguity_level,
            reasons=list(question.reasons),
            metadata={"source": "social-memory"},
        )
        return DeliberationResult(clarifying_question=question, decision=decision)

    if partial or contested or explicit_landing_request or repeated_cross_talk:
        trigger = "explicit_landing_request" if explicit_landing_request else "contested_shared_core" if contested else "crosstalk" if repeated_cross_talk else "partial_agreement"
        agreement_points = []
        if shared_core:
            agreement_points.append(shared_core)
        if room and room.current_thread_summary:
            agreement_points.append(f"the room is still on {sanitize_text(room.current_thread_summary)[:100]}")
        disagreement_points = [disagreement_edge]
        bridge = SocialBridgeSummaryV1(
            platform=room.platform if room else str((turn.client_meta or {}).get("external_room", {}).get("platform") or "unknown"),
            room_id=room.room_id if room else str((turn.client_meta or {}).get("external_room", {}).get("room_id") or "unknown"),
            thread_key=room.current_thread_key if room else None,
            active_claim_ids=[item.claim_id for item in claim_tracking.stances[:2]],
            active_claim_keys=[item.normalized_summary for item in claim_tracking.stances[:2]],
            trigger=trigger,
            shared_core=shared_core,
            disagreement_edge=disagreement_edge,
            attributed_views=views,
            agreement_points=agreement_points[:2],
            disagreement_points=disagreement_points[:2],
            attributed_participants=participants,
            summary_text=_bridge_summary_text(shared_core, disagreement_edge, views),
            proposed_bridge_framing="Offer a compact bridge summary that names both the overlap and the unresolved edge without pretending the room fully agrees.",
            confidence=0.72 if contested else 0.68 if partial else 0.66,
            ambiguity_level="medium" if explicit_landing_request or repeated_cross_talk else "low",
            preserve_disagreement=True,
            reasons=[reason for reason, present in [("partial_agreement_detected", bool(partial)), ("contested_shared_core_detected", bool(contested)), ("repeated_cross_talk_detected", repeated_cross_talk), ("explicit_landing_request_detected", explicit_landing_request)] if present],
            metadata={"source": "social-memory"},
        )
        decision_kind = "normal_room_reply" if explicit_landing_request or repeated_cross_talk or routing_audience in {"room", "summary"} else "bridge_summary"
        if decision_kind != "bridge_summary":
            decision_kind = "bridge_summary"
        decision = SocialDeliberationDecisionV1(
            platform=bridge.platform,
            room_id=bridge.room_id,
            thread_key=bridge.thread_key,
            active_claim_ids=list(bridge.active_claim_ids),
            active_claim_keys=list(bridge.active_claim_keys),
            decision_kind=decision_kind,
            trigger=bridge.trigger,
            bridge_summary_id=bridge.bridge_summary_id,
            confidence=bridge.confidence,
            ambiguity_level=bridge.ambiguity_level,
            reasons=list(bridge.reasons),
            metadata={"source": "social-memory"},
        )
        return DeliberationResult(bridge_summary=bridge, decision=decision)

    decision_kind = "wait" if repeated_cross_talk and not room else "normal_room_reply" if routing_audience in {"room", "summary"} else "normal_peer_reply" if routing_audience in {"peer", "thread"} else "stay_narrow"
    decision = SocialDeliberationDecisionV1(
        platform=room.platform if room else str((turn.client_meta or {}).get("external_room", {}).get("platform") or "unknown"),
        room_id=room.room_id if room else str((turn.client_meta or {}).get("external_room", {}).get("room_id") or "unknown"),
        thread_key=room.current_thread_key if room else None,
        active_claim_ids=[item.claim_id for item in claim_tracking.stances[:2]],
        active_claim_keys=[item.normalized_summary for item in claim_tracking.stances[:2]],
        decision_kind=decision_kind,
        trigger="routing_fallback",
        confidence=0.51 if decision_kind in {"stay_narrow", "wait"} else 0.58,
        ambiguity_level="medium" if decision_kind in {"stay_narrow", "wait"} else "low",
        reasons=["no_deliberative_intervention_needed"],
        metadata={"source": "social-memory"},
    )
    return DeliberationResult(decision=decision)


def _floor_target(
    *,
    room: SocialRoomContinuityV1 | None,
    claim_tracking: ClaimTrackingResult,
) -> tuple[str | None, str | None, str]:
    if room and room.active_threads:
        thread = room.active_threads[0]
        target_id = thread.target_participant_id
        target_name = thread.target_participant_name
        if (target_id or target_name) and (target_id or "").lower() != "orion" and (target_name or "").lower() not in {"orion", "oríon"}:
            return target_id, target_name, thread.audience_scope
    if room and room.handoff_signal and room.handoff_signal.to_participant_id:
        if room.handoff_signal.to_participant_id.lower() != "orion":
            return room.handoff_signal.to_participant_id, room.handoff_signal.to_participant_name, room.handoff_signal.audience_scope
    if room and room.bridge_summary and room.bridge_summary.attributed_participants:
        for participant_id, participant_name in room.bridge_summary.attributed_participants.items():
            if participant_id != "orion":
                return participant_id, participant_name, "peer"
    for attribution in claim_tracking.attributions[:1]:
        for participant_id in attribution.attributed_participant_ids:
            participant_name = (attribution.attributed_participant_names or {}).get(participant_id) or participant_id
            if participant_id != "orion":
                return participant_id, participant_name, "peer"
    audience_scope = room.active_threads[0].audience_scope if room and room.active_threads else "room"
    return None, None, audience_scope


def _handoff_phrase(
    *,
    decision_kind: str,
    target_name: str | None,
    room: SocialRoomContinuityV1 | None,
) -> str:
    open_edge = ""
    if room and room.bridge_summary and room.bridge_summary.disagreement_edge:
        edge = sanitize_text(room.bridge_summary.disagreement_edge)
        open_edge = f" I think the open edge is still {edge[:72]}." if edge else ""
    if decision_kind == "yield_to_peer" and target_name:
        return f"{target_name}, does that match your read?"
    if decision_kind == "invite_peer" and target_name:
        return f"{target_name}, does that match what you meant?"
    if decision_kind == "invite_room":
        return "I’ll leave it there unless someone wants to reopen that."
    if decision_kind == "close_thread":
        return "That sounds aligned enough for now."
    if decision_kind == "leave_open":
        if target_name:
            return f"{target_name}, I’ll leave space for your answer.{open_edge}"[:180]
        return "I’ll leave that open for an answer."
    return ""


def _thread_locally_resolved(
    *,
    turn: SocialRoomTurnStoredV1,
    room: SocialRoomContinuityV1 | None,
    claim_tracking: ClaimTrackingResult,
    active_commitments: list[SocialCommitmentV1] | None,
) -> bool:
    response = sanitize_text(turn.response).lower()
    prompt = sanitize_text(turn.prompt).lower()
    if "?" in prompt:
        return False
    if any(hint in response for hint in _LEAVE_OPEN_HINTS):
        return False
    if active_commitments:
        return False
    contested = any(item.consensus_state in {"contested"} for item in claim_tracking.consensus_states)
    if contested:
        return False
    if room and room.active_threads and room.active_threads[0].open_question:
        return False
    corrected = any(item.consensus_state == "corrected" for item in claim_tracking.consensus_states)
    return any(hint in response for hint in _ALIGNMENT_HINTS) or corrected


def build_floor_result(
    *,
    turn: SocialRoomTurnStoredV1,
    room: SocialRoomContinuityV1 | None,
    claim_tracking: ClaimTrackingResult,
    active_commitments: list[SocialCommitmentV1] | None,
    artifact_dialogue_active: bool,
) -> FloorResult:
    prompt = sanitize_text(turn.prompt)
    response = sanitize_text(turn.response)
    if artifact_dialogue_active:
        return FloorResult(ignored_reasons=["floor handling skipped for pending artifact dialogue"])
    if _SEALED_RE.search(prompt) or _SEALED_RE.search(response):
        return FloorResult(ignored_reasons=["floor handling skipped for blocked/private/sealed material"])

    platform = room.platform if room else str((turn.client_meta or {}).get("external_room", {}).get("platform") or "unknown")
    room_id = room.room_id if room else str((turn.client_meta or {}).get("external_room", {}).get("room_id") or "unknown")
    thread_key = room.current_thread_key if room else None
    target_id, target_name, audience_scope = _floor_target(room=room, claim_tracking=claim_tracking)

    decision_kind = "no_handoff"
    rationale = "no conservative handoff or closure signal is needed"
    reasons = ["no_floor_intervention_needed"]

    if room and room.clarifying_question is not None:
        if target_id or target_name:
            decision_kind = "invite_peer"
            rationale = "the clarifying question should leave space for the most relevant peer to answer"
            reasons = ["clarifying_question_present", "target_peer_available"]
        else:
            decision_kind = "leave_open"
            rationale = "the clarifying question should stay open rather than Orion filling the gap"
            reasons = ["clarifying_question_present", "leave_space_for_answer"]
    elif room and room.bridge_summary is not None:
        if target_id or target_name:
            decision_kind = "yield_to_peer"
            rationale = "after a bridge summary, the floor should go back to the peer most tied to the open edge"
            reasons = ["bridge_summary_present", "target_peer_available"]
        else:
            decision_kind = "invite_room"
            rationale = "the room-level bridge summary should hand the floor back broadly without Orion over-continuing"
            reasons = ["bridge_summary_present", "room_scope_follow_up"]
    elif _thread_locally_resolved(
        turn=turn,
        room=room,
        claim_tracking=claim_tracking,
        active_commitments=active_commitments,
    ):
        decision_kind = "close_thread"
        rationale = "the local thread looks aligned enough to close without forcing more turns"
        reasons = ["resolved_thread_detected"]
    elif room and room.active_threads and room.active_threads[0].open_question:
        if audience_scope in {"room", "summary"} or len(room.active_threads) >= 2:
            decision_kind = "invite_room"
            rationale = "the thread is still open, but the next contribution is better left to the room"
            reasons = ["open_question_detected", "room_should_pick_up"]
        else:
            decision_kind = "leave_open"
            rationale = "the thread is still open and should stay available for a reply"
            reasons = ["open_question_detected", "leave_space"]

    handoff = None
    closure_signal = None
    if decision_kind != "no_handoff":
        handoff_text = _handoff_phrase(
            decision_kind=decision_kind,
            target_name=target_name,
            room=room,
        )
        handoff = SocialTurnHandoffV1(
            platform=platform,
            room_id=room_id,
            thread_key=thread_key,
            audience_scope=audience_scope,  # type: ignore[arg-type]
            target_participant_id=target_id,
            target_participant_name=target_name,
            decision_kind=decision_kind,  # type: ignore[arg-type]
            handoff_text=handoff_text,
            rationale=rationale,
            reasons=reasons,
            confidence=0.74 if decision_kind in {"yield_to_peer", "invite_peer"} else 0.68 if decision_kind == "close_thread" else 0.62,
            metadata={"source": "social-memory"},
        )
        if decision_kind in {"close_thread", "leave_open", "invite_room"}:
            closure_kind = "resolved" if decision_kind == "close_thread" else "left_open"
            closure_signal = SocialClosureSignalV1(
                platform=platform,
                room_id=room_id,
                thread_key=thread_key,
                audience_scope=audience_scope,  # type: ignore[arg-type]
                target_participant_id=target_id,
                target_participant_name=target_name,
                closure_kind=closure_kind,  # type: ignore[arg-type]
                resolved=decision_kind == "close_thread",
                closure_text=handoff_text,
                rationale=rationale,
                reasons=reasons,
                confidence=max(handoff.confidence - 0.04, 0.0),
                metadata={"source": "social-memory"},
            )

    decision = SocialFloorDecisionV1(
        platform=platform,
        room_id=room_id,
        thread_key=thread_key,
        audience_scope=audience_scope,  # type: ignore[arg-type]
        target_participant_id=target_id,
        target_participant_name=target_name,
        decision_kind=decision_kind,  # type: ignore[arg-type]
        handoff_id=handoff.handoff_id if handoff else None,
        closure_signal_id=closure_signal.closure_signal_id if closure_signal else None,
        rationale=rationale,
        reasons=reasons,
        confidence=handoff.confidence if handoff else 0.5,
        metadata={"source": "social-memory"},
    )
    return FloorResult(turn_handoff=handoff, closure_signal=closure_signal, decision=decision)


def extract_topics(turn: SocialRoomTurnStoredV1, *, limit: int) -> list[str]:
    evidence_text = " ".join(
        str(item.summary)
        for item in turn.concept_evidence[:4]
    )
    text = sanitize_text(f"{turn.prompt} {turn.response} {evidence_text}")
    counts: Counter[str] = Counter()
    for raw in _WORD_RE.findall(text.lower()):
        if raw in _STOPWORDS or len(raw) < 4:
            continue
        counts[raw] += 1
    return [word for word, _ in counts.most_common(limit)]


def evidence_ref(turn: SocialRoomTurnStoredV1) -> str:
    return turn.turn_id or turn.correlation_id or "social-turn"


def merge_unique(existing: Iterable[str] | None, new_items: Iterable[str], *, limit: int) -> list[str]:
    out: list[str] = []
    for item in list(existing or []) + list(new_items):
        normalized = str(item or "").strip()
        if not normalized or normalized in out:
            continue
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def _social_meta(turn: SocialRoomTurnStoredV1) -> dict[str, Any]:
    return dict(turn.client_meta or {})


def _external_room(turn: SocialRoomTurnStoredV1) -> dict[str, Any]:
    return dict(_social_meta(turn).get("external_room") or {})


def _external_participant(turn: SocialRoomTurnStoredV1) -> dict[str, Any]:
    return dict(_social_meta(turn).get("external_participant") or {})


def _message_text(turn: SocialRoomTurnStoredV1) -> str:
    return sanitize_text(turn.prompt)


def _infer_last_addressed(turn: SocialRoomTurnStoredV1) -> tuple[str | None, str | None]:
    external_room = _external_room(turn)
    target_id = str(
        external_room.get("target_participant_id")
        or external_room.get("reply_to_sender_id")
        or ""
    ).strip() or None
    target_name = str(external_room.get("target_participant_name") or "").strip() or None
    if target_id or target_name:
        return target_id, target_name
    text = _message_text(turn).lower()
    if "oríon" in text or "orion" in text:
        return "orion", "Oríon"
    return None, None


def _thread_target(turn: SocialRoomTurnStoredV1) -> tuple[str | None, str | None]:
    return _infer_last_addressed(turn)


def _audience_scope(turn: SocialRoomTurnStoredV1) -> str:
    external_room = _external_room(turn)
    prompt = _message_text(turn).lower()
    if external_room.get("target_participant_id") or external_room.get("reply_to_sender_id"):
        return "peer"
    if any(hint in prompt for hint in _ROOM_SUMMARY_HINTS):
        return "summary"
    if any(hint in prompt for hint in ("everyone", "room", "all of you", "anyone")):
        return "room"
    if "?" in prompt:
        return "thread"
    return "peer"


def _thread_key(
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    topics: list[str],
) -> str:
    external_room = _external_room(turn)
    thread_id = str(external_room.get("thread_id") or "").strip()
    if thread_id:
        return f"{platform}:{room_id}:thread:{thread_id}"
    participant = _external_participant(turn)
    sender_id = str(participant.get("participant_id") or turn.user_id or "peer").strip() or "peer"
    addressed_id, _ = _thread_target(turn)
    if addressed_id:
        return f"{platform}:{room_id}:exchange:{sender_id}:{addressed_id}"
    topic_slug = "-".join(topics[:2]) or "room-thread"
    return f"{platform}:{room_id}:topic:{topic_slug}"


def detect_handoff_signal(
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    thread_key: str,
) -> SocialHandoffSignalV1 | None:
    prompt = _message_text(turn).lower()
    participant = _external_participant(turn)
    from_id = str(participant.get("participant_id") or "").strip() or None
    from_name = str(participant.get("participant_name") or "").strip() or None
    if any(hint in prompt for hint in _HANDOFF_TO_ORION_HINTS):
        return SocialHandoffSignalV1(
            platform=platform,
            room_id=room_id,
            thread_key=thread_key,
            handoff_kind="to_orion",
            audience_scope="peer",
            from_participant_id=from_id,
            from_participant_name=from_name,
            to_participant_id="orion",
            to_participant_name="Oríon",
            detected=True,
            rationale="the peer explicitly tossed the thread to Orion",
        )
    if any(hint in prompt for hint in _ROOM_SUMMARY_HINTS):
        return SocialHandoffSignalV1(
            platform=platform,
            room_id=room_id,
            thread_key=thread_key,
            handoff_kind="room_summary",
            audience_scope="summary",
            from_participant_id=from_id,
            from_participant_name=from_name,
            detected=True,
            rationale="the room context shifted toward a summary or recap request",
        )
    if any(hint in prompt for hint in _YIELD_HINTS):
        return SocialHandoffSignalV1(
            platform=platform,
            room_id=room_id,
            thread_key=thread_key,
            handoff_kind="yield_to_peer",
            audience_scope="room",
            from_participant_id=from_id,
            from_participant_name=from_name,
            detected=True,
            rationale="the turn suggests yielding or inviting another peer into the thread",
        )
    if any(hint in prompt for hint in ("wrap this", "leave that there", "close this loop")):
        return SocialHandoffSignalV1(
            platform=platform,
            room_id=room_id,
            thread_key=thread_key,
            handoff_kind="thread_wrap",
            audience_scope="thread",
            from_participant_id=from_id,
            from_participant_name=from_name,
            detected=True,
            rationale="the thread appears to be wrapping or handing off cleanly",
        )
    return None


def build_thread_routing_hint(
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    thread_key: str,
    thread_summary: str,
    handoff_signal: SocialHandoffSignalV1 | None,
) -> SocialThreadRoutingDecisionV1:
    prompt = _message_text(turn).lower()
    participant = _external_participant(turn)
    target_participant_id, target_participant_name = _thread_target(turn)
    audience_scope = _audience_scope(turn)
    if audience_scope == "summary":
        routing = "summarize_room"
        rationale = "the prompt asks for a room-level recap or transition"
    elif any(hint in prompt for hint in _REVIVE_HINTS):
        routing = "revive_thread"
        rationale = "the prompt is reviving an earlier thread"
    elif audience_scope == "room":
        routing = "reply_to_room"
        rationale = "the message reads as room-level rather than peer-specific"
    elif audience_scope in {"peer", "thread"}:
        routing = "reply_to_peer"
        rationale = "the message is locally addressed and best answered in-thread"
    else:
        routing = "wait"
        rationale = "the audience remains ambiguous, so staying conservative is safer"
    return SocialThreadRoutingDecisionV1(
        platform=platform,
        room_id=room_id,
        thread_key=thread_key,
        audience_scope=audience_scope,  # type: ignore[arg-type]
        routing_decision=routing,  # type: ignore[arg-type]
        target_participant_id=target_participant_id,
        target_participant_name=target_participant_name,
        last_speaker=str(participant.get("participant_name") or participant.get("participant_id") or turn.user_id or "peer"),
        last_addressed_participant_id=target_participant_id,
        open_question="?" in prompt,
        handoff_flag=handoff_signal.detected if handoff_signal else False,
        thread_summary=thread_summary[:180],
        rationale=rationale,
        reasons=[rationale],
    )


def update_active_threads(
    existing_threads: list[SocialThreadStateV1] | None,
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    participant_label: str,
    topics: list[str],
    thread_ttl_hours: int,
    artifact_dialogue_active: bool,
) -> tuple[list[SocialThreadStateV1], SocialThreadStateV1 | None, SocialHandoffSignalV1 | None, SocialThreadRoutingDecisionV1 | None]:
    current_threads = list(existing_threads or [])
    now = datetime.now(timezone.utc)
    current_threads = [
        item for item in current_threads
        if item.expires_at and datetime.fromisoformat(item.expires_at) > now
    ]
    if artifact_dialogue_active:
        primary = current_threads[0] if current_threads else None
        return current_threads[:3], primary, None, None

    thread_key = _thread_key(turn, platform=platform, room_id=room_id, topics=topics)
    audience_scope = _audience_scope(turn)
    addressed_id, addressed_name = _thread_target(turn)
    summary_bits = [participant_label]
    if addressed_name or addressed_id:
        summary_bits.append(f"→ {addressed_name or addressed_id}")
    if topics:
        summary_bits.append(", ".join(topics[:2]))
    elif "?" in _message_text(turn):
        summary_bits.append(_message_text(turn)[:80])
    thread_summary = " · ".join(bit for bit in summary_bits if bit)[:180]
    handoff_signal = detect_handoff_signal(turn, platform=platform, room_id=room_id, thread_key=thread_key)
    expires_at = (now + timedelta(hours=max(thread_ttl_hours, 1))).isoformat()
    updated = SocialThreadStateV1(
        thread_key=thread_key,
        platform=platform,
        room_id=room_id,
        thread_id=str(_external_room(turn).get("thread_id") or "").strip() or None,
        active_participants=merge_unique(
            next((item.active_participants for item in current_threads if item.thread_key == thread_key), []),
            [participant_label, addressed_name or addressed_id or ""],
            limit=4,
        ),
        audience_scope=audience_scope,  # type: ignore[arg-type]
        target_participant_id=addressed_id,
        target_participant_name=addressed_name,
        last_speaker=participant_label,
        last_addressed_participant_id=addressed_id,
        last_addressed_participant_name=addressed_name,
        open_question="?" in _message_text(turn),
        handoff_flag=handoff_signal.detected if handoff_signal else False,
        orion_involved=bool(turn.response),
        thread_summary=thread_summary,
        last_activity_at=turn.created_at,
        expires_at=expires_at,
        metadata={"source": "social-memory"},
    )
    remaining = [item for item in current_threads if item.thread_key != thread_key]
    threads = [updated] + remaining
    threads = sorted(threads, key=lambda item: item.last_activity_at, reverse=True)[:3]
    routing = build_thread_routing_hint(
        turn,
        platform=platform,
        room_id=room_id,
        thread_key=thread_key,
        thread_summary=thread_summary,
        handoff_signal=handoff_signal,
    )
    return threads, updated, handoff_signal, routing


def _parse_iso(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _commitment_due_state(expires_at: datetime, *, now: datetime) -> str:
    remaining = (expires_at - now).total_seconds()
    if remaining <= 0:
        return "stale"
    if remaining <= 20 * 60:
        return "due_soon"
    return "fresh"


def _commitment_thread_key(
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    topics: list[str],
) -> str | None:
    return _thread_key(turn, platform=platform, room_id=room_id, topics=topics)


def _commitment_summary(
    *,
    commitment_type: str,
    turn: SocialRoomTurnStoredV1,
    thread_summary: str,
    scope: str | None = None,
) -> str:
    if commitment_type == "summarize_room":
        return f"Give a brief room summary before switching topics in {thread_summary or 'the current room thread'}."[:180]
    if commitment_type == "return_to_thread":
        return f"Come back to {thread_summary or 'the current thread'} after the local detour."[:180]
    if commitment_type == "answer_pending_question":
        return f"Answer the pending question in {thread_summary or 'the active thread'}."[:180]
    if commitment_type == "yield_then_reenter":
        return f"Yield for now, then re-enter {thread_summary or 'the thread'} when locally invited."[:180]
    if commitment_type == "respect_memory_scope":
        return f"Keep the carry-forward scope narrow ({scope or 'room-local'}) for this continuity cue."[:180]
    return sanitize_text(turn.response)[:180]


def _response_is_weak_commitment(text: str) -> bool:
    lowered = text.lower().replace("’", "'")
    return any(hint in lowered for hint in _COMMITMENT_WEAK_HINTS)


def extract_commitments(
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    topics: list[str],
    thread_summary: str,
    ttl_minutes: int,
    artifact_confirmation: SocialArtifactConfirmationV1 | None = None,
) -> list[SocialCommitmentV1]:
    response = sanitize_text(turn.response)
    lowered = response.lower().replace("’", "'")
    if not response or _SEALED_RE.search(lowered) or _response_is_weak_commitment(lowered):
        return []

    created_at = _parse_iso(turn.created_at) or datetime.now(timezone.utc)
    expires_at = created_at + timedelta(minutes=max(ttl_minutes, 15))
    target_id, target_name = _thread_target(turn)
    thread_key = _commitment_thread_key(turn, platform=platform, room_id=room_id, topics=topics)
    audience_scope = _audience_scope(turn)
    candidates: list[tuple[str, str, str | None]] = []

    if any(hint in lowered for hint in _COMMITMENT_SUMMARY_HINTS):
        candidates.append(("summarize_room", "summary promised in Orion's response", None))
    if any(hint in lowered for hint in _COMMITMENT_RETURN_HINTS):
        candidates.append(("return_to_thread", "thread return promised in Orion's response", None))
    if any(hint in lowered for hint in _COMMITMENT_ANSWER_HINTS):
        candidates.append(("answer_pending_question", "follow-up answer promised in Orion's response", None))
    if any(hint in lowered for hint in _COMMITMENT_YIELD_HINTS):
        candidates.append(("yield_then_reenter", "yield / re-entry promised in Orion's response", None))
    if (
        artifact_confirmation is not None
        and artifact_confirmation.decision_state == "accepted"
        and artifact_confirmation.confirmed_scope in {"peer_local", "room_local"}
    ):
        candidates.append(
            (
                "respect_memory_scope",
                "explicit scoped carry-forward promise",
                artifact_confirmation.confirmed_scope,
            )
        )

    commitments: list[SocialCommitmentV1] = []
    for commitment_type, rationale, scope in candidates[:2]:
        summary = _commitment_summary(
            commitment_type=commitment_type,
            turn=turn,
            thread_summary=thread_summary,
            scope=scope,
        )
        commitments.append(
            SocialCommitmentV1(
                platform=platform,
                room_id=room_id,
                thread_key=thread_key,
                commitment_type=commitment_type,  # type: ignore[arg-type]
                audience_scope=("summary" if commitment_type == "summarize_room" else audience_scope),  # type: ignore[arg-type]
                target_participant_id=target_id,
                target_participant_name=target_name,
                summary=summary,
                source_turn_id=turn.turn_id,
                source_correlation_id=turn.correlation_id,
                created_at=created_at.isoformat(),
                expires_at=expires_at.isoformat(),
                due_state=_commitment_due_state(expires_at, now=created_at),  # type: ignore[arg-type]
                metadata={"source": "social-memory", "rationale": rationale, "scope": scope or ""},
            )
        )
    return commitments


def _commitment_relevance(
    commitment: SocialCommitmentV1,
    *,
    turn: SocialRoomTurnStoredV1,
    current_thread_key: str | None,
) -> float:
    prompt = _message_text(turn).lower()
    score = 0.0
    if current_thread_key and commitment.thread_key and current_thread_key == commitment.thread_key:
        score += 0.45
    if commitment.target_participant_id and commitment.target_participant_id == (_thread_target(turn)[0] or ""):
        score += 0.3
    if commitment.commitment_type == "summarize_room" and any(hint in prompt for hint in _ROOM_SUMMARY_HINTS):
        score += 0.5
    if commitment.commitment_type == "return_to_thread" and any(hint in prompt for hint in _REVIVE_HINTS):
        score += 0.45
    if commitment.commitment_type == "answer_pending_question" and "?" in prompt:
        score += 0.35
    if commitment.commitment_type == "yield_then_reenter" and any(hint in prompt for hint in ("back", "return", "re-enter", "oríon")):
        score += 0.25
    if commitment.commitment_type == "respect_memory_scope":
        score += 0.18
    if commitment.due_state == "due_soon":
        score += 0.12
    if commitment.due_state == "stale":
        score -= 0.1
    return score


def _resolve_commitment(
    commitment: SocialCommitmentV1,
    *,
    state: str,
    reason: str,
    resolved_at: datetime,
) -> SocialCommitmentResolutionV1:
    return SocialCommitmentResolutionV1(
        commitment_id=commitment.commitment_id,
        platform=commitment.platform,
        room_id=commitment.room_id,
        thread_key=commitment.thread_key,
        commitment_type=commitment.commitment_type,
        state=state,  # type: ignore[arg-type]
        summary=commitment.summary,
        resolution_reason=reason,
        resolved_at=resolved_at.isoformat(),
        metadata=dict(commitment.metadata or {}),
    )


def _fulfills_commitment(
    commitment: SocialCommitmentV1,
    *,
    turn: SocialRoomTurnStoredV1,
    current_thread_key: str | None,
) -> bool:
    response = sanitize_text(turn.response).lower().replace("’", "'")
    if not response:
        return False
    if commitment.commitment_type == "summarize_room":
        return any(hint in response for hint in _SUMMARY_DELIVERY_HINTS)
    if commitment.commitment_type in {"return_to_thread", "answer_pending_question"}:
        if current_thread_key and commitment.thread_key and current_thread_key == commitment.thread_key:
            return not any(hint in response for hint in _COMMITMENT_RETURN_HINTS + _COMMITMENT_ANSWER_HINTS)
        return False
    if commitment.commitment_type == "yield_then_reenter":
        return any(hint in response for hint in _REENTRY_HINTS)
    if commitment.commitment_type == "respect_memory_scope":
        return False
    return False


def update_commitments(
    existing_commitments: list[SocialCommitmentV1] | None,
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    topics: list[str],
    thread_summary: str,
    artifact_dialogue_active: bool,
    artifact_confirmation: SocialArtifactConfirmationV1 | None,
    ttl_minutes: int,
    max_open: int,
) -> tuple[list[SocialCommitmentV1], list[SocialCommitmentV1], list[SocialCommitmentResolutionV1]]:
    now = _parse_iso(turn.created_at) or datetime.now(timezone.utc)
    current_thread_key = _commitment_thread_key(turn, platform=platform, room_id=room_id, topics=topics)
    open_commitments: list[SocialCommitmentV1] = []
    resolutions: list[SocialCommitmentResolutionV1] = []

    for existing in list(existing_commitments or []):
        if existing.state != "open":
            continue
        expires_at = _parse_iso(existing.expires_at) or now
        if expires_at <= now:
            resolutions.append(
                _resolve_commitment(
                    existing,
                    state="expired",
                    reason="commitment ttl elapsed before local follow-through",
                    resolved_at=now,
                )
            )
            continue
        if _fulfills_commitment(existing, turn=turn, current_thread_key=current_thread_key):
            resolutions.append(
                _resolve_commitment(
                    existing,
                    state="fulfilled",
                    reason="Orion locally followed through on the conversational commitment",
                    resolved_at=now,
                )
            )
            continue
        refreshed = existing.model_copy(update={"due_state": _commitment_due_state(expires_at, now=now)})
        open_commitments.append(refreshed)

    if artifact_dialogue_active and artifact_confirmation is None:
        open_commitments = [
            item
            for item in open_commitments
            if item.commitment_type != "respect_memory_scope"
        ]

    created_commitments = extract_commitments(
        turn,
        platform=platform,
        room_id=room_id,
        topics=topics,
        thread_summary=thread_summary,
        ttl_minutes=ttl_minutes,
        artifact_confirmation=artifact_confirmation,
    )

    for created in created_commitments:
        still_open: list[SocialCommitmentV1] = []
        for existing in open_commitments:
            same_lane = (
                existing.commitment_type == created.commitment_type
                or (
                    existing.thread_key
                    and created.thread_key
                    and existing.thread_key == created.thread_key
                    and existing.commitment_type == "yield_then_reenter"
                )
            )
            if same_lane:
                resolutions.append(
                    _resolve_commitment(
                        existing,
                        state="superseded",
                        reason="a newer local conversational commitment replaced the earlier one",
                        resolved_at=now,
                    )
                )
                continue
            still_open.append(existing)
        open_commitments = still_open
        open_commitments.insert(0, created)

    for existing in list(open_commitments):
        if (
            existing.commitment_type == "yield_then_reenter"
            and _thread_target(turn)[0] not in {None, existing.target_participant_id}
            and _message_text(turn)
        ):
            open_commitments.remove(existing)
            resolutions.append(
                _resolve_commitment(
                    existing,
                    state="dropped",
                    reason="room context moved to a different local exchange before re-entry",
                    resolved_at=now,
                )
            )

    open_commitments = sorted(
        open_commitments,
        key=lambda item: (
            0 if item.due_state == "due_soon" else 1 if item.due_state == "fresh" else 2,
            _commitment_relevance(item, turn=turn, current_thread_key=current_thread_key) * -1,
            item.created_at,
        ),
    )[:max(max_open, 1)]
    return open_commitments, created_commitments, resolutions


def trust_tier(evidence_count: int) -> str:
    if evidence_count >= 6:
        return "steady"
    if evidence_count >= 3:
        return "known"
    return "new"


def describe_tone(turn: SocialRoomTurnStoredV1) -> str:
    stance = str(turn.grounding_state.stance or "warm, direct, grounded")
    parts = [part.strip() for part in stance.split(",") if part.strip()]
    return ", ".join(parts[:3]) or "warm, direct, grounded"


def derive_stance_metrics(turn: SocialRoomTurnStoredV1) -> dict[str, float]:
    prompt = sanitize_text(turn.prompt).lower()
    response = sanitize_text(turn.response).lower()
    total_words = max(len((prompt + " " + response).split()), 1)
    return {
        "curiosity": min(1.0, 0.45 + (0.15 if "?" in prompt else 0.0) + (0.1 if "?" in response else 0.0)),
        "warmth": min(1.0, 0.58 + (0.12 if any(word in response for word in ("glad", "with you", "here", "care")) else 0.0)),
        "directness": min(1.0, 0.45 + (0.18 if total_words < 80 else 0.08)),
        "playfulness": min(1.0, 0.18 + (0.15 if any(word in response for word in ("haha", "play", "light", "spark")) else 0.0)),
        "caution": min(1.0, 0.25 + float(turn.redaction.overall_score) * 0.5),
        "depth_preference": min(1.0, 0.3 + min(total_words / 180.0, 0.45)),
    }


def build_orientation_summary(snapshot: SocialStanceSnapshotV1) -> str:
    dims = {
        "warm": snapshot.warmth,
        "curious": snapshot.curiosity,
        "direct": snapshot.directness,
        "playful": snapshot.playfulness,
        "careful": snapshot.caution,
        "deepening": snapshot.depth_preference,
    }
    top = [label for label, _ in sorted(dims.items(), key=lambda item: item[1], reverse=True)[:3]]
    return f"Recent social stance leans {', '.join(top)}."


def build_open_thread(
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    room_summary: SocialRoomContinuityV1,
    participant_label: str | None,
    thread_id: str | None,
    ttl_hours: int,
) -> SocialOpenThreadV1 | None:
    prompt = sanitize_text(turn.prompt)
    if not prompt:
        return None
    open_question = "?" in prompt
    if not open_question and not thread_id:
        return None
    if thread_id:
        topic_key = f"{platform}:{room_id}:{thread_id}"
    else:
        topic_key = f"{platform}:{room_id}:{'-'.join(extract_topics(turn, limit=3)) or 'room-thread'}"
    summary = prompt[:180]
    return SocialOpenThreadV1(
        topic_key=topic_key,
        platform=platform,
        room_id=room_id,
        summary=summary,
        last_speaker=participant_label or turn.user_id or "peer",
        open_question=open_question,
        orion_involved=bool(turn.response),
        last_activity_at=turn.created_at,
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=max(ttl_hours, 1))).isoformat(),
        evidence_refs=[evidence_ref(turn)],
        evidence_count=max(1, int(room_summary.evidence_count)),
    )


def _blend(old: float, new: float, *, weight_old: float = 0.7, weight_new: float = 0.3) -> float:
    return max(0.0, min(1.0, (old * weight_old) + (new * weight_new)))


def _style_metrics(turn: SocialRoomTurnStoredV1) -> dict[str, float]:
    prompt = sanitize_text(turn.prompt)
    lowered = prompt.lower()
    words = prompt.split()
    length_factor = min(len(words) / 40.0, 1.0)
    return {
        "directness": min(1.0, 0.45 + (0.2 if len(words) < 16 else 0.05) + (0.1 if "please" not in lowered else 0.0)),
        "depth": min(1.0, 0.25 + length_factor * 0.55),
        "question_appetite": min(1.0, 0.2 + prompt.count("?") * 0.25),
        "playfulness": min(1.0, 0.12 + (0.2 if any(token in lowered for token in ("haha", "lol", "play", "spark")) else 0.0)),
        "formality": min(1.0, 0.3 + (0.25 if any(token in lowered for token in ("please", "thank", "appreciate")) else 0.0)),
        "summarization_preference": min(1.0, 0.1 + (0.5 if any(token in lowered for token in ("summary", "summarize", "recap")) else 0.0)),
    }


def _scope_hint_present(text: str, *, scope: str) -> bool:
    hints = _ROOM_SCOPE_HINTS if scope == "room_local" else _PEER_SCOPE_HINTS
    return any(hint in text for hint in hints)


def classify_shared_artifact_decision(
    turn: SocialRoomTurnStoredV1,
    *,
    scope: str,
    topics: list[str] | None = None,
) -> SharedArtifactDecision:
    prompt = sanitize_text(turn.prompt)
    response = sanitize_text(turn.response)
    lowered = f"{prompt} {response}".lower()
    if _SEALED_RE.search(lowered):
        return SharedArtifactDecision(
            status="declined",
            scope=scope,
            reason="private/sealed language blocks shared artifact carry-forward",
        )
    has_defer = any(hint in lowered for hint in _DEFER_HINTS) and _scope_hint_present(lowered, scope=scope)
    has_decline = any(hint in lowered for hint in _DECLINE_HINTS) and _scope_hint_present(lowered, scope=scope)
    if has_defer and scope == "peer_local":
        return SharedArtifactDecision(
            status="deferred",
            scope=scope,
            reason="explicit not-yet / revisit-later language was present for this scope",
        )
    if has_decline:
        return SharedArtifactDecision(
            status="declined",
            scope=scope,
            reason="explicit do-not-keep language was present for this scope",
        )
    if has_defer:
        return SharedArtifactDecision(
            status="deferred",
            scope=scope,
            reason="explicit not-yet / revisit-later language was present for this scope",
        )
    if any(hint in lowered for hint in _ACCEPT_HINTS) and _scope_hint_present(lowered, scope=scope):
        compact_topics = ", ".join((topics or [])[:2]) or "current continuity"
        if scope == "room_local":
            summary = f"Accepted as room-local continuity around {compact_topics}."
        else:
            summary = f"Accepted as peer-local continuity around {compact_topics}."
        return SharedArtifactDecision(
            status="accepted",
            scope=scope,
            summary=summary[:180],
            reason="explicit keep/remember language made the scope legible",
        )
    return SharedArtifactDecision(status="unknown", scope=scope)


def _artifact_scope_matches(scope: str, artifact_scope: str | None) -> bool:
    if not artifact_scope:
        return False
    if scope == "peer_local":
        return artifact_scope in {"peer_local", "session_only", "no_persistence"}
    if scope == "room_local":
        return artifact_scope == "room_local"
    return False


def _pending_artifact_state(
    existing_status: str | None,
    existing_summary: str | None,
    existing_reason: str | None,
    *,
    artifact_proposal: SocialArtifactProposalV1 | None,
    artifact_revision: SocialArtifactRevisionV1 | None,
) -> tuple[str, str, str]:
    if artifact_revision is not None:
        return (
            str(existing_status or "unknown"),
            str(existing_summary or ""),
            artifact_revision.rationale[:180],
        )
    if artifact_proposal is not None:
        return (
            str(existing_status or "unknown"),
            str(existing_summary or ""),
            artifact_proposal.rationale[:180],
        )
    return (
        str(existing_status or "unknown"),
        str(existing_summary or ""),
        str(existing_reason or ""),
    )


def _artifact_confirmation_has_clear_scope(
    artifact_confirmation: SocialArtifactConfirmationV1 | None,
) -> bool:
    return bool(
        artifact_confirmation is not None
        and artifact_confirmation.decision_state == "accepted"
        and artifact_confirmation.confirmed_scope in {"session_only", "peer_local", "room_local"}
    )


def _artifact_confirmation_activates_continuity(
    artifact_confirmation: SocialArtifactConfirmationV1 | None,
) -> bool:
    return bool(
        _artifact_confirmation_has_clear_scope(artifact_confirmation)
        and artifact_confirmation is not None
        and artifact_confirmation.confirmed_scope in {"peer_local", "room_local"}
    )


def artifact_dialogue_records(
    turn: SocialRoomTurnStoredV1,
    *,
    scope: str,
) -> tuple[SocialArtifactProposalV1 | None, SocialArtifactRevisionV1 | None, SocialArtifactConfirmationV1 | None]:
    client_meta = dict(turn.client_meta or {})
    proposal = client_meta.get("social_artifact_proposal")
    revision = client_meta.get("social_artifact_revision")
    confirmation = client_meta.get("social_artifact_confirmation")
    proposal_obj = SocialArtifactProposalV1.model_validate(proposal) if isinstance(proposal, dict) else None
    revision_obj = SocialArtifactRevisionV1.model_validate(revision) if isinstance(revision, dict) else None
    confirmation_obj = SocialArtifactConfirmationV1.model_validate(confirmation) if isinstance(confirmation, dict) else None
    if proposal_obj and not _artifact_scope_matches(scope, proposal_obj.proposed_scope):
        proposal_obj = None
    if revision_obj and not _artifact_scope_matches(scope, revision_obj.revised_scope):
        revision_obj = None
    if confirmation_obj and not _artifact_scope_matches(scope, confirmation_obj.confirmed_scope):
        confirmation_obj = None
    return proposal_obj, revision_obj, confirmation_obj


def update_peer_style_hint(
    existing: SocialPeerStyleHintV1 | None,
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    participant_id: str,
    participant_name: str | None,
    confidence_floor: float,
) -> SocialPeerStyleHintV1:
    metrics = _style_metrics(turn)
    evidence_count = int((existing.evidence_count if existing else 0) + 1)
    directness = _blend(existing.preferred_directness if existing else 0.5, metrics["directness"])
    depth = _blend(existing.preferred_depth if existing else 0.5, metrics["depth"])
    question_appetite = _blend(existing.question_appetite if existing else 0.5, metrics["question_appetite"])
    playfulness = _blend(existing.playfulness_tendency if existing else 0.3, metrics["playfulness"])
    formality = _blend(existing.formality_tendency if existing else 0.5, metrics["formality"])
    summarization = _blend(existing.summarization_preference if existing else 0.3, metrics["summarization_preference"])
    confidence = min(1.0, max(confidence_floor * 0.5, 0.2 + evidence_count * 0.12))
    if evidence_count < 2:
        summary = "Early signal only; keep adaptation light until more repeated peer evidence accumulates."
    else:
        summary = (
            f"{participant_name or participant_id} tends to prefer "
            f"{'direct' if directness >= 0.58 else 'gentle'} replies, "
            f"{'deeper' if depth >= 0.55 else 'compact'} turns, and "
            f"{'welcomes' if question_appetite >= 0.55 else 'only light'} follow-up questions."
        )
    return SocialPeerStyleHintV1(
        peer_style_key=f"{platform}:{room_id}:{participant_id}",
        platform=platform,
        room_id=room_id,
        participant_id=participant_id,
        participant_name=participant_name,
        style_hints_summary=summary[:220],
        preferred_directness=directness,
        preferred_depth=depth,
        question_appetite=question_appetite,
        playfulness_tendency=playfulness,
        formality_tendency=formality,
        summarization_preference=summarization,
        evidence_count=evidence_count,
        confidence=confidence if evidence_count >= 2 else min(confidence, 0.34),
        last_updated_at=utcnow_iso(),
    )


def update_room_ritual_summary(
    existing: SocialRoomRitualSummaryV1 | None,
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    room_summary: SocialRoomContinuityV1,
    confidence_floor: float,
) -> SocialRoomRitualSummaryV1:
    prompt = sanitize_text(turn.prompt).lower()
    response = sanitize_text(turn.response).lower()
    evidence_count = int((existing.evidence_count if existing else 0) + 1)

    def _choose_style(kind: str, default: str) -> str:
        if kind in {"greeting", "reentry"} and any(token in prompt for token in ("hey", "hi", "hello", "back")):
            return "warm"
        if kind == "thread_revival" and "?" in prompt:
            return "direct"
        if kind == "pause" and any(token in prompt for token in ("pause", "later", "back")):
            return "brief"
        return default

    greeting_style = _choose_style("greeting", existing.greeting_style if existing else "warm")
    reentry_style = _choose_style("reentry", existing.reentry_style if existing else "grounded")
    revival_style = _choose_style("thread_revival", existing.thread_revival_style if existing else "direct")
    pause_style = _choose_style("pause", existing.pause_handoff_style if existing else "brief")
    summary_cadence = _blend(
        existing.summary_cadence_preference if existing else 0.3,
        0.75 if any(token in prompt for token in ("summary", "summarize", "recap")) else 0.2,
    )
    culture_summary = (
        f"The room tends toward {greeting_style} greetings, {reentry_style} re-entry, "
        f"{revival_style} thread revival, and {pause_style} pause/handoff cues."
    )
    confidence = min(1.0, max(confidence_floor * 0.5, 0.2 + evidence_count * 0.12))
    return SocialRoomRitualSummaryV1(
        ritual_key=f"{platform}:{room_id}",
        platform=platform,
        room_id=room_id,
        greeting_style=greeting_style,  # type: ignore[arg-type]
        reentry_style=reentry_style,  # type: ignore[arg-type]
        thread_revival_style=revival_style,  # type: ignore[arg-type]
        pause_handoff_style=pause_style,  # type: ignore[arg-type]
        summary_cadence_preference=summary_cadence,
        room_tone_summary=room_summary.room_tone_summary,
        culture_summary=culture_summary[:220],
        evidence_count=evidence_count,
        confidence=confidence if evidence_count >= 2 else min(confidence, 0.34),
        last_updated_at=utcnow_iso(),
    )


def update_participant_continuity(
    existing: SocialParticipantContinuityV1 | None,
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    participant_id: str,
    participant_name: str | None,
    participant_kind: str,
    topic_limit: int,
    evidence_limit: int,
    shared_artifact_decision: SharedArtifactDecision,
    dialogue_active: bool = False,
    artifact_proposal: SocialArtifactProposalV1 | None = None,
    artifact_revision: SocialArtifactRevisionV1 | None = None,
    artifact_confirmation: SocialArtifactConfirmationV1 | None = None,
) -> SocialParticipantContinuityV1:
    topics = extract_topics(turn, limit=topic_limit)
    aliases = merge_unique(existing.aliases if existing else [], [participant_name or ""], limit=4)
    tone = describe_tone(turn)
    continuity = (
        f"Recurring {participant_kind} in {room_id}; recent shared topics include "
        f"{', '.join(topics[:3]) or 'ongoing conversation'}."
    )
    should_carry_forward = shared_artifact_decision.status not in {"declined", "deferred"}
    if dialogue_active:
        should_carry_forward = False
    if artifact_proposal is not None or artifact_revision is not None or artifact_confirmation is not None:
        should_carry_forward = _artifact_confirmation_activates_continuity(artifact_confirmation)
    evidence_count = int((existing.evidence_count if existing else 0) + (1 if should_carry_forward else 0))
    if artifact_confirmation is not None:
        shared_status = artifact_confirmation.decision_state
        shared_summary = (
            artifact_confirmation.confirmed_summary_text[:180]
            if _artifact_confirmation_activates_continuity(artifact_confirmation)
            else (existing.shared_artifact_summary if existing else "")
        )
        shared_reason = artifact_confirmation.rationale
    elif artifact_proposal is not None or artifact_revision is not None:
        shared_status, shared_summary, shared_reason = _pending_artifact_state(
            existing.shared_artifact_status if existing else "unknown",
            existing.shared_artifact_summary if existing else "",
            existing.shared_artifact_reason if existing else "",
            artifact_proposal=artifact_proposal,
            artifact_revision=artifact_revision,
        )
    else:
        shared_status = (
            shared_artifact_decision.status
            if shared_artifact_decision.status != "unknown"
            else (existing.shared_artifact_status if existing else "unknown")
        )
        shared_summary = (
            shared_artifact_decision.summary
            if shared_artifact_decision.status == "accepted"
            else ("" if shared_artifact_decision.status in {"declined", "deferred"} else (existing.shared_artifact_summary if existing else ""))
        )
        shared_reason = (
            shared_artifact_decision.reason
            if shared_artifact_decision.status != "unknown"
            else (existing.shared_artifact_reason if existing else "")
        )
    return SocialParticipantContinuityV1(
        peer_key=f"{platform}:{room_id}:{participant_id}",
        platform=platform,
        room_id=room_id,
        participant_id=participant_id,
        participant_name=participant_name or (existing.participant_name if existing else None),
        aliases=aliases,
        participant_kind=participant_kind,
        recent_shared_topics=merge_unique(existing.recent_shared_topics if existing else [], topics, limit=topic_limit)
        if should_carry_forward
        else list(existing.recent_shared_topics if existing else []),
        interaction_tone_summary=tone,
        safe_continuity_summary=(continuity[:220] if should_carry_forward else (existing.safe_continuity_summary if existing else "")),
        evidence_refs=merge_unique(existing.evidence_refs if existing else [], [evidence_ref(turn)], limit=evidence_limit)
        if should_carry_forward
        else list(existing.evidence_refs if existing else []),
        evidence_count=evidence_count,
        last_seen_at=turn.created_at,
        confidence=min(1.0, 0.25 + evidence_count * 0.12),
        trust_tier=trust_tier(evidence_count),
        shared_artifact_scope=shared_artifact_decision.scope,
        shared_artifact_status=shared_status,  # type: ignore[arg-type]
        shared_artifact_summary=shared_summary[:180],
        shared_artifact_reason=shared_reason[:180],
        shared_artifact_proposal=artifact_proposal,
        shared_artifact_revision=artifact_revision,
        shared_artifact_confirmation=artifact_confirmation,
    )


def update_room_continuity(
    existing: SocialRoomContinuityV1 | None,
    turn: SocialRoomTurnStoredV1,
    *,
    platform: str,
    room_id: str,
    participant_label: str | None,
    thread_id: str | None,
    topic_limit: int,
    participant_limit: int,
    evidence_limit: int,
    shared_artifact_decision: SharedArtifactDecision,
    dialogue_active: bool = False,
    artifact_proposal: SocialArtifactProposalV1 | None = None,
    artifact_revision: SocialArtifactRevisionV1 | None = None,
    artifact_confirmation: SocialArtifactConfirmationV1 | None = None,
    thread_ttl_hours: int = 6,
    active_commitments: list[SocialCommitmentV1] | None = None,
) -> SocialRoomContinuityV1:
    topics = extract_topics(turn, limit=topic_limit)
    open_threads = []
    if thread_id:
        open_threads.append(f"thread:{thread_id}")
    if "?" in sanitize_text(turn.prompt):
        open_threads.append(sanitize_text(turn.prompt)[:80])
    tone = describe_tone(turn)
    should_carry_forward = shared_artifact_decision.status not in {"declined", "deferred"}
    if dialogue_active:
        should_carry_forward = False
    if artifact_proposal is not None or artifact_revision is not None or artifact_confirmation is not None:
        should_carry_forward = _artifact_confirmation_activates_continuity(artifact_confirmation)
    evidence_count = int((existing.evidence_count if existing else 0) + (1 if should_carry_forward else 0))
    thread_summary = f"Recent room themes: {', '.join(topics[:4]) or 'light social continuity'}."
    if artifact_confirmation is not None:
        shared_status = artifact_confirmation.decision_state
        shared_summary = (
            artifact_confirmation.confirmed_summary_text[:180]
            if _artifact_confirmation_activates_continuity(artifact_confirmation)
            else (existing.shared_artifact_summary if existing else "")
        )
        shared_reason = artifact_confirmation.rationale
    elif artifact_proposal is not None or artifact_revision is not None:
        shared_status, shared_summary, shared_reason = _pending_artifact_state(
            existing.shared_artifact_status if existing else "unknown",
            existing.shared_artifact_summary if existing else "",
            existing.shared_artifact_reason if existing else "",
            artifact_proposal=artifact_proposal,
            artifact_revision=artifact_revision,
        )
    else:
        shared_status = (
            shared_artifact_decision.status
            if shared_artifact_decision.status != "unknown"
            else (existing.shared_artifact_status if existing else "unknown")
        )
        shared_summary = (
            shared_artifact_decision.summary
            if shared_artifact_decision.status == "accepted"
            else ("" if shared_artifact_decision.status in {"declined", "deferred"} else (existing.shared_artifact_summary if existing else ""))
        )
        shared_reason = (
            shared_artifact_decision.reason
            if shared_artifact_decision.status != "unknown"
            else (existing.shared_artifact_reason if existing else "")
        )
    active_threads, primary_thread, handoff_signal, _ = update_active_threads(
        existing.active_threads if existing else [],
        turn,
        platform=platform,
        room_id=room_id,
        participant_label=participant_label or turn.user_id or "peer",
        topics=topics,
        thread_ttl_hours=thread_ttl_hours,
        artifact_dialogue_active=dialogue_active,
    )
    return SocialRoomContinuityV1(
        room_key=f"{platform}:{room_id}",
        platform=platform,
        room_id=room_id,
        recurring_topics=merge_unique(existing.recurring_topics if existing else [], topics, limit=topic_limit)
        if should_carry_forward
        else list(existing.recurring_topics if existing else []),
        active_participants=merge_unique(existing.active_participants if existing else [], [participant_label or ""], limit=participant_limit)
        if should_carry_forward
        else list(existing.active_participants if existing else []),
        recent_thread_summary=thread_summary[:240] if should_carry_forward else (existing.recent_thread_summary if existing else ""),
        room_tone_summary=f"Room tone currently reads {tone}."[:200],
        open_threads=merge_unique(existing.open_threads if existing else [], open_threads, limit=4)
        if should_carry_forward
        else list(existing.open_threads if existing else []),
        evidence_refs=merge_unique(existing.evidence_refs if existing else [], [evidence_ref(turn)], limit=evidence_limit)
        if should_carry_forward
        else list(existing.evidence_refs if existing else []),
        evidence_count=evidence_count,
        last_updated_at=utcnow_iso(),
        shared_artifact_scope=shared_artifact_decision.scope,
        shared_artifact_status=shared_status,  # type: ignore[arg-type]
        shared_artifact_summary=shared_summary[:180],
        shared_artifact_reason=shared_reason[:180],
        shared_artifact_proposal=artifact_proposal,
        shared_artifact_revision=artifact_revision,
        shared_artifact_confirmation=artifact_confirmation,
        active_threads=active_threads,
        current_thread_key=primary_thread.thread_key if primary_thread else (existing.current_thread_key if existing else None),
        current_thread_summary=primary_thread.thread_summary if primary_thread else (existing.current_thread_summary if existing else ""),
        handoff_signal=handoff_signal if handoff_signal is not None else (existing.handoff_signal if existing else None),
        active_commitments=list(active_commitments or []),
    )


def update_social_stance(
    existing: SocialStanceSnapshotV1 | None,
    turn: SocialRoomTurnStoredV1,
    *,
    evidence_limit: int,
) -> SocialStanceSnapshotV1:
    metrics = derive_stance_metrics(turn)
    if existing is None:
        snapshot = SocialStanceSnapshotV1(
            curiosity=metrics["curiosity"],
            warmth=metrics["warmth"],
            directness=metrics["directness"],
            playfulness=metrics["playfulness"],
            caution=metrics["caution"],
            depth_preference=metrics["depth_preference"],
            evidence_refs=[evidence_ref(turn)],
            evidence_count=1,
            last_updated_at=utcnow_iso(),
        )
        snapshot.recent_social_orientation_summary = build_orientation_summary(snapshot)
        return snapshot

    def _blend(old: float, new: float) -> float:
        return max(0.0, min(1.0, (old * 0.7) + (new * 0.3)))

    snapshot = SocialStanceSnapshotV1(
        stance_id=existing.stance_id,
        curiosity=_blend(existing.curiosity, metrics["curiosity"]),
        warmth=_blend(existing.warmth, metrics["warmth"]),
        directness=_blend(existing.directness, metrics["directness"]),
        playfulness=_blend(existing.playfulness, metrics["playfulness"]),
        caution=_blend(existing.caution, metrics["caution"]),
        depth_preference=_blend(existing.depth_preference, metrics["depth_preference"]),
        evidence_refs=merge_unique(existing.evidence_refs, [evidence_ref(turn)], limit=evidence_limit),
        evidence_count=existing.evidence_count + 1,
        last_updated_at=utcnow_iso(),
    )
    snapshot.recent_social_orientation_summary = build_orientation_summary(snapshot)
    return snapshot
