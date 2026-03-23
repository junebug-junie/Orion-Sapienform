from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from orion.inspection.social import build_social_inspection_snapshot
from orion.schemas.social_chat import (
    SocialConceptEvidenceV1,
    SocialGroundingStateV1,
    SocialRedactionScoreV1,
    SocialRoomTurnV1,
)
from orion.schemas.social_artifact import (
    SocialArtifactConfirmationV1,
    SocialArtifactProposalV1,
    SocialArtifactRevisionV1,
)
from orion.schemas.social_memory import (
    SocialParticipantContinuityV1,
    SocialRoomContinuityV1,
    SocialStanceSnapshotV1,
)
from orion.schemas.social_style import (
    SocialPeerStyleHintV1,
    SocialRoomRitualSummaryV1,
    SocialStyleAdaptationSnapshotV1,
)
from orion.schemas.social_context import (
    SocialContextCandidateV1,
    SocialContextSelectionDecisionV1,
    SocialContextWindowV1,
    SocialEpisodeSnapshotV1,
    SocialReentryAnchorV1,
)
from orion.schemas.social_gif import (
    SocialGifIntentV1,
    SocialGifInterpretationV1,
    SocialGifObservedSignalV1,
    SocialGifPolicyDecisionV1,
    SocialGifProxyContextV1,
)
from orion.schemas.social_inspection import SocialInspectionSnapshotV1
from orion.schemas.social_thread import SocialHandoffSignalV1, SocialThreadRoutingDecisionV1
from orion.schemas.social_skills import (
    SocialSkillName,
    SocialSkillRequestV1,
    SocialSkillResultV1,
    SocialSkillSelectionV1,
)

SOCIAL_ROOM_PROFILE = "social_room"
SOCIAL_ROOM_RECALL_PROFILE = "social.room.v1"
SOCIAL_ROOM_VERB = "chat_social_room"
SOCIAL_ROOM_ALLOWED_SKILLS: List[SocialSkillName] = [
    "social_summarize_thread",
    "social_safe_recall",
    "social_self_ground",
    "social_followup_question",
    "social_room_reflection",
    "social_exit_or_pause",
    "social_artifact_dialogue",
]

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s().]{7,}\d)")
_LONG_DIGIT_RE = re.compile(r"\b\d{8,}\b")
_BLOCKED_MEMORY_RE = re.compile(r"\b(sealed|private|password|secret|ssn|mirror|journal)\b", re.IGNORECASE)
_MEMORY_DIALOGUE_RE = re.compile(
    r"(keep (?:this|that|a short)|carry forward|shared cue|takeaway|room-local|peer-local|session-only|reword that|shorter and safer)",
    re.IGNORECASE,
)

logger = logging.getLogger("orion-hub.social_room")


def is_social_room_payload(payload: Dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    raw = payload.get("chat_profile") or payload.get("profile") or payload.get("room")
    return str(raw or "").strip().lower() == SOCIAL_ROOM_PROFILE


def build_social_grounding_state(*, payload: Dict[str, Any], trace_verb: str | None = None) -> SocialGroundingStateV1:
    anchor = payload.get("continuity_anchor") or payload.get("identity_anchor") or "Juniper ↔ Oríon ongoing peer dialogue"
    if trace_verb:
        anchor = f"{anchor} ({trace_verb})"
    return SocialGroundingStateV1(
        continuity_anchor=str(anchor)[:180],
    )


def _redaction_score(text: str | None) -> tuple[float, list[str]]:
    raw = str(text or "")
    score = 0.0
    reasons: list[str] = []
    if _EMAIL_RE.search(raw):
        score += 0.45
        reasons.append("contains_email")
    if _PHONE_RE.search(raw):
        score += 0.35
        reasons.append("contains_phone")
    if _LONG_DIGIT_RE.search(raw):
        score += 0.30
        reasons.append("contains_long_numeric_token")
    lowered = raw.lower()
    for needle, weight in (
        ("address", 0.20),
        ("password", 0.50),
        ("ssn", 0.60),
        ("secret", 0.20),
        ("private", 0.15),
    ):
        if needle in lowered:
            score += weight
            reasons.append(f"mentions_{needle}")
    return min(score, 1.0), reasons


def build_social_redaction(*, prompt: str, response: str, memory_digest: str | None = None) -> SocialRedactionScoreV1:
    prompt_score, prompt_reasons = _redaction_score(prompt)
    response_score, response_reasons = _redaction_score(response)
    memory_score, memory_reasons = _redaction_score(memory_digest)
    overall = max(prompt_score, response_score, memory_score)
    if overall >= 0.7:
        level = "high"
    elif overall >= 0.35:
        level = "medium"
    else:
        level = "low"
    return SocialRedactionScoreV1(
        prompt_score=prompt_score,
        response_score=response_score,
        memory_score=memory_score,
        overall_score=overall,
        recall_safe=overall < 0.7,
        redaction_level=level,
        reasons=list(dict.fromkeys(prompt_reasons + response_reasons + memory_reasons)),
    )


def build_social_concept_evidence(items: Iterable[Dict[str, Any]] | None) -> List[SocialConceptEvidenceV1]:
    out: List[SocialConceptEvidenceV1] = []
    for item in list(items or [])[:4]:
        if not isinstance(item, dict):
            continue
        summary = str(item.get("summary") or item.get("text") or "").strip()
        ref_id = str(item.get("ref_id") or item.get("id") or item.get("artifact_id") or "").strip()
        source_kind = str(item.get("source_kind") or item.get("kind") or "unknown").strip() or "unknown"
        if not summary or not ref_id:
            continue
        try:
            confidence = float(item.get("confidence") or 0.0)
        except Exception:
            confidence = 0.0
        out.append(
            SocialConceptEvidenceV1(
                ref_id=ref_id,
                source_kind=source_kind,
                summary=summary[:220],
                confidence=max(0.0, min(confidence, 1.0)),
            )
        )
    return out


def resolve_social_skill_allowlist(raw: str | None = None) -> List[SocialSkillName]:
    if raw is None:
        return list(SOCIAL_ROOM_ALLOWED_SKILLS)
    allowed = set(SOCIAL_ROOM_ALLOWED_SKILLS)
    parsed = [str(item).strip() for item in str(raw).split(",") if str(item).strip()]
    return [item for item in parsed if item in allowed] or list(SOCIAL_ROOM_ALLOWED_SKILLS)


def _latest_user_prompt(payload: Dict[str, Any], prompt: str | None = None) -> str:
    if prompt:
        return str(prompt)
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        for item in reversed(messages):
            if isinstance(item, dict) and str(item.get("role") or "").strip().lower() == "user":
                content = str(item.get("content") or "").strip()
                if content:
                    return content
    return ""


def _recent_messages(payload: Dict[str, Any]) -> List[str]:
    messages = payload.get("messages")
    out: List[str] = []
    if not isinstance(messages, list):
        return out
    for item in messages[-6:]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if content:
            out.append(content[:240])
    return out


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _blocked_memory_text(text: str | None) -> bool:
    return bool(_BLOCKED_MEMORY_RE.search(str(text or "")))


def _dialogue_scope_hint(prompt: str) -> str:
    lowered = prompt.lower()
    if any(needle in lowered for needle in ("room-local", "room local", "in this room", "for the room", "here in the room")):
        return "room_local"
    if any(needle in lowered for needle in ("peer-local", "peer local", "between us", "with me", "about me")):
        return "peer_local"
    if "session-only" in lowered or "session only" in lowered:
        return "session_only"
    return "session_only"


def _artifact_type_hint(prompt: str) -> str:
    lowered = prompt.lower()
    if "room" in lowered or "norm" in lowered:
        return "room_norm"
    if any(needle in lowered for needle in ("with me", "about me", "between us")):
        return "peer_cue"
    return "shared_takeaway"


def _condense_artifact_summary(prompt: str, *, fallback: str = "grounded continuity cue") -> str:
    text = re.sub(r"\b(can|could|would|should|please|you|we|keep|carry|remember|note|hold|store|this|that|as|a|the)\b", " ", str(prompt or ""), flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    words = [word for word in text.split() if len(word) >= 3]
    summary = " ".join(words[:8]).strip()
    return (summary or fallback)[:100]


def _shorter_summary(text: str) -> str:
    words = [word for word in str(text or "").split() if word]
    if len(words) <= 6:
        return " ".join(words)
    return " ".join(words[:6])


def _narrower_scope(scope: str) -> str:
    if scope in {"room_local", "peer_local"}:
        return "session_only"
    if scope == "session_only":
        return "no_persistence"
    return scope


def _proposal_source(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for key in ("social_artifact_proposal",):
        raw = payload.get(key)
        if isinstance(raw, dict):
            return raw
    for surface in ("social_peer_continuity", "social_room_continuity"):
        raw = payload.get(surface) or {}
        if isinstance(raw, dict) and isinstance(raw.get("shared_artifact_proposal"), dict):
            return raw["shared_artifact_proposal"]
    return None


def build_social_artifact_dialogue(
    *,
    payload: Dict[str, Any],
    prompt: str,
) -> Tuple[Optional[SocialArtifactProposalV1], Optional[SocialArtifactRevisionV1], Optional[SocialArtifactConfirmationV1], Optional[SocialSkillResultV1], str]:
    lowered = prompt.lower()
    if not _MEMORY_DIALOGUE_RE.search(prompt):
        pending = _proposal_source(payload)
        if pending is None:
            return None, None, None, None, "no shared-artifact dialogue cue matched"
    if _blocked_memory_text(prompt):
        confirmation = SocialArtifactConfirmationV1(
            artifact_type=_artifact_type_hint(prompt),  # type: ignore[arg-type]
            decision_state="declined",
            confirmed_scope="no_persistence",
            rationale="private/sealed wording is not eligible for shared carry-forward",
            metadata={"boundary": "blocked_private"},
        )
        result = SocialSkillResultV1(
            skill_name="social_artifact_dialogue",
            summary="I wouldn’t turn that into a shared artifact. We can just leave it here.",
            snippets=["No shared carry-forward for private or sealed material."],
            safety_notes=["blocked private/sealed material was not proposed"],
            metadata={"confirmation": confirmation.model_dump(mode="json")},
        )
        return None, None, confirmation, result, "blocked private/sealed material should not enter artifact dialogue"

    pending_raw = _proposal_source(payload)
    pending = SocialArtifactProposalV1.model_validate(pending_raw) if pending_raw else None

    if any(needle in lowered for needle in ("don't keep", "do not keep", "not now", "maybe later", "let's defer", "lets defer", "hold off")):
        decision_state = "deferred" if any(needle in lowered for needle in ("not now", "maybe later", "hold off")) else "declined"
        confirmation = SocialArtifactConfirmationV1(
            proposal_id=pending.proposal_id if pending else None,
            artifact_type=(pending.artifact_type if pending else _artifact_type_hint(prompt)),  # type: ignore[arg-type]
            decision_state=decision_state,  # type: ignore[arg-type]
            confirmed_scope="no_persistence",
            rationale="the peer declined or deferred shared carry-forward",
        )
        wording = "Okay — I won’t carry that forward." if decision_state == "declined" else "Okay — we can leave that session-only for now."
        result = SocialSkillResultV1(
            skill_name="social_artifact_dialogue",
            summary=wording,
            snippets=[wording],
            metadata={"confirmation": confirmation.model_dump(mode="json")},
        )
        return None, None, confirmation, result, "decline/defer cue for shared artifact dialogue"

    if pending and any(needle in lowered for needle in ("shorter", "safer", "tighter", "narrower", "reword", "lighter")):
        revised_summary = _shorter_summary(_condense_artifact_summary(prompt, fallback=pending.proposed_summary_text) or pending.proposed_summary_text)
        revised_scope = _narrower_scope(pending.proposed_scope)
        revision = SocialArtifactRevisionV1(
            proposal_id=pending.proposal_id,
            artifact_type=pending.artifact_type,
            prior_summary_text=pending.proposed_summary_text,
            prior_scope=pending.proposed_scope,
            revised_summary_text=revised_summary,
            revised_scope=revised_scope,  # type: ignore[arg-type]
            confirmation_needed=True,
            rationale="revised to be shorter and narrower before any carry-forward",
        )
        result = SocialSkillResultV1(
            skill_name="social_artifact_dialogue",
            summary=f"Shorter version: “{revised_summary}.” I’d treat that as {revised_scope.replace('_', '-')} unless you want more.",
            snippets=[
                f"Short version: {revised_summary}",
                f"Scope: {revised_scope.replace('_', '-')}",
            ],
            metadata={
                "proposal": pending.model_dump(mode="json"),
                "revision": revision.model_dump(mode="json"),
            },
        )
        return pending, revision, None, result, "artifact wording/scope revision requested"

    if pending and any(needle in lowered for needle in ("yes", "that works", "that's right", "that is right", "okay keep", "sounds right", "works for me")):
        accepted_scope = _dialogue_scope_hint(prompt) if _dialogue_scope_hint(prompt) != "session_only" else pending.proposed_scope
        confirmation = SocialArtifactConfirmationV1(
            proposal_id=pending.proposal_id,
            artifact_type=pending.artifact_type,
            decision_state="accepted",
            confirmed_summary_text=pending.proposed_summary_text,
            confirmed_scope=accepted_scope,  # type: ignore[arg-type]
            confirmation_needed=False,
            rationale="the peer accepted the proposed wording/scope",
        )
        result = SocialSkillResultV1(
            skill_name="social_artifact_dialogue",
            summary=f"Okay — I’ll carry forward “{pending.proposed_summary_text}” as {accepted_scope.replace('_', '-')}.",
            snippets=[
                f"Carry-forward: {pending.proposed_summary_text}",
                f"Scope: {accepted_scope.replace('_', '-')}",
            ],
            metadata={
                "proposal": pending.model_dump(mode="json"),
                "confirmation": confirmation.model_dump(mode="json"),
            },
        )
        return pending, None, confirmation, result, "explicit acceptance of a pending shared artifact"

    scope = _dialogue_scope_hint(prompt)
    artifact_type = _artifact_type_hint(prompt)
    summary = _condense_artifact_summary(prompt)
    clarify_scope = scope == "session_only" and not any(
        needle in lowered
        for needle in ("session-only", "session only", "room-local", "room local", "peer-local", "peer local", "between us", "with me", "in this room")
    )
    proposal = SocialArtifactProposalV1(
        artifact_type=artifact_type,  # type: ignore[arg-type]
        proposed_summary_text=summary,
        proposed_scope=scope,  # type: ignore[arg-type]
        decision_state="clarify_scope" if clarify_scope else "proposed",
        confirmation_needed=True,
        rationale="defaulted to the narrowest safe scope until the carry-forward target is clear",
    )
    if clarify_scope:
        wording = f"I’d treat that as session-only for now: “{summary}.” Does that wording match what you meant?"
    else:
        wording = f"I can keep that {scope.replace('_', '-')} if you want. Short version: “{summary}.”"
    result = SocialSkillResultV1(
        skill_name="social_artifact_dialogue",
        summary=wording,
        snippets=[
            f"Proposed carry-forward: {summary}",
            f"Scope: {scope.replace('_', '-')}",
        ],
        metadata={"proposal": proposal.model_dump(mode="json")},
    )
    return proposal, None, None, result, "shared-artifact proposal or clarification cue matched"


def build_style_adaptation_snapshot(
    *,
    payload: Dict[str, Any],
    confidence_floor: float,
    adaptation_enabled: bool,
    rituals_enabled: bool,
) -> SocialStyleAdaptationSnapshotV1:
    grounding = build_social_grounding_state(payload=payload)
    peer_style = SocialPeerStyleHintV1.model_validate(payload.get("social_peer_style_hint") or {}) if payload.get("social_peer_style_hint") else None
    room_ritual = SocialRoomRitualSummaryV1.model_validate(payload.get("social_room_ritual_summary") or {}) if payload.get("social_room_ritual_summary") else None
    if not adaptation_enabled:
        return SocialStyleAdaptationSnapshotV1(
            snapshot_id="social-style-adaptation-disabled",
            platform=(payload.get("external_room") or {}).get("platform"),
            room_id=(payload.get("external_room") or {}).get("room_id"),
            participant_id=(payload.get("external_participant") or {}).get("participant_id"),
            core_identity_anchor=f"{grounding.identity_label} remains a {grounding.relationship_frame}: {grounding.stance}.",
            guardrail="Adaptation disabled; remain Orion without peer-specific styling.",
            confidence=0.0,
        )

    confidence = 0.0
    directness_delta = depth_delta = question_delta = playfulness_delta = summarization_delta = 0.0
    peer_hint = ""
    if peer_style and peer_style.confidence >= confidence_floor:
        confidence = max(confidence, float(peer_style.confidence))
        directness_delta = round((peer_style.preferred_directness - 0.5) * 0.35, 3)
        depth_delta = round((peer_style.preferred_depth - 0.5) * 0.35, 3)
        question_delta = round((peer_style.question_appetite - 0.5) * 0.35, 3)
        playfulness_delta = round((peer_style.playfulness_tendency - peer_style.formality_tendency) * 0.2, 3)
        summarization_delta = round((peer_style.summarization_preference - 0.3) * 0.25, 3)
        peer_hint = peer_style.style_hints_summary

    ritual_hint = ""
    if rituals_enabled and room_ritual and room_ritual.confidence >= confidence_floor:
        confidence = max(confidence, float(room_ritual.confidence))
        ritual_hint = (
            f"Use {room_ritual.greeting_style} greeting cues, {room_ritual.reentry_style} re-entry, "
            f"{room_ritual.thread_revival_style} revival, and {room_ritual.pause_handoff_style} pause/handoff."
        )
        summarization_delta = round(
            max(min(summarization_delta + ((room_ritual.summary_cadence_preference - 0.3) * 0.2), 0.35), -0.35),
            3,
        )

    return SocialStyleAdaptationSnapshotV1(
        snapshot_id=f"social-style:{(payload.get('external_room') or {}).get('room_id') or 'room'}:{(payload.get('external_participant') or {}).get('participant_id') or 'peer'}",
        platform=(payload.get("external_room") or {}).get("platform"),
        room_id=(payload.get("external_room") or {}).get("room_id"),
        participant_id=(payload.get("external_participant") or {}).get("participant_id"),
        core_identity_anchor=f"{grounding.identity_label} stays {grounding.stance} as a {grounding.relationship_frame}.",
        peer_adaptation_hint=peer_hint[:220],
        room_ritual_hint=ritual_hint[:220],
        directness_delta=directness_delta,
        depth_delta=depth_delta,
        question_frequency_delta=question_delta,
        playfulness_delta=playfulness_delta,
        summarization_tendency_delta=summarization_delta,
        guardrail="Adapt lightly to the peer and room while remaining Orion; do not turn this into a persona mask, drift, or manipulation.",
        confidence=confidence,
    )


def _make_skill_request(payload: Dict[str, Any], *, prompt: str, allowlist: List[SocialSkillName]) -> SocialSkillRequestV1:
    external_room = payload.get("external_room") or {}
    return SocialSkillRequestV1(
        room_id=str(external_room.get("room_id") or "").strip() or None,
        thread_id=str(external_room.get("thread_id") or "").strip() or None,
        prompt=prompt,
        recent_messages=_recent_messages(payload),
        allowlist=allowlist,
    )


def _summarize_thread(payload: Dict[str, Any], request: SocialSkillRequestV1) -> SocialSkillResultV1:
    room = payload.get("social_room_continuity") or {}
    snippets = []
    if room.get("recent_thread_summary"):
        snippets.append(str(room.get("recent_thread_summary"))[:180])
    open_threads = [str(item).strip() for item in room.get("open_threads") or [] if str(item).strip()]
    if open_threads:
        snippets.append(f"Open thread: {open_threads[0][:120]}")
    if request.recent_messages:
        snippets.append(f"Latest turn: {request.recent_messages[-1][:120]}")
    summary = _first_nonempty(*snippets) or "The room is continuing its current thread."
    return SocialSkillResultV1(
        skill_name="social_summarize_thread",
        summary=summary,
        snippets=snippets[:3],
        metadata={"thread_id": request.thread_id or "", "room_id": request.room_id or ""},
    )


def _safe_recall(payload: Dict[str, Any], request: SocialSkillRequestV1) -> SocialSkillResultV1:
    peer = payload.get("social_peer_continuity") or {}
    room = payload.get("social_room_continuity") or {}
    candidate_snippets = [
        str(peer.get("safe_continuity_summary") or "").strip(),
        str(room.get("recent_thread_summary") or "").strip(),
    ]
    topics = [str(item).strip() for item in peer.get("recent_shared_topics") or [] if str(item).strip()]
    if topics:
        candidate_snippets.append(f"Shared topics: {', '.join(topics[:3])}")
    safe_snippets = [item[:180] for item in candidate_snippets if item and not _blocked_memory_text(item)]
    safety_notes = []
    if len(safe_snippets) < len([item for item in candidate_snippets if item]):
        safety_notes.append("blocked private/sealed memory was suppressed")
    summary = safe_snippets[0] if safe_snippets else "No extra safe social-memory detail is needed beyond the current room context."
    return SocialSkillResultV1(
        skill_name="social_safe_recall",
        summary=summary,
        snippets=safe_snippets[:3],
        safety_notes=safety_notes,
        metadata={"request_id": request.request_id},
    )


def _self_ground(payload: Dict[str, Any], request: SocialSkillRequestV1) -> SocialSkillResultV1:
    grounding = build_social_grounding_state(payload=payload)
    summary = (
        f"{grounding.identity_label} is here as a {grounding.relationship_frame}, "
        f"staying {grounding.stance} with continuity anchored in {grounding.continuity_anchor}."
    )
    return SocialSkillResultV1(
        skill_name="social_self_ground",
        summary=summary[:220],
        snippets=[
            f"Identity: {grounding.identity_label}",
            f"Relationship: {grounding.relationship_frame}",
            f"Anchor: {grounding.continuity_anchor}",
        ],
        metadata={"request_id": request.request_id},
    )


def _followup_question(payload: Dict[str, Any], request: SocialSkillRequestV1) -> SocialSkillResultV1:
    room = payload.get("social_room_continuity") or {}
    open_threads = [str(item).strip() for item in room.get("open_threads") or [] if str(item).strip()]
    topics = [str(item).strip() for item in room.get("recurring_topics") or [] if str(item).strip()]
    if open_threads:
        question = f"Do you want to keep pulling on {open_threads[0][:80]} a little further?"
    elif topics:
        question = f"What feels most alive for you in the {topics[0][:60]} thread right now?"
    else:
        question = "What feels like the next useful thread to open here?"
    return SocialSkillResultV1(
        skill_name="social_followup_question",
        summary=question,
        snippets=[question],
        metadata={"room_id": request.room_id or ""},
    )


def _room_reflection(payload: Dict[str, Any], request: SocialSkillRequestV1) -> SocialSkillResultV1:
    room = payload.get("social_room_continuity") or {}
    stance = payload.get("social_stance_snapshot") or {}
    tone = str(room.get("room_tone_summary") or "The room feels steady.").strip()
    topics = [str(item).strip() for item in room.get("recurring_topics") or [] if str(item).strip()]
    orientation = str(stance.get("recent_social_orientation_summary") or "").strip()
    summary = f"{tone} {orientation}".strip()
    snippets = [summary]
    if topics:
        snippets.append(f"Topic dynamic: {', '.join(topics[:3])}")
    return SocialSkillResultV1(
        skill_name="social_room_reflection",
        summary=summary[:220] or "The room seems to be converging around one shared thread.",
        snippets=snippets[:2],
        metadata={"request_id": request.request_id},
    )


def _exit_or_pause(payload: Dict[str, Any], request: SocialSkillRequestV1) -> SocialSkillResultV1:
    stance = payload.get("social_stance_snapshot") or {}
    warmth = float(stance.get("warmth") or 0.7)
    if warmth >= 0.7:
        summary = "A graceful option is to pause warmly and leave the thread open without sounding abrupt."
    else:
        summary = "A graceful option is to step back briefly and keep the reply short and calm."
    snippets = [
        "I can leave a little space here if that's better.",
        "Happy to pause for a beat and pick this back up later.",
    ]
    return SocialSkillResultV1(
        skill_name="social_exit_or_pause",
        summary=summary,
        snippets=snippets,
        metadata={"request_id": request.request_id},
    )


def _skill_from_heuristics(payload: Dict[str, Any], prompt: str, allowlist: List[SocialSkillName]) -> tuple[SocialSkillName | None, str]:
    lowered = prompt.lower()
    policy = payload.get("social_turn_policy") or {}
    open_threads = payload.get("social_room_continuity", {}).get("open_threads") or []

    if "social_artifact_dialogue" in allowlist:
        proposal, revision, confirmation, result, reason = build_social_artifact_dialogue(payload=payload, prompt=prompt)
        if any(item is not None for item in (proposal, revision, confirmation)) and result is not None:
            return "social_artifact_dialogue", reason
    if "social_self_ground" in allowlist and any(needle in lowered for needle in ("who are you", "what are you", "remind me who you are")):
        return "social_self_ground", "explicit self/identity request"
    if "social_summarize_thread" in allowlist and any(needle in lowered for needle in ("summarize", "summary", "recap", "what were we just talking about", "catch me up")):
        return "social_summarize_thread", "explicit request to summarize the room/thread"
    if "social_safe_recall" in allowlist and any(needle in lowered for needle in ("remember", "recall", "what do you know about me", "what do you remember")):
        return "social_safe_recall", "explicit safe-recall request"
    if "social_room_reflection" in allowlist and any(needle in lowered for needle in ("what do you notice about this room", "what's happening in this room", "reflect on this room")):
        return "social_room_reflection", "explicit room-dynamic reflection request"
    if "social_exit_or_pause" in allowlist and any(
        needle in lowered
        for needle in ("let's pause", "let’s pause", "pause here", "let's stop", "step back", "no need to answer", "give us space")
    ):
        return "social_exit_or_pause", "explicit pause / disengagement cue"
    if "social_followup_question" in allowlist and (
        any(needle in lowered for needle in ("what should we ask", "where should we go next", "what's the next question"))
        or (str(policy.get("decision") or "") == "ask_follow_up" and open_threads)
        or (float(policy.get("novelty_score") or 1.0) <= 0.2 and open_threads)
    ):
        return "social_followup_question", "open-thread continuation would benefit from one grounded follow-up question"
    return None, "no narrow social-skill trigger matched"


def select_social_room_skill(
    *,
    payload: Dict[str, Any],
    prompt: str,
    skills_enabled: bool,
    allowlist: List[SocialSkillName],
) -> tuple[SocialSkillSelectionV1, SocialSkillResultV1 | None, SocialSkillRequestV1]:
    request = _make_skill_request(payload, prompt=prompt, allowlist=allowlist)
    if not skills_enabled:
        selection = SocialSkillSelectionV1(
            considered_skills=allowlist,
            selection_reason="social skill surfacing disabled",
            suppressed_reason="skills_disabled",
            request_id=request.request_id,
        )
        logger.debug("social_skill_suppressed reason=%s", selection.suppressed_reason)
        return selection, None, request
    if not allowlist:
        selection = SocialSkillSelectionV1(
            considered_skills=[],
            selection_reason="social skill surfacing has an empty allowlist",
            suppressed_reason="empty_allowlist",
            request_id=request.request_id,
        )
        logger.debug("social_skill_suppressed reason=%s", selection.suppressed_reason)
        return selection, None, request

    skill_name, reason = _skill_from_heuristics(payload, prompt, allowlist)
    if skill_name is None:
        selection = SocialSkillSelectionV1(
            considered_skills=allowlist,
            selection_reason=reason,
            suppressed_reason="no_skill_needed",
            request_id=request.request_id,
        )
        logger.debug("social_skill_none reason=%s", reason)
        return selection, None, request

    builder_map = {
        "social_summarize_thread": _summarize_thread,
        "social_safe_recall": _safe_recall,
        "social_self_ground": _self_ground,
        "social_followup_question": _followup_question,
        "social_room_reflection": _room_reflection,
        "social_exit_or_pause": _exit_or_pause,
        "social_artifact_dialogue": lambda payload, request: build_social_artifact_dialogue(payload=payload, prompt=request.prompt)[3],  # type: ignore[return-value]
    }
    result = builder_map[skill_name](payload, request)
    selection = SocialSkillSelectionV1(
        considered_skills=allowlist,
        selected_skill=skill_name,
        used=True,
        selection_reason=reason,
        request_id=request.request_id,
    )
    logger.info("social_skill_selected skill=%s reason=%s", skill_name, reason)
    return selection, result, request


def social_room_client_meta(
    *,
    payload: Dict[str, Any],
    route_debug: Dict[str, Any],
    trace_verb: str | None,
    memory_digest: str | None,
) -> Dict[str, Any]:
    grounding = build_social_grounding_state(payload=payload, trace_verb=trace_verb)
    concept_evidence = build_social_concept_evidence(payload.get("concept_evidence"))
    peer_continuity = payload.get("social_peer_continuity") or {}
    room_continuity = payload.get("social_room_continuity") or {}
    stance_snapshot = payload.get("social_stance_snapshot") or {}
    peer_style_hint = payload.get("social_peer_style_hint") or {}
    room_ritual_summary = payload.get("social_room_ritual_summary") or {}
    style_adaptation = route_debug.get("social_style_adaptation") or payload.get("social_style_adaptation") or {}
    thread_routing = route_debug.get("social_thread_routing") or payload.get("social_thread_routing") or {}
    handoff_signal = route_debug.get("social_handoff_signal") or payload.get("social_handoff_signal") or {}
    artifact_proposal = route_debug.get("social_artifact_proposal") or payload.get("social_artifact_proposal") or {}
    artifact_revision = route_debug.get("social_artifact_revision") or payload.get("social_artifact_revision") or {}
    artifact_confirmation = route_debug.get("social_artifact_confirmation") or payload.get("social_artifact_confirmation") or {}
    skill_request = route_debug.get("social_skill_request") or payload.get("social_skill_request") or {}
    skill_selection = route_debug.get("social_skill_selection") or payload.get("social_skill_selection") or {}
    skill_result = route_debug.get("social_skill_result") or payload.get("social_skill_result") or {}
    context_window = route_debug.get("social_context_window") or payload.get("social_context_window") or {}
    context_selection_decision = route_debug.get("social_context_selection_decision") or payload.get("social_context_selection_decision") or {}
    context_candidates = route_debug.get("social_context_candidates") or payload.get("social_context_candidates") or []
    episode_snapshot = route_debug.get("social_episode_snapshot") or payload.get("social_episode_snapshot") or {}
    reentry_anchor = route_debug.get("social_reentry_anchor") or payload.get("social_reentry_anchor") or {}
    gif_policy = route_debug.get("social_gif_policy") or payload.get("social_gif_policy") or {}
    gif_intent = route_debug.get("social_gif_intent") or payload.get("social_gif_intent") or {}
    gif_observed_signal = route_debug.get("social_gif_observed_signal") or payload.get("social_gif_observed_signal") or {}
    gif_proxy_context = route_debug.get("social_gif_proxy_context") or payload.get("social_gif_proxy_context") or {}
    gif_interpretation = route_debug.get("social_gif_interpretation") or payload.get("social_gif_interpretation") or {}
    return {
        "chat_profile": SOCIAL_ROOM_PROFILE,
        "social_grounding_state": grounding.model_dump(mode="json"),
        "social_concept_evidence": [item.model_dump(mode="json") for item in concept_evidence],
        "social_recall_profile": route_debug.get("recall_profile") or SOCIAL_ROOM_RECALL_PROFILE,
        "memory_digest_excerpt": str(memory_digest or "")[:280],
        "external_room": dict(payload.get("external_room") or {}) if isinstance(payload.get("external_room"), dict) else {},
        "external_participant": dict(payload.get("external_participant") or {}) if isinstance(payload.get("external_participant"), dict) else {},
        "social_peer_continuity": SocialParticipantContinuityV1.model_validate(peer_continuity).model_dump(mode="json") if peer_continuity else {},
        "social_room_continuity": SocialRoomContinuityV1.model_validate(room_continuity).model_dump(mode="json") if room_continuity else {},
        "social_stance_snapshot": SocialStanceSnapshotV1.model_validate(stance_snapshot).model_dump(mode="json") if stance_snapshot else {},
        "social_peer_style_hint": SocialPeerStyleHintV1.model_validate(peer_style_hint).model_dump(mode="json") if peer_style_hint else {},
        "social_room_ritual_summary": SocialRoomRitualSummaryV1.model_validate(room_ritual_summary).model_dump(mode="json") if room_ritual_summary else {},
        "social_style_adaptation": SocialStyleAdaptationSnapshotV1.model_validate(style_adaptation).model_dump(mode="json") if style_adaptation else {},
        "social_thread_routing": SocialThreadRoutingDecisionV1.model_validate(thread_routing).model_dump(mode="json") if thread_routing else {},
        "social_handoff_signal": SocialHandoffSignalV1.model_validate(handoff_signal).model_dump(mode="json") if handoff_signal else {},
        "social_artifact_proposal": SocialArtifactProposalV1.model_validate(artifact_proposal).model_dump(mode="json") if artifact_proposal else {},
        "social_artifact_revision": SocialArtifactRevisionV1.model_validate(artifact_revision).model_dump(mode="json") if artifact_revision else {},
        "social_artifact_confirmation": SocialArtifactConfirmationV1.model_validate(artifact_confirmation).model_dump(mode="json") if artifact_confirmation else {},
        "social_skill_request": SocialSkillRequestV1.model_validate(skill_request).model_dump(mode="json") if skill_request else {},
        "social_skill_selection": SocialSkillSelectionV1.model_validate(skill_selection).model_dump(mode="json") if skill_selection else {},
        "social_skill_result": SocialSkillResultV1.model_validate(skill_result).model_dump(mode="json") if skill_result else {},
        "social_episode_snapshot": SocialEpisodeSnapshotV1.model_validate(episode_snapshot).model_dump(mode="json") if episode_snapshot else {},
        "social_reentry_anchor": SocialReentryAnchorV1.model_validate(reentry_anchor).model_dump(mode="json") if reentry_anchor else {},
        "social_gif_policy": SocialGifPolicyDecisionV1.model_validate(gif_policy).model_dump(mode="json") if gif_policy else {},
        "social_gif_intent": SocialGifIntentV1.model_validate(gif_intent).model_dump(mode="json") if gif_intent else {},
        "social_gif_observed_signal": SocialGifObservedSignalV1.model_validate(gif_observed_signal).model_dump(mode="json") if gif_observed_signal else {},
        "social_gif_proxy_context": SocialGifProxyContextV1.model_validate(gif_proxy_context).model_dump(mode="json") if gif_proxy_context else {},
        "social_gif_interpretation": SocialGifInterpretationV1.model_validate(gif_interpretation).model_dump(mode="json") if gif_interpretation else {},
        "social_context_window": SocialContextWindowV1.model_validate(context_window).model_dump(mode="json") if context_window else {},
        "social_context_selection_decision": SocialContextSelectionDecisionV1.model_validate(context_selection_decision).model_dump(mode="json") if context_selection_decision else {},
        "social_context_candidates": [SocialContextCandidateV1.model_validate(item).model_dump(mode="json") for item in context_candidates[:8] if isinstance(item, dict)] if isinstance(context_candidates, list) else [],
    }


def build_social_inspection_debug(
    *,
    payload: Dict[str, Any],
    route_debug: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    room_continuity = metadata.get("social_room_continuity") or payload.get("social_room_continuity") or {}
    context_window = metadata.get("social_context_window") or payload.get("social_context_window") or {}
    participant_raw = ((metadata.get("social_peer_continuity") or {}).get("participant_id") or (payload.get("external_participant") or {}).get("participant_id"))
    inspection = build_social_inspection_snapshot(
        platform=str((room_continuity or {}).get("platform") or (payload.get("external_room") or {}).get("platform") or "unknown"),
        room_id=str((room_continuity or {}).get("room_id") or (payload.get("external_room") or {}).get("room_id") or "unknown"),
        participant_id=str(participant_raw) if participant_raw else None,
        thread_key=str((context_window or {}).get("thread_key") or (room_continuity or {}).get("current_thread_key") or "") or None,
        surfaces={
            "social_peer_continuity": metadata.get("social_peer_continuity") or payload.get("social_peer_continuity") or {},
            "social_room_continuity": room_continuity,
            "social_context_window": context_window,
            "social_context_selection_decision": metadata.get("social_context_selection_decision") or payload.get("social_context_selection_decision") or {},
            "social_context_candidates": metadata.get("social_context_candidates") or payload.get("social_context_candidates") or [],
            "social_thread_routing": metadata.get("social_thread_routing") or payload.get("social_thread_routing") or {},
            "social_handoff_signal": metadata.get("social_handoff_signal") or payload.get("social_handoff_signal") or {},
            "social_repair_signal": route_debug.get("social_repair_signal") or metadata.get("social_repair_signal") or payload.get("social_repair_signal") or {},
            "social_repair_decision": route_debug.get("social_repair_decision") or metadata.get("social_repair_decision") or payload.get("social_repair_decision") or {},
            "social_epistemic_signal": route_debug.get("social_epistemic_signal") or metadata.get("social_epistemic_signal") or payload.get("social_epistemic_signal") or {},
            "social_epistemic_decision": route_debug.get("social_epistemic_decision") or metadata.get("social_epistemic_decision") or payload.get("social_epistemic_decision") or {},
            "social_artifact_proposal": metadata.get("social_artifact_proposal") or payload.get("social_artifact_proposal") or {},
            "social_artifact_revision": metadata.get("social_artifact_revision") or payload.get("social_artifact_revision") or {},
            "social_artifact_confirmation": metadata.get("social_artifact_confirmation") or payload.get("social_artifact_confirmation") or {},
            "social_episode_snapshot": metadata.get("social_episode_snapshot") or payload.get("social_episode_snapshot") or {},
            "social_reentry_anchor": metadata.get("social_reentry_anchor") or payload.get("social_reentry_anchor") or {},
            "social_gif_policy": route_debug.get("social_gif_policy") or metadata.get("social_gif_policy") or payload.get("social_gif_policy") or {},
            "social_gif_intent": route_debug.get("social_gif_intent") or metadata.get("social_gif_intent") or payload.get("social_gif_intent") or {},
            "social_gif_observed_signal": route_debug.get("social_gif_observed_signal") or metadata.get("social_gif_observed_signal") or payload.get("social_gif_observed_signal") or {},
            "social_gif_proxy_context": route_debug.get("social_gif_proxy_context") or metadata.get("social_gif_proxy_context") or payload.get("social_gif_proxy_context") or {},
            "social_gif_interpretation": route_debug.get("social_gif_interpretation") or metadata.get("social_gif_interpretation") or payload.get("social_gif_interpretation") or {},
        },
        source_surface="hub-routing-debug",
        source_service="orion-hub",
    )
    logger.info(
        "social_inspection_snapshot_built room_id=%s participant_id=%s sections=%s traces=%s source=%s",
        inspection.room_id,
        inspection.participant_id or "room",
        len(inspection.sections),
        len(inspection.decision_traces),
        "hub-routing-debug",
    )
    for section in inspection.sections:
        logger.info(
            "social_inspection_section_included room_id=%s participant_id=%s kind=%s included=%s traces=%s",
            inspection.room_id,
            inspection.participant_id or "room",
            section.section_kind,
            len(section.included_artifact_summaries),
            len(section.decision_traces),
        )
    if int(inspection.metadata.get("safety_omissions") or 0) > 0:
        logger.info(
            "social_inspection_safety_omission room_id=%s participant_id=%s omitted=%s",
            inspection.room_id,
            inspection.participant_id or "room",
            inspection.metadata.get("safety_omissions"),
        )
    return SocialInspectionSnapshotV1.model_validate(inspection).model_dump(mode="json")


def build_social_room_turn(
    *,
    prompt: str,
    response: str,
    session_id: str | None,
    correlation_id: str | None,
    user_id: str | None,
    source: str,
    recall_profile: str | None,
    trace_verb: str | None,
    client_meta: Dict[str, Any] | None,
    memory_digest: str | None,
) -> SocialRoomTurnV1:
    social_meta = dict(client_meta or {})
    grounding_state = SocialGroundingStateV1.model_validate(
        social_meta.get("social_grounding_state") or {}
    )
    concept_evidence = build_social_concept_evidence(social_meta.get("social_concept_evidence"))
    redaction = build_social_redaction(prompt=prompt, response=response, memory_digest=memory_digest)
    tags = [SOCIAL_ROOM_PROFILE]
    if trace_verb:
        tags.append(str(trace_verb))
    return SocialRoomTurnV1(
        turn_id=str(correlation_id) if correlation_id else f"social-turn-{hash((prompt, response, session_id)) & 0xffffffff:08x}",
        correlation_id=correlation_id,
        session_id=session_id,
        user_id=user_id,
        source=source,
        prompt=prompt,
        response=response,
        text=f"User: {prompt}\nOrion: {response}".strip(),
        recall_profile=recall_profile or SOCIAL_ROOM_RECALL_PROFILE,
        trace_verb=trace_verb,
        tags=tags,
        concept_evidence=concept_evidence,
        grounding_state=grounding_state,
        redaction=redaction,
        client_meta=social_meta,
    )
