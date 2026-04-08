from __future__ import annotations

import re
from typing import Any

from orion.schemas.social_autonomy import SocialTurnPolicyDecisionV1
from orion.schemas.social_bridge import CallSyneRoomMessageV1
from orion.schemas.social_gif import (
    SocialGifIntentKind,
    SocialGifIntentV1,
    SocialGifPolicyDecisionV1,
    SocialGifUsageStateV1,
)


_LIGHT_RE = re.compile(r"\b(lol|lmao|haha|hehe|mood|same|fair|nice|yay|woo|nailed it|we did it|exactly|yep)\b", re.IGNORECASE)
_LAUGH_RE = re.compile(r"\b(lol|lmao|haha|hehe|😂|🤣)\b", re.IGNORECASE)
_CELEBRATE_RE = re.compile(r"\b(yay|woo|nailed it|we did it|finally|yes!|lets go|let's go|win|won)\b", re.IGNORECASE)
_SYMPATHY_RE = re.compile(r"\b(oof|ugh|rough|sorry|ouch|brutal)\b", re.IGNORECASE)
_AGREEMENT_RE = re.compile(r"\b(exactly|same|totally|yep|yes|100%)\b", re.IGNORECASE)
_FACEPALM_RE = re.compile(r"\b(welp|oops|whoops|facepalm)\b", re.IGNORECASE)
_CONFUSION_RE = re.compile(r"\b(wait|what|huh|which one|confused)\b", re.IGNORECASE)
_BLOCKED_RE = re.compile(r"\b(private|sealed|blocked|secret|off[- ]record|journal|mirror)\b", re.IGNORECASE)
_REDUNDANT_GIF_TEXT_RE = re.compile(r"\b(gif|reaction gif|reaction image|meme|insert .*gif|posting a gif)\b", re.IGNORECASE)
_STAGE_DIRECTION_RE = re.compile(r"^\s*[\[*_].+[\]*_]\s*$")
_EMOJI_BURST_RE = re.compile(r"[😂🤣😅😭🙃🤦🎉🔥✨]{2,}")

_MIN_TEXT_ONLY_TURNS_BEFORE_GIF = 2
_MAX_GIFS_PER_WINDOW = 2
_DENSITY_BLOCK_THRESHOLD = 0.21
_MEDIUM_CHAOS_DENSITY_BLOCK_THRESHOLD = 0.12


def _boolish(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _light_exchange(message: CallSyneRoomMessageV1, *, addressed: bool) -> bool:
    text = str(message.text or "").strip()
    if not addressed:
        return False
    if len(text) > 180:
        return False
    return bool(_LIGHT_RE.search(text) or "!" in text)


def _peer_used_gif(message: CallSyneRoomMessageV1) -> bool:
    metadata = dict(message.metadata or {})
    raw_payload = dict(message.raw_payload or {})
    return any(
        _boolish(candidate)
        for candidate in (
            metadata.get("peer_used_gif"),
            metadata.get("contains_gif"),
            raw_payload.get("peer_used_gif"),
            raw_payload.get("contains_gif"),
        )
    )


def _transport_supports_media_hints(message: CallSyneRoomMessageV1) -> bool:
    metadata = dict(message.metadata or {})
    raw_payload = dict(message.raw_payload or {})
    explicit = next(
        (
            candidate
            for candidate in (
                metadata.get("supports_media_hints"),
                metadata.get("supports_gif_hints"),
                raw_payload.get("supports_media_hints"),
                raw_payload.get("supports_gif_hints"),
            )
            if candidate is not None
        ),
        None,
    )
    return _boolish(explicit, default=True)


def _artifact_boundary_active(summary: dict[str, Any]) -> bool:
    participant = dict(summary.get("participant") or {})
    room = dict(summary.get("room") or {})
    for container in (participant, room):
        status = str(container.get("shared_artifact_status") or "").strip().lower()
        if status in {"declined", "deferred"}:
            return True
        if any(isinstance(container.get(key), dict) for key in ("shared_artifact_proposal", "shared_artifact_revision")):
            return True
    return False


def _contested_claims(summary: dict[str, Any], decision: SocialTurnPolicyDecisionV1) -> bool:
    room = dict(summary.get("room") or {})
    claim_kind = str(decision.epistemic_signal.claim_kind if decision.epistemic_signal else "").strip().lower()
    if claim_kind in {
        "recall",
        "inference",
        "speculation",
        "proposal",
        "clarification_needed",
    }:
        return True
    if str((room.get("deliberation_decision") or {}).get("decision_kind") or "").strip().lower() == "bridge_summary":
        return True
    if room.get("claim_divergence_signals"):
        return True
    consensus_states = {
        str(item.get("consensus_state") or "").strip().lower()
        for item in (room.get("claim_consensus_states") or [])
        if isinstance(item, dict)
    }
    return bool(consensus_states & {"partial", "contested", "corrected"})


def _chaotic_room(summary: dict[str, Any]) -> bool:
    room = dict(summary.get("room") or {})
    return len(room.get("active_participants") or []) >= 4 or len(room.get("open_threads") or []) >= 4 or int(room.get("evidence_count") or 0) >= 7


def _medium_chaos_room(summary: dict[str, Any]) -> bool:
    room = dict(summary.get("room") or {})
    active_threads = room.get("active_threads") or room.get("open_threads") or []
    return (
        len(room.get("active_participants") or []) >= 3
        or len(active_threads) >= 2
        or int(room.get("evidence_count") or 0) >= 5
    )


def _stale_resumptive_context(summary: dict[str, Any]) -> bool:
    for candidate in (summary.get("context_candidates") or []):
        if not isinstance(candidate, dict):
            continue
        if candidate.get("candidate_kind") not in {"episode_snapshot", "reentry_anchor"}:
            continue
        if candidate.get("inclusion_decision") in {"include", "soften"} and candidate.get("freshness_band") in {"stale", "refresh_needed", "expired"}:
            return True
    return False


def _freshness_state(summary: dict[str, Any], *, artifact_kind: str) -> str | None:
    room = dict(summary.get("room") or {})
    participant = dict(summary.get("participant") or {})
    for item in list(room.get("memory_freshness") or []) + list(participant.get("memory_freshness") or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("artifact_kind") or "").strip().lower() != artifact_kind:
            continue
        state = str(item.get("freshness_state") or "").strip().lower()
        if state:
            return state
    for candidate in summary.get("context_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        if artifact_kind == "room_ritual" and candidate.get("candidate_kind") == "ritual":
            state = str(candidate.get("freshness_band") or "").strip().lower()
            if state:
                return state
    return None


def _playful_room_bonus(summary: dict[str, Any]) -> tuple[int, list[str]]:
    room = dict(summary.get("room") or {})
    ritual = dict(summary.get("room_ritual") or {})
    peer_style = dict(summary.get("peer_style") or {})
    state = _freshness_state(summary, artifact_kind="room_ritual")
    stale = state in {"stale", "refresh_needed", "expired"}
    phrases = " ".join(
        str(value or "")
        for value in (
            ritual.get("culture_summary"),
            ritual.get("room_tone_summary"),
            room.get("room_tone_summary"),
        )
    ).lower()
    reasons: list[str] = []
    if stale:
        return 0, ["stale_ritual_does_not_push_gif"]

    bonus = 0
    if any(word in phrases for word in ("playful", "warm", "light", "funny", "expressive")):
        bonus += 1
        reasons.append("fresh_room_ritual_supports_playfulness")
    if float(peer_style.get("playfulness_tendency") or 0.0) >= 0.65:
        bonus += 1
        reasons.append("peer_style_supports_playfulness")
    return bonus, reasons


def _caution_context_present(summary: dict[str, Any]) -> bool:
    room = dict(summary.get("room") or {})
    if any(
        isinstance(item, dict) and str(item.get("freshness_state") or "").strip().lower() in {"refresh_needed", "stale"}
        for item in list(room.get("memory_freshness") or []) + list((summary.get("participant") or {}).get("memory_freshness") or [])
    ):
        return True
    for item in summary.get("context_candidates") or []:
        if not isinstance(item, dict):
            continue
        if item.get("candidate_kind") != "freshness_hint":
            continue
        if item.get("inclusion_decision") in {"include", "soften"}:
            return True
    return False


def _intent_loop_detected(
    usage: SocialGifUsageStateV1,
    *,
    intent_kind: SocialGifIntentKind,
    target_participant_id: str | None,
) -> bool:
    recent = list(usage.recent_intent_kinds or [])[-3:]
    if recent.count(intent_kind) >= 2:
        return True
    if usage.last_intent_kind == intent_kind and recent[-1:] == [intent_kind]:
        return True
    recent_targets = list(usage.recent_target_participant_ids or [])[-3:]
    if target_participant_id and recent_targets.count(target_participant_id) >= 2 and recent.count(intent_kind) >= 1:
        return True
    pairs = list(zip(recent, recent_targets[-len(recent) :]))
    if target_participant_id and sum(1 for kind, target in pairs if kind == intent_kind and target == target_participant_id) >= 2:
        return True
    return False


def _select_intent(
    message: CallSyneRoomMessageV1,
    *,
    platform: str,
    room_id: str,
    thread_key: str | None,
    audience_scope: str,
    target_participant_id: str | None,
    target_participant_name: str | None,
    peer_used_gif: bool,
) -> SocialGifIntentV1:
    text = str(message.text or "")
    intent_kind: SocialGifIntentKind
    if _CELEBRATE_RE.search(text):
        intent_kind = "celebrate"
    elif _LAUGH_RE.search(text):
        intent_kind = "laugh_with"
    elif _SYMPATHY_RE.search(text):
        intent_kind = "sympathetic_reaction"
    elif _FACEPALM_RE.search(text):
        intent_kind = "soft_facepalm"
    elif _CONFUSION_RE.search(text):
        intent_kind = "playful_confusion"
    elif "we did it" in text.lower() or "finally" in text.lower():
        intent_kind = "victory_lap"
    elif _AGREEMENT_RE.search(text):
        intent_kind = "dramatic_agreement"
    elif peer_used_gif:
        intent_kind = "laugh_with"
    else:
        intent_kind = "dramatic_agreement"

    query_map = {
        "celebrate": "subtle celebratory reaction gif",
        "laugh_with": "warm laugh with you reaction gif",
        "sympathetic_reaction": "gentle sympathetic reaction gif",
        "dramatic_agreement": "playful dramatic agreement reaction gif",
        "soft_facepalm": "soft facepalm reaction gif",
        "playful_confusion": "playful confusion reaction gif",
        "victory_lap": "tiny victory lap reaction gif",
    }
    return SocialGifIntentV1(
        platform=platform,
        room_id=room_id,
        thread_key=thread_key,
        intent_kind=intent_kind,
        gif_query=query_map[intent_kind],
        audience_scope=audience_scope or "peer",
        target_participant_id=target_participant_id,
        target_participant_name=target_participant_name,
        rationale="Intent was chosen first so any GIF stays a bounded expressive garnish rather than carrying meaning.",
        reasons=["intent_first_policy", f"intent={intent_kind}"],
    )


def evaluate_social_gif_policy(
    *,
    message: CallSyneRoomMessageV1,
    turn_policy: SocialTurnPolicyDecisionV1,
    social_memory: dict[str, Any],
    usage_state: SocialGifUsageStateV1 | None,
) -> SocialGifPolicyDecisionV1:
    room = dict(social_memory.get("room") or {})
    thread_key = str(
        (turn_policy.thread_routing.thread_key if turn_policy.thread_routing else None)
        or room.get("current_thread_key")
        or ""
    ).strip() or None
    audience_scope = turn_policy.thread_routing.audience_scope if turn_policy.thread_routing else "peer"
    if audience_scope == "peer":
        target_participant_id = message.sender_id
        target_participant_name = message.sender_name
    else:
        target_participant_id = turn_policy.thread_routing.target_participant_id if turn_policy.thread_routing else message.target_participant_id
        target_participant_name = turn_policy.thread_routing.target_participant_name if turn_policy.thread_routing else message.target_participant_name
    usage = usage_state or SocialGifUsageStateV1(platform=message.platform, room_id=message.room_id, thread_key=thread_key)
    peer_used_gif = _peer_used_gif(message)
    transport_supported = _transport_supports_media_hints(message)
    reasons: list[str] = ["gif_policy_considered", f"turn_decision={turn_policy.decision}"]
    hard_blocks: list[str] = []
    playful_bonus, playful_reasons = _playful_room_bonus(social_memory)

    repair_active = bool(turn_policy.repair_signal or turn_policy.repair_decision)
    clarification_active = turn_policy.decision == "ask_follow_up" or (
        turn_policy.epistemic_decision is not None and turn_policy.epistemic_decision.decision == "ask_clarifying_question"
    ) or str((room.get("deliberation_decision") or {}).get("decision_kind") or "") == "ask_clarifying_question"
    bridge_summary_active = str((room.get("deliberation_decision") or {}).get("decision_kind") or "").strip().lower() == "bridge_summary" or bool(room.get("bridge_summary"))
    contested_or_sensitive = _contested_claims(social_memory, turn_policy)
    artifact_boundary = _artifact_boundary_active(social_memory)
    private_or_blocked = bool(_BLOCKED_RE.search(message.text or "")) or not transport_supported and _boolish((message.metadata or {}).get("private_boundary"))
    ambiguity_high = bool(turn_policy.thread_routing and turn_policy.thread_routing.ambiguity_level == "high")
    cooldown_active = usage.turns_since_last_orion_gif < _MIN_TEXT_ONLY_TURNS_BEFORE_GIF or usage.consecutive_gif_turns > 0
    density_threshold = _MEDIUM_CHAOS_DENSITY_BLOCK_THRESHOLD if _medium_chaos_room(social_memory) else _DENSITY_BLOCK_THRESHOLD
    density_blocked = usage.recent_gif_turn_count >= _MAX_GIFS_PER_WINDOW or usage.recent_gif_density > density_threshold
    first_turn_block = usage.orion_turn_count == 0 and not peer_used_gif

    for condition, label in (
        (repair_active, "repair_active_turn"),
        (clarification_active, "clarification_turn_text_only"),
        (artifact_boundary, "shared_artifact_or_scope_boundary"),
        (contested_or_sensitive, "epistemically_sensitive_or_contested"),
        (bridge_summary_active, "bridge_summary_turn"),
        (private_or_blocked, "private_or_blocked_material"),
        (ambiguity_high, "high_thread_ambiguity"),
        (usage.consecutive_gif_turns > 0, "previous_orion_turn_used_gif"),
        (usage.turns_since_last_orion_gif < _MIN_TEXT_ONLY_TURNS_BEFORE_GIF, "gif_cooldown_active"),
        (density_blocked, "gif_density_cap_reached"),
        (first_turn_block, "first_orion_turn_in_thread_defaults_text_only"),
    ):
        if condition:
            hard_blocks.append(label)
    reasons.extend(hard_blocks)

    intent = _select_intent(
        message,
        platform=message.platform,
        room_id=message.room_id,
        thread_key=thread_key,
        audience_scope=audience_scope,
        target_participant_id=target_participant_id,
        target_participant_name=target_participant_name,
        peer_used_gif=peer_used_gif,
    )
    if _intent_loop_detected(usage, intent_kind=intent.intent_kind, target_participant_id=target_participant_id):
        hard_blocks.append("gif_intent_loop_detected")
        reasons.append("gif_intent_loop_detected")

    if hard_blocks:
        return SocialGifPolicyDecisionV1(
            platform=message.platform,
            room_id=message.room_id,
            thread_key=thread_key,
            gif_allowed=False,
            decision_kind="text_only",
            intent_kind=intent.intent_kind,
            rationale="GIFs stay disabled because this turn is sensitive, too soon after a recent GIF, or not bounded enough for expressive garnish.",
            reasons=reasons,
            cooldown_active=cooldown_active,
            consecutive_gif_turns=usage.consecutive_gif_turns,
            turns_since_last_orion_gif=usage.turns_since_last_orion_gif,
            recent_gif_density=usage.recent_gif_density,
            audience_scope=audience_scope,
            target_participant_id=target_participant_id,
            target_participant_name=target_participant_name,
            metadata={
                "transport_supports_media_hints": "true" if transport_supported else "false",
                "blocked": "true",
                "density_threshold": str(density_threshold),
            },
        )

    score = 0
    if peer_used_gif:
        score += 2
        reasons.append("peer_used_gif_recently")
    if _light_exchange(message, addressed=turn_policy.addressed):
        score += 2
        reasons.append("light_affiliative_low_stakes")
    if usage.orion_turn_count >= 3 and usage.turns_since_last_orion_gif >= 3:
        score += 1
        reasons.append("orion_has_been_text_only")
    if playful_bonus:
        score += playful_bonus
        reasons.extend(playful_reasons)
    else:
        reasons.extend(playful_reasons)
    if "stale_ritual_does_not_push_gif" in playful_reasons:
        score -= 1
        reasons.append("stale_ritual_downrank")

    if usage.last_intent_kind and usage.last_intent_kind == intent.intent_kind:
        score -= 2
        reasons.append("repeated_intent_suppressed")
    if target_participant_id and usage.last_target_participant_id == target_participant_id:
        score -= 1
        reasons.append("same_peer_got_last_gif")
    if (
        usage.last_intent_kind
        and usage.last_intent_kind == intent.intent_kind
        and target_participant_id
        and usage.last_target_participant_id == target_participant_id
    ):
        score -= 3
        reasons.append("repeated_intent_same_peer_penalty")
    if _chaotic_room(social_memory):
        score -= 1
        reasons.append("chaotic_room_downrank")
    elif _medium_chaos_room(social_memory):
        score -= 1
        reasons.append("medium_chaos_downrank")
    if _stale_resumptive_context(social_memory):
        score -= 1
        reasons.append("stale_resumptive_context_downrank")
    if _caution_context_present(social_memory) and not peer_used_gif:
        score -= 1
        reasons.append("fresh_caution_context_prefers_text")
    if audience_scope != "peer" or not target_participant_id:
        score -= 1
        reasons.append("room_or_scope_wide_turn_prefers_text")
    if bool(turn_policy.handoff_signal) or str((room.get("floor_decision") or {}).get("decision_kind") or "") in {"yield_to_peer", "leave_open", "close_thread"}:
        score -= 1
        reasons.append("handoff_or_closure_prefers_plain_text")

    gif_allowed = score >= 2 and transport_supported
    if score >= 2 and not transport_supported:
        reasons.append("transport_degraded_to_text_only")

    return SocialGifPolicyDecisionV1(
        platform=message.platform,
        room_id=message.room_id,
        thread_key=thread_key,
        gif_allowed=gif_allowed,
        decision_kind="text_plus_gif" if gif_allowed else "text_only",
        intent_kind=intent.intent_kind,
        selected_intent=intent if gif_allowed else None,
        rationale=(
            "A text-plus-GIF reply is allowed because the turn is light, affiliative, and safely bounded."
            if gif_allowed
            else "The turn stays text-only because GIF expression did not clear the bounded social-expression threshold."
        ),
        reasons=reasons,
        cooldown_active=cooldown_active,
        consecutive_gif_turns=usage.consecutive_gif_turns,
        turns_since_last_orion_gif=usage.turns_since_last_orion_gif,
        recent_gif_density=usage.recent_gif_density,
        audience_scope=audience_scope,
        target_participant_id=target_participant_id,
        target_participant_name=target_participant_name,
        metadata={
            "transport_supports_media_hints": "true" if transport_supported else "false",
            "score": str(score),
            "peer_used_gif": "true" if peer_used_gif else "false",
            "transport_degraded": "true" if (score >= 2 and not transport_supported) else "false",
            "density_threshold": str(density_threshold),
        },
    )


def reconcile_gif_policy_with_reply_text(
    *,
    policy: SocialGifPolicyDecisionV1,
    reply_text: str,
) -> SocialGifPolicyDecisionV1:
    text = str(reply_text or "").strip()
    if not policy.gif_allowed or policy.decision_kind != "text_plus_gif" or not text:
        return policy
    if (
        _REDUNDANT_GIF_TEXT_RE.search(text)
        or _STAGE_DIRECTION_RE.search(text)
        or _EMOJI_BURST_RE.search(text)
    ):
        reasons = list(policy.reasons)
        reasons.append("reply_text_already_carries_reaction")
        return policy.model_copy(
            update={
                "gif_allowed": False,
                "decision_kind": "text_only",
                "selected_intent": None,
                "rationale": "The reply already carries the reaction in text, so the GIF stays off to keep media secondary.",
                "reasons": reasons,
                "metadata": dict(policy.metadata or {}, redundant_reply_text="true"),
            }
        )
    return policy


def update_live_gif_usage_state(
    *,
    usage_state: SocialGifUsageStateV1 | None,
    policy: SocialGifPolicyDecisionV1 | None,
    platform: str,
    room_id: str,
    thread_key: str | None,
    target_participant_id: str | None,
    target_participant_name: str | None,
) -> SocialGifUsageStateV1:
    existing = usage_state or SocialGifUsageStateV1(platform=platform, room_id=room_id, thread_key=thread_key)
    used_gif = bool(policy and policy.gif_allowed and policy.decision_kind == "text_plus_gif")
    recent_turns = list(existing.recent_turn_was_gif)
    recent_turns.append(used_gif)
    recent_turns = recent_turns[-existing.recent_turn_window_size :]
    gif_count = sum(1 for item in recent_turns if item)
    density = gif_count / float(len(recent_turns) or 1)
    recent_intents = list(existing.recent_intent_kinds)
    recent_target_ids = list(existing.recent_target_participant_ids)
    recent_target_names = list(existing.recent_target_participant_names)
    if used_gif and policy and policy.intent_kind:
        recent_intents.append(policy.intent_kind)
        if target_participant_id:
            recent_target_ids.append(target_participant_id)
        if target_participant_name:
            recent_target_names.append(target_participant_name)
    return SocialGifUsageStateV1(
        usage_state_id=existing.usage_state_id,
        platform=platform,
        room_id=room_id,
        thread_key=thread_key or existing.thread_key,
        consecutive_gif_turns=(existing.consecutive_gif_turns + 1) if used_gif else 0,
        turns_since_last_orion_gif=0 if used_gif else int(existing.turns_since_last_orion_gif) + 1,
        recent_gif_density=density,
        recent_gif_turn_count=gif_count,
        recent_turn_window_size=existing.recent_turn_window_size,
        orion_turn_count=int(existing.orion_turn_count) + 1,
        recent_turn_was_gif=recent_turns,
        recent_intent_kinds=recent_intents[-4:],
        recent_target_participant_ids=recent_target_ids[-4:],
        recent_target_participant_names=recent_target_names[-4:],
        last_intent_kind=policy.intent_kind if used_gif and policy else existing.last_intent_kind,
        last_target_participant_id=target_participant_id if used_gif and target_participant_id else existing.last_target_participant_id,
        last_target_participant_name=target_participant_name if used_gif and target_participant_name else existing.last_target_participant_name,
        last_gif_at=policy.created_at if used_gif and policy else existing.last_gif_at,
        metadata={
            "source": "social-room-bridge",
            "last_decision_kind": policy.decision_kind if policy else "text_only",
        },
    )
