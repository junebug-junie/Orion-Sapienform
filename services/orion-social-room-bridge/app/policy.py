from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from orion.schemas.social_autonomy import (
    SocialAutonomyMode,
    SocialOpenThreadV1,
    SocialTurnPolicyDecisionV1,
)
from orion.schemas.social_bridge import CallSyneRoomMessageV1
from orion.schemas.social_epistemic import SocialEpistemicDecisionV1, SocialEpistemicSignalV1
from orion.schemas.social_repair import SocialRepairDecisionV1, SocialRepairSignalV1
from orion.schemas.social_thread import SocialHandoffSignalV1, SocialThreadRoutingDecisionV1

from .settings import Settings


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'-]{2,}")
_SEALED_RE = re.compile(r"\b(private|sealed|secret|off[- ]record)\b", re.IGNORECASE)
_STOPWORDS = {
    "about",
    "again",
    "been",
    "from",
    "here",
    "just",
    "like",
    "more",
    "really",
    "room",
    "same",
    "that",
    "this",
    "thread",
    "what",
    "when",
    "where",
    "with",
    "would",
    "your",
}
_REPAIR_DIRECT_CORRECTION_HINTS = (
    "that was for",
    "not you",
    "not for you",
    "i meant",
    "not orion",
    "not oríon",
    "wrong person",
)
_REPAIR_THREAD_MISMATCH_HINTS = (
    "wrong thread",
    "other thread",
    "different thread",
    "not this thread",
    "wrong channel",
)
_REPAIR_CONTRADICTION_HINTS = (
    "you just said",
    "that's not what you said",
    "that contradicts",
    "contradicts",
    "inconsistent",
)
_REPAIR_SCOPE_HINTS = (
    "room-local, not peer-local",
    "room local, not peer local",
    "peer-local, not room-local",
    "peer local, not room local",
    "session-only",
    "session only",
    "room-local",
    "peer-local",
    "private",
    "off the record",
)
_REPAIR_REDIRECT_HINTS = (
    "let archivist take this one",
    "let cadence take this one",
    "let them take this one",
    "let someone else take this one",
    "let them answer",
    "let archivist answer",
    "let cadence answer",
    "step back",
    "pause there",
)
_REPAIR_CLARIFY_HINTS = (
    "who are you answering",
    "which thread",
    "do you mean me or",
    "wait, who is this for",
    "who is this for",
)
_REPAIR_LOW_CONFIDENCE_HINTS = (
    "maybe not you",
    "not sure who that's for",
    "not sure who that was for",
)
_EPISTEMIC_MEMORY_HINTS = (
    "what do you remember",
    "do you remember",
    "what do you recall",
    "remember about",
)
_EPISTEMIC_SUMMARY_HINTS = (
    "summary",
    "summarize",
    "recap",
    "catch us up",
    "where are we",
)
_EPISTEMIC_INFERENCE_HINTS = (
    "what's your read",
    "what is your read",
    "why do you think",
    "do you think",
    "seems like",
    "sounds like",
)
_EPISTEMIC_SPECULATION_HINTS = (
    "guess",
    "maybe",
    "might be",
    "could be",
    "speculate",
)
_EPISTEMIC_CLARIFY_HINTS = (
    "what do you mean",
    "which one",
    "which thread",
    "who do you mean",
    "who is this for",
)


@dataclass
class PolicyContext:
    social_memory: dict[str, Any]
    is_duplicate: bool
    is_self_message: bool
    addressed: bool
    cooldown_active: bool
    consecutive_limit_hit: bool


def _tokens(text: str | None) -> set[str]:
    found = {
        token.lower()
        for token in _TOKEN_RE.findall(str(text or ""))
        if len(token) >= 4 and token.lower() not in _STOPWORDS
    }
    return found


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a = set(left)
    b = set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _trim_reason(value: str, *, limit: int = 120) -> str:
    text = " ".join(str(value).split())
    return text[:limit]


def _parse_dt(value: Any) -> datetime | None:
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


class SocialTurnPolicyEvaluator:
    def __init__(self, *, settings: Settings) -> None:
        self.settings = settings

    def evaluate(
        self,
        *,
        message: CallSyneRoomMessageV1,
        context: PolicyContext,
        mode: SocialAutonomyMode | None = None,
    ) -> SocialTurnPolicyDecisionV1:
        active_mode = mode or self.settings.social_bridge_autonomy_mode
        participant = context.social_memory.get("participant") or {}
        room = context.social_memory.get("room") or {}
        open_thread = self._derive_open_thread(message=message, room=room)
        novelty_score = self._novelty_score(message=message, participant=participant, room=room)
        continuity_score = self._continuity_score(message=message, room=room, open_thread=open_thread)
        quiet_room = self._quiet_room(room=room)
        low_novelty = novelty_score < float(self.settings.social_bridge_min_novelty_score)
        open_question = self._is_open_question(message)
        thread_routing = self._thread_routing(
            message=message,
            room=room,
            open_thread=open_thread,
            addressed=context.addressed,
            mode=active_mode,
        )
        repair_signal = self._repair_signal(
            message=message,
            room=room,
            thread_routing=thread_routing,
            addressed=context.addressed,
        )
        repair_decision = self._repair_decision(
            message=message,
            thread_routing=thread_routing,
            repair_signal=repair_signal,
            addressed=context.addressed,
        )
        if repair_decision is not None and repair_decision.decision != "ignore":
            thread_routing = self._apply_repair_to_routing(
                thread_routing=thread_routing,
                repair_decision=repair_decision,
                message=message,
            )
        handoff_signal = self._handoff_signal(
            message=message,
            room=room,
            thread_routing=thread_routing,
            repair_decision=repair_decision,
        )
        epistemic_signal = self._epistemic_signal(
            message=message,
            participant=participant,
            room=room,
            thread_routing=thread_routing,
            repair_signal=repair_signal,
            repair_decision=repair_decision,
            addressed=context.addressed,
        )
        epistemic_decision = self._epistemic_decision(
            message=message,
            thread_routing=thread_routing,
            epistemic_signal=epistemic_signal,
            addressed=context.addressed,
        )
        initiative_allowed = (
            active_mode == "light_initiative"
            and open_thread is not None
            and continuity_score >= float(self.settings.social_bridge_light_initiative_min_continuity)
        )

        reasons: list[str] = [
            f"mode={active_mode}",
            f"addressed={context.addressed}",
            f"novelty={novelty_score:.2f}",
            f"continuity={continuity_score:.2f}",
            f"ambiguity={thread_routing.ambiguity_level}",
        ]
        if open_thread is not None:
            reasons.append(f"open_thread={open_thread.topic_key}")
        reasons.append(f"audience={thread_routing.audience_scope}")
        reasons.append(f"thread_route={thread_routing.routing_decision}")
        reasons.extend(thread_routing.reasons[:3])
        if repair_signal is not None:
            reasons.append(f"repair_type={repair_signal.repair_type}")
            reasons.append(f"repair_confidence={repair_signal.confidence:.2f}")
        if repair_decision is not None:
            reasons.append(f"repair_decision={repair_decision.decision}")
        if epistemic_signal is not None:
            reasons.append(f"claim_kind={epistemic_signal.claim_kind}")
            reasons.append(f"epistemic_confidence={epistemic_signal.confidence_level}")
        if epistemic_decision is not None:
            reasons.append(f"epistemic_decision={epistemic_decision.decision}")
        if quiet_room:
            reasons.append("room context is sparse")

        decision_kwargs = dict(
            mode=active_mode,
            cooldown_active=context.cooldown_active,
            consecutive_limit_hit=context.consecutive_limit_hit,
            quiet_room=quiet_room,
            novelty_score=novelty_score,
            continuity_score=continuity_score,
            open_thread=open_thread,
            thread_routing=thread_routing,
            handoff_signal=handoff_signal,
            repair_signal=repair_signal,
            repair_decision=repair_decision,
            epistemic_signal=epistemic_signal,
            epistemic_decision=epistemic_decision,
        )

        if context.is_duplicate:
            return self._decision(
                message=message,
                decision="skip",
                should_speak=False,
                reasons=reasons + ["duplicate inbound transport message"],
                addressed=context.addressed,
                **decision_kwargs,
            )

        if context.is_self_message:
            return self._decision(
                message=message,
                decision="skip",
                should_speak=False,
                reasons=reasons + ["self-loop suppression for Orion-authored traffic"],
                addressed=context.addressed,
                **decision_kwargs,
            )

        if context.cooldown_active:
            return self._decision(
                message=message,
                decision="wait",
                should_speak=False,
                reasons=reasons + ["room-local cooldown is active"],
                addressed=context.addressed,
                **decision_kwargs,
            )

        if context.consecutive_limit_hit:
            return self._decision(
                message=message,
                decision="wait",
                should_speak=False,
                reasons=reasons + ["max consecutive Orion turns reached"],
                addressed=context.addressed,
                **decision_kwargs,
            )

        if repair_decision is not None and repair_decision.decision != "ignore":
            if repair_decision.decision == "yield" and not self._repair_needs_ack(
                message=message,
                addressed=context.addressed,
            ):
                return self._decision(
                    message=message,
                    decision="wait",
                    should_speak=False,
                    reasons=reasons + ["repair redirect does not require an additional Orion turn"],
                    addressed=context.addressed,
                    **decision_kwargs,
                )
            if (
                epistemic_decision is not None
                and epistemic_decision.decision == "ask_clarifying_question"
                and repair_decision.decision in {"repair", "clarify"}
            ):
                return self._decision(
                    message=message,
                    decision="ask_follow_up",
                    should_speak=True,
                    reasons=reasons + [epistemic_decision.rationale or "the safer repair is a brief clarification before carrying anything forward"],
                    addressed=context.addressed,
                    **decision_kwargs,
                )
            return self._decision(
                message=message,
                decision="ask_follow_up" if repair_decision.decision == "clarify" else "reply",
                should_speak=True,
                reasons=reasons + [repair_decision.rationale or "compact conversational repair is locally warranted"],
                addressed=context.addressed,
                **decision_kwargs,
            )

        if epistemic_decision is not None:
            if epistemic_decision.decision == "ask_clarifying_question" and (
                context.addressed or open_question or thread_routing.routing_decision != "wait"
            ):
                return self._decision(
                    message=message,
                    decision="ask_follow_up",
                    should_speak=True,
                    reasons=reasons + [epistemic_decision.rationale or "clarity is safer than false certainty here"],
                    addressed=context.addressed,
                    **decision_kwargs,
                )
            if epistemic_decision.decision == "defer_narrowly" and not context.addressed:
                return self._decision(
                    message=message,
                    decision="wait",
                    should_speak=False,
                    reasons=reasons + [epistemic_decision.rationale or "the epistemic basis is too thin to answer broadly"],
                    addressed=context.addressed,
                    **decision_kwargs,
                )

        if active_mode == "addressed_only" and not context.addressed:
            return self._decision(
                message=message,
                decision="wait",
                should_speak=False,
                reasons=reasons + ["addressed_only mode requires a direct mention or reply target"],
                addressed=False,
                **decision_kwargs,
            )

        if (
            not context.addressed
            and thread_routing is not None
            and thread_routing.routing_decision == "wait"
        ):
            wait_reasons = list(reasons)
            if "peer_targeted_elsewhere" in thread_routing.reasons:
                wait_reasons.append("peer-targeted exchange appears aimed at another participant")
            if "ambiguous_multi_thread" in thread_routing.reasons:
                wait_reasons.append("multiple active threads are plausible and the audience is unclear")
            return self._decision(
                message=message,
                decision="wait",
                should_speak=False,
                reasons=wait_reasons,
                addressed=False,
                **decision_kwargs,
            )

        if not context.addressed and low_novelty and not open_question and not initiative_allowed:
            return self._decision(
                message=message,
                decision="wait",
                should_speak=False,
                reasons=reasons + ["low novelty / redundant with recent room continuity"],
                addressed=False,
                **decision_kwargs,
            )

        if context.addressed:
            return self._decision(
                message=message,
                decision="reply",
                should_speak=True,
                reasons=reasons + ["directly addressed to Orion"],
                addressed=True,
                **decision_kwargs,
            )

        if active_mode == "responsive":
            if thread_routing.routing_decision == "summarize_room":
                return self._decision(
                    message=message,
                    decision="reply",
                    should_speak=True,
                    reasons=reasons + ["responsive mode accepts a compact room summary handoff"],
                    addressed=False,
                    **decision_kwargs,
                )
            if open_question:
                return self._decision(
                    message=message,
                    decision="ask_follow_up",
                    should_speak=True,
                    reasons=reasons + ["responsive mode allows a reply to an open room question"],
                    addressed=False,
                    **decision_kwargs,
                )
            return self._decision(
                message=message,
                decision="wait",
                should_speak=False,
                reasons=reasons + ["responsive mode stays quiet unless addressed or asked a room question"],
                addressed=False,
                **decision_kwargs,
            )

        if active_mode == "light_initiative":
            if thread_routing.routing_decision == "summarize_room":
                return self._decision(
                    message=message,
                    decision="reply",
                    should_speak=True,
                    reasons=reasons + ["room_summary_preferred"],
                    addressed=False,
                    **decision_kwargs,
                )
            if thread_routing.routing_decision == "revive_thread":
                return self._decision(
                    message=message,
                    decision="initiate_lightly",
                    should_speak=True,
                    reasons=reasons + ["revival_allowed"],
                    addressed=False,
                    **decision_kwargs,
                )
            if open_question:
                return self._decision(
                    message=message,
                    decision="ask_follow_up",
                    should_speak=True,
                    reasons=reasons + ["light_initiative mode can respond to open room questions"],
                    addressed=False,
                    **decision_kwargs,
                )
            if initiative_allowed:
                return self._decision(
                    message=message,
                    decision="initiate_lightly",
                    should_speak=True,
                    reasons=reasons + ["light_initiative mode allows a bounded follow-on to an active open thread"],
                    addressed=False,
                    **decision_kwargs,
                )
            return self._decision(
                message=message,
                decision="wait",
                should_speak=False,
                reasons=reasons + ["light_initiative found no active open thread worth extending"],
                addressed=False,
                **decision_kwargs,
            )

        return self._decision(
            message=message,
            decision="wait",
            should_speak=False,
            reasons=reasons + ["no autonomy policy matched"],
            addressed=context.addressed,
            **decision_kwargs,
        )

    def _decision(
        self,
        *,
        message: CallSyneRoomMessageV1,
        decision: str,
        should_speak: bool,
        reasons: list[str],
        mode: SocialAutonomyMode | None = None,
        addressed: bool,
        cooldown_active: bool,
        consecutive_limit_hit: bool,
        quiet_room: bool,
        novelty_score: float,
        continuity_score: float,
        open_thread: SocialOpenThreadV1 | None,
        thread_routing: SocialThreadRoutingDecisionV1 | None = None,
        handoff_signal: SocialHandoffSignalV1 | None = None,
        repair_signal: SocialRepairSignalV1 | None = None,
        repair_decision: SocialRepairDecisionV1 | None = None,
        epistemic_signal: SocialEpistemicSignalV1 | None = None,
        epistemic_decision: SocialEpistemicDecisionV1 | None = None,
    ) -> SocialTurnPolicyDecisionV1:
        return SocialTurnPolicyDecisionV1(
            mode=mode or self.settings.social_bridge_autonomy_mode,
            platform=message.platform,
            room_id=message.room_id,
            thread_id=message.thread_id,
            participant_id=message.sender_id,
            decision=decision,
            should_speak=should_speak,
            reasons=[_trim_reason(reason) for reason in reasons],
            addressed=addressed,
            cooldown_active=cooldown_active,
            consecutive_limit_hit=consecutive_limit_hit,
            quiet_room=quiet_room,
            novelty_score=novelty_score,
            continuity_score=continuity_score,
            open_thread_key=open_thread.topic_key if open_thread is not None else None,
            thread_routing=thread_routing,
            handoff_signal=handoff_signal,
            repair_signal=repair_signal,
            repair_decision=repair_decision,
            epistemic_signal=epistemic_signal,
            epistemic_decision=epistemic_decision,
        )

    def _repair_needs_ack(
        self,
        *,
        message: CallSyneRoomMessageV1,
        addressed: bool,
    ) -> bool:
        if addressed or message.mentions_orion:
            return True
        return self._participant_is_orion(
            participant_id=message.reply_to_sender_id,
            participant_name=message.target_participant_name,
        )

    def _repair_signal(
        self,
        *,
        message: CallSyneRoomMessageV1,
        room: dict[str, Any],
        thread_routing: SocialThreadRoutingDecisionV1,
        addressed: bool,
    ) -> SocialRepairSignalV1 | None:
        prompt = str(message.text or "").lower()
        if not prompt:
            return None

        target_participant_id = message.target_participant_id or thread_routing.target_participant_id
        target_participant_name = message.target_participant_name or thread_routing.target_participant_name
        active_commitments = [item for item in room.get("active_commitments") or [] if isinstance(item, dict)]
        participant_hint = (target_participant_name or target_participant_id or "").strip()

        if any(token in prompt for token in _REPAIR_THREAD_MISMATCH_HINTS):
            return SocialRepairSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                repair_type="thread_mismatch",
                trigger="thread_routing",
                source_participant_id=message.sender_id,
                source_participant_name=message.sender_name,
                target_participant_id=target_participant_id,
                target_participant_name=target_participant_name,
                confidence=0.9,
                detected=True,
                rationale="the peer explicitly signaled that Orion is in the wrong thread",
                reasons=["explicit thread correction", thread_routing.thread_summary[:80]],
                metadata={"source": "social-turn-policy"},
            )

        if any(token in prompt for token in _REPAIR_DIRECT_CORRECTION_HINTS):
            repair_type = "peer_correction" if "that was for" in prompt or "i meant" in prompt else "audience_mismatch"
            confidence = 0.92 if ("not you" in prompt or "that was for" in prompt) else 0.78
            if participant_hint and participant_hint.lower() in {"oríon", "orion"}:
                target_participant_id = None
                target_participant_name = None
            return SocialRepairSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                repair_type=repair_type,  # type: ignore[arg-type]
                trigger="peer_message",
                source_participant_id=message.sender_id,
                source_participant_name=message.sender_name,
                target_participant_id=target_participant_id,
                target_participant_name=target_participant_name,
                confidence=confidence,
                detected=True,
                rationale="the peer corrected Orion's audience or participation target",
                reasons=["direct audience correction", participant_hint[:80] or "target reset"],
                metadata={"source": "social-turn-policy"},
            )

        if any(token in prompt for token in _REPAIR_REDIRECT_HINTS) or (
            thread_routing.routing_decision == "wait"
            and "peer_targeted_elsewhere" in thread_routing.reasons
            and addressed
        ):
            return SocialRepairSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                repair_type="redirect",
                trigger="handoff_redirect",
                source_participant_id=message.sender_id,
                source_participant_name=message.sender_name,
                target_participant_id=target_participant_id,
                target_participant_name=target_participant_name,
                confidence=0.88 if any(token in prompt for token in _REPAIR_REDIRECT_HINTS) else 0.72,
                detected=True,
                rationale="the room appears to redirect the exchange away from Orion",
                reasons=["yield requested", participant_hint[:80] or thread_routing.rationale[:80]],
                metadata={"source": "social-turn-policy"},
            )

        if any(token in prompt for token in _REPAIR_CONTRADICTION_HINTS):
            confidence = 0.84 if active_commitments else 0.58
            trigger = "active_commitment" if active_commitments else "peer_message"
            return SocialRepairSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                repair_type="commitment_contradiction",
                trigger=trigger,  # type: ignore[arg-type]
                source_participant_id=message.sender_id,
                source_participant_name=message.sender_name,
                confidence=confidence,
                detected=True,
                rationale="the peer signaled a contradiction against Orion's recent local continuity",
                reasons=[str((active_commitments[0] if active_commitments else {}).get("summary") or "")[:120] or "recent contradiction cue"],
                metadata={"source": "social-turn-policy"},
            )

        if any(token in prompt for token in _REPAIR_SCOPE_HINTS):
            corrected_scope = self._corrected_scope(prompt=prompt, room=room)
            confidence = 0.91 if corrected_scope in {"session_only", "room_local", "peer_local", "private"} else 0.62
            return SocialRepairSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                repair_type="scope_correction",
                trigger="scope_boundary",
                source_participant_id=message.sender_id,
                source_participant_name=message.sender_name,
                confidence=confidence,
                detected=True,
                rationale="the peer narrowed how Orion should treat continuity or scope in this room",
                reasons=[f"corrected_scope={corrected_scope}", "narrow rather than broaden"],
                metadata={"source": "social-turn-policy", "corrected_scope": corrected_scope},
            )

        if any(token in prompt for token in _REPAIR_CLARIFY_HINTS):
            return SocialRepairSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                repair_type="clarification_after_misalignment",
                trigger="peer_message",
                source_participant_id=message.sender_id,
                source_participant_name=message.sender_name,
                confidence=0.74,
                detected=True,
                rationale="the peer asked for a quick clarification after a local mismatch",
                reasons=["clarification requested", thread_routing.rationale[:80]],
                metadata={"source": "social-turn-policy"},
            )

        if any(token in prompt for token in _REPAIR_LOW_CONFIDENCE_HINTS):
            return SocialRepairSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                repair_type="audience_mismatch",
                trigger="low_confidence",
                source_participant_id=message.sender_id,
                source_participant_name=message.sender_name,
                confidence=0.34,
                detected=True,
                rationale="a weak audience-mismatch cue was present, but it remains low confidence",
                reasons=["weak mismatch hint only"],
                metadata={"source": "social-turn-policy"},
            )

        return None

    def _repair_decision(
        self,
        *,
        message: CallSyneRoomMessageV1,
        thread_routing: SocialThreadRoutingDecisionV1,
        repair_signal: SocialRepairSignalV1 | None,
        addressed: bool,
    ) -> SocialRepairDecisionV1 | None:
        if repair_signal is None:
            return None
        decision = "ignore"
        rationale = "the signal is too weak to justify a repair turn"
        reasons = list(repair_signal.reasons)
        metadata = dict(repair_signal.metadata or {})
        if repair_signal.confidence < 0.55:
            reasons.append("ignored low-confidence signal")
        elif repair_signal.repair_type == "scope_correction":
            decision = "repair"
            rationale = "scope correction should be acknowledged briefly while staying narrower, not broader"
        elif repair_signal.repair_type in {"redirect", "audience_mismatch", "peer_correction"}:
            decision = "yield" if repair_signal.target_participant_id or repair_signal.target_participant_name else "repair"
            rationale = "a compact correction or yield is safer than continuing in the wrong audience lane"
        elif repair_signal.repair_type == "thread_mismatch":
            decision = "reset_thread"
            rationale = "the thread should be reset before Orion continues so it does not answer across threads"
        elif repair_signal.repair_type == "commitment_contradiction":
            decision = "repair" if repair_signal.confidence >= 0.75 else "clarify"
            rationale = "Orion should correct or briefly clarify the contradiction without re-litigating it"
        elif repair_signal.repair_type == "clarification_after_misalignment":
            decision = "clarify"
            rationale = "a single compact clarification is better than pressing forward"
        if decision == "yield" and not (addressed or message.mentions_orion):
            metadata.setdefault("passive_yield", "true")
        return SocialRepairDecisionV1(
            platform=message.platform,
            room_id=message.room_id,
            thread_key=thread_routing.thread_key,
            repair_type=repair_signal.repair_type,
            trigger=repair_signal.trigger,
            signal_id=repair_signal.repair_id,
            decision=decision,  # type: ignore[arg-type]
            target_participant_id=repair_signal.target_participant_id,
            target_participant_name=repair_signal.target_participant_name,
            confidence=repair_signal.confidence,
            rationale=rationale,
            reasons=reasons + [f"routing={thread_routing.routing_decision}"],
            metadata=metadata,
        )

    def _corrected_scope(self, *, prompt: str, room: dict[str, Any]) -> str:
        if "off the record" in prompt or "private" in prompt:
            return "private"
        if "session-only" in prompt or "session only" in prompt:
            return "session_only"
        if "room-local" in prompt or "room local" in prompt:
            return "room_local"
        if "peer-local" in prompt or "peer local" in prompt:
            return "peer_local"
        scope = str(room.get("shared_artifact_scope") or "")
        return scope or "narrower"

    def _apply_repair_to_routing(
        self,
        *,
        thread_routing: SocialThreadRoutingDecisionV1,
        repair_decision: SocialRepairDecisionV1,
        message: CallSyneRoomMessageV1,
    ) -> SocialThreadRoutingDecisionV1:
        metadata = dict(thread_routing.metadata or {})
        metadata.update({
            "repair_decision": repair_decision.decision,
            "repair_type": str(repair_decision.repair_type or ""),
        })
        reasons = list(thread_routing.reasons or [])
        reasons.append(f"repair_{repair_decision.decision}")
        rationale = thread_routing.rationale
        routing = thread_routing.routing_decision
        audience_scope = thread_routing.audience_scope
        target_participant_id = thread_routing.target_participant_id
        target_participant_name = thread_routing.target_participant_name
        if repair_decision.decision in {"yield", "reset_thread"}:
            routing = "wait"
            audience_scope = "peer" if (repair_decision.target_participant_id or repair_decision.target_participant_name) else "none"
            target_participant_id = repair_decision.target_participant_id
            target_participant_name = repair_decision.target_participant_name
            rationale = "repair handling suppresses the wrong-thread reply and yields or resets cleanly"
            reasons.append("repair_redirect_wait")
        elif repair_decision.decision == "clarify":
            routing = "reply_to_peer"
            audience_scope = "peer"
            target_participant_id = message.sender_id
            target_participant_name = message.sender_name
            rationale = "repair handling prefers a compact clarification to the correcting peer"
            reasons.append("repair_clarify")
        elif repair_decision.decision == "repair":
            routing = "reply_to_peer"
            audience_scope = "peer"
            target_participant_id = message.sender_id
            target_participant_name = message.sender_name
            rationale = "repair handling prefers a compact correction before moving on"
            reasons.append("repair_brief_ack")
        return thread_routing.model_copy(
            update={
                "routing_decision": routing,
                "audience_scope": audience_scope,
                "target_participant_id": target_participant_id,
                "target_participant_name": target_participant_name,
                "rationale": rationale,
                "reasons": reasons[:6],
                "metadata": metadata,
            }
        )

    def _confidence_bucket(self, evidence_count: int, *, ambiguity_level: str) -> str:
        if ambiguity_level == "high":
            return "low"
        if evidence_count >= 4:
            return "high"
        if evidence_count >= 2:
            return "medium"
        return "low"

    def _epistemic_signal(
        self,
        *,
        message: CallSyneRoomMessageV1,
        participant: dict[str, Any],
        room: dict[str, Any],
        thread_routing: SocialThreadRoutingDecisionV1,
        repair_signal: SocialRepairSignalV1 | None,
        repair_decision: SocialRepairDecisionV1 | None,
        addressed: bool,
    ) -> SocialEpistemicSignalV1 | None:
        prompt = str(message.text or "").lower()
        if not prompt:
            return None

        participant_evidence = int(participant.get("evidence_count") or 0)
        room_evidence = int(room.get("evidence_count") or 0)
        evidence_count = max(participant_evidence, room_evidence)
        ambiguity_level = (
            "high"
            if repair_decision is not None and repair_decision.decision in {"clarify", "reset_thread"}
            else thread_routing.ambiguity_level
        )
        has_pending_artifact = any(
            isinstance(container.get(key), dict)
            for container in (participant, room)
            for key in ("shared_artifact_proposal", "shared_artifact_revision")
        )
        declined_or_deferred = any(
            str(container.get("shared_artifact_status") or "") in {"declined", "deferred", "unknown"}
            for container in (participant, room)
        )

        if any(token in prompt for token in _EPISTEMIC_CLARIFY_HINTS):
            return SocialEpistemicSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                claim_kind="clarification_needed",
                confidence_level="low",
                ambiguity_level="high",
                source_basis="explicit_peer_request",
                audience_scope=thread_routing.audience_scope,
                target_participant_id=thread_routing.target_participant_id,
                target_participant_name=thread_routing.target_participant_name,
                rationale="the peer explicitly asked for clarification, so clarity should come before confidence",
                reasons=["explicit clarification prompt"],
                metadata={"source": "social-turn-policy"},
            )

        if repair_signal is not None and repair_signal.confidence >= 0.55:
            return SocialEpistemicSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                claim_kind="clarification_needed",
                confidence_level="low",
                ambiguity_level="high",
                source_basis="repair_context",
                audience_scope=thread_routing.audience_scope,
                target_participant_id=thread_routing.target_participant_id,
                target_participant_name=thread_routing.target_participant_name,
                rationale="repair-active turns should narrow confidence until the thread and audience are stable again",
                reasons=[repair_signal.repair_type, repair_signal.rationale[:100]],
                metadata={"source": "social-turn-policy"},
            )

        if any(token in prompt for token in _EPISTEMIC_MEMORY_HINTS):
            if has_pending_artifact or declined_or_deferred:
                return SocialEpistemicSignalV1(
                    platform=message.platform,
                    room_id=message.room_id,
                    thread_key=thread_routing.thread_key,
                    claim_kind="proposal",
                    confidence_level="low",
                    ambiguity_level=ambiguity_level,  # type: ignore[arg-type]
                    source_basis="pending_artifact",
                    audience_scope=thread_routing.audience_scope,
                    target_participant_id=thread_routing.target_participant_id,
                    target_participant_name=thread_routing.target_participant_name,
                    rationale="pending or non-active artifact state should not be framed as accepted recall",
                    reasons=["artifact state is unresolved"],
                    metadata={"source": "social-turn-policy"},
                )
            if _SEALED_RE.search(prompt):
                return SocialEpistemicSignalV1(
                    platform=message.platform,
                    room_id=message.room_id,
                    thread_key=thread_routing.thread_key,
                    claim_kind="clarification_needed",
                    confidence_level="low",
                    ambiguity_level="medium",
                    source_basis="low_evidence",
                    audience_scope=thread_routing.audience_scope,
                    rationale="private or sealed language blocks confident recall framing",
                    reasons=["blocked/private language present"],
                    metadata={"source": "social-turn-policy"},
                )
            if evidence_count >= 2:
                return SocialEpistemicSignalV1(
                    platform=message.platform,
                    room_id=message.room_id,
                    thread_key=thread_routing.thread_key,
                    claim_kind="recall",
                    confidence_level=self._confidence_bucket(evidence_count, ambiguity_level=ambiguity_level),  # type: ignore[arg-type]
                    ambiguity_level=ambiguity_level,  # type: ignore[arg-type]
                    source_basis="social_memory",
                    audience_scope=thread_routing.audience_scope,
                    target_participant_id=thread_routing.target_participant_id,
                    target_participant_name=thread_routing.target_participant_name,
                    rationale="the peer explicitly asked for memory and social-memory evidence is available",
                    reasons=[f"evidence_count={evidence_count}", "memory request"],
                    metadata={"source": "social-turn-policy"},
                )
            return SocialEpistemicSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                claim_kind="speculation",
                confidence_level="low",
                ambiguity_level=ambiguity_level,  # type: ignore[arg-type]
                source_basis="low_evidence",
                audience_scope=thread_routing.audience_scope,
                target_participant_id=thread_routing.target_participant_id,
                target_participant_name=thread_routing.target_participant_name,
                rationale="the prompt asks for memory, but there is not enough evidence to present recall confidently",
                reasons=[f"evidence_count={evidence_count}", "memory request with thin evidence"],
                metadata={"source": "social-turn-policy"},
            )

        if any(token in prompt for token in _EPISTEMIC_SUMMARY_HINTS) or thread_routing.routing_decision == "summarize_room":
            return SocialEpistemicSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                claim_kind="summary",
                confidence_level="medium" if evidence_count < 2 else "high",
                ambiguity_level=ambiguity_level,  # type: ignore[arg-type]
                source_basis="active_thread" if room.get("active_threads") else "explicit_peer_request",
                audience_scope=thread_routing.audience_scope,
                target_participant_id=thread_routing.target_participant_id,
                target_participant_name=thread_routing.target_participant_name,
                rationale="the room is asking for a compact summary rather than a memory claim",
                reasons=["summary request", thread_routing.rationale[:100]],
                metadata={"source": "social-turn-policy"},
            )

        if thread_routing.ambiguity_level in {"medium", "high"} and (addressed or message.mentions_orion):
            return SocialEpistemicSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                claim_kind="clarification_needed",
                confidence_level="low",
                ambiguity_level="high",
                source_basis="active_thread",
                audience_scope=thread_routing.audience_scope,
                target_participant_id=thread_routing.target_participant_id,
                target_participant_name=thread_routing.target_participant_name,
                rationale="the thread/audience ambiguity is too high for a clean answer without clarification",
                reasons=["ambiguous multi-thread context"],
                metadata={"source": "social-turn-policy"},
            )

        if any(token in prompt for token in _EPISTEMIC_INFERENCE_HINTS):
            claim_kind = "inference" if evidence_count >= 2 else "speculation"
            source_basis = "social_memory" if evidence_count >= 2 else "low_evidence"
            confidence_level = "medium" if evidence_count >= 2 else "low"
            return SocialEpistemicSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                claim_kind=claim_kind,  # type: ignore[arg-type]
                confidence_level=confidence_level,  # type: ignore[arg-type]
                ambiguity_level=ambiguity_level,  # type: ignore[arg-type]
                source_basis=source_basis,  # type: ignore[arg-type]
                audience_scope=thread_routing.audience_scope,
                target_participant_id=thread_routing.target_participant_id,
                target_participant_name=thread_routing.target_participant_name,
                rationale="the prompt asks for a read or interpretation rather than strict recall",
                reasons=[f"evidence_count={evidence_count}", "interpretive prompt"],
                metadata={"source": "social-turn-policy"},
            )

        if any(token in prompt for token in _EPISTEMIC_SPECULATION_HINTS):
            return SocialEpistemicSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                claim_kind="speculation",
                confidence_level="low",
                ambiguity_level=ambiguity_level,  # type: ignore[arg-type]
                source_basis="recent_turns" if evidence_count else "low_evidence",
                audience_scope=thread_routing.audience_scope,
                target_participant_id=thread_routing.target_participant_id,
                target_participant_name=thread_routing.target_participant_name,
                rationale="the peer is inviting a tentative read rather than certain knowledge",
                reasons=["speculative language in the prompt"],
                metadata={"source": "social-turn-policy"},
            )

        return None

    def _epistemic_decision(
        self,
        *,
        message: CallSyneRoomMessageV1,
        thread_routing: SocialThreadRoutingDecisionV1,
        epistemic_signal: SocialEpistemicSignalV1 | None,
        addressed: bool,
    ) -> SocialEpistemicDecisionV1 | None:
        if epistemic_signal is None:
            return None
        decision = "defer_narrowly"
        rationale = "stay narrow when the epistemic basis is thin"
        if epistemic_signal.claim_kind == "recall":
            decision = "answer_recall"
            rationale = "the room asked for memory and there is enough evidence to answer as recall"
        elif epistemic_signal.claim_kind == "summary":
            decision = "answer_summary"
            rationale = "a summary is the cleanest epistemic frame for this request"
        elif epistemic_signal.claim_kind == "inference":
            decision = "answer_inference"
            rationale = "the safest answer is an inference grounded in the visible room context"
        elif epistemic_signal.claim_kind == "speculation":
            decision = "answer_speculation"
            rationale = "evidence is thin, so Orion should stay tentative rather than present recall as fact"
        elif epistemic_signal.claim_kind == "proposal":
            decision = "ask_clarifying_question" if addressed or message.mentions_orion else "defer_narrowly"
            rationale = "pending or unconfirmed artifact state should be clarified before it is treated as memory"
        elif epistemic_signal.claim_kind == "clarification_needed":
            decision = "ask_clarifying_question" if addressed or message.mentions_orion or thread_routing.routing_decision != "wait" else "defer_narrowly"
            rationale = "clarity is preferable to false certainty in this turn"
        return SocialEpistemicDecisionV1(
            platform=message.platform,
            room_id=message.room_id,
            thread_key=thread_routing.thread_key,
            signal_id=epistemic_signal.epistemic_id,
            claim_kind=epistemic_signal.claim_kind,
            decision=decision,  # type: ignore[arg-type]
            confidence_level=epistemic_signal.confidence_level,
            ambiguity_level=epistemic_signal.ambiguity_level,
            source_basis=epistemic_signal.source_basis,
            audience_scope=epistemic_signal.audience_scope,
            target_participant_id=epistemic_signal.target_participant_id,
            target_participant_name=epistemic_signal.target_participant_name,
            rationale=rationale,
            reasons=list(epistemic_signal.reasons) + [f"routing={thread_routing.routing_decision}"],
            metadata=dict(epistemic_signal.metadata or {}),
        )

    def _thread_routing(
        self,
        *,
        message: CallSyneRoomMessageV1,
        room: dict[str, Any],
        open_thread: SocialOpenThreadV1 | None,
        addressed: bool,
        mode: SocialAutonomyMode,
    ) -> SocialThreadRoutingDecisionV1:
        active_threads = [item for item in room.get("active_threads") or [] if isinstance(item, dict)]
        prompt = str(message.text or "").lower()
        exact_thread = None
        if message.thread_id:
            exact_thread = next((item for item in active_threads if item.get("thread_id") == message.thread_id), None)
        target_participant_id, target_participant_name = self._thread_target(message=message, matching_thread=exact_thread)
        target_is_orion = self._participant_is_orion(
            participant_id=target_participant_id,
            participant_name=target_participant_name,
        )
        ranked_threads = self._rank_threads(
            message=message,
            room=room,
            active_threads=active_threads,
            target_participant_id=target_participant_id,
            addressed=addressed,
        )
        matching_thread = ranked_threads[0][0] if ranked_threads else exact_thread
        relevant_commitments = self._relevant_commitments(
            room=room,
            message=message,
            matching_thread=matching_thread,
            target_participant_id=target_participant_id,
        )
        active_commitment = relevant_commitments[0] if relevant_commitments else None
        candidate_threads = [thread for thread, score in ranked_threads if score >= 0.18][:3]
        secondary_candidates = [
            str(thread.get("thread_summary") or "")[:180]
            for thread in candidate_threads[1:3]
            if str(thread.get("thread_summary") or "").strip()
        ]
        ambiguity_level = self._ambiguity_level(ranked_threads)
        audience_scope = "peer" if addressed or target_participant_id else "room"
        routing = "reply_to_peer" if audience_scope == "peer" else "reply_to_room"
        rationale = "defaulting to the most local reply audience"
        reason_tags: list[str] = []
        if any(token in prompt for token in ("summary", "summarize", "recap", "catch us up")):
            audience_scope = "summary"
            routing = "summarize_room"
            rationale = "the room is asking for a summary or handoff-aware recap"
            reason_tags.append("room_summary_preferred")
        elif active_commitment and str(active_commitment.get("commitment_type") or "") == "summarize_room" and not target_participant_id:
            audience_scope = "summary"
            routing = "summarize_room"
            rationale = "an open room-summary commitment is due soon and locally relevant"
            reason_tags.extend(["commitment_influenced_routing", "room_summary_preferred"])
        elif any(token in prompt for token in ("back to", "pick this back up", "again")):
            audience_scope = "thread"
            if self._revival_allowed(
                primary_thread=matching_thread,
                ranked_threads=ranked_threads,
                mode=mode,
            ):
                routing = "revive_thread"
                rationale = "the prompt is reviving a still-relevant open thread"
                reason_tags.append("revival_allowed")
                if active_commitment and str(active_commitment.get("commitment_type") or "") in {"return_to_thread", "answer_pending_question"}:
                    reason_tags.append("commitment_influenced_routing")
            else:
                routing = "wait"
                rationale = "revival was requested but the thread is stale, unresolved context is weak, or a fresher thread competes"
                reason_tags.extend(["revival_suppressed", "unclear_audience_wait"])
        elif active_commitment and str(active_commitment.get("commitment_type") or "") == "yield_then_reenter" and not addressed:
            audience_scope = "none"
            routing = "wait"
            rationale = "an active yield commitment is still locally relevant, so waiting is safer than forcing re-entry"
            reason_tags.extend(["commitment_influenced_routing", "unclear_audience_wait"])
        elif target_participant_id and not addressed and not target_is_orion:
            audience_scope = "peer"
            routing = "wait"
            rationale = "the turn appears to be directed at another peer, so Orion should stay out"
            reason_tags.append("peer_targeted_elsewhere")
        elif addressed or target_is_orion:
            audience_scope = "peer"
            routing = "reply_to_peer"
            rationale = "the most local relevant reply is to the peer-facing thread involving Orion"
            reason_tags.append("peer_reply_preferred")
        elif ambiguity_level in {"medium", "high"} and len(candidate_threads) > 1:
            audience_scope = "none"
            routing = "wait"
            rationale = "multiple active threads are plausible and the audience is unclear"
            reason_tags.extend(["ambiguous_multi_thread", "unclear_audience_wait"])
        elif self._room_summary_preferred(prompt=prompt):
            audience_scope = "summary"
            routing = "summarize_room"
            rationale = "the room context is asking for a brief state-of-the-room summary"
            reason_tags.append("room_summary_preferred")
        elif self._room_reply_preferred(prompt=prompt):
            audience_scope = "room"
            routing = "reply_to_room"
            rationale = "the room-facing thread is the clearest local context"
            reason_tags.append("room_reply_preferred")
        elif not addressed and not self._is_open_question(message) and open_thread is None:
            audience_scope = "none"
            routing = "wait"
            rationale = "the audience is ambiguous and no active thread clearly fits"
            reason_tags.append("unclear_audience_wait")
        if not reason_tags:
            if routing == "reply_to_peer":
                reason_tags.append("peer_reply_preferred")
            elif routing == "reply_to_room":
                reason_tags.append("room_reply_preferred")
        return SocialThreadRoutingDecisionV1(
            platform=message.platform,
            room_id=message.room_id,
            thread_key=(matching_thread or {}).get("thread_key") or (open_thread.topic_key if open_thread else None),
            audience_scope=audience_scope,  # type: ignore[arg-type]
            routing_decision=routing,  # type: ignore[arg-type]
            target_participant_id=target_participant_id if target_participant_id else None,
            target_participant_name=target_participant_name if target_participant_name else None,
            last_speaker=message.sender_name or message.sender_id,
            last_addressed_participant_id=target_participant_id,
            open_question=self._is_open_question(message),
            handoff_flag=bool("what do you think" in prompt or "over to you" in prompt or "summary" in prompt),
            thread_summary=str((matching_thread or {}).get("thread_summary") or (open_thread.summary if open_thread else message.text[:160]))[:180],
            primary_thread_key=str((matching_thread or {}).get("thread_key") or "") or None,
            primary_thread_summary=str((matching_thread or {}).get("thread_summary") or "")[:180],
            candidate_thread_summaries=secondary_candidates,
            ambiguity_level=ambiguity_level,  # type: ignore[arg-type]
            rationale=rationale,
            reasons=reason_tags + [rationale],
            metadata={
                "source": "social-turn-policy",
                "commitment_summary": str((active_commitment or {}).get("summary") or "")[:160],
                "commitment_due_state": str((active_commitment or {}).get("due_state") or ""),
            },
        )

    def _handoff_signal(
        self,
        *,
        message: CallSyneRoomMessageV1,
        room: dict[str, Any],
        thread_routing: SocialThreadRoutingDecisionV1,
        repair_decision: SocialRepairDecisionV1 | None = None,
    ) -> SocialHandoffSignalV1 | None:
        existing = room.get("handoff_signal") if isinstance(room.get("handoff_signal"), dict) else None
        prompt = str(message.text or "").lower()
        if repair_decision is not None and repair_decision.decision == "yield":
            return SocialHandoffSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                handoff_kind="yield_to_peer",
                audience_scope="peer",
                from_participant_id=message.sender_id,
                from_participant_name=message.sender_name,
                to_participant_id=repair_decision.target_participant_id,
                to_participant_name=repair_decision.target_participant_name,
                detected=True,
                rationale="repair handling yielded the exchange to another peer after a correction or redirect",
            )
        if (
            thread_routing.routing_decision == "wait"
            and thread_routing.audience_scope == "peer"
            and thread_routing.target_participant_id
            and not self._participant_is_orion(
                participant_id=thread_routing.target_participant_id,
                participant_name=thread_routing.target_participant_name,
            )
        ):
            return SocialHandoffSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                handoff_kind="yield_to_peer",
                audience_scope="peer",
                from_participant_id=message.sender_id,
                from_participant_name=message.sender_name,
                to_participant_id=thread_routing.target_participant_id,
                to_participant_name=thread_routing.target_participant_name,
                detected=True,
                rationale="the turn appears to hand the exchange to another peer rather than Orion",
            )
        if any(token in prompt for token in ("what do you think", "over to you", "your take")):
            return SocialHandoffSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                handoff_kind="to_orion",
                audience_scope=thread_routing.audience_scope,
                from_participant_id=message.sender_id,
                from_participant_name=message.sender_name,
                to_participant_id="orion",
                to_participant_name="Oríon",
                detected=True,
                rationale="the room explicitly handed the thread to Orion",
            )
        if thread_routing.routing_decision == "summarize_room":
            return SocialHandoffSignalV1(
                platform=message.platform,
                room_id=message.room_id,
                thread_key=thread_routing.thread_key,
                handoff_kind="room_summary",
                audience_scope="summary",
                from_participant_id=message.sender_id,
                from_participant_name=message.sender_name,
                detected=True,
                rationale="the room appears to be transitioning into summary mode",
            )
        if existing:
            return SocialHandoffSignalV1.model_validate(existing)
        return None

    def _thread_target(
        self,
        *,
        message: CallSyneRoomMessageV1,
        matching_thread: dict[str, Any] | None,
    ) -> tuple[str | None, str | None]:
        target_participant_id = (
            str(message.target_participant_id or message.reply_to_sender_id or "").strip() or None
        )
        target_participant_name = str(message.target_participant_name or "").strip() or None
        if target_participant_id or target_participant_name:
            return target_participant_id, target_participant_name
        if matching_thread:
            thread_target_id = str(matching_thread.get("target_participant_id") or "").strip() or None
            thread_target_name = str(matching_thread.get("target_participant_name") or "").strip() or None
            if thread_target_id or thread_target_name:
                return thread_target_id, thread_target_name
        if message.mentions_orion:
            return "orion", self.settings.social_bridge_self_name
        return None, None

    def _participant_is_orion(
        self,
        *,
        participant_id: str | None,
        participant_name: str | None,
    ) -> bool:
        if participant_id and participant_id in set(self.settings.social_bridge_self_participant_ids):
            return True
        if participant_id and participant_id.lower() == "orion":
            return True
        normalized_name = str(participant_name or "").strip().lower()
        normalized_self_name = self.settings.social_bridge_self_name.strip().lower()
        return bool(normalized_name and normalized_name in {normalized_self_name, "orion", "oríon"})

    def _derive_open_thread(self, *, message: CallSyneRoomMessageV1, room: dict[str, Any]) -> SocialOpenThreadV1 | None:
        open_threads = [str(item).strip() for item in room.get("open_threads") or [] if str(item).strip()]
        active_threads = [item for item in room.get("active_threads") or [] if isinstance(item, dict)]
        if not open_threads and not self._is_open_question(message):
            if not active_threads:
                return None
        summary = open_threads[0] if open_threads else str((active_threads[0] if active_threads else {}).get("thread_summary") or message.text[:160])
        topic_key = str((active_threads[0] if active_threads else {}).get("thread_key") or f"{message.platform}:{message.room_id}:{message.thread_id or summary.lower()[:48]}")
        return SocialOpenThreadV1(
            topic_key=topic_key,
            platform=message.platform,
            room_id=message.room_id,
            summary=summary[:200],
            last_speaker=message.sender_name or message.sender_id,
            open_question=self._is_open_question(message),
            orion_involved=bool(room.get("evidence_count")),
            evidence_refs=list(room.get("evidence_refs") or [])[:4],
            evidence_count=int(room.get("evidence_count") or 0),
        )

    def _rank_threads(
        self,
        *,
        message: CallSyneRoomMessageV1,
        room: dict[str, Any],
        active_threads: list[dict[str, Any]],
        target_participant_id: str | None,
        addressed: bool,
    ) -> list[tuple[dict[str, Any], float]]:
        current_thread_key = str(room.get("current_thread_key") or "").strip() or None
        message_tokens = _tokens(message.text)
        ranked: list[tuple[dict[str, Any], float]] = []
        for index, thread in enumerate(active_threads):
            score = 0.0
            if message.thread_id and str(thread.get("thread_id") or "").strip() == message.thread_id:
                score += 1.2
            if current_thread_key and str(thread.get("thread_key") or "") == current_thread_key:
                score += 0.18
            if target_participant_id and target_participant_id in {
                str(thread.get("target_participant_id") or "").strip(),
                str(thread.get("last_addressed_participant_id") or "").strip(),
            }:
                score += 0.45
            if addressed and self._participant_is_orion(
                participant_id=str(thread.get("target_participant_id") or "").strip() or None,
                participant_name=str(thread.get("target_participant_name") or "").strip() or None,
            ):
                score += 0.25
            if str(thread.get("last_speaker") or "").strip() == str(message.sender_name or message.sender_id):
                score += 0.12
            if self._is_open_question(message) and bool(thread.get("open_question")):
                score += 0.08
            overlap = _jaccard(message_tokens, _tokens(thread.get("thread_summary")))
            score += min(0.35, overlap * 0.6)
            score += self._thread_freshness(thread) * 0.22
            if bool(thread.get("handoff_flag")):
                score += 0.08
            ranked.append((thread, score - (index * 0.01)))
        return sorted(
            ranked,
            key=lambda item: (
                item[1],
                str(item[0].get("last_activity_at") or ""),
            ),
            reverse=True,
        )

    def _thread_freshness(self, thread: dict[str, Any]) -> float:
        expires_at = _parse_dt(thread.get("expires_at"))
        last_activity_at = _parse_dt(thread.get("last_activity_at"))
        now = datetime.now(timezone.utc)
        if expires_at is not None:
            remaining = (expires_at - now).total_seconds()
            if remaining <= 0:
                return 0.0
            return max(0.0, min(1.0, remaining / (6 * 3600)))
        if last_activity_at is None:
            return 0.2
        age_hours = max((now - last_activity_at).total_seconds(), 0.0) / 3600.0
        return max(0.0, min(1.0, 1.0 - (age_hours / 6.0)))

    def _ambiguity_level(self, ranked_threads: list[tuple[dict[str, Any], float]]) -> str:
        if len(ranked_threads) < 2:
            return "low"
        primary_score = ranked_threads[0][1]
        secondary_score = ranked_threads[1][1]
        if primary_score <= 0.22:
            return "high"
        if secondary_score >= max(primary_score - 0.10, 0.22):
            return "high"
        if secondary_score >= max(primary_score - 0.22, 0.12):
            return "medium"
        return "low"

    def _room_summary_preferred(self, *, prompt: str) -> bool:
        return any(token in prompt for token in ("summary", "summarize", "recap", "catch us up", "where are we"))

    def _room_reply_preferred(self, *, prompt: str) -> bool:
        return any(token in prompt for token in ("everyone", "room", "all of you", "anyone"))

    def _relevant_commitments(
        self,
        *,
        room: dict[str, Any],
        message: CallSyneRoomMessageV1,
        matching_thread: dict[str, Any] | None,
        target_participant_id: str | None,
    ) -> list[dict[str, Any]]:
        commitments = [item for item in room.get("active_commitments") or [] if isinstance(item, dict)]
        prompt = str(message.text or "").lower()
        ranked: list[tuple[float, dict[str, Any]]] = []
        for commitment in commitments:
            if str(commitment.get("state") or "open") != "open":
                continue
            score = 0.0
            if matching_thread and commitment.get("thread_key") == matching_thread.get("thread_key"):
                score += 0.45
            if target_participant_id and commitment.get("target_participant_id") == target_participant_id:
                score += 0.35
            if str(commitment.get("commitment_type") or "") == "summarize_room" and any(
                token in prompt for token in ("summary", "recap", "where are we", "what do you think")
            ):
                score += 0.4
            if str(commitment.get("commitment_type") or "") in {"return_to_thread", "answer_pending_question"} and any(
                token in prompt for token in ("back to", "again", "?")
            ):
                score += 0.3
            if str(commitment.get("due_state") or "") == "due_soon":
                score += 0.12
            if str(commitment.get("due_state") or "") == "stale":
                score -= 0.08
            ranked.append((score, commitment))
        return [item for _, item in sorted(ranked, key=lambda pair: pair[0], reverse=True) if _ > 0.08][:2]

    def _revival_allowed(
        self,
        *,
        primary_thread: dict[str, Any] | None,
        ranked_threads: list[tuple[dict[str, Any], float]],
        mode: SocialAutonomyMode,
    ) -> bool:
        if primary_thread is None:
            return False
        if mode != "light_initiative":
            return False
        if self._thread_freshness(primary_thread) < 0.18:
            return False
        if not (
            bool(primary_thread.get("orion_involved"))
            or bool(primary_thread.get("open_question"))
            or bool(primary_thread.get("handoff_flag"))
        ):
            return False
        if len(ranked_threads) > 1:
            primary_score = ranked_threads[0][1]
            secondary_score = ranked_threads[1][1]
            if secondary_score >= max(primary_score - 0.08, 0.24):
                return False
        return True

    def _novelty_score(self, *, message: CallSyneRoomMessageV1, participant: dict[str, Any], room: dict[str, Any]) -> float:
        message_tokens = _tokens(message.text)
        if not message_tokens:
            return 0.0
        reference_sets = [
            _tokens(participant.get("safe_continuity_summary")),
            _tokens(participant.get("interaction_tone_summary")),
            _tokens(room.get("recent_thread_summary")),
            _tokens(room.get("room_tone_summary")),
        ]
        reference_sets.extend(_tokens(str(item)) for item in room.get("open_threads") or [])
        overlap = max((_jaccard(message_tokens, ref_tokens) for ref_tokens in reference_sets), default=0.0)
        novelty = max(0.0, min(1.0, 1.0 - overlap))
        if self._is_open_question(message):
            novelty = min(1.0, novelty + 0.08)
        return novelty

    def _continuity_score(
        self,
        *,
        message: CallSyneRoomMessageV1,
        room: dict[str, Any],
        open_thread: SocialOpenThreadV1 | None,
    ) -> float:
        score = 0.0
        if message.thread_id:
            score += 0.35
        if open_thread is not None:
            score += 0.25
        if room.get("current_thread_key"):
            score += 0.15
        if room.get("evidence_count"):
            score += min(float(room.get("evidence_count")) * 0.05, 0.2)
        if room.get("active_participants"):
            score += 0.1
        return max(0.0, min(1.0, score))

    def _quiet_room(self, *, room: dict[str, Any]) -> bool:
        evidence_count = int(room.get("evidence_count") or 0)
        participant_count = len(room.get("active_participants") or [])
        return evidence_count <= 1 and participant_count <= 1

    def _is_open_question(self, message: CallSyneRoomMessageV1) -> bool:
        text = str(message.text or "")
        return "?" in text or bool(message.metadata.get("open_question"))
