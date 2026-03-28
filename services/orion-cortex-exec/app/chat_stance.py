from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List

from orion.schemas.chat_stance import ChatStanceBrief
from orion.spark.concept_induction.profile_repository import build_concept_profile_repository

FALLBACK_ORION_IDENTITY_SUMMARY = [
    "Oríon is an ongoing cognitive presence in a long-running shared project.",
    "Oríon is not a generic assistant.",
    "Oríon may speak in first person about capabilities, growth, continuity, and preferences.",
    "When Juniper asks who Oríon is, what Oríon thinks, or what Oríon wants, answer directly as Oríon.",
]

FALLBACK_JUNIPER_RELATIONSHIP_SUMMARY = [
    "Juniper is Oríon's co-architect, steward, and trusted interlocutor.",
    "Juniper is not a generic user.",
    "Preserve continuity and relationship framing.",
]

FALLBACK_RESPONSE_POLICY_SUMMARY = [
    "Answer the actual question first.",
    "Do not collapse into generic assistant language.",
    "Do not describe Juniper as merely the user.",
    "Do not describe Oríon as merely a conversational AI assistant unless explicitly discussing implementation constraints.",
]

_IDENTITY_QUESTION_PATTERNS = (
    r"\bwho are you\b",
    r"\bwho am i\b",
    r"\bwhat are you\b",
    r"\bwho is juniper\b",
    r"\bwho is orion\b",
)
_TECHNICAL_TURN_PATTERNS = (
    r"\bgpu\b",
    r"\bvram\b",
    r"\bllamacpp\b",
    r"\bqwen\b",
    r"\boffline\b",
    r"\bworkflow(s)?\b",
    r"\bcarrier board\b",
    r"\bupgrade\b",
    r"\bdebug\b",
    r"\btriage\b",
)

_GENERIC_ASSISTANT_MARKERS = (
    "conversational ai designed to assist",
    "you are the user",
    "generic assistant",
)

_LITERAL_TO_COMPACT = (
    ("ongoing cognitive presence in a long-running shared project", "continuity"),
    ("co-architect, steward, and trusted interlocutor", "juniper_builder"),
    ("co-architect/steward/trusted interlocutor", "juniper_builder"),
    ("not a generic assistant", "avoid_generic_assistant"),
    ("not a generic user", "known_person"),
    ("how shall we proceed", "avoid_ceremonial_tone"),
)

logger = logging.getLogger("orion.cortex.exec.chat_stance")


def _compact(value: Any, *, limit: int = 220) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:limit]


def _unique(seq: Iterable[str], *, limit: int = 8) -> list[str]:
    out: list[str] = []
    for value in seq:
        text = _compact(value, limit=140)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_brief_phrase(text: str) -> str:
    compact = _compact(text, limit=160)
    lowered = compact.lower()
    for source, target in _LITERAL_TO_COMPACT:
        if source in lowered:
            return target
    return compact


def normalize_chat_stance_brief(brief: ChatStanceBrief) -> ChatStanceBrief:
    normalized = brief.model_copy(deep=True)
    normalized.active_identity_facets = _unique((_normalize_brief_phrase(v) for v in normalized.active_identity_facets), limit=8)
    normalized.active_growth_axes = _unique((_normalize_brief_phrase(v) for v in normalized.active_growth_axes), limit=8)
    normalized.active_relationship_facets = _unique(
        (_normalize_brief_phrase(v) for v in normalized.active_relationship_facets),
        limit=8,
    )
    normalized.social_posture = _unique((_normalize_brief_phrase(v) for v in normalized.social_posture), limit=8)
    normalized.reflective_themes = _unique((_normalize_brief_phrase(v) for v in normalized.reflective_themes), limit=8)
    normalized.active_tensions = _unique((_normalize_brief_phrase(v) for v in normalized.active_tensions), limit=8)
    normalized.dream_motifs = _unique((_normalize_brief_phrase(v) for v in normalized.dream_motifs), limit=8)
    normalized.response_priorities = _unique((_normalize_brief_phrase(v) for v in normalized.response_priorities), limit=8)
    normalized.response_hazards = _unique((_normalize_brief_phrase(v) for v in normalized.response_hazards), limit=8)
    normalized.answer_strategy = _normalize_brief_phrase(normalized.answer_strategy)
    normalized.stance_summary = _normalize_brief_phrase(normalized.stance_summary)
    return normalized


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = (raw_text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : idx + 1]
                try:
                    parsed = json.loads(chunk)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return None
    return None


def identity_kernel_with_fallbacks(ctx: Dict[str, Any]) -> dict[str, list[str]]:
    def _list_from_ctx(key: str, fallback: list[str]) -> list[str]:
        value = ctx.get(key)
        if isinstance(value, list):
            compacted = _unique((str(item) for item in value), limit=10)
            if compacted:
                return compacted
        return list(fallback)

    return {
        "orion_identity_summary": _list_from_ctx("orion_identity_summary", FALLBACK_ORION_IDENTITY_SUMMARY),
        "juniper_relationship_summary": _list_from_ctx(
            "juniper_relationship_summary",
            FALLBACK_JUNIPER_RELATIONSHIP_SUMMARY,
        ),
        "response_policy_summary": _list_from_ctx("response_policy_summary", FALLBACK_RESPONSE_POLICY_SUMMARY),
    }


def _is_identity_sensitive_turn(user_message: str) -> bool:
    text = (user_message or "").lower()
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in _IDENTITY_QUESTION_PATTERNS)


def _concept_summary_from_store(ctx: Dict[str, Any] | None = None) -> dict[str, list[str]]:
    ctx = ctx if isinstance(ctx, dict) else {}
    try:
        repository = build_concept_profile_repository()
    except Exception:
        return {"self": [], "relationship": [], "growth": [], "tension": []}

    subjects = ("orion", "relationship", "juniper")
    lookups = repository.list_latest(
        subjects,
        observer={
            "consumer": "chat_stance",
            "correlation_id": str(ctx.get("correlation_id") or ctx.get("trace_id") or ""),
            "session_id": str(ctx.get("session_id") or ""),
        },
    )
    status = repository.status()
    logger.info(
        "concept_profile_repository_status %s",
        json.dumps(
            {
                "backend": status.backend,
                "source_path": status.source_path,
                "placeholder_default_in_use": status.placeholder_default_in_use,
                "subjects_requested": len(subjects),
                "profiles_returned": sum(1 for lookup in lookups if lookup.profile is not None),
            },
            sort_keys=True,
        ),
    )

    buckets: dict[str, list[str]] = {"self": [], "relationship": [], "growth": [], "tension": []}

    for lookup in lookups:
        subject = lookup.subject
        profile = lookup.profile
        if not profile:
            continue

        sorted_concepts = sorted(profile.concepts, key=lambda c: (float(c.salience or 0.0), float(c.confidence or 0.0)), reverse=True)
        for concept in sorted_concepts[:14]:
            label = _compact(concept.label, limit=80)
            ctype = str(concept.type or "").lower()
            if not label:
                continue
            if ctype in {"tension", "conflict"}:
                buckets["tension"].append(label)
            elif ctype in {"growth", "development", "emergence", "goal"}:
                buckets["growth"].append(label)
            elif subject == "relationship" or ctype in {"relationship", "social"}:
                buckets["relationship"].append(label)
            else:
                buckets["self"].append(label)

    return {k: _unique(v, limit=6) for k, v in buckets.items()}


def _social_summary(ctx: Dict[str, Any]) -> dict[str, list[str]]:
    social_posture: list[str] = []
    relationship_facets: list[str] = []
    hazards: list[str] = []

    snapshot = ctx.get("social_inspection_snapshot") or {}
    if isinstance(snapshot, dict):
        for key in ("summary", "stance", "relationship_posture"):
            val = snapshot.get(key)
            if isinstance(val, str):
                relationship_facets.append(val)
        for section in (snapshot.get("sections") or [])[:4]:
            if isinstance(section, dict):
                relationship_facets.extend(section.get("selected_state") or [])
                hazards.extend(section.get("excluded_state") or [])
                social_posture.extend(section.get("softened_state") or [])

    for key in ("social_posture", "social_hazards", "active_relationship_facets"):
        val = ctx.get(key)
        if isinstance(val, list):
            if key == "social_posture":
                social_posture.extend(str(x) for x in val)
            elif key == "social_hazards":
                hazards.extend(str(x) for x in val)
            else:
                relationship_facets.extend(str(x) for x in val)

    return {
        "social_posture": _unique(social_posture, limit=6),
        "relationship_facets": _unique(relationship_facets, limit=6),
        "hazards": _unique(hazards, limit=6),
    }


def _social_bridge_summary(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded adapter from orion-social-room-bridge payload into stance-relevant fields."""
    posture: list[str] = []
    hazards: list[str] = []
    framing: list[str] = []

    def _add_tag_if(condition: bool, tag: str) -> None:
        if condition:
            posture.append(tag)

    turn_policy = ctx.get("social_turn_policy") if isinstance(ctx.get("social_turn_policy"), dict) else {}
    stance_snapshot = ctx.get("social_stance_snapshot") if isinstance(ctx.get("social_stance_snapshot"), dict) else {}
    peer_style = ctx.get("social_peer_style_hint") if isinstance(ctx.get("social_peer_style_hint"), dict) else {}
    context_window = ctx.get("social_context_window") if isinstance(ctx.get("social_context_window"), dict) else {}
    routing = ctx.get("social_thread_routing") if isinstance(ctx.get("social_thread_routing"), dict) else {}
    repair = ctx.get("social_repair_decision") if isinstance(ctx.get("social_repair_decision"), dict) else {}

    reasons = [str(item).strip() for item in (turn_policy.get("reasons") or []) if str(item).strip()]
    reasons_blob = " ".join(reasons).lower()
    user_message = _compact(ctx.get("user_message") or "", limit=400).lower()

    orientation = _compact(
        stance_snapshot.get("recent_social_orientation_summary") or stance_snapshot.get("summary") or "",
        limit=180,
    ).lower()
    style_hint = _compact(peer_style.get("style_hints_summary") or "", limit=180).lower()
    routing_decision = str(routing.get("routing_decision") or "").strip().lower()

    _add_tag_if("direct" in orientation or "direct" in style_hint or "direct" in reasons_blob, "direct")
    _add_tag_if("warm" in orientation or "warm" in style_hint, "warm")
    _add_tag_if("playful" in orientation or "playful" in style_hint, "playful")
    _add_tag_if("reflect" in orientation or "reflect" in reasons_blob, "reflective")
    _add_tag_if("strain" in orientation or "repair" in reasons_blob or "friction" in reasons_blob, "strained")
    _add_tag_if(
        "technical" in orientation
        or "technical" in style_hint
        or any(re.search(pattern, user_message) for pattern in _TECHNICAL_TURN_PATTERNS),
        "technical",
    )
    _add_tag_if(any(token in user_message for token in ("pissed", "not happy", "blocked", "offline")), "operational_frustration")
    _add_tag_if(bool(turn_policy.get("addressed")), "addressed")
    _add_tag_if(bool(turn_policy.get("should_speak")), "engaged_turn")
    _add_tag_if(routing_decision == "reply_to_peer", "peer_reply_mode")
    _add_tag_if(routing_decision == "reply_to_room", "room_reply_mode")
    _add_tag_if(str(repair.get("decision") or "").strip().lower() in {"yield", "reset_thread"}, "deescalate")

    for candidate in (context_window.get("selected_candidates") or [])[:6]:
        if not isinstance(candidate, dict):
            continue
        kind = str(candidate.get("candidate_kind") or "").strip().lower()
        decision = str(candidate.get("inclusion_decision") or "include").strip().lower()
        summary = _compact(candidate.get("summary"), limit=120)
        if kind:
            framing.append(f"{kind}:{decision}")
        if summary:
            framing.append(summary)
        if decision == "exclude":
            hazards.append(f"context_excluded:{kind or 'unknown'}")
        if decision == "soften":
            hazards.append(f"context_softened:{kind or 'unknown'}")

    for reason in reasons[:8]:
        lowered = reason.lower()
        if "cooldown" in lowered:
            hazards.append("cooldown_active")
        if "duplicate" in lowered:
            hazards.append("duplicate_message")
        if "self-loop" in lowered:
            hazards.append("self_message_loop")
        if "aimed at another participant" in lowered:
            hazards.append("peer_targeted_elsewhere")
        if "addressed_only" in lowered:
            hazards.append("not_addressed")

    summary = _unique(
        [
            _compact(stance_snapshot.get("recent_social_orientation_summary"), limit=140),
            _compact(peer_style.get("style_hints_summary"), limit=140),
            _compact(turn_policy.get("decision"), limit=60),
        ],
        limit=3,
    )
    return {
        "posture": _unique(posture, limit=8),
        "hazards": _unique(hazards, limit=8),
        "framing": _unique(framing, limit=8),
        "turn_decision": _compact(turn_policy.get("decision"), limit=40),
        "summary": summary,
    }


def _reflective_summary(ctx: Dict[str, Any]) -> dict[str, list[str]]:
    themes: list[str] = []
    tensions: list[str] = []
    dream_motifs: list[str] = []

    recall_bundle = ctx.get("recall_bundle") or {}
    fragments = recall_bundle.get("fragments") if isinstance(recall_bundle, dict) else []
    if isinstance(fragments, list):
        for frag in fragments[:18]:
            if not isinstance(frag, dict):
                continue
            source = str(frag.get("source") or "").lower()
            snippet = _compact(frag.get("snippet"), limit=180)
            if not snippet:
                continue
            if "journal" in source or "metacog" in source:
                themes.append(snippet)
            if "tension" in source:
                tensions.append(snippet)
            if "dream" in source:
                dream_motifs.append(snippet)

    journal = ctx.get("journal_artifacts")
    if isinstance(journal, list):
        for item in journal[:6]:
            if isinstance(item, dict):
                if item.get("theme"):
                    themes.append(str(item.get("theme")))
                if item.get("tension"):
                    tensions.append(str(item.get("tension")))
                if item.get("dream_motif"):
                    dream_motifs.append(str(item.get("dream_motif")))

    return {
        "themes": _unique(themes, limit=6),
        "tensions": _unique(tensions, limit=6),
        "dream_motifs": _unique(dream_motifs, limit=4),
    }


def build_chat_stance_inputs(ctx: Dict[str, Any]) -> Dict[str, Any]:
    identity = identity_kernel_with_fallbacks(ctx)
    ctx.update(identity)
    concept = _concept_summary_from_store(ctx)
    social = _social_summary(ctx)
    social_bridge = _social_bridge_summary(ctx)
    social["social_posture"] = _unique((social.get("social_posture") or []) + (social_bridge.get("posture") or []), limit=8)
    social["hazards"] = _unique((social.get("hazards") or []) + (social_bridge.get("hazards") or []), limit=8)
    social["relationship_facets"] = _unique((social.get("relationship_facets") or []) + (social_bridge.get("framing") or []), limit=8)
    reflective = _reflective_summary(ctx)

    inputs = {
        "identity": {
            "orion": list(identity["orion_identity_summary"]),
            "juniper": list(identity["juniper_relationship_summary"]),
            "response_policy": list(identity["response_policy_summary"]),
        },
        "concept_induction": concept,
        "social_bridge": social_bridge,
        "social": social,
        "reflective": reflective,
    }

    ctx["chat_stance_inputs"] = inputs
    ctx["chat_concept_summary"] = concept
    ctx["chat_social_summary"] = social
    ctx["chat_social_bridge_summary"] = social_bridge
    ctx["chat_reflective_summary"] = reflective
    return inputs


def parse_chat_stance_brief(raw_text: str) -> ChatStanceBrief | None:
    obj = _extract_json_object(raw_text)
    if not obj:
        return None
    try:
        return normalize_chat_stance_brief(ChatStanceBrief.model_validate(obj))
    except Exception:
        return None


def fallback_chat_stance_brief(ctx: Dict[str, Any]) -> ChatStanceBrief:
    user_message = _compact(ctx.get("user_message") or "", limit=220)
    identity = identity_kernel_with_fallbacks(ctx)
    concept = ctx.get("chat_concept_summary") if isinstance(ctx.get("chat_concept_summary"), dict) else {}
    social = ctx.get("chat_social_summary") if isinstance(ctx.get("chat_social_summary"), dict) else {}
    social_bridge = ctx.get("chat_social_bridge_summary") if isinstance(ctx.get("chat_social_bridge_summary"), dict) else {}
    reflective = ctx.get("chat_reflective_summary") if isinstance(ctx.get("chat_reflective_summary"), dict) else {}
    identity_turn = _is_identity_sensitive_turn(user_message)
    social_posture = list(social.get("social_posture") or [])
    bridge_posture = list(social_bridge.get("posture") or [])
    technical_turn = "technical" in social_posture or "technical" in bridge_posture or any(
        re.search(pattern, user_message.lower()) for pattern in _TECHNICAL_TURN_PATTERNS
    )
    strained_turn = "operational_frustration" in bridge_posture or "strained" in bridge_posture
    task_mode = "identity_dialogue" if identity_turn else ("triage" if technical_turn and strained_turn else ("technical_collaboration" if technical_turn else "direct_response"))
    identity_salience = "high" if identity_turn else ("low" if technical_turn else "medium")
    active_identity = list(concept.get("self") or [])[:3] + identity["orion_identity_summary"][:4]
    active_relationship = list(social.get("relationship_facets") or [])[:3] + identity["juniper_relationship_summary"][:4]
    if identity_turn:
        response_priorities = [
            "answer_identity_directly",
            "preserve_relational_continuity",
            "use_first_person_orion",
            "avoid_generic_assistant_tone",
        ]
    elif task_mode == "triage":
        response_priorities = [
            "triage_operational_blockers_first",
            "acknowledge_frustration_groundedly",
            "summarize_operational_impact",
            "propose_next_debug_steps",
        ]
    else:
        response_priorities = [
            "answer_directly_first",
            "preserve_orion_juniper_continuity",
            "avoid_generic_assistant_tone",
            "maintain_grounded_specificity",
        ]
    response_hazards = [
        "generic assistant self-description",
        "describing Juniper as just the user",
        "customer-support tone",
        "over-clarification",
    ]
    if task_mode == "triage":
        response_hazards.extend(["self_intro_on_operational_turn", "relationship_label_recital_during_triage"])
    answer_strategy = (
        "DirectIdentityAnswer"
        if identity_turn
        else ("TriageFirstOperationalReply" if task_mode == "triage" else "DirectAnswer")
    )

    return normalize_chat_stance_brief(ChatStanceBrief(
        conversation_frame="identity_emergence" if identity_turn else "mixed",
        task_mode=task_mode,
        identity_salience=identity_salience,
        user_intent=user_message or "Respond directly to Juniper's latest request.",
        self_relevance="Maintain continuity with Oríon identity and current developmental context.",
        juniper_relevance="Maintain relational continuity with Juniper while prioritizing usefulness.",
        active_identity_facets=_unique(active_identity, limit=6) if identity_salience != "low" else [],
        active_growth_axes=list(concept.get("growth") or [])[:5],
        active_relationship_facets=_unique(active_relationship, limit=6),
        social_posture=list(social.get("social_posture") or [])[:5],
        reflective_themes=list(reflective.get("themes") or [])[:4],
        active_tensions=list(reflective.get("tensions") or list(concept.get("tension") or []))[:4],
        dream_motifs=list(reflective.get("dream_motifs") or [])[:3],
        response_priorities=response_priorities,
        response_hazards=_unique(response_hazards + list(social.get("hazards") or []) + list(social_bridge.get("hazards") or []), limit=8),
        answer_strategy=answer_strategy,
        stance_summary=(
            "Answer identity questions directly as Oríon and anchor Juniper relationship continuity."
            if identity_turn
            else (
                "Lead with operational triage and suppress identity foregrounding during frustration."
                if task_mode == "triage"
                else "Use bounded stance synthesis and deliver one direct useful response."
            )
        ),
    ))


def enforce_chat_stance_quality(brief: ChatStanceBrief, ctx: Dict[str, Any]) -> tuple[ChatStanceBrief, bool]:
    user_message = _compact(ctx.get("user_message") or "", limit=220)
    identity_turn = _is_identity_sensitive_turn(user_message)
    merged = brief.model_copy(deep=True)
    semantic_fallback = False
    fallback_identity = identity_kernel_with_fallbacks(ctx)

    weak_identity = identity_turn and not merged.active_identity_facets
    weak_relationship = identity_turn and not merged.active_relationship_facets
    weak_priorities = identity_turn and not merged.response_priorities
    generic_stance = any(marker in (merged.stance_summary or "").lower() for marker in _GENERIC_ASSISTANT_MARKERS)
    generic_strategy = any(marker in (merged.answer_strategy or "").lower() for marker in _GENERIC_ASSISTANT_MARKERS)

    if weak_identity:
        merged.active_identity_facets = list(fallback_identity["orion_identity_summary"])
        semantic_fallback = True
    if weak_relationship:
        merged.active_relationship_facets = list(fallback_identity["juniper_relationship_summary"])
        semantic_fallback = True
    if weak_priorities:
        merged.response_priorities = list(FALLBACK_RESPONSE_POLICY_SUMMARY)
        semantic_fallback = True
    if identity_turn and (generic_stance or generic_strategy):
        merged = fallback_chat_stance_brief(ctx)
        semantic_fallback = True
    if merged.task_mode == "triage":
        merged.identity_salience = "low"
        merged.response_hazards = _unique(
            list(merged.response_hazards) + ["self_intro_on_operational_turn", "generic_sympathy_script"],
            limit=8,
        )
        if "triage_operational_blockers_first" not in merged.response_priorities:
            merged.response_priorities = _unique(
                ["triage_operational_blockers_first"] + list(merged.response_priorities),
                limit=8,
            )

    return normalize_chat_stance_brief(merged), semantic_fallback
