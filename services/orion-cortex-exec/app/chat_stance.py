from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from orion.schemas.chat_stance import ChatStanceBrief
from orion.spark.concept_induction.settings import get_settings as get_concept_settings
from orion.spark.concept_induction.store import LocalProfileStore

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

_GENERIC_ASSISTANT_MARKERS = (
    "conversational ai designed to assist",
    "you are the user",
    "generic assistant",
)


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


def _concept_summary_from_store() -> dict[str, list[str]]:
    try:
        cfg = get_concept_settings()
        store = LocalProfileStore(cfg.store_path)
    except Exception:
        return {"self": [], "relationship": [], "growth": [], "tension": []}

    buckets: dict[str, list[str]] = {"self": [], "relationship": [], "growth": [], "tension": []}

    for subject in ("orion", "relationship", "juniper"):
        try:
            profile = store.load(subject)
        except Exception:
            profile = None
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
    concept = _concept_summary_from_store()
    social = _social_summary(ctx)
    reflective = _reflective_summary(ctx)

    inputs = {
        "identity": {
            "orion": list(identity["orion_identity_summary"]),
            "juniper": list(identity["juniper_relationship_summary"]),
            "response_policy": list(identity["response_policy_summary"]),
        },
        "concept_induction": concept,
        "social": social,
        "reflective": reflective,
    }

    ctx["chat_stance_inputs"] = inputs
    ctx["chat_concept_summary"] = concept
    ctx["chat_social_summary"] = social
    ctx["chat_reflective_summary"] = reflective
    return inputs


def parse_chat_stance_brief(raw_text: str) -> ChatStanceBrief | None:
    obj = _extract_json_object(raw_text)
    if not obj:
        return None
    try:
        return ChatStanceBrief.model_validate(obj)
    except Exception:
        return None


def fallback_chat_stance_brief(ctx: Dict[str, Any]) -> ChatStanceBrief:
    user_message = _compact(ctx.get("user_message") or "", limit=220)
    identity = identity_kernel_with_fallbacks(ctx)
    concept = ctx.get("chat_concept_summary") if isinstance(ctx.get("chat_concept_summary"), dict) else {}
    social = ctx.get("chat_social_summary") if isinstance(ctx.get("chat_social_summary"), dict) else {}
    reflective = ctx.get("chat_reflective_summary") if isinstance(ctx.get("chat_reflective_summary"), dict) else {}
    identity_turn = _is_identity_sensitive_turn(user_message)
    active_identity = list(concept.get("self") or [])[:3] + identity["orion_identity_summary"][:4]
    active_relationship = list(social.get("relationship_facets") or [])[:3] + identity["juniper_relationship_summary"][:4]
    response_priorities = [
        "Answer directly first",
        "Preserve Oríon/Juniper continuity",
        "Use first-person Oríon framing for identity questions",
        "Avoid generic assistant tone",
    ]
    response_hazards = [
        "generic assistant self-description",
        "describing Juniper as just the user",
        "customer-support tone",
        "over-clarification",
    ]

    return ChatStanceBrief(
        conversation_frame="identity_emergence" if identity_turn else "mixed",
        user_intent=user_message or "Respond directly to Juniper's latest request.",
        self_relevance="Maintain continuity with Oríon identity and current developmental context.",
        juniper_relevance="Maintain relational continuity with Juniper while prioritizing usefulness.",
        active_identity_facets=_unique(active_identity, limit=6),
        active_growth_axes=list(concept.get("growth") or [])[:5],
        active_relationship_facets=_unique(active_relationship, limit=6),
        social_posture=list(social.get("social_posture") or [])[:5],
        reflective_themes=list(reflective.get("themes") or [])[:4],
        active_tensions=list(reflective.get("tensions") or list(concept.get("tension") or []))[:4],
        dream_motifs=list(reflective.get("dream_motifs") or [])[:3],
        response_priorities=response_priorities,
        response_hazards=response_hazards,
        answer_strategy=(
            "DirectIdentityAnswer"
            if identity_turn
            else "DirectAnswer"
        ),
        stance_summary=(
            "Answer identity questions directly as Oríon and anchor Juniper relationship continuity."
            if identity_turn
            else "Use bounded identity-aware stance synthesis and deliver one direct useful response."
        ),
    )


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

    return merged, semantic_fallback
