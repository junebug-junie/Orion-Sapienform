from __future__ import annotations

import re
from typing import Any

from orion.memory.recall_skip_gate import RecallSkipGateResult

_ENTITY_QUERY_RE = re.compile(
    r"[A-Z][A-Za-z0-9_]+(?:\s+[A-Z][A-Za-z0-9_]+)?"
)

_RELATIONAL_TASK_MODES = frozenset({"reflective_dialogue", "playful_exchange"})
_RELATIONAL_CONVERSATION_FRAMES = frozenset({"reflective", "playful_relational"})
_PLANNING_LIKE_MARKERS = ("plan", "step", "procedure", "roadmap", "debug_step")


def _stance_field(stance_brief: Any, key: str) -> str:
    if isinstance(stance_brief, dict):
        return str(stance_brief.get(key) or "").strip()
    return str(getattr(stance_brief, key, None) or "").strip()


def _response_priorities(stance_brief: Any) -> list[Any]:
    if isinstance(stance_brief, dict):
        priorities = stance_brief.get("response_priorities")
    else:
        priorities = getattr(stance_brief, "response_priorities", None)
    return priorities if isinstance(priorities, list) else []


def _open_loops(attention_frame: dict | None) -> list[Any]:
    if not isinstance(attention_frame, dict):
        return []
    loops = attention_frame.get("open_loops")
    return loops if isinstance(loops, list) else []


def _coerce_novelty(appraisal: dict | None) -> float | None:
    raw_novelty = (appraisal or {}).get("novelty_score")
    if isinstance(raw_novelty, (int, float)):
        return float(raw_novelty)
    return None


def _shift_kind(appraisal: dict | None) -> str:
    return str((appraisal or {}).get("shift_kind") or "NONE").upper()


def _novelty_meets_floor(appraisal: dict | None, floor: float) -> bool:
    novelty_score = _coerce_novelty(appraisal)
    return novelty_score is not None and novelty_score >= floor


def _is_relational_mode(stance_brief: Any) -> bool:
    task_mode = _stance_field(stance_brief, "task_mode")
    conversation_frame = _stance_field(stance_brief, "conversation_frame")
    return task_mode in _RELATIONAL_TASK_MODES or conversation_frame in _RELATIONAL_CONVERSATION_FRAMES


def _is_instrumental_mode(stance_brief: Any) -> bool:
    task_mode = _stance_field(stance_brief, "task_mode")
    interaction_regime = _stance_field(stance_brief, "interaction_regime")
    return task_mode == "instrumental" or interaction_regime == "instrumental"


def _has_entity_query(user_message: str) -> bool:
    """Capitalized entity / anchor token in user message (existing recall anchor pattern)."""
    text = str(user_message or "").strip()
    if not text:
        return False
    entities = [m.strip() for m in _ENTITY_QUERY_RE.findall(text) if m.strip()]
    if entities:
        return True
    return bool(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\d+\b", text))


def _has_contradiction_seed(
    *,
    seed_crystallization_id: str | None,
    attention_frame: dict | None,
) -> bool:
    if str(seed_crystallization_id or "").strip():
        return True
    if not isinstance(attention_frame, dict):
        return False
    for key in ("contradiction_refs", "contradiction_crystallization_ids"):
        refs = attention_frame.get(key)
        if isinstance(refs, list) and refs:
            return True
    return False


def _has_planning_like_priority(stance_brief: Any) -> bool:
    if _stance_field(stance_brief, "conversation_frame") == "planning":
        return True
    for priority in _response_priorities(stance_brief):
        normalized = str(priority).lower().replace("-", "_").replace(" ", "_")
        if any(marker in normalized for marker in _PLANNING_LIKE_MARKERS):
            return True
    return False


def derive_retrieval_intent(
    *,
    skip_gate: RecallSkipGateResult,
    stance_brief: Any,
    attention_frame: dict | None,
    appraisal: dict | None,
    hub_chat_lane: str | None,
    user_message: str,
    shift_novelty_floor: float = 0.35,
    seed_crystallization_id: str | None = None,
) -> tuple[str, str]:
    """Derive PCR retrieval intent from stance, appraisal, and attention signals."""
    _ = hub_chat_lane

    if skip_gate.skip:
        return "none", "phase0_skip"

    if _open_loops(attention_frame):
        return "open_loop", "open_loops_present"

    shift_kind = _shift_kind(appraisal)
    if shift_kind == "REPAIR" and _novelty_meets_floor(appraisal, shift_novelty_floor):
        return "open_loop", "repair_shift"

    if _is_relational_mode(stance_brief):
        return "relational", "relational_mode"

    if shift_kind == "STANCE" and _novelty_meets_floor(appraisal, shift_novelty_floor):
        return "relational", "stance_shift"

    if _has_contradiction_seed(
        seed_crystallization_id=seed_crystallization_id,
        attention_frame=attention_frame,
    ):
        return "contradiction", "contradiction_seed"

    if _is_instrumental_mode(stance_brief) and _has_planning_like_priority(stance_brief):
        return "procedural", "procedural_mode"

    if shift_kind == "TOPIC" and _novelty_meets_floor(appraisal, shift_novelty_floor):
        return "semantic", "topic_shift"

    if _has_entity_query(user_message):
        return "semantic", "entity_query"

    return "continuity", "continuity_only"
