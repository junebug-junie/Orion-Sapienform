"""PCR phase 3 retrieval intent: which purposeful recall profile a chat turn gets.

Every input is a model judgment or a structured id. Nothing here matches
words in the user's message:

* ``stance_brief``: the stance LLM's ``task_mode`` / ``conversation_frame`` /
  ``interaction_regime`` (closed literals in ``chat_stance_brief.j2``).
* ``appraisal``: the turn-change LLM classifier's ``shift_kind`` and
  ``novelty_score`` (orion-memory-consolidation ``classify.py``), when present.
* ``turn_signals``: ``ctx["current_turn_llm_signals"]``, the same-turn LLM's
  typed reading of what Juniper's message names (``person``, ``place``,
  ``plan``, ``belief``, ``concept``, ``activity``, ``other``;
  ``services/orion-cortex-exec/app/current_turn_llm_signals.py``). Empty on
  Orion's own turns, which never run that call.
* ``seed_crystallization_id`` / ``contradiction_refs``: explicit ids.

Memory Stage 2 (2026-10-06) fixed the classifier that returned ``open_loop``
for 630 of 630 purposeful recalls in 7 days: it fired whenever the attention
frame listed any open loop, and live chat frames always list several
(mostly substrate prediction-error concepts, not conversational threads).
``open_loop`` now needs a repair shift. The spec's other two open-loop
triggers (a loop that persists across turns; an open follow-up memory whose
referents appear in the turn) have no in-turn input yet and arrive with
recall-by-referent (Stage 2 PR F). The capitalized-word "entity" regex and the
"plan/step/..." substring list over response priorities are gone; the turn's
LLM-typed signals replace them.
"""
from __future__ import annotations

from typing import Any

from orion.memory.recall_skip_gate import RecallSkipGateResult

_RELATIONAL_TASK_MODES = frozenset({"reflective_dialogue", "playful_exchange"})
_RELATIONAL_CONVERSATION_FRAMES = frozenset({"reflective", "playful_relational"})

# The referent rule over the turn's LLM-typed signals (Stage 2 spec 4.2):
# a person -> relational; a plan -> procedural; anything else it names ->
# semantic. Checked in this order, so a turn naming a person and a plan is
# relational.
_TURN_SIGNAL_RULES: tuple[tuple[str, str, str], ...] = (
    ("person", "relational", "turn_names_person"),
    ("plan", "procedural", "turn_names_plan"),
)


def _stance_field(stance_brief: Any, key: str) -> str:
    if isinstance(stance_brief, dict):
        return str(stance_brief.get(key) or "").strip()
    return str(getattr(stance_brief, key, None) or "").strip()


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


def _is_procedural_mode(stance_brief: Any) -> bool:
    # The stance LLM's own literals for "this turn is about doing/building":
    # conversation_frame=planning or task_mode=technical_collaboration. (The
    # old rule required task_mode=instrumental, which is not one of the
    # template's task_mode literals, plus a substring word list.)
    return (_stance_field(stance_brief, "conversation_frame") == "planning"
            or _stance_field(stance_brief, "task_mode") == "technical_collaboration")


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


def turn_signal_types(turn_signals: Any) -> set[str]:
    """Types the same-turn LLM assigned to what the message names."""
    if not isinstance(turn_signals, list):
        return set()
    types = set()
    for item in turn_signals:
        if isinstance(item, dict) and str(item.get("phrase") or "").strip():
            types.add(str(item.get("type") or "other").strip().lower() or "other")
    return types


def derive_retrieval_intent(
    *,
    skip_gate: RecallSkipGateResult,
    stance_brief: Any,
    attention_frame: dict | None,
    appraisal: dict | None,
    hub_chat_lane: str | None,
    turn_signals: Any = None,
    shift_novelty_floor: float = 0.35,
    seed_crystallization_id: str | None = None,
    eligible_belief_count: int = 0,
    brain_belief_default_enabled: bool = True,
) -> tuple[str, str]:
    """Return ``(intent, rule_id)``. ``rule_id`` names the rule that fired."""

    if skip_gate.skip:
        return "none", "phase0_skip"

    shift_kind = _shift_kind(appraisal)
    shifted = _novelty_meets_floor(appraisal, shift_novelty_floor)

    # Relational and topic rules first (Stage 2 spec 4.2).
    if _is_relational_mode(stance_brief):
        return "relational", "relational_mode"
    if shift_kind == "STANCE" and shifted:
        return "relational", "stance_shift"
    if shift_kind == "TOPIC" and shifted:
        return "semantic", "topic_shift"
    if shift_kind == "REPAIR" and shifted:
        return "open_loop", "repair_shift"

    if _has_contradiction_seed(
        seed_crystallization_id=seed_crystallization_id,
        attention_frame=attention_frame,
    ):
        return "contradiction", "contradiction_seed"

    if _is_procedural_mode(stance_brief):
        return "procedural", "procedural_mode"

    types = turn_signal_types(turn_signals)
    for signal_type, intent, rule_id in _TURN_SIGNAL_RULES:
        if signal_type in types:
            return intent, rule_id
    if types:
        return "semantic", "turn_names_topic"

    if (
        brain_belief_default_enabled
        and hub_chat_lane in ("brain", "orion")
        and eligible_belief_count > 0
    ):
        return "semantic", "brain_lane_belief_default"

    return "continuity", "continuity_only"
