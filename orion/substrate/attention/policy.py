from __future__ import annotations

import re

from orion.schemas.attention_frame import CuriosityCandidateActionV1, CuriositySuppressionV1, OpenLoopV1
from orion.substrate.attention.questions import question_for
from orion.substrate.attention.scoring import score_loop

GENERIC_QUESTION_RE = re.compile(
    r"\b(how (?:are|was) (?:you|your day)|what about you|and you\??|how about you\??)\b",
    flags=re.IGNORECASE,
)
QUESTION_RE = re.compile(r"\?\s*$")
DIRECT_WORK_RE = re.compile(
    r"^\s*(please\s+)?(implement|fix|debug|review|explain|summarize|show|tell|write|add|remove|update|run|paste)\b",
    flags=re.IGNORECASE,
)


def generic_reversal_present(user_text: str) -> bool:
    return bool(GENERIC_QUESTION_RE.search(user_text))


def direct_work_turn(user_text: str) -> bool:
    return bool(DIRECT_WORK_RE.search(user_text) or QUESTION_RE.search(user_text))


def base_suppressions(*, user_text: str, stale_thread_active: bool) -> list[CuriositySuppressionV1]:
    suppressions: list[CuriositySuppressionV1] = []
    if generic_reversal_present(user_text):
        suppressions.append(CuriositySuppressionV1(reason="generic_reciprocity", target_ref="generic_reversal", rationale="current turn matches a generic reversal pattern", confidence=0.9))
    if direct_work_turn(user_text):
        suppressions.append(CuriositySuppressionV1(reason="user_needs_direct_answer", target_ref="current_turn", rationale="current turn asks for work or a direct answer", confidence=0.78))
    if stale_thread_active:
        suppressions.append(CuriositySuppressionV1(reason="stale_thread", target_ref="situation.conversation_phase", rationale="conversation phase marks thread as stale", confidence=0.7))
    return suppressions


def select_actions(
    *,
    open_loops: list[OpenLoopV1],
    suppressions: list[CuriositySuppressionV1],
    min_ask: float,
    max_asks: int,
    generic_reversal: bool,
    stale_thread_active: bool,
) -> tuple[list[CuriosityCandidateActionV1], CuriosityCandidateActionV1, list[CuriositySuppressionV1], list[str]]:
    actions: list[CuriosityCandidateActionV1] = []
    suppressions = list(suppressions)
    for loop in open_loops:
        score = score_loop(loop)
        if loop.already_known:
            action_type = "suppress"
            rationale = "already-known target should not be asked about again"
            question = None
            suppressions.append(CuriositySuppressionV1(reason="already_known", target_ref=loop.id, rationale=f"{loop.description} appears in current memory/concept context", confidence=0.78))
        elif (score >= min_ask or (score >= (min_ask - 0.08) and loop.autonomy_value >= 0.5 and loop.predictive_value >= 0.5)) and loop.askability >= 0.45 and not generic_reversal and not stale_thread_active:
            action_type = "ask"
            rationale = "highest-value unresolved target is askable in this turn"
            question = question_for(loop)
        elif score >= 0.48:
            action_type = "watch"
            rationale = "target is useful but not worth a question now"
            question = None
        elif score >= 0.35:
            action_type = "defer"
            rationale = "target is unresolved but low priority"
            question = None
        else:
            action_type = "none"
            rationale = "target below curiosity threshold"
            question = None
        actions.append(
            CuriosityCandidateActionV1(
                action_type=action_type,  # type: ignore[arg-type]
                open_loop_id=loop.id,
                score=score,
                rationale=rationale,
                question_text=question,
                provenance={"policy": "deterministic_attention_frame_v1", "min_ask_score": min_ask},
            )
        )

    ask_actions = sorted([a for a in actions if a.action_type == "ask"], key=lambda a: a.score, reverse=True)
    selected = ask_actions[0] if ask_actions and max_asks >= 1 else None
    if len(ask_actions) > max_asks:
        for extra in ask_actions[max_asks:]:
            suppressions.append(CuriositySuppressionV1(reason="too_many_questions", target_ref=extra.open_loop_id, rationale="policy allows at most one selected ask", confidence=0.95))
            idx = actions.index(extra)
            actions[idx] = extra.model_copy(update={"action_type": "watch", "question_text": None, "rationale": "ask suppressed because policy allows at most one selected ask"})
    if selected is None:
        non_none = sorted([a for a in actions if a.action_type in {"watch", "defer", "suppress"}], key=lambda a: a.score, reverse=True)
        selected = non_none[0] if non_none else CuriosityCandidateActionV1(action_type="none", score=0.0, rationale="no qualifying open loop")

    deferred = [str(a.open_loop_id) for a in actions if a.open_loop_id and a.action_type in {"defer", "watch"}]
    return actions, selected, suppressions, deferred
