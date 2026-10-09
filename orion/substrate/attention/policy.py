from __future__ import annotations

from typing import Any, Literal

from orion.schemas.attention_frame import CuriosityCandidateActionV1, CuriositySuppressionV1, OpenLoopV1
from orion.substrate.attention.questions import NATURAL_QUESTION_KEY, question_for
from orion.substrate.attention.scoring import score_loop

DirectAnswerCause = Literal["judged", "unavailable"]
TURN_READ_UNAVAILABLE_REF = "turn_read_unavailable"
# Targets Orion's own threads, not the turn: a follow-up on something the user
# shared can still be the selected ask on a direct turn, and prompts read a
# suppression aimed at the turn itself as "don't ask".
BACKGROUND_THREADS_REF = "background_threads"


def direct_answer_cause(ctx: dict[str, Any], user_text: str) -> DirectAnswerCause | None:
    """Whether this chat turn asks Orion to do or answer something, as judged by the
    same-turn LLM read (`ctx["current_turn_llm_read"]`, populated by cortex-exec's
    current_turn_llm_signals.py). Replaces the deleted verb-prefix / trailing-"?" /
    "what about you" regexes, which misread shared news that happened to end in a
    question and could not tell "tell me about X" from "told my boss about X".

    No user text (substrate/background frames) -> None. A chat turn whose read failed
    or is missing -> "unavailable", which callers treat as a direct turn: without a
    read there is no evidence Orion's own threads are welcome, so fail closed.
    """
    if not user_text.strip():
        return None
    read = ctx.get("current_turn_llm_read")
    if not isinstance(read, dict) or read.get("ok") is not True:
        return "unavailable"
    wants = read.get("wants_direct_answer")
    if wants is True:
        return "judged"
    if wants is False:
        return None
    return "unavailable"


def base_suppressions(*, direct_cause: DirectAnswerCause | None, stale_thread_active: bool) -> list[CuriositySuppressionV1]:
    suppressions: list[CuriositySuppressionV1] = []
    if direct_cause == "judged":
        suppressions.append(CuriositySuppressionV1(reason="user_needs_direct_answer", target_ref=BACKGROUND_THREADS_REF, rationale="turn read judged the user wants work or a direct answer; answer first and keep Orion's own background threads out of it", confidence=0.78))
    elif direct_cause == "unavailable":
        suppressions.append(CuriositySuppressionV1(reason="user_needs_direct_answer", target_ref=TURN_READ_UNAVAILABLE_REF, rationale="no same-turn read available; fail closed on Orion's own background threads", confidence=0.6))
    if stale_thread_active:
        suppressions.append(CuriositySuppressionV1(reason="stale_thread", target_ref="situation.conversation_phase", rationale="conversation phase marks thread as stale", confidence=0.7))
    return suppressions


def select_actions(
    *,
    open_loops: list[OpenLoopV1],
    suppressions: list[CuriositySuppressionV1],
    min_ask: float,
    max_asks: int,
    stale_thread_active: bool,
) -> tuple[list[CuriosityCandidateActionV1], CuriosityCandidateActionV1, list[CuriositySuppressionV1], list[str]]:
    # NOTE (2026-07-31, disclosed not fixed): min_ask (caller-supplied,
    # default 0.65 -- see AttentionFrameV1/build_attention_frame) and the
    # inline 0.48/0.35 thresholds below were tuned against the OLD
    # SEED_WEIGHTS absolute per-loop score, not the new Borda
    # rank-aggregated score. Borda salience is fundamentally a RELATIVE
    # rank measure -- the same underlying evidence can score differently
    # depending on how many other loops happen to compete in the same tick
    # (e.g. an all-tied field always lands every loop at exactly 0.5;
    # adjacent-rank gaps compress as ~1/(n-1)). Whether these absolute
    # cutoffs still make sense against the new score distribution is an
    # open, real question -- explicitly out of scope for this patch, same
    # as `SURFACE_MIN_SALIENCE`'s deferred recalibration (see
    # orion/sentience_striving_program/README.md's 2026-07-31 entry) --
    # the new formula needs to run for real before there is a distribution
    # to recalibrate against.
    #
    # A loop carrying a natural follow-up question was explicitly judged, by the
    # same-turn LLM read, as something Juniper shared that a friend would ask
    # about. That judgment is the salience evidence for it; the relative Borda
    # score cannot supply it (a lone current-turn loop's n==1 fallback tops out
    # near 0.5, below min_ask, so a disclosure on a quiet turn could never be
    # asked). Invited loops also outrank Orion's own background threads when
    # both are askable: following up on what the person said comes first.
    actions: list[CuriosityCandidateActionV1] = []
    invited_ids: set[str] = set()
    suppressions = list(suppressions)
    for loop in open_loops:
        score = score_loop(loop)
        invited = bool(loop.provenance.get(NATURAL_QUESTION_KEY))
        clears_threshold = score >= min_ask or (score >= (min_ask - 0.08) and loop.autonomy_value >= 0.5 and loop.predictive_value >= 0.5)
        if loop.already_known:
            action_type = "suppress"
            rationale = "already-known target should not be asked about again"
            question = None
            suppressions.append(CuriositySuppressionV1(reason="already_known", target_ref=loop.id, rationale=f"{loop.description} appears in current memory/concept context", confidence=0.78))
        elif (clears_threshold or invited) and loop.askability >= 0.45 and not stale_thread_active:
            action_type = "ask"
            rationale = (
                "follow-up on something the user shared this turn"
                if invited
                else "highest-value unresolved target is askable in this turn"
            )
            question = question_for(loop)
            if invited:
                invited_ids.add(loop.id)
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

    ask_actions = sorted(
        [a for a in actions if a.action_type == "ask"],
        key=lambda a: (a.open_loop_id in invited_ids, a.score),
        reverse=True,
    )
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
