"""Resource wait around the self-sense-eval graph.

Same admission shell as `admitted_graph.py` (request a GPU pool hold → wait
for the pool's grant → work under the hold → release), but the work is
`ask_questions → publish → finish`, not the curiosity harness/journal pipeline.

Live 2026-09-22: every admitted run — including `workflow=self_sense_eval` —
was driven through `build_admitted_graph` (curiosity only). Orion got the
placeholder brief.prompt ("self-sense eval: four fixed questions") as a
single investigation turn, wrote a curiosity journal, and never scored.
This module is the missing second admitted graph.

Durable (2026-10-23): ``ask_questions`` asks ONE question per node run, so each answer is
checkpointed as it lands, and a failed attempt (node timeout, transport, hung Hub turn) goes through
the shared ``failed_attempt`` policy -- ``retrying`` with backoff, then resume at the next
unanswered question -- instead of failing the run on the first exception. Live: run
20261023T180730Z-0842e5 held agent-gpu2 for its whole 8840 s budget on one hung question, then
failed with every answer lost.

    resource_request -> resource_wait -> ask_questions (xN) -> publish -> finish
                                         ask_questions (failed attempt) -> retry_wait -> resource_request
"""

from __future__ import annotations

import uuid
from typing import Any

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, failed_attempt,
    replay_if_requeued, resource_nodes, retry_wait_node, taken_back,
)
from app.graph import attempt_timeout_sec
from app.self_sense_graph import (
    TURN_REPLY_GRACE_SEC, Deps, SelfSenseRunState, ask_one, make_nodes, next_unanswered,
    no_answers_error, question_share_sec,
)
from orion.schemas.durable_run import SELF_SENSE_EVAL_NODES


def build_admitted_self_sense_graph(deps: Deps, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    original = make_nodes(deps)

    resource_request, resource_wait, after_wait = resource_nodes(admission)
    retry_wait = retry_wait_node(admission)

    async def ask_questions(state: SelfSenseRunState) -> dict:
        """ONE question per node run, so each answer is checkpointed the moment it lands: a
        timeout, a crash or a pool take-back resumes at the next unanswered question instead of
        re-asking (or losing) the ones already answered. The node's timer is the question's share
        of the brief's budget, not the whole budget (``question_share_sec``)."""
        state = dict(state)
        pending = next_unanswered(state)
        if pending is None:
            return {"status": "running"}   # resumed after the last answer's checkpoint
        question_key, question = pending
        share = question_share_sec(state["brief"])
        turn_timeout = attempt_timeout_sec(state, max(1.0, share - TURN_REPLY_GRACE_SEC), admission.now())
        correlation_id = str(uuid.uuid4())   # a real uuid4 per question turn (see ask_one)
        last_turn = {"question_key": question_key, "turn_correlation_id": correlation_id,
                     "timeout_sec": turn_timeout}

        async def operation(held):
            return {"answer": await ask_one(
                deps, held, question_key, question, correlation_id=correlation_id,
                timeout_sec=turn_timeout, attempt=int(held.get("attempt") or 0) + 1)}

        try:
            result = await admission.execute(
                {**state, "step_correlation_id": correlation_id, "node_timeout_sec": share}, operation)
        except WorkflowDeadline:
            released = await admission.release(state, "workflow_deadline")
            return {**released, "status": "failed", "last_error": "workflow_deadline", "last_turn": last_turn}
        except RunControlPending:
            raise
        except HoldRecalled:
            return {"status": "waiting_resource", "lease": None, "hold": None, "last_turn": last_turn}
        except HoldLost as exc:
            # The pool took the hold back (urgent pause, recall past its grace, lost heartbeat): wait
            # for the same hold (or a fresh one if the pool ended it) and replay. Not an attempt.
            update = await taken_back(admission, state, exc.release_reason, f"{type(exc).__name__}: {exc}",
                                      {"status": "waiting_resource"})
            return {**update, "last_turn": last_turn}
        except Exception as exc:  # noqa: BLE001 -- timeout, transport, hung turn: the shared policy
            # Answers already checkpointed stay; the retry resumes at this same question. A
            # deterministic error fails at once; a spent budget fails the run.
            return await failed_attempt(admission, state, exc,
                                        waiting={"status": "retrying", "retry_node": None},
                                        extra={"last_turn": last_turn}, attempt_sec=share)
        answer = result["answer"]
        if not answer.get("text"):
            # An empty answer may be the pool's: if it took the hold, drop this answer and replay.
            replay = await replay_if_requeued(admission, state, {"status": "waiting_resource"})
            if replay is not None:
                return {**replay, "last_turn": last_turn}
        answers = {**(state.get("answers") or {}), question_key: answer}
        update = {"answers": answers, "last_turn": last_turn, "status": "running", "last_error": None}
        if next_unanswered({**state, "answers": answers}) is None:
            no_answers = no_answers_error(answers)
            if no_answers is not None:
                # Every question came back empty (e.g. every LLM call 500'd): nothing was measured.
                # Fail the run with the turns' own errors instead of publishing non-measurements.
                released = await admission.release(state, "attempt_failed")
                return {**update, **released, "status": "failed", "last_error": no_answers}
        return update

    def after_ask(state) -> str:
        status = state.get("status")
        if status == "waiting_resource":
            return "resource_request"
        if status == "retrying":
            return "retry_wait"
        if status == "failed":
            return "failed"
        return "ask_questions" if next_unanswered(state) is not None else "publish"

    async def publish(state: SelfSenseRunState) -> dict:
        state = dict(state)
        if admission.guard is not None:
            # Node boundary: deadline/control, and the hold is let go if the pool recalled it.
            state["lease"] = await admission.guard(state)
        result = await original["publish"](state)
        if admission.guard is not None and state["lease"] is None:
            result = {**result, "lease": None, "hold": None}   # let go at this boundary: persist it
        return result

    async def finish(state: SelfSenseRunState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: SelfSenseRunState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    g = StateGraph(SelfSenseRunState)
    g.add_node("resource_request", resource_request)
    g.add_node("resource_wait", resource_wait)
    g.add_node("ask_questions", ask_questions)
    g.add_node("retry_wait", retry_wait)
    g.add_node("publish", publish)
    g.add_node("finish", finish)
    g.add_node("failed", failed)
    g.add_edge(START, "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_wait,
                            {"granted": "ask_questions", "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("ask_questions", after_ask)
    g.add_edge("retry_wait", "resource_request")
    g.add_edge("publish", "finish")
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    # Keep the contract name visible for inspectors; not used as a node list
    # walker here (admission emits events from astream updates).
    _ = SELF_SENSE_EVAL_NODES
    return g.compile(checkpointer=checkpointer)
