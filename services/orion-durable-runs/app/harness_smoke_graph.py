"""harness.echo_smoke: one admitted durable run whose one step is one harness turn.

    resource_request -> resource_wait -> harness_step -> finish
                                         harness_step (failed attempt) -> retry_wait -> resource_request

The smoke for the durable-run -> harness step seam (``app/harness_step.py``). It proves, on the live
rail, that a durable run can wait for a GPU pool hold, run a step as a harness turn under it, and
checkpoint the turn's final text as the step's result -- with retry/resume intact:

* waiting for the hold is ``resource_wait``'s job and never an attempt;
* a refusal / timeout / empty reply raises ``HarnessStepFailed`` and goes through the shared
  ``failed_attempt`` policy (backoff in ``retry_wait``, then a fresh hold); past
  ``DURABLE_RUNS_RETRY_MAX_ATTEMPTS`` the run fails with the turn's error, never an empty success;
* a pool take-back replays the step without spending an attempt;
* a checkpointed step is never re-run (a restart after it resumes at ``finish``).

Nothing is persisted beyond the durable-run checkpoint, the admission row and its state rows: no
journal, no memory card, no recall write (the turn runs with Hub's ``no_write``).

Contract: orion/schemas/harness_step_run.py.
"""
from __future__ import annotations

from typing import Any, TypedDict

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, failed_attempt,
    replay_if_requeued, resource_nodes, retry_wait_node, taken_back,
)
from app.graph import attempt_timeout_sec
from app.harness_step import (
    TURN_REPLY_GRACE_SEC, RunTurn, harness_step_correlation_id, run_harness_step, step_detail,
)
from orion.schemas.harness_step_run import (
    HARNESS_ECHO_SMOKE_SESSION_ID, HARNESS_ECHO_SMOKE_STEP_KEY, HARNESS_ECHO_SMOKE_TAG, HarnessEchoSmokeBriefV1,
)


class HarnessSmokeState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]
    admission: dict[str, Any]
    requested_at: str
    attempt: int            # failed attempts of the current step; reset when the step lands
    hold_takebacks: int
    lease: dict[str, Any] | None
    hold: dict[str, Any] | None
    hold_seq: int
    turn_fence: int
    status: str
    retry_at: str | None
    retry_node: str | None
    last_error: str | None
    steps: dict[str, dict[str, Any]]    # step_key -> the step's result record (harness_step.py)
    harness_turn_meta: dict[str, Any]   # the last issued turn's identity (failed-run detail, cancel)


def finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    steps = state.get("steps") or {}
    echo = steps.get(HARNESS_ECHO_SMOKE_STEP_KEY) or {}
    return {
        "line": "harness_smoke",
        "turn_correlation_id": echo.get("turn_correlation_id"),
        "steps": {key: step_detail(record) for key, record in steps.items()},
    }


def build_harness_smoke_graph(run_turn: RunTurn, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    resource_request, resource_wait, after_wait = resource_nodes(admission)
    retry_wait = retry_wait_node(admission)
    step_key = HARNESS_ECHO_SMOKE_STEP_KEY

    async def harness_step(state: HarnessSmokeState) -> dict:
        state = dict(state)
        if step_key in (state.get("steps") or {}):
            return {"status": "running"}    # resumed after the step's checkpoint: never re-run it
        brief = HarnessEchoSmokeBriefV1.model_validate(state["brief"])
        window = attempt_timeout_sec(state, brief.timeout_sec, admission.now())   # the node timer, deadline-clamped
        turn_timeout = max(1.0, window - TURN_REPLY_GRACE_SEC)
        correlation_id = harness_step_correlation_id(state, step_key)
        turn_meta = {"turn_correlation_id": correlation_id, "step_key": step_key}

        async def operation(held: dict[str, Any]) -> dict[str, Any]:
            return await run_harness_step(
                run_turn, held, step_key=step_key, prompt=brief.prompt, correlation_id=correlation_id,
                timeout_sec=turn_timeout, source_tag=HARNESS_ECHO_SMOKE_TAG,
                session_id=HARNESS_ECHO_SMOKE_SESSION_ID)

        try:
            record = await admission.execute({**state, "step_correlation_id": correlation_id}, operation)
        except WorkflowDeadline:
            released = await admission.release(state, "workflow_deadline")
            return {**released, "status": "failed", "last_error": "workflow_deadline", "harness_turn_meta": turn_meta}
        except RunControlPending:
            raise
        except HoldRecalled:
            return {"status": "waiting_resource", "lease": None, "hold": None}
        except HoldLost as exc:   # HoldPreempted too: the pool keeps its place, no attempt spent
            update = await taken_back(admission, state, exc.release_reason, f"{type(exc).__name__}: {exc}",
                                      {"status": "waiting_resource"})
            return {**update, "harness_turn_meta": turn_meta}
        except Exception as exc:  # noqa: BLE001 -- refusal, timeout, empty turn, transport: the shared policy
            # A failed turn may be the pool's (it took the hold back mid-turn): replay, no attempt.
            replay = await replay_if_requeued(admission, state, {"status": "waiting_resource"})
            if replay is not None:
                return {**replay, "harness_turn_meta": turn_meta}
            return await failed_attempt(admission, state, exc, waiting={"status": "retrying", "retry_node": None},
                                        extra={"harness_turn_meta": turn_meta}, attempt_sec=brief.timeout_sec)
        record["attempts"] = int(state.get("attempt") or 0) + 1
        return {"steps": {**(state.get("steps") or {}), step_key: record}, "harness_turn_meta": turn_meta,
                "status": "running", "attempt": 0, "retry_at": None, "last_error": None}

    async def finish(state: HarnessSmokeState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: HarnessSmokeState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    def after_step(state: HarnessSmokeState) -> str:
        status = state.get("status")
        if status == "waiting_resource":
            return "resource_request"
        if status == "retrying":
            return "retry_wait"
        if status == "failed":
            return "failed"
        return "finish"

    g = StateGraph(HarnessSmokeState)
    for name, node in {"resource_request": resource_request, "resource_wait": resource_wait,
                       "harness_step": harness_step, "retry_wait": retry_wait,
                       "finish": finish, "failed": failed}.items():
        g.add_node(name, node)
    g.add_edge(START, "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_wait,
                            {"granted": "harness_step", "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("harness_step", after_step)
    g.add_edge("retry_wait", "resource_request")
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
