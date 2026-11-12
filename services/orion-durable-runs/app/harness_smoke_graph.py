"""harness.echo_smoke: one admitted durable run whose one step is one harness turn.

    resource_request -> resource_wait -> harness_step -> finish
                                         harness_step (failed attempt) -> retry_wait -> resource_request
                         harness_step (cut short, consult only) -> retry_wait -> resource_request

The smoke for the durable-run -> harness step seam (``app/harness_step.py``). It proves, on the live
rail, that a durable run can wait for a GPU pool hold, run a step as a harness turn under it, and
checkpoint the turn's final text as the step's result -- with retry/resume intact:

* waiting for the hold is ``resource_wait``'s job and never an attempt;
* a refusal / timeout / empty reply raises ``HarnessStepFailed`` and goes through the shared
  ``failed_attempt`` policy (backoff in ``retry_wait``, then a fresh hold); past
  ``DURABLE_RUNS_RETRY_MAX_ATTEMPTS`` the run fails with the turn's error, never an empty success;
* a pool take-back replays the step without spending an attempt;
* a checkpointed step is never re-run (a restart after it resumes at ``finish``);
* ``retry_cut_short`` (consult.orion_turn): a turn the FCC motor stopped mid-work
  (``orion.harness.cut_short.CUT_SHORT_CODES``) is a failed attempt while a retry will still run. The
  retry does NOT get the cut session back: the motor records its own stop as a finished session
  (``fcc_motor._record_session_end`` -> ``completed``; a context ceiling -> ``ceiling``), so the retry
  starts fresh. What the cut attempt found (its cut-short draft lists it) is kept in ``cut_short_text``
  and handed to the retry in its prompt. The last attempt keeps the honest cut-short text rather than
  failing.

Nothing is persisted beyond the durable-run checkpoint, the admission row and its state rows: no
journal, no memory card, no recall write (the turn runs with Hub's ``no_write``).

Contract: orion/schemas/harness_step_run.py.
"""
from __future__ import annotations

from typing import Any, TypedDict

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, failed_attempt,
    replay_if_requeued, resource_nodes, retry_wait_node, retry_would_run, taken_back,
)
from app.graph import attempt_timeout_sec
from app.harness_step import (
    TURN_REPLY_GRACE_SEC, HarnessStepFailed, RunTurn, harness_step_correlation_id, run_harness_step, step_detail,
)
from orion.harness.cut_short import CUT_SHORT_CODES
from orion.schemas.consult_turn_run import (
    CONSULT_ORION_TURN_STEP_KEY, CONSULT_ORION_TURN_TAG, ConsultOrionTurnBriefV1,
)
from orion.schemas.harness_step_run import (
    HARNESS_ECHO_SMOKE_SESSION_ID, HARNESS_ECHO_SMOKE_STEP_KEY, HARNESS_ECHO_SMOKE_TAG, HarnessEchoSmokeBriefV1,
)


# ``last_error`` of a cut-short attempt starts with this (after the exception type).
CUT_SHORT_ERROR = "harness_step_cut_short"
# The retry starts a fresh session, so the earlier attempts' findings go in its prompt, newest kept
# when they are long.
CUT_SHORT_CARRY_MAX_CHARS = 16000
CUT_SHORT_CONTINUE_NOTE = (
    "Your earlier attempt at this turn was cut short before you finished (time ran out). This is a new "
    "session: you do not have that attempt's tool results. What it found is below. Do not redo checks it "
    "already made; continue from there, and write your finished reply."
)


def carry_cut_short(previous: str | None, text: str, attempt: int) -> str:
    """The findings carried to the next attempt: every cut attempt's text, newest last, capped from
    the oldest end."""
    joined = "\n\n".join(x for x in (previous or "", f"--- Attempt {attempt} (cut short) ---\n{text.strip()}") if x)
    return joined[-CUT_SHORT_CARRY_MAX_CHARS:]


def continue_prompt(carried: str, prompt: str) -> str:
    return f"{CUT_SHORT_CONTINUE_NOTE}\n\n{carried}\n\n--- The turn ---\n{prompt}"


def cut_short_status(record: dict[str, Any]) -> str | None:
    """The motor's cut-short code for a landed step, else None."""
    status = (record.get("debug") or {}).get("harness_grounding_status")
    return status if status in CUT_SHORT_CODES else None


class HarnessSmokeState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]
    admission: dict[str, Any]
    requested_at: str
    attempt: int            # failed attempts of the current step; reset when the step lands
    hold_takebacks: int
    listener_waits: int   # RPCs published to nobody this run waited out (listener_wait; never an attempt)
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
    cut_short_text: str | None          # retry_cut_short: what the cut attempts found, for the next prompt


def _finish_detail(state: dict[str, Any], *, line: str, step_key: str) -> dict[str, Any]:
    steps = state.get("steps") or {}
    step = steps.get(step_key) or {}
    return {
        "line": line,
        "turn_correlation_id": step.get("turn_correlation_id"),
        "steps": {key: step_detail(record) for key, record in steps.items()},
    }


def finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    return _finish_detail(state, line="harness_smoke", step_key=HARNESS_ECHO_SMOKE_STEP_KEY)


def consult_turn_finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    """consult.orion_turn's finish detail. Hub reads the full text from ``GET /runs/{run_id}``."""
    return _finish_detail(state, line="consult_orion_turn", step_key=CONSULT_ORION_TURN_STEP_KEY)


def _smoke_step(brief: HarnessEchoSmokeBriefV1) -> dict[str, Any]:
    return {"prompt": brief.prompt, "source_tag": HARNESS_ECHO_SMOKE_TAG, "session_id": HARNESS_ECHO_SMOKE_SESSION_ID}


def _consult_step(brief: ConsultOrionTurnBriefV1) -> dict[str, Any]:
    # require_lookup=False: Orion may reply to Claude without tool use. The source tag carries the
    # turn so each consult turn has its own resumed session key (harness_step.py constraints).
    return {"prompt": brief.prompt, "source_tag": f"{CONSULT_ORION_TURN_TAG}:t{brief.turn_no}",
            "session_id": brief.session_id, "retrieval_query": brief.retrieval_query, "require_lookup": False}


def build_harness_smoke_graph(run_turn: RunTurn, admission: AdmissionDeps, checkpointer: Any):
    return build_single_step_harness_graph(run_turn, admission, checkpointer, step_key=HARNESS_ECHO_SMOKE_STEP_KEY,
                                           brief_model=HarnessEchoSmokeBriefV1, step_args=_smoke_step)


def build_consult_turn_graph(run_turn: RunTurn, admission: AdmissionDeps, checkpointer: Any):
    """consult.orion_turn: the same one-step graph; the step is Orion's consult turn."""
    return build_single_step_harness_graph(run_turn, admission, checkpointer, step_key=CONSULT_ORION_TURN_STEP_KEY,
                                           brief_model=ConsultOrionTurnBriefV1, step_args=_consult_step,
                                           retry_cut_short=True)


def build_single_step_harness_graph(run_turn: RunTurn, admission: AdmissionDeps, checkpointer: Any, *,
                                    step_key: str, brief_model: Any, step_args: Any, retry_cut_short: bool = False):
    """One admitted run, one harness turn: the graph harness.echo_smoke and consult.orion_turn share.
    ``step_args(brief)`` gives the turn's prompt, source_tag, session_id and optional
    retrieval_query / require_lookup. ``retry_cut_short``: see the module docstring."""
    from langgraph.graph import END, START, StateGraph

    resource_request, resource_wait, after_wait = resource_nodes(admission)
    retry_wait = retry_wait_node(admission)

    async def harness_step(state: HarnessSmokeState) -> dict:
        state = dict(state)
        if step_key in (state.get("steps") or {}):
            return {"status": "running"}    # resumed after the step's checkpoint: never re-run it
        brief = brief_model.model_validate(state["brief"])
        window = attempt_timeout_sec(state, brief.timeout_sec, admission.now())   # the node timer, deadline-clamped
        turn_timeout = max(1.0, window - TURN_REPLY_GRACE_SEC)
        correlation_id = harness_step_correlation_id(state, step_key)
        turn_meta = {"turn_correlation_id": correlation_id, "step_key": step_key}
        args = dict(step_args(brief))
        if retry_cut_short and state.get("cut_short_text"):
            args["prompt"] = continue_prompt(state["cut_short_text"], args["prompt"])

        async def operation(held: dict[str, Any]) -> dict[str, Any]:
            return await run_harness_step(
                run_turn, held, step_key=step_key, correlation_id=correlation_id,
                timeout_sec=turn_timeout, **args)

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
        cut = cut_short_status(record) if retry_cut_short else None
        if cut and retry_would_run(admission, state, attempt_sec=brief.timeout_sec):
            # The pool took the hold back mid-turn: replay, no attempt spent (as on the exception path).
            replay = await replay_if_requeued(admission, state, {"status": "waiting_resource"})
            if replay is not None:
                return {**replay, "harness_turn_meta": turn_meta}
            carried = carry_cut_short(state.get("cut_short_text"), record["text"], int(state.get("attempt") or 0) + 1)
            return await failed_attempt(admission, state, HarnessStepFailed(f"{CUT_SHORT_ERROR}:{step_key}:{cut}"),
                                        waiting={"status": "retrying", "retry_node": None},
                                        extra={"harness_turn_meta": turn_meta, "cut_short_text": carried},
                                        attempt_sec=brief.timeout_sec)
        record["attempts"] = int(state.get("attempt") or 0) + 1
        if cut:
            record["cut_short"] = cut    # the last attempt: kept, labelled, never an empty failure
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
