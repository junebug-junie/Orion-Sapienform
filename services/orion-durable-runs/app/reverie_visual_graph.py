"""One admitted visual reverie (``reverie.visual``): orion-thought does every stage's work.

Design: docs/superpowers/specs/2026-09-28-visual-reverie-durable-graph-design.md.

    prepare -> resource_request -> resource_wait -> generate -> caption -> finish
       ^            ^                                   |           |
       |            +----------- retry_wait <-----------+-----------+
       +-- retry_wait

Only ``generate`` runs under the run's pool hold (``diffusion`` class); the hold is released as
soon as generate's result is checkpointed, so interpret (inside prepare) and caption never carry
it. A step ``retry`` (deferral, transient failure, transport error) never spends the run's attempt
budget: the graph releases any hold, backs off, and resumes at the same stage. The only terminal
failure is the run deadline (``admission.deadline_at``): ``failed`` with ``retry_window_expired``,
and thought is told to ``abandon`` the attempt so it does not block the next claim.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, TypedDict
from uuid import NAMESPACE_URL, uuid5

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, resource_nodes,
)
from orion.schemas.reverie_visual_run import (
    NEEDS_GENERATE,
    ReverieVisualRunBriefV1,
    ReverieVisualStepRequestV1,
    ReverieVisualStepResultV1,
)

logger = logging.getLogger(__name__)

RETRY_WINDOW_EXPIRED = "retry_window_expired"
ABANDON_TIMEOUT_SEC = 30.0
# Floor for any backoff, including a thought-supplied retry_after_sec of 0: a retry is never an
# immediate loop.
MIN_BACKOFF_SEC = 1.0

# (request, budget_sec) -> result. budget_sec is brief.timeout_sec for generate, else None.
RunStep = Callable[[ReverieVisualStepRequestV1, float | None], Awaitable[ReverieVisualStepResultV1]]


class ReverieVisualState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict
    admission: dict
    requested_at: str
    attempt: int
    lease: dict | None
    hold: dict | None
    hold_seq: int
    turn_fence: int
    status: str
    last_error: str | None
    retry_at: str | None
    retry_node: str | None
    # Which node the last node's conditional edge goes to.
    route: str
    # Thought's attempt row for this dispatch (== chain_id); returned by prepare.
    attempt_id: str | None
    # step -> RPCs issued so far; part of each step's correlation id.
    step_calls: dict
    retries: int
    retry_streak: int
    outcome: str | None
    reason: str | None
    chain_id: str | None
    artifact_sha256: str | None
    execution_receipt: dict | None
    generate_elapsed_sec: float
    visual_elapsed_sec: float
    started_at: str | None
    finished_at: str | None
    abandon_sent: bool


def step_correlation_id(state: dict, step: str) -> str:
    """Per-RPC identity: a late reply for an earlier call of the same step (or a fenced generate)
    can never be read as the answer to this one."""
    n = int((state.get("step_calls") or {}).get(step, 0))
    fence = int(state.get("turn_fence") or 0)
    return str(uuid5(NAMESPACE_URL, f"orion:reverie-visual:{state['run_id']}:{step}:{n}:{fence}"))


def _bumped(state: dict, step: str) -> dict:
    calls = dict(state.get("step_calls") or {})
    calls[step] = int(calls.get(step, 0)) + 1
    return calls


def _visual_request(state: dict):
    return ReverieVisualRunBriefV1.model_validate(state["brief"]).visual_request


def _expired(state: dict, now: datetime) -> bool:
    deadline = (state.get("admission") or {}).get("deadline_at")
    return bool(deadline) and now >= datetime.fromisoformat(str(deadline))


def _deadline_failure(extra: dict | None = None) -> dict:
    return {**(extra or {}), "status": "failed", "route": "failed", "last_error": RETRY_WINDOW_EXPIRED}


def _detail_common(state: dict) -> dict[str, Any]:
    try:
        request = _visual_request(state)
        ids = {"dispatch_id": request.dispatch_id, "proposal_id": request.proposal_id,
               "decision_id": request.decision_id}
    except Exception:  # noqa: BLE001 -- a malformed brief must still produce a terminal detail
        ids = {"dispatch_id": None, "proposal_id": None, "decision_id": None}
    return {**ids, "attempt_id": state.get("attempt_id"), "chain_id": state.get("chain_id"),
            "retries": int(state.get("retries") or 0), "started_at": state.get("started_at"),
            "finished_at": state.get("finished_at")}


def finish_detail(state: dict) -> dict[str, Any]:
    """``run.completed`` detail: what dispatch settles the render result from. visual_elapsed_sec is
    thought's own real-work seconds across done steps -- never queue or hold-wait time."""
    return {
        **_detail_common(state),
        "outcome": state.get("outcome"),
        "reason": state.get("reason"),
        "artifact_sha256": state.get("artifact_sha256"),
        "execution_receipt": state.get("execution_receipt"),
        "generate_elapsed_sec": round(float(state.get("generate_elapsed_sec") or 0.0), 3),
        "visual_elapsed_sec": round(float(state.get("visual_elapsed_sec") or 0.0), 3),
    }


def terminal_detail(state: dict, status: str) -> dict[str, Any]:
    """``run.failed`` / ``run.cancelled`` detail, read from the checkpoint at terminal time.
    ``reason`` keeps the last stage reason (e.g. the deferral that was being retried)."""
    error = "cancelled" if status == "cancelled" else (state.get("last_error") or status)
    return {**_detail_common(state), "error": error, "last_error": error, "reason": state.get("reason")}


def build_reverie_visual_graph(run_step: RunStep, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    shared_request, resource_wait, after_wait = resource_nodes(admission)

    def backoff(state: dict, retry_after: float | None) -> float:
        if retry_after is not None:
            return max(MIN_BACKOFF_SEC, float(retry_after))
        streak = int(state.get("retry_streak") or 0)
        delay = min(admission.retry_max_seconds, admission.retry_base_seconds * 2 ** min(streak, 30))
        return max(MIN_BACKOFF_SEC, delay)

    def retry(state: dict, node: str, reason: str, retry_after: float | None = None, **extra) -> dict:
        """A deferral: back off, resume at ``node``. Counted, never an attempt."""
        delay = backoff(state, retry_after)
        return {**extra, "status": "retrying", "route": "retry_wait", "retry_node": node,
                "retry_at": (admission.now() + timedelta(seconds=delay)).isoformat(),
                "reason": reason, "last_error": reason[:500],
                "retries": int(state.get("retries") or 0) + 1,
                "retry_streak": int(state.get("retry_streak") or 0) + 1}

    def done_update(state: dict, result: ReverieVisualStepResultV1) -> dict:
        elapsed = float(result.elapsed_sec or 0.0)
        return {"retry_streak": 0, "last_error": None, "reason": result.reason,
                "visual_elapsed_sec": float(state.get("visual_elapsed_sec") or 0.0) + elapsed}

    def request(state: dict, step: str, correlation_id: str, **extra) -> ReverieVisualStepRequestV1:
        return ReverieVisualStepRequestV1(run_id=state["run_id"], correlation_id=correlation_id, step=step,
                                          visual_request=_visual_request(state),
                                          attempt_id=state.get("attempt_id"), **extra)

    async def call(req: ReverieVisualStepRequestV1, budget_sec: float | None = None) -> ReverieVisualStepResultV1:
        result = await run_step(req, budget_sec)
        if (result.run_id != req.run_id or result.correlation_id != req.correlation_id
                or result.step != req.step):
            raise ValueError("reverie visual step identity mismatch")
        return result

    async def boundary(state: dict) -> bool:
        """Deadline and operator control before a no-hold step. True = the window is over."""
        if _expired(state, admission.now()):
            return True
        if admission.guard is not None:
            try:
                await admission.guard(state)
            except WorkflowDeadline:
                return True
        return False

    async def resource_request(state):
        if _expired(dict(state), admission.now()):
            return _deadline_failure()
        update = await shared_request(state)
        return {**update, "route": "failed" if update.get("status") == "failed" else "resource_wait"}

    async def prepare(state):
        state = dict(state)
        if await boundary(state):
            return _deadline_failure()
        started = {"started_at": state.get("started_at") or admission.now().isoformat()}
        req = request(state, "prepare", step_correlation_id(state, "prepare"))
        calls = {"step_calls": _bumped(state, "prepare")}
        try:
            result = await call(req)
        except RunControlPending:
            raise
        except Exception as exc:  # noqa: BLE001 -- transport/identity: a deferral, not a failure
            return retry(state, "prepare", f"transport:{type(exc).__name__}: {exc}"[:500], **started, **calls)
        if result.status == "retry":
            return retry(state, "prepare", result.reason or "retry", result.retry_after_sec, **started, **calls)
        if result.status == "terminal":
            return {**started, **calls, "status": "running", "route": "finish", "outcome": result.outcome,
                    "reason": result.reason, "attempt_id": result.attempt_id or state.get("attempt_id"),
                    "execution_receipt": result.execution_receipt, "last_error": None}
        return {**started, **calls, **done_update(state, result), "status": "running",
                "route": "resource_request", "attempt_id": result.attempt_id, "hold": None, "lease": None}

    async def generate(state):
        state = dict(state)
        corr = step_correlation_id(state, "generate")
        calls = {"step_calls": _bumped(state, "generate")}

        async def operation(held: dict) -> ReverieVisualStepResultV1:
            budget = ReverieVisualRunBriefV1.model_validate(held["brief"]).timeout_sec
            return await call(request(held, "generate", corr, gpu_lease=held["lease"]), budget)

        try:
            result = await admission.execute({**state, "step_correlation_id": corr}, operation)
        except RunControlPending:
            raise
        except WorkflowDeadline:
            released = await admission.release(state, "workflow_deadline")
            return _deadline_failure({**released, **calls})
        except HoldRecalled:
            # Released by the runtime before generate started: queue afresh.
            return {**calls, "status": "waiting_resource", "route": "resource_request",
                    "lease": None, "hold": None}
        except HoldLost:
            released = await admission.release(state, "hold_lost", keep_requeued=True)
            return {**released, **calls, "status": "waiting_resource", "route": "resource_request"}
        except Exception as exc:  # noqa: BLE001 -- timeout/transport: a deferral, not a failure
            if str(exc).startswith("run_control:"):
                raise RunControlPending(str(exc)) from exc
            released = await admission.release(state, "step_retry")
            return retry(state, "resource_request", f"transport:{type(exc).__name__}: {exc}"[:500],
                         **released, **calls)
        # Caption (and anything after) never carries the diffusion hold.
        if result.status == "done":
            released = await admission.release(state, "generated")
            elapsed = float(result.elapsed_sec or 0.0)
            return {**released, **calls, **done_update(state, result), "status": "running", "route": "caption",
                    "attempt_id": result.attempt_id or state.get("attempt_id"),
                    "artifact_sha256": result.artifact_sha256 or state.get("artifact_sha256"),
                    "generate_elapsed_sec": float(state.get("generate_elapsed_sec") or 0.0) + elapsed}
        if result.status == "retry":
            released = await admission.release(state, "step_retry")
            return retry(state, "resource_request", result.reason or "retry", result.retry_after_sec,
                         **released, **calls)
        released = await admission.release(state, "terminal")
        return {**released, **calls, "status": "running", "route": "finish", "outcome": result.outcome,
                "reason": result.reason, "execution_receipt": result.execution_receipt, "last_error": None}

    async def caption(state):
        state = dict(state)
        if await boundary(state):
            return _deadline_failure()
        req = request(state, "caption", step_correlation_id(state, "caption"))
        calls = {"step_calls": _bumped(state, "caption")}
        try:
            result = await call(req)
        except RunControlPending:
            raise
        except Exception as exc:  # noqa: BLE001
            return retry(state, "caption", f"transport:{type(exc).__name__}: {exc}"[:500], **calls)
        if result.status == "retry":
            if result.reason == NEEDS_GENERATE:
                # The recorded image is gone from disk: back through the hold to regenerate.
                return {**calls, "status": "waiting_resource", "route": "resource_request",
                        "reason": NEEDS_GENERATE, "last_error": NEEDS_GENERATE, "lease": None, "hold": None,
                        "retries": int(state.get("retries") or 0) + 1}
            return retry(state, "caption", result.reason or "retry", result.retry_after_sec, **calls)
        update = {**calls, "status": "running", "route": "finish", "outcome": result.outcome,
                  "chain_id": result.chain_id or state.get("chain_id"),
                  "execution_receipt": result.execution_receipt, "reason": result.reason, "last_error": None}
        if result.status == "done":
            update.update(done_update(state, result))
        return update

    async def retry_wait(state):
        state = dict(state)
        if _expired(state, admission.now()):
            return _deadline_failure()
        until = datetime.fromisoformat(state["retry_at"])
        if admission.now() < until:
            interrupt({"reason": "retrying", "until": state["retry_at"]})
        if _expired(state, admission.now()):
            return _deadline_failure()
        return {"status": "retrying", "route": state.get("retry_node") or "resource_request"}

    async def finish(state):
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed", "finished_at": admission.now().isoformat()}

    async def failed(state):
        state = dict(state)
        released = await admission.release(state, "failed")
        error = state.get("last_error")
        if error == "workflow_deadline":
            error = RETRY_WINDOW_EXPIRED
        sent = await send_abandon(run_step, state, reason=error or "failed")
        return {**released, "status": "failed", "last_error": error,
                "finished_at": admission.now().isoformat(), "abandon_sent": sent or bool(state.get("abandon_sent"))}

    graph = StateGraph(ReverieVisualState)
    for name, node in {
        "prepare": prepare, "resource_request": resource_request, "resource_wait": resource_wait,
        "generate": generate, "caption": caption, "retry_wait": retry_wait,
        "finish": finish, "failed": failed,
    }.items():
        graph.add_node(name, node)
    graph.add_edge(START, "prepare")
    route = lambda s: s.get("route") or "failed"  # noqa: E731
    graph.add_conditional_edges("prepare", route, ["retry_wait", "resource_request", "finish", "failed"])
    # By status, not route: the runtime's restart fence re-enters here via aupdate_state(as_node=...).
    graph.add_conditional_edges("resource_request",
                                lambda s: "failed" if s.get("status") == "failed" else "resource_wait",
                                ["resource_wait", "failed"])
    graph.add_conditional_edges("resource_wait", after_wait,
                                {"granted": "generate", "request": "resource_request", "failed": "failed"})
    graph.add_conditional_edges("generate", route, ["caption", "retry_wait", "resource_request", "finish", "failed"])
    graph.add_conditional_edges("caption", route, ["finish", "retry_wait", "resource_request", "failed"])
    graph.add_conditional_edges("retry_wait", route, ["prepare", "resource_request", "caption", "failed"])
    graph.add_edge("finish", END)
    graph.add_edge("failed", END)
    return graph.compile(checkpointer=checkpointer)


async def send_abandon(run_step: RunStep | None, state: dict, *, reason: str) -> bool:
    """Tell thought to close the attempt (best effort, bounded; never raises). An attempt left
    ``active`` blocks every later claim for the dispatch. False when there is nothing to abandon
    or thought could not be reached."""
    if run_step is None or not state.get("attempt_id") or state.get("abandon_sent"):
        return False
    try:
        req = ReverieVisualStepRequestV1(
            run_id=state["run_id"], correlation_id=step_correlation_id(state, "abandon"), step="abandon",
            visual_request=_visual_request(state), attempt_id=state["attempt_id"])
        result = await asyncio.wait_for(run_step(req, None), ABANDON_TIMEOUT_SEC)
        return result.correlation_id == req.correlation_id and result.run_id == req.run_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("reverie_visual_abandon_failed run=%s attempt=%s reason=%s err=%s",
                       state.get("run_id"), state.get("attempt_id"), reason, exc)
        return False
