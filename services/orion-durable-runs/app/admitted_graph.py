"""Resource and retry boundaries around the established Curiosity graph.

The graph is the only workflow state machine. Each wait exits via interrupt; the runtime later
supplies a wakeup, which is never trusted as a grant: ``resource_wait`` always re-reads the GPU
pool's own answer for the run's hold (stage 4.5 -- the pool is the only scheduler).

State keys this shell owns (all checkpointed, so a restart resumes with them):

* ``hold``     -- the run's current pool hold request ``{"lease_id", "request_id"}``, queued or
                  granted. Re-asking with the same ``request_id`` is idempotent at the pool.
* ``hold_seq`` -- which ``<run_id>:<seq>`` request id the hold was taken under; bumped only when a
                  previous hold is gone (released, cancelled), never on a re-ask.
* ``lease``    -- the granted hold's ``GpuLeaseRefV1`` fields; every LLM call of the run carries it.
* ``turn_fence`` -- bumped when a restarted driver fences a turn still running under the SAME hold
                  generation, so the replay gets a new turn identity (graph.turn_correlation_id).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from app.graph import CuriosityRunState, Deps, failed_turn_meta, make_nodes
from orion.schemas.durable_run import CURIOSITY_NODES

logger = logging.getLogger(__name__)


class RunControlPending(RuntimeError):
    """Operator control: preserve the graph position without treating it as failure."""


class WorkflowDeadline(RuntimeError):
    """The optional overall deadline expired, independently of inference timeout."""


# ``AdmissionDeps.lease`` outcomes (resource_wait's decision).
GRANTED, WAITING, GONE, REFUSED = "granted", "waiting", "gone", "refused"


@dataclass
class AdmissionDeps:
    # resource_request: make sure the run has a live pool hold request -> state update
    register: Callable[[dict], Awaitable[dict]]
    # resource_wait: the pool's answer for the run's hold -> (GRANTED|WAITING|GONE|REFUSED, state update)
    lease: Callable[[dict], Awaitable[tuple[str, dict]]]
    execute: Callable[[dict, Callable], Awaitable[dict]]
    # (state, reason, keep_requeued=False) -> state update clearing ``lease`` (and ``hold`` unless kept)
    release: Callable[..., Awaitable[dict]]
    event: Callable[[dict, str, dict], Awaitable[None]]
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    max_attempts: int = 3
    retry_base_seconds: float = 30.0
    retry_max_seconds: float = 300.0
    # Node boundary: deadline + control, then heartbeat the hold. Returns the (possibly now
    # released) lease; never waits for a GPU -- the tail nodes do not use one.
    guard: Callable[[dict], Awaitable[dict | None]] | None = None
    # Door-A: keep heartbeating the run's hold after ``finish`` until Hub releases it.
    keep_for_outreach: Callable[[dict], Awaitable[None]] | None = None


def resource_nodes(admission: AdmissionDeps):
    """``resource_request`` / ``resource_wait`` shared by every admitted workflow graph."""
    from langgraph.types import interrupt

    async def resource_request(state) -> dict:
        return {"lease": None, **await admission.register(dict(state))}

    async def resource_wait(state) -> dict:
        # The hold was requested (and checkpointed) by the preceding node. Re-evaluation after the
        # interrupt always asks the pool again; a wakeup payload is never a grant.
        outcome, update = await admission.lease(dict(state))
        if outcome == WAITING:
            interrupt({"reason": "waiting_resource", "run_id": state["run_id"]})
            outcome, update = await admission.lease(dict(state))
        return update

    def after_wait(state) -> str:
        if state.get("lease"):
            return "granted"
        return "failed" if state.get("status") == "failed" else "request"

    return resource_request, resource_wait, after_wait


def build_admitted_graph(deps: Deps, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    original = make_nodes(deps)
    resource_request, resource_wait, after_wait = resource_nodes(admission)

    async def harness_turn(state: CuriosityRunState) -> dict:
        # The identity this attempt was (or would have been) issued under,
        # captured from the same state the wrapped node derives it from.
        # Every failure return below clears `lease`, after which it can no
        # longer be re-derived -- so a failed attempt stashes it here for the
        # terminal `failed` detail. Guarded: a malformed lease must still
        # raise inside the try below and take the release/attempt path.
        failed_meta = failed_turn_meta(state)
        try:
            result = await admission.execute(dict(state), original["harness_turn"])
            return {**result, "status": "running", "last_error": None, "retry_at": None}
        except WorkflowDeadline:
            released = await admission.release(dict(state), "workflow_deadline")
            return {**released, "status": "failed", "last_error": "workflow_deadline", **failed_meta}
        except RunControlPending:
            raise
        except Exception as exc:
            # GraphBubbleUp/interrupt is a BaseException and is not caught here.
            attempt = int(state.get("attempt") or 0) + 1
            error = f"{type(exc).__name__}: {exc}"[:500]
            if attempt >= admission.max_attempts:
                released = await admission.release(dict(state), "attempt_failed")
                return {**released, "status": "failed", "attempt": attempt, "last_error": error, **failed_meta}
            # A hold the pool already re-queued (lost heartbeat, recall past its grace) keeps its
            # lease_id and place; one still granted is handed back for the backoff.
            released = await admission.release(dict(state), "attempt_failed", keep_requeued=True)
            delay = min(admission.retry_max_seconds, admission.retry_base_seconds * 2 ** (attempt - 1))
            return {**released, "status": "retrying", "attempt": attempt, "last_error": error,
                    "retry_node": None, "retry_at": (admission.now() + timedelta(seconds=delay)).isoformat(),
                    **failed_meta}

    async def run_started(state: CuriosityRunState) -> dict:
        return {"status": "running"}

    async def retry_wait(state: CuriosityRunState) -> dict:
        until = datetime.fromisoformat(state["retry_at"])
        if admission.now() < until:
            interrupt({"reason": "retrying", "until": state["retry_at"]})
        return {"status": "retrying"}

    def guarded_tail(name, operation):
        async def tail(state):
            state = dict(state)
            if admission.guard is not None:
                # Deadline/control, and the hold's heartbeat. A recalled or lost hold is let go here
                # (the node boundary the pool's clawback grace waits for); the tail itself needs no GPU.
                state["lease"] = await admission.guard(state)
                if state["lease"] is None:
                    state["hold"] = None
            try:
                result = await operation(state)
            except RunControlPending:
                raise
            except Exception as exc:
                attempts = dict(state.get("tail_attempts") or {})
                attempts[name] = attempts.get(name, 0) + 1
                released = await admission.release(state, "node_failed")
                delay = min(admission.retry_max_seconds, admission.retry_base_seconds * 2 ** (attempts[name]-1))
                return {**released, "status": "failed" if attempts[name] >= admission.max_attempts else "retrying",
                        "last_error": f"{type(exc).__name__}: {exc}"[:500],
                        "tail_attempts": attempts, "retry_node": name,
                        "retry_at": (admission.now()+timedelta(seconds=delay)).isoformat()}
            # A control received on another replica while the side effect was in flight stops
            # every subsequent node and completion handoff.
            if admission.guard is not None:
                state["lease"] = await admission.guard(state)
                if state["lease"] is None:
                    state["hold"] = None
            return {**result, "status": "running", "lease": state.get("lease"), "hold": state.get("hold"),
                    "retry_node": None}
        return tail

    async def failed(state: CuriosityRunState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    async def finish(state: CuriosityRunState) -> dict:
        # Door-A (2026-09-22): when Orion asked to share, keep the run's pool hold through Hub's
        # composition turn. finish_detail carries the hold's ref (``gpu_lease``); Hub releases it
        # via /runs/{id}/release-outreach-lease, and durable-runs heartbeats it until then.
        outcome = state.get("outcome") or {}
        lease = state.get("lease")
        if bool(outcome.get("reach_out")) and lease:
            if admission.guard is not None:
                try:
                    lease = await admission.guard(dict(state))
                except Exception:  # noqa: BLE001 -- the run is done; Hub re-checks the hold with the pool
                    logger.warning("door_a_hold_check_failed run=%s", state.get("run_id"), exc_info=True)
            if lease:
                await admission.event(state, "run.outreach_pending", {**lease, "lane": lease.get("role")})
                if admission.keep_for_outreach is not None:
                    await admission.keep_for_outreach({**dict(state), "lease": lease})
                return {"status": "completed", "lease": lease}
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    g = StateGraph(CuriosityRunState)
    nodes = {**original, "resource_request": resource_request, "resource_wait": resource_wait,
             "run_started": run_started, "harness_turn": harness_turn, "retry_wait": retry_wait, "failed": failed, "finish": finish}
    for name in CURIOSITY_NODES[1:-1]:
        nodes[name] = guarded_tail(name, original[name])
    for name, node in nodes.items():
        g.add_node(name, node)
    g.add_edge(START, "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_wait,
                            {"granted": "run_started", "request": "resource_request", "failed": "failed"})
    g.add_edge("run_started", "harness_turn")
    g.add_conditional_edges("harness_turn", lambda s: {"retrying": "retry_wait", "failed": "failed"}.get(s.get("status"), "read_turn_result"))
    g.add_conditional_edges("retry_wait", lambda s: s.get("retry_node") or "resource_request")
    for a, b in zip(CURIOSITY_NODES[1:], CURIOSITY_NODES[2:]):
        g.add_conditional_edges(a, lambda s, successor=b: "retry_wait" if s.get("status") == "retrying" else "failed" if s.get("status") == "failed" else successor)
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
