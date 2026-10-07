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

from app.graph import CuriosityRunState, Deps, failed_turn_meta, is_urgent, make_nodes
from app.pool_hold import URGENT_PREEMPT
from orion.schemas.durable_run import CURIOSITY_NODES

logger = logging.getLogger(__name__)


class RunControlPending(RuntimeError):
    """Operator control: preserve the graph position without treating it as failure."""


class WorkflowDeadline(RuntimeError):
    """The optional overall deadline expired, independently of inference timeout."""


class HoldRecalled(RuntimeError):
    """The hold was already being recalled when the work node was about to start. The runtime has
    released it; the run queues afresh. Not a failed attempt."""


class HoldLost(RuntimeError):
    """The pool took this run's hold back mid-node: a recall ran out its grace (gpu2's max_hold
    seat limit, an owner reclaiming its card, an unlend), the heartbeat was lost, or the pool ended
    the hold. The turn is stopped. NOT a failed attempt (stage 4.3/4.5: "re-queued under the same
    lease id"; "a lost hold stops the turn, and the run keeps the same lease id and its place in
    line"): every admitted graph releases with ``keep_requeued=True`` -- the run waits for the SAME
    hold when the pool kept it in line, and asks afresh when the pool ended it -- then replays the
    node. A refusal that is a property of the run (deadline, ...) is never raised as HoldLost.

    Live 2026-09-26..28: 19 runs failed on ``HoldLost`` (each recall spent one of three attempts;
    self-sense failed on the first), all of them the pool taking a seat back, none the run's fault."""

    release_reason = "hold_lost"


class HoldPreempted(HoldLost):
    """An urgent run took this run's slot mid-node; the pool re-queued the hold in its original
    place. The node replays when it is granted again. Not a failed attempt. The one HoldLost that
    records ``run.preempted`` (via release's pool read) and is released as ``urgent_preempt``."""

    release_reason = URGENT_PREEMPT


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
    # One pool read: did the pool take this run's hold back (re-queued it: urgent pause, a recall past
    # its grace, a lost heartbeat)? The release reason to use (``urgent_preempt`` / ``hold_lost``), or
    # None. For a node whose failed turn comes back as a result rather than an exception.
    requeued: Callable[[dict], Awaitable[str | None]] | None = None
    # Bound on pool take-backs per run (DURABLE_RUNS_HOLD_MAX_TAKEBACKS); 0 = unbounded. A take-back
    # is not a failed attempt, but a step that never fits the seats it lands on must not replay
    # forever: past this many, the run fails with ``hold_takeback_limit``.
    max_takebacks: int = 0


# Urgent runs (brief.urgent) must end in a report within minutes: at most two
# attempts per node, 10 s·2^n backoff. Never looser than the service budget.
URGENT_MAX_ATTEMPTS = 2
URGENT_RETRY_BASE_SECONDS = 10.0


def retry_cannot_finish(state: dict, retry_at: datetime) -> bool:
    """A retry starting at ``retry_at`` cannot get a whole attempt before the run's deadline.

    An attempt is ``brief.timeout_sec`` long, and the motor inside it is budgeted from that
    same figure (Hub keeps a finalize reserve out of it), so a retry with less left than one
    attempt is cut by ``workflow_deadline`` mid-motor and ends with nothing: run a153451fe423's
    attempt 2 restarted from zero with ~290 s of a 900 s attempt left. Runs with no deadline
    (or no timeout on the brief) always retry, as before."""
    deadline = (state.get("admission") or {}).get("deadline_at")
    timeout = (state.get("brief") or {}).get("timeout_sec")
    if not deadline or not timeout:
        return False
    left = (datetime.fromisoformat(deadline) - retry_at).total_seconds()
    return left < float(timeout)


def retry_budget(admission: AdmissionDeps, state: dict) -> tuple[int, float]:
    """(max_attempts, retry_base_seconds) for this run."""
    if is_urgent(state):
        return min(admission.max_attempts, URGENT_MAX_ATTEMPTS), URGENT_RETRY_BASE_SECONDS
    return admission.max_attempts, admission.retry_base_seconds


async def taken_back(admission: AdmissionDeps, state: dict, reason: str, error: str, waiting: dict) -> dict:
    """The pool took the run's hold back mid-node: keep the re-queued hold (``keep_requeued``) and
    return ``waiting`` so the node replays -- no attempt spent. Counted in ``hold_takebacks``; past
    ``admission.max_takebacks`` the run fails instead (``hold_takeback_limit:<n>``)."""
    count = int(state.get("hold_takebacks") or 0) + 1
    if admission.max_takebacks and count > admission.max_takebacks:
        released = await admission.release(state, "hold_takeback_limit")
        return {**released, "status": "failed", "hold_takebacks": count,
                "last_error": f"hold_takeback_limit:{admission.max_takebacks}: {error}"[:500]}
    released = await admission.release(state, reason, keep_requeued=True)
    return {**released, **waiting, "hold_takebacks": count, "last_error": error[:500]}


async def replay_if_requeued(admission: AdmissionDeps, state: dict, waiting: dict) -> dict | None:
    """A node's turn came back failed: if the pool says it took the hold back meanwhile, the failure
    is the pool's (the turn's calls could not attach to the aborted hold). Returns ``taken_back``'s
    update (the node replays, or the take-back limit fails the run); None when it was a real failure."""
    reason = None if admission.requeued is None else await admission.requeued(state)
    if not reason:
        return None
    return await taken_back(admission, state, reason, f"HoldLost: gpu_hold_requeued:{reason}", waiting)


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
        except HoldRecalled:
            # Released by the runtime before the turn started: straight back to resource_request.
            return {"status": "retrying", "lease": None, "hold": None, "retry_node": None,
                    "retry_at": admission.now().isoformat()}
        except HoldLost as exc:
            # The pool took the hold back (urgent pause, recall past its grace, lost heartbeat): wait
            # for the same hold again now (or ask afresh if the pool ended it), attempt untouched.
            update = await taken_back(admission, dict(state), exc.release_reason, f"{type(exc).__name__}: {exc}",
                                      {"status": "retrying", "retry_node": None,
                                       "retry_at": admission.now().isoformat()})
            return {**update, **failed_meta}
        except Exception as exc:
            # GraphBubbleUp/interrupt is a BaseException and is not caught here.
            attempt = int(state.get("attempt") or 0) + 1
            error = f"{type(exc).__name__}: {exc}"[:500]
            max_attempts, retry_base = retry_budget(admission, state)
            if attempt >= max_attempts:
                released = await admission.release(dict(state), "attempt_failed")
                return {**released, "status": "failed", "attempt": attempt, "last_error": error, **failed_meta}
            delay = min(admission.retry_max_seconds, retry_base * 2 ** (attempt - 1))
            retry_at = admission.now() + timedelta(seconds=delay)
            if retry_cannot_finish(dict(state), retry_at):
                # A doomed retry only burns the GPU until the deadline kills it: end now.
                released = await admission.release(dict(state), "attempt_failed")
                return {**released, "status": "failed", "attempt": attempt,
                        "last_error": f"retry_skipped_insufficient_time: {error}"[:500], **failed_meta}
            # A hold the pool already re-queued meanwhile keeps its lease_id and place; one still
            # granted (at this generation or, since nothing heartbeats it through the backoff, a newer
            # one) is handed back.
            released = await admission.release(dict(state), "attempt_failed", keep_requeued=True)
            return {**released, "status": "retrying", "attempt": attempt, "last_error": error,
                    "retry_node": None, "retry_at": retry_at.isoformat(),
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
                max_attempts, retry_base = retry_budget(admission, state)
                delay = min(admission.retry_max_seconds, retry_base * 2 ** (attempts[name]-1))
                return {**released, "status": "failed" if attempts[name] >= max_attempts else "retrying",
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
        # Urgent runs end in a report, never in Hub's Door-A composition, so nothing would ever
        # release a kept hold: always hand it back here.
        outcome = state.get("outcome") or {}
        lease = state.get("lease")
        if bool(outcome.get("reach_out")) and lease and not is_urgent(state):
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
