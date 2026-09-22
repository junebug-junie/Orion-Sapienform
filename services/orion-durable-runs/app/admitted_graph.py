"""Resource and retry boundaries around the established Curiosity graph.

The graph is the only workflow state machine. Each wait exits via interrupt;
the runtime later supplies a wakeup, which is never trusted as a lease grant.
Demand registration is separately checkpointed and idempotent in Postgres.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

from app.graph import CuriosityRunState, Deps, failed_turn_meta, make_nodes
from orion.schemas.durable_run import CURIOSITY_NODES


class RunControlPending(RuntimeError):
    """Operator control: preserve the graph position without treating it as failure."""


class WorkflowDeadline(RuntimeError):
    """The optional overall deadline expired, independently of inference timeout."""


@dataclass
class AdmissionDeps:
    register: Callable[[dict], Awaitable[None]]
    lease: Callable[[str], Awaitable[dict | None]]
    execute: Callable[[dict, Callable], Awaitable[dict]]
    release: Callable[[str, str], Awaitable[None]]
    event: Callable[[dict, str, dict], Awaitable[None]]
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    max_attempts: int = 3
    retry_base_seconds: float = 30.0
    retry_max_seconds: float = 300.0
    guard: Callable[[dict], Awaitable[dict | None]] | None = None


def build_admitted_graph(deps: Deps, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    original = make_nodes(deps)

    async def resource_request(state: CuriosityRunState) -> dict:
        await admission.register(dict(state))
        return {"status": "waiting_resource", "lease": None}

    async def resource_wait(state: CuriosityRunState) -> dict:
        # The request was committed in the preceding node. Re-evaluation after
        # interrupt always looks up the authoritative lease, never the payload.
        lease = await admission.lease(state["run_id"])
        if lease is None:
            interrupt({"reason": "waiting_resource", "run_id": state["run_id"]})
            lease = await admission.lease(state["run_id"])
        if lease is None:
            return {"status": "waiting_resource", "lease": None}
        return {"status": "admitted", "lease": lease}

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
            await admission.release(state["run_id"], "workflow_deadline")
            return {"status": "failed", "last_error": "workflow_deadline", "lease": None, **failed_meta}
        except RunControlPending:
            raise
        except Exception as exc:
            # GraphBubbleUp/interrupt is a BaseException and is not caught here.
            attempt = int(state.get("attempt") or 0) + 1
            error = f"{type(exc).__name__}: {exc}"[:500]
            await admission.release(state["run_id"], "attempt_failed")
            if attempt >= admission.max_attempts:
                return {"status": "failed", "attempt": attempt, "last_error": error, "lease": None, **failed_meta}
            delay = min(admission.retry_max_seconds, admission.retry_base_seconds * 2 ** (attempt - 1))
            return {"status": "retrying", "attempt": attempt, "last_error": error,
                    "lease": None, "retry_node": None, "retry_at": (admission.now() + timedelta(seconds=delay)).isoformat(),
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
                lease = await admission.guard(state)
                while lease is None:
                    interrupt({"reason": "waiting_resource", "run_id": state["run_id"]})
                    lease = await admission.guard(state)
                state["lease"] = lease
            try:
                result = await operation(state)
            except RunControlPending:
                raise
            except Exception as exc:
                attempts = dict(state.get("tail_attempts") or {})
                attempts[name] = attempts.get(name, 0) + 1
                await admission.release(state["run_id"], "node_failed")
                delay = min(admission.retry_max_seconds, admission.retry_base_seconds * 2 ** (attempts[name]-1))
                return {"status": "failed" if attempts[name] >= admission.max_attempts else "retrying",
                        "last_error": f"{type(exc).__name__}: {exc}"[:500], "lease": None,
                        "tail_attempts": attempts, "retry_node": name,
                        "retry_at": (admission.now()+timedelta(seconds=delay)).isoformat()}
            # A control received on another replica while the side effect was
            # in flight stops every subsequent node and completion handoff.
            if admission.guard is not None:
                await admission.guard(state)
            return {**result, "status": "running", "lease": state.get("lease"), "retry_node": None}
        return tail

    async def failed(state: CuriosityRunState) -> dict:
        await admission.release(state["run_id"], "failed")
        return {"status": "failed"}

    async def finish(state: CuriosityRunState) -> dict:
        await admission.release(state["run_id"], "completed")
        return {"status": "completed"}

    g = StateGraph(CuriosityRunState)
    nodes = {**original, "resource_request": resource_request, "resource_wait": resource_wait,
             "run_started": run_started, "harness_turn": harness_turn, "retry_wait": retry_wait, "failed": failed, "finish": finish}
    for name in CURIOSITY_NODES[1:-1]:
        nodes[name] = guarded_tail(name, original[name])
    for name, node in nodes.items():
        g.add_node(name, node)
    g.add_edge(START, "resource_request")
    g.add_edge("resource_request", "resource_wait")
    g.add_conditional_edges("resource_wait", lambda s: "run_started" if s.get("lease") else "resource_request")
    g.add_edge("run_started", "harness_turn")
    g.add_conditional_edges("harness_turn", lambda s: {"retrying": "retry_wait", "failed": "failed"}.get(s.get("status"), "read_turn_result"))
    g.add_conditional_edges("retry_wait", lambda s: s.get("retry_node") or "resource_request")
    for a, b in zip(CURIOSITY_NODES[1:], CURIOSITY_NODES[2:]):
        g.add_conditional_edges(a, lambda s, successor=b: "retry_wait" if s.get("status") == "retrying" else "failed" if s.get("status") == "failed" else successor)
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
