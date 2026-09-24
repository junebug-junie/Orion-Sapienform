"""The lease lifecycle as a checkpointed LangGraph run (``thread_id = lease_id``).

Shape: ``admit -> wait -> wait -> ... -> END``. ``wait`` is an ``interrupt``: the lease
holds no task or socket while it sits in line, is backlogged, is held, or waits to retry. The
pool runtime resumes it with one event (from the scheduler or an RPC verb); ``wait`` then runs
the pure transition table below and appends to ``history``. A pool restart resumes every thread
from its checkpoint with its original ``created_at`` and priority, so it keeps its place.

Only ``released`` is final. ``dead_letter`` and ``unavailable`` stay interrupted so an operator
can replay them; the checkpoint history is what the Hub graph walker draws.
"""
from __future__ import annotations

import operator
from datetime import datetime, timedelta
from typing import Annotated, Any, TypedDict

from orion.gpu_pool.config import PoolConfig

FINAL = frozenset({"released"})

# (status, event) -> next status. Anything not listed is rejected, never guessed.
_TABLE: dict[tuple[str, str], str] = {
    ("queued", "grant"): "granted",
    ("queued", "backlog"): "backlogged",
    ("queued", "unavailable"): "unavailable",
    ("queued", "cancel"): "released",
    ("queued", "release_ok"): "released",
    ("backlogged", "requeue"): "queued",
    ("backlogged", "dead_letter"): "dead_letter",
    ("backlogged", "cancel"): "released",
    ("backlogged", "release_ok"): "released",
    ("granted", "heartbeat"): "granted",
    ("granted", "release_ok"): "released",
    ("granted", "release_failed"): "retry_wait",
    ("granted", "recall"): "recalling",
    ("granted", "expire"): "retry_wait",
    ("granted", "cancel"): "released",
    ("recalling", "heartbeat"): "recalling",
    ("recalling", "release_ok"): "released",
    ("recalling", "release_failed"): "retry_wait",
    ("recalling", "abort"): "retry_wait",
    ("recalling", "expire"): "retry_wait",
    ("recalling", "cancel"): "released",
    ("retry_wait", "requeue"): "queued",
    ("retry_wait", "unavailable"): "unavailable",
    ("retry_wait", "cancel"): "released",
    ("retry_wait", "release_ok"): "released",
    ("dead_letter", "replay"): "queued",
    ("dead_letter", "cancel"): "released",
    ("unavailable", "replay"): "queued",
    ("unavailable", "cancel"): "released",
}

# Events that consume an attempt when they send a lease back to retry.
_FAILURES = frozenset({"release_failed", "expire", "abort"})


class InvalidTransition(ValueError):
    pass


class LeaseState(TypedDict, total=False):
    lease_id: str
    request: dict[str, Any]          # the GpuLeaseRequestV1 acquire payload (+ operator flag)
    status: str
    role: str | None
    generation: int
    attempt: int
    replays: int
    created_at: str
    queued_since: str | None
    granted_at: str | None
    recall_by: str | None
    not_before: str | None
    expires_at: str | None
    reason: str | None
    history: Annotated[list[dict[str, Any]], operator.add]


def initial_state(lease_id: str, request: dict[str, Any], now: datetime) -> LeaseState:
    return LeaseState(
        lease_id=lease_id, request=request, status="queued", role=None, generation=0,
        attempt=1, replays=0, created_at=now.isoformat(), queued_since=now.isoformat(),
        granted_at=None, recall_by=None, not_before=None, expires_at=None, reason=None,
        history=[{"event": "admit", "status": "queued", "at": now.isoformat()}],
    )


def transition(state: LeaseState, event: dict[str, Any], cfg: PoolConfig) -> dict[str, Any]:
    """Pure: the state update for one event. Raises InvalidTransition on an illegal pair."""
    kind = event["type"]
    now = datetime.fromisoformat(event["at"])
    status = state["status"]
    nxt = _TABLE.get((status, kind))
    if nxt is None:
        raise InvalidTransition(f"{status} -/-> {kind}")

    upd: dict[str, Any] = {"status": nxt, "reason": event.get("reason")}
    ttl = cfg.defaults.hold_lease_ttl_sec if state["request"].get("kind") == "hold" \
        else cfg.defaults.request_lease_ttl_sec

    # Operator holds (the Hub button, the experiment seat) have no heartbeat; max_hold_sec bounds them.
    operator_hold = bool(state["request"].get("operator"))
    expiry = None if operator_hold else (now + timedelta(seconds=ttl)).isoformat()
    if kind == "grant":
        upd.update(role=event["role"], generation=int(state.get("generation") or 0) + 1,
                   granted_at=now.isoformat(), expires_at=expiry, recall_by=None)
    elif kind == "heartbeat":
        upd.update(expires_at=expiry, reason=state.get("reason"))
    elif kind == "recall":
        upd.update(recall_by=event["recall_by"])
    elif kind in ("requeue", "replay"):
        upd.update(queued_since=now.isoformat(), not_before=None, role=None)
        if kind == "replay":
            upd.update(attempt=1, replays=int(state.get("replays") or 0) + 1)

    if nxt == "retry_wait" and kind in _FAILURES:
        attempt = int(state.get("attempt") or 1)
        if not state["request"].get("retryable"):
            # Nobody will use a re-grant: end here, keeping the cause.
            upd.update(status="released", reason=f"{kind}:{event.get('reason') or kind}")
        elif attempt >= cfg.defaults.retry.max_attempts:
            upd.update(status="dead_letter", role=None, reason=event.get("reason") or kind)
        else:
            delay = cfg.defaults.retry.delay(attempt)
            upd.update(attempt=attempt + 1, role=None,
                       not_before=(now + timedelta(seconds=delay)).isoformat())
    if upd["status"] in ("released", "dead_letter", "unavailable", "backlogged", "retry_wait"):
        upd.setdefault("role", None if upd["status"] != "released" else state.get("role"))
        upd.update(expires_at=None, recall_by=None)

    upd["history"] = [{
        "event": kind, "from": status, "status": upd["status"], "at": now.isoformat(),
        "role": event.get("role") or state.get("role"), "reason": event.get("reason"),
        "attempt": upd.get("attempt", state.get("attempt")),
    }]
    return upd


def build_lease_graph(cfg_getter, checkpointer: Any):
    """``cfg_getter`` returns the live PoolConfig, so a YAML reload applies to the next event."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    async def admit(state: LeaseState) -> dict:
        return {}

    async def wait(state: LeaseState) -> dict:
        # Re-executed from the top on resume; interrupt() then returns the resume event.
        event = interrupt({"lease_id": state["lease_id"], "status": state["status"]})
        return transition(state, event, cfg_getter())

    def after(state: LeaseState) -> str:
        return END if state["status"] in FINAL else "wait"

    g = StateGraph(LeaseState)
    g.add_node("admit", admit)
    g.add_node("wait", wait)
    g.add_edge(START, "admit")
    g.add_edge("admit", "wait")
    g.add_conditional_edges("wait", after, {"wait": "wait", END: END})
    return g.compile(checkpointer=checkpointer)
