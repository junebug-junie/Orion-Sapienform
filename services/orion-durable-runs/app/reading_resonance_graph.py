"""One tool-free judgment under a background hold; frozen brief and checked result survive restart.

Transport failures retry through durable admission. Rejected model output is recorded once.
The persist checkpoint runs after releasing the hold and never regenerates the judgment.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, TypedDict

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, listener_wait, replay_if_requeued,
    resource_nodes, taken_back,
)
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.reading_resonance import ReadingResonanceRunBriefV1, ReadingResonanceResultV1
from orion.world_pulse_read.resonance import build_judge_prompt, check_judgment

CallFn = Callable[..., Awaitable[dict[str, Any]]]
PersistFn = Callable[[ReadingResonanceResultV1], Awaitable[None]]
RESONANCE_MIN_ATTEMPTS = 4


class ReadingResonanceState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]
    admission: dict[str, Any]
    requested_at: str
    attempt: int
    hold_takebacks: int
    listener_waits: int   # RPCs published to nobody this run waited out (listener_wait; never an attempt)
    lease: dict[str, Any] | None
    hold: dict[str, Any] | None
    hold_seq: int
    turn_fence: int
    retry_at: str | None
    retry_node: str | None
    tail_attempts: dict[str, int]
    status: str
    last_error: str | None
    result: dict[str, Any] | None
    persisted: bool


def finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    result = state.get("result") or {}
    return {"line": "reading_resonance", "seed_id": result.get("seed_id"),
            "resonance_id": result.get("resonance_id"), "check": result.get("check"),
            "persisted": bool(state.get("persisted")), "attempts": state.get("attempt", 0)}


def build_reading_resonance_graph(call: CallFn, persist: PersistFn, admission: AdmissionDeps,
                                  checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    resource_request, resource_wait, after_wait = resource_nodes(admission)

    async def operation(state: dict[str, Any]) -> dict[str, Any]:
        lease = state["lease"]
        ref = GpuLeaseRefV1.model_validate({k: lease[k] for k in ("lease_id", "generation", "role", "holder")})
        brief = ReadingResonanceRunBriefV1.model_validate(state["brief"])
        prompt = build_judge_prompt(brief.packet, brief.retained_texts)
        response = await call(prompt, brief=brief, run_id=state["run_id"],
                              correlation_id=state["correlation_id"], gpu_lease=ref)
        raw = str(response.get("text") or "")
        if not raw.strip():
            raise ValueError("empty_generation")
        judgment, check = check_judgment(brief.packet, raw, retained_texts=brief.retained_texts, prompt=prompt)
        # Rejected answers are terminal receipts, never re-asked until one says yes.
        result = ReadingResonanceResultV1(
            resonance_id=brief.packet.packet_id, seed_id=brief.packet.seed_id, run_id=state["run_id"],
            packet=brief.packet, judgment=judgment, check=check, raw=raw,
            model=response.get("model"), lane=brief.llm_route,
            latency_ms=int(response.get("latency_ms") or 0), completed_at=admission.now())
        return {"result": result.model_dump(mode="json"), "attempt": int(state.get("attempt") or 0) + 1}

    async def judge_node(state: ReadingResonanceState) -> dict:
        try:
            result = await admission.execute(dict(state), operation)
            return {**result, "status": "running", "last_error": None}
        except WorkflowDeadline:
            released = await admission.release(dict(state), "workflow_deadline")
            return {**released, "status": "failed", "last_error": "workflow_deadline"}
        except RunControlPending:
            raise
        except HoldRecalled:
            return {"status": "waiting_resource", "lease": None, "hold": None}
        except HoldLost as exc:   # HoldPreempted too: the pool keeps its place, no attempt spent
            return await taken_back(admission, dict(state), exc.release_reason, f"{type(exc).__name__}: {exc}",
                                    {"status": "waiting_resource"})
        except Exception as exc:  # noqa: BLE001 -- a real compose attempt failed: bounded re-try
            # The answering service restarting: the call reached nobody. Wait it out, replay, no attempt.
            parked = await listener_wait(admission, dict(state), exc,
                                         waiting={"status": "retrying", "retry_node": None})
            if parked is not None:
                return parked
            # An urgent preemption can surface as a failed call: replay it, never spend an attempt.
            replay = await replay_if_requeued(admission, dict(state), {"status": "waiting_resource"})
            if replay is not None:
                return replay
            attempt = int(state.get("attempt") or 0) + 1
            error = f"{type(exc).__name__}: {exc}"[:500]
            if attempt >= max(admission.max_attempts, RESONANCE_MIN_ATTEMPTS):
                released = await admission.release(dict(state), "attempt_failed")
                return {**released, "status": "failed", "attempt": attempt, "last_error": error}
            released = await admission.release(dict(state), "attempt_failed", keep_requeued=True)
            delay = min(admission.retry_max_seconds, admission.retry_base_seconds * 2 ** (attempt - 1))
            return {**released, "status": "retrying", "attempt": attempt, "last_error": error,
                    "retry_node": None, "retry_at": (admission.now() + timedelta(seconds=delay)).isoformat()}

    async def retry_wait(state: ReadingResonanceState) -> dict:
        # The driver resumes this interrupt only once retry_at has passed (admission_runtime).
        from langgraph.types import interrupt

        if admission.now() < datetime.fromisoformat(state["retry_at"]):
            interrupt({"reason": "retrying", "until": state["retry_at"]})
        return {"status": "retrying"}

    async def persist_node(state: ReadingResonanceState) -> dict:
        released = await admission.release(dict(state), "completed")
        await persist(ReadingResonanceResultV1.model_validate(state["result"]))
        return {**released, "persisted": True, "status": "running"}

    async def finish(state: ReadingResonanceState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: ReadingResonanceState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    g = StateGraph(ReadingResonanceState)
    for name, node in {"resource_request": resource_request, "resource_wait": resource_wait,
                       "judge": judge_node, "retry_wait": retry_wait, "persist": persist_node,
                       "finish": finish, "failed": failed}.items():
        g.add_node(name, node)
    g.add_edge(START, "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_wait,
                            {"granted": "judge", "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("judge", lambda s: (
        "resource_request" if s.get("status") == "waiting_resource"
        else "retry_wait" if s.get("status") == "retrying"
        else "failed" if s.get("status") == "failed" else "persist"))
    g.add_edge("retry_wait", "resource_request")
    g.add_edge("persist", "finish")
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
