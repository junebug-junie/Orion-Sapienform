"""One admitted reading turn; Hub retains parsing, evidence gates and landing."""

from typing import Any, TypedDict

from app.admitted_graph import (
    HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, replay_if_requeued, resource_nodes, taken_back,
)
from app.graph import turn_correlation_id
from orion.schemas.reading_turn import (
    ReadingRunBriefV1,
    ReadingTurnRequestV1,
    ReadingTurnResultV1,
)


class ReadingState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict
    admission: dict
    requested_at: str
    attempt: int
    hold_takebacks: int   # times the pool took the hold back mid-node (never an attempt)
    lease: dict | None
    hold: dict | None
    hold_seq: int
    turn_fence: int
    status: str
    last_error: str | None
    result: dict | None
    harness_turn_meta: dict


def finish_detail(state):
    result = ReadingTurnResultV1.model_validate(state["result"])
    return {
        "turn_correlation_id": result.correlation_id,
        "reading_result": result.model_dump(mode="json"),
    }


def build_reading_graph(run_turn, admission, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    resource_request, resource_wait, after_wait = resource_nodes(admission)

    async def operation(state):
        request = ReadingTurnRequestV1(
            run_id=state["run_id"],
            correlation_id=turn_correlation_id(state),
            brief=ReadingRunBriefV1.model_validate(state["brief"]),
            gpu_lease=state["lease"],
        )
        result = await run_turn(request)
        if (
            result.run_id != request.run_id
            or result.correlation_id != request.correlation_id
        ):
            raise ValueError("reading turn identity mismatch")
        if result.ok and not result.text.strip():
            raise ValueError("empty_generation")
        return {
            "result": result.model_dump(mode="json"),
            "harness_turn_meta": {"turn_correlation_id": result.correlation_id},
            "attempt": int(state.get("attempt") or 0) + 1,
            "status": "running" if result.ok else "failed",
            "last_error": None if result.ok else result.error or "empty_generation",
        }

    async def reading_turn(state):
        try:
            result = await admission.execute(dict(state), operation)
            if result.get("status") == "failed":
                # A failed turn is a result here, not an exception: an urgent preemption's too.
                replay = await replay_if_requeued(admission, dict(state), {"status": "waiting_resource"})
                if replay is not None:
                    return replay
            return result
        except RunControlPending:
            raise
        except HoldRecalled:
            return {"status": "waiting_resource", "lease": None, "hold": None}
        except WorkflowDeadline:
            released = await admission.release(dict(state), "workflow_deadline")
            return {**released, "status": "failed", "last_error": "workflow_deadline"}
        except HoldLost as exc:   # HoldPreempted too: the pool keeps its place, no attempt spent
            return await taken_back(admission, dict(state), exc.release_reason,
                                    f"{type(exc).__name__}: {exc}", {"status": "waiting_resource"})
        except Exception as exc:
            # Reading's existing queue owns bounded actual-work retries. Waiting
            # and recalls never spend those attempts; do not nest another retry loop.
            released = await admission.release(dict(state), "attempt_failed")
            return {
                **released,
                "status": "failed",
                "last_error": f"turn_exception:{type(exc).__name__}: {exc}"[:500],
                "harness_turn_meta": {
                    "turn_correlation_id": turn_correlation_id(state)
                },
            }

    async def finish(state):
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state):
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    graph = StateGraph(ReadingState)
    for name, node in {
        "resource_request": resource_request,
        "resource_wait": resource_wait,
        "reading_turn": reading_turn,
        "finish": finish,
        "failed": failed,
    }.items():
        graph.add_node(name, node)
    graph.add_edge(START, "resource_request")
    graph.add_conditional_edges(
        "resource_request",
        lambda s: "failed" if s.get("status") == "failed" else "resource_wait",
    )
    graph.add_conditional_edges(
        "resource_wait",
        after_wait,
        {"granted": "reading_turn", "request": "resource_request", "failed": "failed"},
    )
    graph.add_conditional_edges(
        "reading_turn",
        lambda s: (
            "resource_request"
            if s.get("status") == "waiting_resource"
            else "failed"
            if s.get("status") == "failed"
            else "finish"
        ),
    )
    graph.add_edge("finish", END)
    graph.add_edge("failed", END)
    return graph.compile(checkpointer=checkpointer)
