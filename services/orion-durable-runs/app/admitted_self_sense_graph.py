"""Resource wait around the self-sense-eval graph.

Same admission shell as `admitted_graph.py` (register → wait for lease →
work → release), but the work is `ask_questions → publish → finish`, not
the curiosity harness/journal pipeline.

Live 2026-09-22: every admitted run — including `workflow=self_sense_eval` —
was driven through `build_admitted_graph` (curiosity only). Orion got the
placeholder brief.prompt ("self-sense eval: four fixed questions") as a
single investigation turn, wrote a curiosity journal, and never scored.
This module is the missing second admitted graph.
"""

from __future__ import annotations

from typing import Any

from app.admitted_graph import AdmissionDeps, RunControlPending, WorkflowDeadline
from app.self_sense_graph import Deps, SelfSenseAskFailed, SelfSenseRunState, make_nodes
from orion.schemas.durable_run import SELF_SENSE_EVAL_NODES


def build_admitted_self_sense_graph(deps: Deps, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    original = make_nodes(deps)

    async def resource_request(state: SelfSenseRunState) -> dict:
        await admission.register(dict(state))
        return {"status": "waiting_resource", "lease": None}

    async def resource_wait(state: SelfSenseRunState) -> dict:
        lease = await admission.lease(state["run_id"])
        if lease is None:
            from langgraph.types import interrupt

            interrupt({"reason": "waiting_resource", "run_id": state["run_id"]})
            lease = await admission.lease(state["run_id"])
        if lease is None:
            return {"status": "waiting_resource", "lease": None}
        return {"status": "admitted", "lease": lease}

    async def ask_questions(state: SelfSenseRunState) -> dict:
        try:
            result = await admission.execute(dict(state), original["ask_questions"])
            return {**result, "status": "running", "last_error": None}
        except WorkflowDeadline:
            await admission.release(state["run_id"], "workflow_deadline")
            return {"status": "failed", "last_error": "workflow_deadline", "lease": None}
        except RunControlPending:
            raise
        except SelfSenseAskFailed as exc:
            # Transport blip: release the lease and re-queue for a fresh grant.
            # Partial answers stay on state; ask_questions skips keys already done.
            await admission.release(state["run_id"], "attempt_failed")
            return {
                "status": "waiting_resource",
                "lease": None,
                "last_error": f"{type(exc).__name__}: {exc}"[:500],
            }
        except Exception as exc:  # noqa: BLE001
            await admission.release(state["run_id"], "attempt_failed")
            return {
                "status": "failed",
                "last_error": f"{type(exc).__name__}: {exc}"[:500],
                "lease": None,
            }

    async def publish(state: SelfSenseRunState) -> dict:
        if admission.guard is not None:
            await admission.guard(state)
        return await original["publish"](state)

    async def finish(state: SelfSenseRunState) -> dict:
        await admission.release(state["run_id"], "completed")
        return {"status": "completed"}

    async def failed(state: SelfSenseRunState) -> dict:
        await admission.release(state["run_id"], "failed")
        return {"status": "failed"}

    g = StateGraph(SelfSenseRunState)
    g.add_node("resource_request", resource_request)
    g.add_node("resource_wait", resource_wait)
    g.add_node("ask_questions", ask_questions)
    g.add_node("publish", publish)
    g.add_node("finish", finish)
    g.add_node("failed", failed)
    g.add_edge(START, "resource_request")
    g.add_edge("resource_request", "resource_wait")
    g.add_conditional_edges(
        "resource_wait",
        lambda s: "ask_questions" if s.get("lease") else "resource_request",
    )
    g.add_conditional_edges(
        "ask_questions",
        lambda s: (
            "resource_request" if s.get("status") == "waiting_resource"
            else ("failed" if s.get("status") == "failed" else "publish")
        ),
    )
    g.add_edge("publish", "finish")
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    # Keep the contract name visible for inspectors; not used as a node list
    # walker here (admission emits events from astream updates).
    _ = SELF_SENSE_EVAL_NODES
    return g.compile(checkpointer=checkpointer)
