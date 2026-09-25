"""Resource wait around the self-sense-eval graph.

Same admission shell as `admitted_graph.py` (request a GPU pool hold → wait
for the pool's grant → work under the hold → release), but the work is
`ask_questions → publish → finish`, not the curiosity harness/journal pipeline.

Live 2026-09-22: every admitted run — including `workflow=self_sense_eval` —
was driven through `build_admitted_graph` (curiosity only). Orion got the
placeholder brief.prompt ("self-sense eval: four fixed questions") as a
single investigation turn, wrote a curiosity journal, and never scored.
This module is the missing second admitted graph.
"""

from __future__ import annotations

from typing import Any

from app.admitted_graph import AdmissionDeps, RunControlPending, WorkflowDeadline, resource_nodes
from app.self_sense_graph import Deps, SelfSenseAskFailed, SelfSenseRunState, make_nodes
from orion.schemas.durable_run import SELF_SENSE_EVAL_NODES


def build_admitted_self_sense_graph(deps: Deps, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    original = make_nodes(deps)

    resource_request, resource_wait, after_wait = resource_nodes(admission)

    async def ask_questions(state: SelfSenseRunState) -> dict:
        try:
            result = await admission.execute(dict(state), original["ask_questions"])
            return {**result, "status": "running", "last_error": None}
        except WorkflowDeadline:
            released = await admission.release(dict(state), "workflow_deadline")
            return {**released, "status": "failed", "last_error": "workflow_deadline"}
        except RunControlPending:
            raise
        except SelfSenseAskFailed as exc:
            # Transport blip: hand the hold back (or keep it if the pool already re-queued it) and
            # wait for a fresh grant. Partial answers stay on state; ask_questions skips keys done.
            released = await admission.release(dict(state), "attempt_failed", keep_requeued=True)
            return {
                **released,
                "status": "waiting_resource",
                "last_error": f"{type(exc).__name__}: {exc}"[:500],
            }
        except Exception as exc:  # noqa: BLE001
            released = await admission.release(dict(state), "attempt_failed")
            return {
                **released,
                "status": "failed",
                "last_error": f"{type(exc).__name__}: {exc}"[:500],
            }

    async def publish(state: SelfSenseRunState) -> dict:
        state = dict(state)
        if admission.guard is not None:
            # Node boundary: deadline/control, and the hold is let go if the pool recalled it.
            state["lease"] = await admission.guard(state)
        return await original["publish"](state)

    async def finish(state: SelfSenseRunState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: SelfSenseRunState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    g = StateGraph(SelfSenseRunState)
    g.add_node("resource_request", resource_request)
    g.add_node("resource_wait", resource_wait)
    g.add_node("ask_questions", ask_questions)
    g.add_node("publish", publish)
    g.add_node("finish", finish)
    g.add_node("failed", failed)
    g.add_edge(START, "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_wait,
                            {"granted": "ask_questions", "request": "resource_request", "failed": "failed"})
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
