"""GPU pool hold around the self_study.reflect graph (stage 4.5).

cortex-exec submits ``workflow="self_study.reflect"`` WITH admission, but before this module the
admission runtime had no reflect graph and drove such a run through the curiosity graph (the
registry fell back to it). This is the reflect workflow's own admitted shell:

    resource_request -> resource_wait -> llm_call -> finish

``llm_call`` runs under ``AdmissionRuntime.execute`` (the hold is heartbeated while cortex answers)
and passes the hold's ref to the one LLM call, so the gateway attaches it to the run's own hold
instead of queueing it behind that hold (spec, "Corrections from building 4.4" item 4).
"""
from __future__ import annotations

from typing import Any

from app.admitted_graph import AdmissionDeps, RunControlPending, WorkflowDeadline, resource_nodes
from app.reflect_graph import Deps, ReflectRunState, make_nodes


def build_admitted_reflect_graph(deps: Deps, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    original = make_nodes(deps)
    resource_request, resource_wait, after_wait = resource_nodes(admission)

    async def llm_call(state: ReflectRunState) -> dict:
        try:
            result = await admission.execute(dict(state), original["llm_call"])
            return {**result, "status": "running", "last_error": None}
        except WorkflowDeadline:
            released = await admission.release(dict(state), "workflow_deadline")
            return {**released, "status": "failed", "last_error": "workflow_deadline"}
        except RunControlPending:
            raise
        except Exception as exc:  # noqa: BLE001 -- transport failure / lost hold: bounded re-try
            attempt = int(state.get("attempt") or 0) + 1
            error = f"{type(exc).__name__}: {exc}"[:500]
            if attempt >= admission.max_attempts:
                released = await admission.release(dict(state), "attempt_failed")
                return {**released, "status": "failed", "attempt": attempt, "last_error": error}
            released = await admission.release(dict(state), "attempt_failed", keep_requeued=True)
            return {**released, "status": "waiting_resource", "attempt": attempt, "last_error": error}

    async def finish(state: ReflectRunState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: ReflectRunState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    g = StateGraph(ReflectRunState)
    for name, node in {"resource_request": resource_request, "resource_wait": resource_wait,
                       "llm_call": llm_call, "finish": finish, "failed": failed}.items():
        g.add_node(name, node)
    g.add_edge(START, "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_wait,
                            {"granted": "llm_call", "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("llm_call", lambda s: (
        "resource_request" if s.get("status") == "waiting_resource"
        else "failed" if s.get("status") == "failed" else "finish"))
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
