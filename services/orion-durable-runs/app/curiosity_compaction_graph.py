"""curiosity.compaction: topic passes -> overall synthesis, one held call per node.

The shared Postgres artifact checkpoints every accepted answer BEFORE the node
returns. A replay reloads that cursor, so even a crash between artifact commit
and LangGraph checkpoint never repeats a saved call. Store writes fence run_id;
a later sleep may retry a terminal failure under a new generation, but an old
worker cannot overwrite it. No LLM work executes in the dream service.
"""
from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, failed_attempt,
    replay_if_requeued, resource_nodes, taken_back,
)
from orion.curiosity.compaction import journal_entry, next_prompt, record_answer
from orion.schemas.curiosity_compaction import CURIOSITY_COMPACTION_VERB, CuriosityCompactionBriefV1
from orion.schemas.gpu_pool import GpuLeaseRefV1


class CompactionState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict
    admission: dict
    requested_at: str
    attempt: int
    hold_takebacks: int
    lease: dict | None
    hold: dict | None
    hold_seq: int
    turn_fence: int
    status: str
    last_error: str | None
    digested: bool
    result: dict


def finish_detail(state):
    return {"line": "compactor", "compaction_id": state["brief"]["compaction_id"], **(state.get("result") or {})}


def build_curiosity_compaction_graph(store, call_verb_text, publish_journal, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph
    resource_request, resource_wait, after_wait = resource_nodes(admission)

    async def load(state):
        brief = CuriosityCompactionBriefV1.model_validate(state["brief"])
        return await asyncio.to_thread(store.load, brief.compaction_id, state["run_id"])

    async def prepare(state):
        row = await load(state)
        if not row["state"]["batches"] and row["status"] == "pending":
            row["status"] = "empty"
            await asyncio.to_thread(store.persist, row)
        return {"digested": row["status"] in {"completed", "empty"}, "status": "running"}

    async def compact(state):
        row = await load(state)
        if row["status"] == "completed":
            return {"digested": True, "attempt": 0, "status": "running"}
        brief = CuriosityCompactionBriefV1.model_validate(state["brief"])
        async def operation(held):
            raw = await call_verb_text(
                CURIOSITY_COMPACTION_VERB, {"curiosity_compaction_prompt": next_prompt(row)}, brief.llm_route,
                gpu_lease=GpuLeaseRefV1.model_validate(held["lease"]), timeout_sec=brief.timeout_sec - 30,
                user_text="Consolidate the dated Curiosity evidence.",
            )
            return {"artifact": record_answer(row, raw)}
        try:
            answer = await admission.execute(dict(state), operation)
            artifact = answer["artifact"]
            await asyncio.to_thread(store.persist, artifact)
            return {"digested": artifact["status"] == "completed", "attempt": 0,
                    "status": "running", "last_error": None}
        except RunControlPending:
            raise
        except WorkflowDeadline:
            released = await admission.release(dict(state), "workflow_deadline")
            return {**released, "status": "failed", "last_error": "workflow_deadline"}
        except HoldRecalled:
            return {"status": "waiting_resource", "lease": None, "hold": None}
        except HoldLost as exc:
            return await taken_back(admission, dict(state), exc.release_reason, str(exc), {"status": "waiting_resource"})
        except Exception as exc:
            replay = await replay_if_requeued(admission, dict(state), {"status": "waiting_resource"})
            if replay is not None:
                return replay
            # Shared policy: transport/timeouts are bounded attempts, re-queued at once. A rejected
            # answer (``record_answer``'s ValueError/ValidationError) changes nothing, so the retry
            # would re-send the same prompt: it fails now (14 days to 2026-10-23: 22 runs ended on
            # these after up to 3 attempts, 0 recovered; none were JSON parse errors).
            return await failed_attempt(admission, dict(state), exc, waiting={"status": "waiting_resource"},
                                        backoff=False)

    async def finalize(state):
        if admission.guard is not None:
            await admission.guard(dict(state))
        released = await admission.release(dict(state), "completed")
        row = await load(state)
        entry_id = None
        if row["status"] == "completed":
            entry = journal_entry(row)
            entry.correlation_id = state["correlation_id"]
            if not await publish_journal(entry):
                raise RuntimeError("curiosity compaction journal publish failed")
            entry_id = entry.entry_id
        elif row["status"] != "empty":
            raise RuntimeError("cannot finalize unfinished compaction")
        return {**released, "status": "running", "result": {
            "artifact_status": row["status"], "topics": len(row["state"].get("topics") or []),
            "source_count": len(row["state"]["sources"]), "journal_entry_id": entry_id,
        }}

    async def finish(state):
        return {**await admission.release(dict(state), "completed"), "status": "completed"}

    async def failed(state):
        return {**await admission.release(dict(state), "failed"), "status": "failed"}

    def after_compact(state):
        if state.get("status") == "failed":
            return "failed"
        if state.get("status") == "waiting_resource":
            return "resource_request"
        return "finalize" if state.get("digested") else "compact"

    graph = StateGraph(CompactionState)
    for name, node in {"prepare": prepare, "resource_request": resource_request, "resource_wait": resource_wait,
                       "compact": compact, "finalize": finalize, "finish": finish, "failed": failed}.items():
        graph.add_node(name, node)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", lambda s: "finalize" if s.get("digested") else "resource_request")
    graph.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    graph.add_conditional_edges("resource_wait", after_wait,
                                {"granted": "compact", "request": "resource_request", "failed": "failed"})
    graph.add_conditional_edges("compact", after_compact)
    graph.add_edge("finalize", "finish")
    graph.add_edge("finish", END)
    graph.add_edge("failed", END)
    # September's live corpus needs 182 map calls before reduction. LangGraph's
    # default 25-step ceiling is not a window-size limit; no input may be dropped
    # to fit it. A failed ceiling still leaves the artifact cursor resumable.
    return graph.compile(checkpointer=checkpointer).with_config(recursion_limit=10000)
