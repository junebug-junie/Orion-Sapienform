"""journal.compose: one journal entry, composed under a GPU pool hold (admitted only).

    resource_request -> resource_wait -> compose -> publish -> finish

* ``compose`` runs under ``AdmissionRuntime.execute`` (the hold is heartbeated while cortex answers)
  and sends the ``journal.compose`` cortex verb with the hold's ref, so the gateway attaches the
  call to this run's hold. Waiting for the hold is ``resource_wait``'s job and is never an attempt:
  a busy pool at 06:00 means the run waits, bounded by ``admission.deadline_at``.
* ``publish`` lets the hold go (no GPU needed), then publishes ``journal.entry.write.v1`` with the
  brief's fixed ``entry_id`` and the ``created_at`` checkpointed by ``compose``. A crash after the
  publish resumes at ``publish`` and re-sends the identical write: sql-writer upserts by entry_id
  and orion-actions' post-persist email path dedupes by entry_id, so it cannot double-send.
  A publish failure raises; the driver's bounded checkpoint-resume retries it without recomposing.

Contract: orion/schemas/journal_compose_run.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, TypedDict

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, replay_if_requeued,
    resource_nodes, taken_back,
)
from orion.journaler.schemas import JournalEntryDraftV1, JournalEntryWriteV1
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.journal_compose_run import JournalComposeRunBriefV1

ComposeFn = Callable[..., Awaitable[JournalEntryDraftV1]]
"""(brief, *, run_id, correlation_id, gpu_lease) -> draft. Raises on ANY failure (transport,
non-ok result, empty or unparseable text): each is a bounded, retried attempt."""
PublishFn = Callable[[JournalEntryWriteV1], Awaitable[bool]]


class JournalComposeState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]
    admission: dict[str, Any]
    requested_at: str
    attempt: int
    hold_takebacks: int
    lease: dict[str, Any] | None
    hold: dict[str, Any] | None
    hold_seq: int
    turn_fence: int
    retry_at: str | None
    retry_node: str | None
    tail_attempts: dict[str, int]
    status: str
    last_error: str | None
    draft: dict[str, Any] | None
    created_at: str | None
    published: bool


def build_write(state: dict[str, Any]) -> JournalEntryWriteV1:
    """The write this run publishes -- a pure function of checkpointed state, so a replay is
    byte-identical (same entry_id, same created_at)."""
    brief = JournalComposeRunBriefV1.model_validate(state["brief"])
    draft = JournalEntryDraftV1.model_validate(state["draft"])
    trigger = brief.trigger
    return JournalEntryWriteV1(
        entry_id=brief.entry_id,
        created_at=datetime.fromisoformat(state["created_at"]),
        author=brief.author,
        mode=draft.mode,
        title=draft.title,
        body=draft.body,
        source_kind=trigger.source_kind,
        source_ref=trigger.source_ref,
        correlation_id=state["correlation_id"],
        spawned_correlation_id=trigger.spawned_correlation_id,
        trigger_kind=trigger.trigger_kind,
    )


def finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    brief = state.get("brief") or {}
    return {
        "line": "journal",
        "entry_id": brief.get("entry_id"),
        "trigger_kind": (brief.get("trigger") or {}).get("trigger_kind"),
        "published": bool(state.get("published")),
        "attempts": int(state.get("attempt") or 0),
    }


def build_journal_compose_graph(compose: ComposeFn, publish: PublishFn, admission: AdmissionDeps,
                                checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    resource_request, resource_wait, after_wait = resource_nodes(admission)

    async def operation(state: dict[str, Any]) -> dict[str, Any]:
        lease = state["lease"]
        ref = GpuLeaseRefV1.model_validate({k: lease[k] for k in ("lease_id", "generation", "role", "holder")})
        draft = await compose(JournalComposeRunBriefV1.model_validate(state["brief"]),
                              run_id=state["run_id"], correlation_id=state["correlation_id"], gpu_lease=ref)
        if not draft.body.strip():
            raise ValueError("empty_generation")
        return {"draft": draft.model_dump(mode="json"), "created_at": datetime.now(timezone.utc).isoformat(),
                "attempt": int(state.get("attempt") or 0) + 1}

    async def compose_node(state: JournalComposeState) -> dict:
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
            # An urgent preemption can surface as a failed call: replay it, never spend an attempt.
            replay = await replay_if_requeued(admission, dict(state), {"status": "waiting_resource"})
            if replay is not None:
                return replay
            attempt = int(state.get("attempt") or 0) + 1
            error = f"{type(exc).__name__}: {exc}"[:500]
            if attempt >= admission.max_attempts:
                released = await admission.release(dict(state), "attempt_failed")
                return {**released, "status": "failed", "attempt": attempt, "last_error": error}
            released = await admission.release(dict(state), "attempt_failed", keep_requeued=True)
            return {**released, "status": "waiting_resource", "attempt": attempt, "last_error": error}

    async def publish_node(state: JournalComposeState) -> dict:
        released = await admission.release(dict(state), "completed")
        write = build_write(dict(state))
        if not await publish(write):
            raise RuntimeError("journal_publish_failed")   # resumed here by the driver, not recomposed
        return {**released, "published": True, "status": "running"}

    async def finish(state: JournalComposeState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: JournalComposeState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    g = StateGraph(JournalComposeState)
    for name, node in {"resource_request": resource_request, "resource_wait": resource_wait,
                       "compose": compose_node, "publish": publish_node, "finish": finish,
                       "failed": failed}.items():
        g.add_node(name, node)
    g.add_edge(START, "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_wait,
                            {"granted": "compose", "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("compose", lambda s: (
        "resource_request" if s.get("status") == "waiting_resource"
        else "failed" if s.get("status") == "failed" else "publish"))
    g.add_edge("publish", "finish")
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
