"""compactor.digest: a compactor day's LLM digest calls under one GPU pool hold (admitted only).

    resource_request -> resource_wait -> digest (loops: one chunk digest, or the merge, per run)
                                     -> finalize -> finish

* ``digest`` makes exactly ONE LLM call per node run, under ``AdmissionRuntime.execute`` (the hold
  is heartbeated while cortex answers), and checkpoints its result before the next call: a restart
  resumes at the first chunk not yet digested, never re-digesting a finished one. The call carries
  the hold's ref (``options.gpu_lease``) so the gateway attaches it to this run's hold. Waiting for
  the hold is ``resource_wait``'s job and is never an attempt: a busy pool at 06:00 is a wait,
  bounded only by ``admission.deadline_at``. Which call is next is decided by the pure step machine
  in ``orion.cognition.compactor.map_reduce`` (chunks, then one merge if it fits; a merge that
  drops a chunk's refs or fails every attempt falls back to the deterministic join).
* A failed call (verb failure, empty / rejected / invalid JSON, transport) is one of
  ``DURABLE_RUNS_RETRY_MAX_ATTEMPTS`` attempts for THAT call; the count resets after each success.
  A chunk that exhausts them fails the run. A merge that exhausts them does not: the day's chunk
  digests are joined (``merge_mode=concatenated``, ``merge_skipped_reason=merge_failed:...``).
* ``finalize`` lets the hold go (no GPU needed), assembles the digest, and sends
  ``CompactorDigestResultV1`` back to cortex-orch as ``workflow_request.durable_digest``; orch writes
  the memory card and the journal entry (stable ids, so a replayed finalize is an upsert) and
  notifies per the schedule's policy. A finalize failure raises: the driver's bounded
  checkpoint-resume retries it without re-running any LLM call.

Contract: orion/schemas/compactor_digest_run.py.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, TypedDict

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, replay_if_requeued,
    resource_nodes, taken_back,
)
from orion.cognition.compactor.constants import COMPACTOR_FINALIZE_RPC_TIMEOUT_SEC
from orion.cognition.compactor.map_reduce import (
    SPECS, CompactorDigestCallError, assemble, build_digest_request_payload, digest_from_payload,
    finalize_request_payload, merge_gave_up, next_call, record_merge, resolve_merge_without_call,
)
from orion.schemas.compactor_digest_run import CompactorDigestResultV1, CompactorDigestRunBriefV1

CortexRpc = Callable[..., Awaitable[dict[str, Any]]]
"""(request_payload, *, timeout_sec, label) -> the decoded cortex-orch result payload. Raises on a
transport / decode failure."""


class CompactorDigestState(TypedDict, total=False):
    run_id: str
    correlation_id: str
    workflow: str
    brief: dict[str, Any]
    admission: dict[str, Any]
    requested_at: str
    attempt: int            # failed attempts of the CURRENT call; reset after each success
    hold_takebacks: int
    lease: dict[str, Any] | None
    hold: dict[str, Any] | None
    hold_seq: int
    turn_fence: int
    status: str
    last_error: str | None
    partials: list[dict[str, Any]]      # chunk digests, in chunk order (checkpointed per call)
    merge: dict[str, Any] | None        # {"status": "done"|"skipped", "digest", "reason"}
    call_log: list[dict[str, Any]]      # one row per call attempt; pool waits are not rows
    gpu_roles: list[str]
    digested: bool                      # every call made; finalize next
    result: dict[str, Any] | None       # small summary of what finalize sent + orch's answer


def finish_detail(state: dict[str, Any]) -> dict[str, Any]:
    brief = state.get("brief") or {}
    result = state.get("result") or {}
    return {
        "line": "compactor",
        "workflow_id": brief.get("workflow_id"),
        "window_label": brief.get("window_label"),
        "chunk_count": result.get("chunk_count"),
        "merge_mode": result.get("merge_mode"),
        "merge_skipped_reason": result.get("merge_skipped_reason"),
        "journal_entry_id": result.get("journal_entry_id"),
        "card_id": result.get("card_id"),
        "attempts": len(state.get("call_log") or []),
        "gpu_roles": list(state.get("gpu_roles") or []),
    }


def build_compactor_digest_graph(cortex_rpc: CortexRpc, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph

    resource_request, resource_wait, after_wait = resource_nodes(admission)

    def _progress(state: dict[str, Any]):
        brief = CompactorDigestRunBriefV1.model_validate(state["brief"])
        return brief, SPECS[brief.kind], list(brief.inputs), list(state.get("partials") or []), state.get("merge")

    async def digest(state: CompactorDigestState) -> dict:
        brief, spec, inputs, partials, merge = _progress(dict(state))
        decided = resolve_merge_without_call(spec, inputs, partials, merge)
        if decided is not None:
            return {"merge": decided, "status": "running"}
        call = next_call(spec, inputs, partials, merge)
        if call is None:
            # Not a new lifecycle status (the driver records "run.<status>" events): a flag.
            return {"status": "running", "digested": True}
        log = list(state.get("call_log") or [])
        roles = list(state.get("gpu_roles") or [])

        async def operation(held: dict[str, Any]) -> dict[str, Any]:
            lease = held["lease"]
            ref = {k: lease[k] for k in ("lease_id", "generation", "role", "holder")}
            payload = await cortex_rpc(
                build_digest_request_payload(
                    spec, call["input"], workflow_id=brief.workflow_id, correlation_id=held["correlation_id"],
                    session_id=brief.session_id, user_id=brief.user_id, llm_route=brief.llm_route,
                    timeout_sec=brief.timeout_sec, gpu_lease=ref),
                timeout_sec=float(brief.timeout_sec), label=call["label"])
            parsed, error = digest_from_payload(spec, payload if isinstance(payload, dict) else {})
            if error is not None:
                raise CompactorDigestCallError(error)
            role = str(lease.get("role") or "")
            update: dict[str, Any] = {
                "attempt": 0,
                "gpu_roles": roles + ([role] if role and role not in roles else []),
            }
            row = {"step": call["label"], "ok": True, "role": role or None}
            if call["kind"] == "chunk":
                update["partials"] = partials + [parsed.model_dump(mode="json")]
                update["call_log"] = log + [row]
            else:
                record, refs_error = record_merge(spec, partials, parsed)
                update["merge"] = record
                update["call_log"] = log + [{**row, "ok": refs_error is None,
                                             **({"error": refs_error} if refs_error else {})}]
            return update

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
        except Exception as exc:  # noqa: BLE001 -- a real digest call failed: bounded re-try
            # An urgent preemption can surface as a failed call: replay it, never spend an attempt.
            replay = await replay_if_requeued(admission, dict(state), {"status": "waiting_resource"})
            if replay is not None:
                return replay
            attempt = int(state.get("attempt") or 0) + 1
            error = f"{type(exc).__name__}: {exc}"[:500]
            failed_row = {"step": call["label"], "ok": False, "error": error[:300]}
            if attempt >= admission.max_attempts:
                if call["kind"] == "merge":
                    # The chunk digests are real output over real input: join them, keep the hold
                    # (finalize lets it go), and finish the day.
                    return {"merge": merge_gave_up(error), "attempt": 0, "call_log": log + [failed_row],
                            "status": "running", "last_error": None}
                released = await admission.release(dict(state), "attempt_failed")
                return {**released, "status": "failed", "attempt": attempt, "last_error": error,
                        "call_log": log + [failed_row]}
            released = await admission.release(dict(state), "attempt_failed", keep_requeued=True)
            return {**released, "status": "waiting_resource", "attempt": attempt, "last_error": error,
                    "call_log": log + [failed_row]}

    async def finalize(state: CompactorDigestState) -> dict:
        if admission.guard is not None:
            # Operator pause/cancel wins before the card + journal are written (RunControlPending
            # keeps this position). A passed deadline does not: every LLM call is already done,
            # and throwing the day's digest away now would only waste it.
            try:
                await admission.guard(dict(state))
            except WorkflowDeadline:
                pass
        released = await admission.release(dict(state), "completed")
        brief, spec, inputs, partials, merge = _progress(dict(state))
        assembled = assemble(spec, inputs, partials, merge, window_label=brief.window_label)
        result = CompactorDigestResultV1(
            run_id=state["run_id"], kind=brief.kind, workflow_id=brief.workflow_id,
            window_label=brief.window_label, llm_route=brief.llm_route,
            attempts=list(state.get("call_log") or []), gpu_roles=list(state.get("gpu_roles") or []),
            finalize=dict(brief.finalize), **assembled)
        payload = await cortex_rpc(finalize_request_payload(brief, result, correlation_id=state["correlation_id"]),
                                   timeout_sec=float(COMPACTOR_FINALIZE_RPC_TIMEOUT_SEC), label="finalize")
        if not isinstance(payload, dict) or not payload.get("ok"):
            error = (payload or {}).get("error") if isinstance(payload, dict) else None
            # Resumed here by the driver (bounded), never re-digested.
            raise RuntimeError(f"compactor_finalize_failed:{error or (payload or {}).get('status')}")
        workflow = ((payload.get("metadata") or {}).get("workflow") or {})
        journal = workflow.get("journal_entry") if isinstance(workflow.get("journal_entry"), dict) else {}
        return {**released, "status": "running", "result": {
            "chunk_count": assembled["chunk_count"], "merge_mode": assembled["merge_mode"],
            "merge_skipped_reason": assembled["merge_skipped_reason"],
            "journal_entry_id": journal.get("entry_id"), "card_id": workflow.get("card_id"),
            "journal_body_chars": workflow.get("journal_body_chars"),
        }}

    async def finish(state: CompactorDigestState) -> dict:
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed"}

    async def failed(state: CompactorDigestState) -> dict:
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed"}

    def after_digest(state: CompactorDigestState) -> str:
        status = state.get("status")
        if status == "waiting_resource":
            return "resource_request"
        if status == "failed":
            return "failed"
        return "finalize" if state.get("digested") else "digest"

    g = StateGraph(CompactorDigestState)
    for name, node in {"resource_request": resource_request, "resource_wait": resource_wait,
                       "digest": digest, "finalize": finalize, "finish": finish, "failed": failed}.items():
        g.add_node(name, node)
    g.add_edge(START, "resource_request")
    g.add_conditional_edges("resource_request", lambda s: "failed" if s.get("status") == "failed" else "resource_wait")
    g.add_conditional_edges("resource_wait", after_wait,
                            {"granted": "digest", "request": "resource_request", "failed": "failed"})
    g.add_conditional_edges("digest", after_digest)
    g.add_edge("finalize", "finish")
    g.add_edge("finish", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)
