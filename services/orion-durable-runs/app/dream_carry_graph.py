"""dream.carry: one dream carried through words and pictures (admitted only).

Design: docs/superpowers/specs/2026-10-10-dream-carry-through-design.md ("As built").
Contract: orion/schemas/dream_carry.py.

    next_hop -> resource_request -> resource_wait -> text_hop -> next_hop        (even hop: text)
    next_hop -> image_submit -> image_wait (polls itself) -> next_hop             (odd hop: image)
    next_hop -> finish_dream -> finish                                            (all hops made)
    any hop stage: retry_wait -> same stage;  deadline / terminal -> finish_dream (partial)

* **text hop** (even index): one ``DreamCarryStepRequestV1(step="text")`` to orion-dream, run
  under the run's own LLM hold (``llm.route.metacog_background``) exactly like journal.compose's
  compose node: ``resource_request -> resource_wait -> text_hop`` under ``AdmissionRuntime.execute``,
  the hold released as soon as the answer is checkpointed. A ``retry`` answer (or transport trouble)
  hands the hold back, backs off and asks again; it never spends an attempt (as reverie.visual).
* **image hop** (odd index): a child ``reverie.visual`` run submitted inside durable-runs (the
  store, not the bus) with ``dream_hop`` set: it paints the previous text hop's ``image_prompt`` and
  captions it. Its run id is deterministic per (carry, hop), so a replayed submit dedupes. The carry
  holds nothing while it waits: ``image_wait`` re-reads the child's terminal fact every
  ``child_poll_sec`` (an interrupt, woken by the driver at ``retry_at``).
* **finish_dream**: ``step="finish"`` with every hop made and ``stopped_reason``; orion-dream
  publishes the dream. It keeps retrying for ``finish_grace_sec`` past the deadline so a partial
  carry is still written; after that the run fails with the last error.
* **Deadline** (``admission.deadline_at``): any hop stage past it goes to ``finish_dream`` with
  ``stopped_reason="deadline at hop N: <last reason>"``: partial, never failed-and-empty. A carry
  that made no hop at all has nothing to write and fails instead (no empty dream).
* Hops already in the checkpoint are never redone: the next hop index is ``len(hops)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, TypedDict
from uuid import NAMESPACE_URL, uuid5

from app.admitted_graph import (
    AdmissionDeps, HoldLost, HoldRecalled, RunControlPending, WorkflowDeadline, resource_nodes, taken_back,
)
from orion.schemas.dream_carry import (
    DreamCarryBriefV1,
    DreamCarryHopV1,
    DreamCarryStepRequestV1,
    DreamCarryStepResultV1,
    clip_image_prompt,
    dream_hop_dispatch_id,
)
from orion.schemas.durable_run import DurableRunRequestV1
from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.resource_admission import ResourceRequirementV1
from orion.schemas.reverie_visual import VisualRunRequestV1
from orion.schemas.reverie_visual_run import (
    REVERIE_VISUAL_HOLD_LANE,
    REVERIE_VISUAL_MAX_RETRY_WINDOW_SEC,
    REVERIE_VISUAL_WORKFLOW,
    DreamHopImageV1,
    ReverieVisualRunBriefV1,
    reverie_visual_run_id,
)

# How often a waiting image hop re-reads its child's terminal fact (one indexed row read).
CHILD_POLL_SEC = 30.0
DEFAULT_FINISH_GRACE_SEC = 1800.0
# Floor for any backoff, including an orion-dream retry_after_sec of 0.
MIN_BACKOFF_SEC = 1.0
# The child outcome that means "a picture was painted and seen".
PRODUCED = "produced"

# (request, budget_sec) -> result. Raises on transport/identity trouble (a retry, never an attempt).
CarryStep = Callable[[DreamCarryStepRequestV1, float | None], Awaitable[DreamCarryStepResultV1]]
# Submit one child reverie.visual run (idempotent per run_id).
SubmitChild = Callable[[DurableRunRequestV1], Awaitable[Any]]
# (child run_id) -> (terminal status, terminal detail), or None while it is still running.
ChildTerminal = Callable[[str], Awaitable[tuple[str, dict] | None]]


@dataclass
class DreamCarryDeps:
    run_step: CarryStep
    submit_child: SubmitChild
    child_terminal: ChildTerminal
    finish_grace_sec: float = DEFAULT_FINISH_GRACE_SEC
    child_poll_sec: float = CHILD_POLL_SEC


class DreamCarryState(TypedDict, total=False):
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
    retry_at: str | None
    retry_node: str | None
    route: str
    # Every hop made, in order (DreamCarryHopV1 dumps). The next hop's index is len(hops).
    hops: list
    step_calls: dict
    retries: int
    retry_streak: int
    # The last stage reason (a deferral being retried, ...): what a deadline stop names.
    reason: str | None
    # Why the carry finished short; None when every hop was made.
    stopped_reason: str | None
    # The image hop currently waited on, and every child submitted.
    child_run_id: str | None
    child_hop: int | None
    child_run_ids: list
    dream_id: str | None
    started_at: str | None
    finished_at: str | None


def _brief(state: dict) -> DreamCarryBriefV1:
    return DreamCarryBriefV1.model_validate(state["brief"])


def _hops(state: dict) -> list[DreamCarryHopV1]:
    return [DreamCarryHopV1.model_validate(h) for h in (state.get("hops") or [])]


def _deadline(state: dict) -> datetime | None:
    value = (state.get("admission") or {}).get("deadline_at")
    return datetime.fromisoformat(str(value)) if value else None


def _expired(state: dict, now: datetime) -> bool:
    deadline = _deadline(state)
    return deadline is not None and now >= deadline


def step_correlation_id(state: dict, step: str, hop: int) -> str:
    """Per-RPC identity: a late reply for an earlier call can never be read as this one's."""
    n = int((state.get("step_calls") or {}).get(f"{step}:{hop}", 0))
    fence = int(state.get("turn_fence") or 0)
    return str(uuid5(NAMESPACE_URL, f"orion:dream-carry:{state['run_id']}:{step}:{hop}:{n}:{fence}"))


def _bumped(state: dict, step: str, hop: int) -> dict:
    calls = dict(state.get("step_calls") or {})
    key = f"{step}:{hop}"
    calls[key] = int(calls.get(key, 0)) + 1
    return calls


def _stop(reason: str) -> dict:
    """Go to finish_dream with the hops made so far and why the carry stopped."""
    return {"status": "stopping", "route": "finish_dream", "stopped_reason": reason[:500],
            "retry_at": None, "retry_node": None, "lease": None}


def _deadline_reason(state: dict, waiting_on: str = "no reason recorded") -> str:
    last = state.get("reason") or state.get("last_error") or waiting_on
    return f"deadline at hop {len(state.get('hops') or [])}: {last}"


def child_request(state: dict, hop_index: int, prompt: str, now: datetime) -> DurableRunRequestV1:
    """The child reverie.visual run for one image hop. A pure function of checkpointed state (plus
    ``requested_at``/``deadline_at``, which the store ignores on a resubmit), so a replay dedupes."""
    run_id = state["run_id"]
    dispatch_id = dream_hop_dispatch_id(run_id, hop_index)
    window_end = now + timedelta(seconds=REVERIE_VISUAL_MAX_RETRY_WINDOW_SEC)
    deadline = _deadline(state)
    child_deadline = min(deadline, window_end) if deadline is not None else window_end
    return DurableRunRequestV1(
        run_id=reverie_visual_run_id(dispatch_id),
        workflow=REVERIE_VISUAL_WORKFLOW,
        correlation_id=state["correlation_id"],
        requested_at=now,
        brief=ReverieVisualRunBriefV1(
            visual_request=VisualRunRequestV1(dispatch_id=dispatch_id, correlation_id=state["correlation_id"]),
            dream_hop=DreamHopImageV1(carry_run_id=run_id, hop_index=hop_index, prompt=prompt)),
        admission=ResourceRequirementV1(
            resource=f"service.route.{REVERIE_VISUAL_HOLD_LANE}", preferred_lane=REVERIE_VISUAL_HOLD_LANE,
            deadline_at=child_deadline),
    )


def _common_detail(state: dict) -> dict[str, Any]:
    brief = state.get("brief") or {}
    return {"line": "dream", "trigger_id": brief.get("trigger_id"),
            "hops_planned": brief.get("hops"), "hops_made": len(state.get("hops") or []),
            "stopped_reason": state.get("stopped_reason"),
            "child_run_ids": list(state.get("child_run_ids") or []),
            "retries": int(state.get("retries") or 0)}


def finish_detail(state: dict) -> dict[str, Any]:
    """``run.completed`` detail."""
    return {**_common_detail(state), "dream_id": state.get("dream_id")}


def terminal_detail(state: dict, status: str) -> dict[str, Any]:
    """``run.failed`` / ``run.cancelled`` detail, read from the checkpoint at terminal time."""
    error = "cancelled" if status == "cancelled" else (state.get("last_error") or status)
    return {**_common_detail(state), "error": error, "last_error": error,
            "child_run_id": state.get("child_run_id")}


def build_dream_carry_graph(carry: DreamCarryDeps, admission: AdmissionDeps, checkpointer: Any):
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    shared_request, shared_wait, _ = resource_nodes(admission)

    def hard_deadline(state: dict) -> datetime | None:
        deadline = _deadline(state)
        return None if deadline is None else deadline + timedelta(seconds=carry.finish_grace_sec)

    def retry(state: dict, node: str, reason: str, retry_after: float | None = None, **extra) -> dict:
        """A deferral: back off, resume at ``node``. Counted, never an attempt. Never sleeps past the
        bound that applies to ``node`` (the deadline for a hop, deadline + grace for finish)."""
        if retry_after is not None:
            delay = max(MIN_BACKOFF_SEC, float(retry_after))
        else:
            streak = int(state.get("retry_streak") or 0)
            delay = max(MIN_BACKOFF_SEC, min(admission.retry_max_seconds,
                                             admission.retry_base_seconds * 2 ** min(streak, 30)))
        now = admission.now()
        retry_at = now + timedelta(seconds=delay)
        bound = hard_deadline(state) if node == "finish_dream" else _deadline(state)
        if bound is not None and bound > now:
            retry_at = min(retry_at, bound)
        return {**extra, "status": "retrying", "route": "retry_wait", "retry_node": node,
                "retry_at": retry_at.isoformat(), "reason": reason[:500], "last_error": reason[:500],
                "retries": int(state.get("retries") or 0) + 1,
                "retry_streak": int(state.get("retry_streak") or 0) + 1}

    async def call(req: DreamCarryStepRequestV1, budget_sec: float | None) -> DreamCarryStepResultV1:
        result = await carry.run_step(req, budget_sec)
        if (result.run_id, result.correlation_id, result.step) != (req.run_id, req.correlation_id, req.step):
            raise ValueError("dream carry step identity mismatch")
        return result

    async def next_hop(state):
        state = dict(state)
        started = {"started_at": state.get("started_at") or admission.now().isoformat()}
        made = len(state.get("hops") or [])
        if made >= _brief(state).hops:
            return {**started, "status": "running", "route": "finish_dream", "stopped_reason": None}
        if _expired(state, admission.now()):
            return {**started, **_stop(_deadline_reason(state))}
        return {**started, "status": "running", "route": "resource_request" if made % 2 == 0 else "image_submit"}

    # --- text hop: under the run's LLM hold ------------------------------------------------------
    async def resource_request(state):
        state = dict(state)
        if _expired(state, admission.now()):
            released = await admission.release(state, "workflow_deadline")
            return {**released, **_stop(_deadline_reason(state, "waiting for the LLM hold"))}
        update = await shared_request(state)
        if update.get("status") == "failed":
            return {**update, **_stop(f"text hop {len(state.get('hops') or [])}: {update.get('last_error')}")}
        return update

    async def resource_wait(state):
        state = dict(state)
        if _expired(state, admission.now()):
            released = await admission.release(state, "workflow_deadline")
            return {**released, **_stop(_deadline_reason(state, "waiting for the LLM hold"))}
        update = await shared_wait(state)
        if update.get("status") == "failed":
            if update.get("last_error") == "workflow_deadline":
                return {**update, **_stop(_deadline_reason(state))}
            return {**update, **_stop(f"text hop {len(state.get('hops') or [])}: {update.get('last_error')}")}
        return update

    def after_wait(state) -> str:
        if state.get("lease"):
            return "text_hop"
        return "finish_dream" if state.get("status") == "stopping" else "resource_request"

    async def text_hop(state):
        state = dict(state)
        hops = list(state.get("hops") or [])
        idx = len(hops)
        if idx % 2 or idx >= _brief(state).hops:
            # This text hop is already made (a replay): never redo it.
            released = await admission.release(state, "completed")
            return {**released, "status": "running", "route": "next_hop"}
        corr = step_correlation_id(state, "text", idx)
        calls = {"step_calls": _bumped(state, "text", idx)}

        async def operation(held: dict) -> DreamCarryStepResultV1:
            lease = held["lease"]
            ref = GpuLeaseRefV1.model_validate({k: lease[k] for k in ("lease_id", "generation", "role", "holder")})
            brief = _brief(held)
            req = DreamCarryStepRequestV1(run_id=held["run_id"], correlation_id=corr, step="text", brief=brief,
                                          hops=_hops(held), hop_index=idx, gpu_lease=ref)
            return await call(req, brief.timeout_sec)

        try:
            result = await admission.execute({**state, "step_correlation_id": corr}, operation)
        except RunControlPending:
            raise
        except WorkflowDeadline:
            released = await admission.release(state, "workflow_deadline")
            return {**released, **calls, **_stop(_deadline_reason(state))}
        except HoldRecalled:
            # Released by the runtime before the hop started: queue afresh, no retry counted.
            return {**calls, "status": "waiting_resource", "route": "resource_request", "lease": None, "hold": None}
        except HoldLost as exc:   # a pool take-back: wait for the same hold again, not a retry
            update = await taken_back(admission, state, exc.release_reason, f"{type(exc).__name__}: {exc}",
                                      {"status": "waiting_resource", "route": "resource_request"})
            if update.get("status") == "failed":
                return {**update, **calls, **_stop(f"text hop {idx}: {update.get('last_error')}")}
            return {**update, **calls}
        except Exception as exc:  # noqa: BLE001 -- timeout/transport/identity: a deferral, not a failure
            if str(exc).startswith("run_control:"):
                raise RunControlPending(str(exc)) from exc
            released = await admission.release(state, "step_retry")
            return retry(state, "resource_request", f"transport:{type(exc).__name__}: {exc}", None,
                         **released, **calls)
        if result.status == "done":
            released = await admission.release(state, "text_done")
            hop = result.hop
            if hop is None or hop.index != idx:
                return retry(state, "resource_request", f"hop_index_mismatch:{getattr(hop, 'index', None)}!={idx}",
                             None, **released, **calls)
            hop = hop.model_copy(update={"image_prompt": clip_image_prompt(hop.image_prompt or "")})
            return {**released, **calls, "hops": hops + [hop.model_dump(mode="json")], "status": "running",
                    "route": "next_hop", "retry_streak": 0, "reason": None, "last_error": None}
        if result.status == "retry":
            released = await admission.release(state, "step_retry")
            return retry(state, "resource_request", result.reason or "retry", result.retry_after_sec,
                         **released, **calls)
        released = await admission.release(state, "terminal")
        return {**released, **calls, **_stop(f"text hop {idx}: {result.reason or 'terminal'}")}

    # --- image hop: a child reverie.visual run, nothing held -------------------------------------
    async def image_submit(state):
        state = dict(state)
        hops = _hops(state)
        idx = len(hops)
        if idx % 2 == 0 or idx >= _brief(state).hops:
            return {"status": "running", "route": "next_hop"}
        if _expired(state, admission.now()):
            return _stop(_deadline_reason(state))
        request = child_request(state, idx, hops[-1].image_prompt or "", admission.now())
        try:
            await carry.submit_child(request)
        except Exception as exc:  # noqa: BLE001
            from orion.durable_runs.registry_store import SubmissionConflict

            if isinstance(exc, SubmissionConflict):   # deterministic: retrying cannot fix it
                return _stop(f"image hop {idx}: child_submission_conflict")
            return retry(state, "image_submit", f"child_submit:{type(exc).__name__}: {exc}")
        children = list(state.get("child_run_ids") or [])
        if request.run_id not in children:
            children.append(request.run_id)
        return {"status": "waiting_child", "route": "image_wait", "child_run_id": request.run_id,
                "child_hop": idx, "child_run_ids": children, "retry_streak": 0,
                "retry_at": (admission.now() + timedelta(seconds=carry.child_poll_sec)).isoformat()}

    async def image_wait(state):
        state = dict(state)
        child = state["child_run_id"]
        idx = int(state.get("child_hop") if state.get("child_hop") is not None else len(state.get("hops") or []))
        expired = _expired(state, admission.now())
        if not expired and state.get("retry_at") and admission.now() < datetime.fromisoformat(state["retry_at"]):
            interrupt({"reason": "waiting_child", "child_run_id": child, "until": state["retry_at"]})
        try:
            terminal = await carry.child_terminal(child)
        except Exception as exc:  # noqa: BLE001 -- a store read failed: look again next poll
            terminal, read_error = None, f"child_read:{type(exc).__name__}: {exc}"
        else:
            read_error = None
        if terminal is None:
            if _expired(state, admission.now()):
                return _stop(f"deadline at hop {idx}: image hop still running ({child})")
            return {"status": "waiting_child", "route": "image_wait", "reason": read_error or state.get("reason"),
                    "retry_at": (admission.now() + timedelta(seconds=carry.child_poll_sec)).isoformat()}
        status, detail = terminal
        detail = detail or {}
        sha, caption = detail.get("artifact_sha256"), (detail.get("caption") or "").strip()
        if status == "completed" and detail.get("outcome") == PRODUCED and sha and caption:
            hop = DreamCarryHopV1(index=idx, kind="image", sha256=sha, caption=caption, child_run_id=child,
                                  elapsed_sec=float(detail.get("visual_elapsed_sec") or 0.0))
            hops = list(state.get("hops") or [])
            if len(hops) == idx:
                hops.append(hop.model_dump(mode="json"))
            return {"hops": hops, "status": "running", "route": "next_hop", "child_run_id": None,
                    "child_hop": None, "reason": None, "last_error": None, "retry_at": None}
        if status != "completed":
            why = detail.get("last_error") or detail.get("error") or detail.get("reason") or status
        elif detail.get("outcome") != PRODUCED:
            why = f"outcome:{detail.get('outcome')}" + (f" ({detail['reason']})" if detail.get("reason") else "")
        else:
            why = "completed without an image or caption"
        return {**_stop(f"image hop {idx}: {why}"), "child_run_id": None, "child_hop": None}

    async def retry_wait(state):
        state = dict(state)
        node = state.get("retry_node") or "resource_request"
        hop_stage = node != "finish_dream"
        if hop_stage and _expired(state, admission.now()):
            return _stop(_deadline_reason(state))
        if admission.now() < datetime.fromisoformat(state["retry_at"]):
            interrupt({"reason": "retrying", "until": state["retry_at"]})
        if hop_stage and _expired(state, admission.now()):
            return _stop(_deadline_reason(state))
        return {"status": "retrying", "route": node}

    # --- the end -----------------------------------------------------------------------------------
    async def finish_dream(state):
        state = dict(state)
        released = await admission.release(state, "completed")   # nothing should be held here
        hops = _hops(state)
        if not hops:
            # Nothing was made: an empty dream is not a dream. Fail with why it stopped.
            return {**released, "status": "failed", "route": "failed",
                    "last_error": state.get("stopped_reason") or state.get("last_error") or "no_hops"}
        bound = hard_deadline(state)
        if bound is not None and admission.now() >= bound:
            return {**released, "status": "failed", "route": "failed",
                    "last_error": f"finish_grace_expired: {state.get('last_error') or 'finish never answered'}"[:500]}
        corr = step_correlation_id(state, "finish", len(hops))
        calls = {"step_calls": _bumped(state, "finish", len(hops))}
        brief = _brief(state)
        req = DreamCarryStepRequestV1(run_id=state["run_id"], correlation_id=corr, step="finish", brief=brief,
                                      hops=hops, stopped_reason=state.get("stopped_reason"))
        try:
            result = await call(req, brief.timeout_sec)
        except RunControlPending:
            raise
        except Exception as exc:  # noqa: BLE001 -- transport: retry, bounded by deadline + grace
            return retry(state, "finish_dream", f"finish_transport:{type(exc).__name__}: {exc}", None,
                         **released, **calls)
        if result.status == "done":
            return {**released, **calls, "status": "running", "route": "finish", "dream_id": result.dream_id,
                    "last_error": None, "retry_at": None}
        if result.status == "retry":
            return retry(state, "finish_dream", f"finish:{result.reason or 'retry'}", result.retry_after_sec,
                         **released, **calls)
        return {**released, **calls, "status": "failed", "route": "failed",
                "last_error": f"finish:{result.reason or 'terminal'}"[:500]}

    async def finish(state):
        released = await admission.release(dict(state), "completed")
        return {**released, "status": "completed", "finished_at": admission.now().isoformat()}

    async def failed(state):
        released = await admission.release(dict(state), "failed")
        return {**released, "status": "failed", "finished_at": admission.now().isoformat()}

    graph = StateGraph(DreamCarryState)
    for name, node in {
        "next_hop": next_hop, "resource_request": resource_request, "resource_wait": resource_wait,
        "text_hop": text_hop, "image_submit": image_submit, "image_wait": image_wait,
        "retry_wait": retry_wait, "finish_dream": finish_dream, "finish": finish, "failed": failed,
    }.items():
        graph.add_node(name, node)
    route = lambda s: s.get("route") or "failed"  # noqa: E731
    graph.add_edge(START, "next_hop")
    graph.add_conditional_edges("next_hop", route, ["resource_request", "image_submit", "finish_dream"])
    # By status, not route: the runtime's restart fence re-enters here via aupdate_state(as_node=...).
    graph.add_conditional_edges("resource_request",
                                lambda s: "finish_dream" if s.get("status") == "stopping" else "resource_wait",
                                ["resource_wait", "finish_dream"])
    graph.add_conditional_edges("resource_wait", after_wait, ["text_hop", "resource_request", "finish_dream"])
    graph.add_conditional_edges("text_hop", route, ["next_hop", "retry_wait", "resource_request", "finish_dream"])
    graph.add_conditional_edges("image_submit", route, ["image_wait", "retry_wait", "finish_dream", "next_hop"])
    graph.add_conditional_edges("image_wait", route, ["image_wait", "next_hop", "finish_dream"])
    graph.add_conditional_edges("retry_wait", route, ["resource_request", "image_submit", "finish_dream"])
    graph.add_conditional_edges("finish_dream", route, ["finish", "retry_wait", "failed"])
    graph.add_edge("finish", END)
    graph.add_edge("failed", END)
    return graph.compile(checkpointer=checkpointer)
