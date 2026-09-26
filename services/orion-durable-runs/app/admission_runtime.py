"""Drive admitted LangGraph threads on GPU pool holds (stage 4.5).

The GPU pool is the only scheduler (spec: docs/superpowers/specs/2026-09-25-gpu-pool-stage4-
durable-runs-and-actuation.md, Decision 1). A run asks the pool for one *hold* for its whole life
and every LLM call it makes attaches to that hold. This runtime only:

* drives each run's graph (``resource_request`` asks the pool, ``resource_wait`` interrupts until
  the pool grants, the work nodes run under the hold, the last node releases it);
* heartbeats the hold while a node works, and at each node boundary lets a recalled hold go;
* wakes a waiting run on the pool's ``granted`` event for its holder (``on_pool_event``), falling
  back to one ``status`` read per run per DURABLE_RUNS_HOLD_STATUS_POLL_SEC -- never a poll storm;
* keeps a Door-A hold alive for Hub's outreach composition until Hub releases it.

Deleted here in 4.5 (kill means kill, no fallback): the durable broker, its lane policy and
widening, the gpu2 elastic decider and its controller calls, ``/leases/validate``,
``/admission`` and ``/elastic/*``. The run registry and lifecycle outbox
(``durable_admission_runs`` / ``durable_resource_events``) are unchanged, so Hub's run views read
the same event names (``run.waiting_resource``, ``run.resource_granted`` + ``run.lane_assigned``
with lane = the hold's role, ``resource.lease_released``, ``resource.lease_expired``).
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from langgraph.graph import START
from langgraph.types import Command

from app.admitted_graph import (
    GONE, GRANTED, REFUSED, WAITING, AdmissionDeps, HoldRecalled, RunControlPending, WorkflowDeadline,
    build_admitted_graph,
)
from app.admitted_reflect_graph import build_admitted_reflect_graph
from app.admitted_self_sense_graph import build_admitted_self_sense_graph
from app.graph import failed_turn_meta, finish_detail, recorded_turn_correlation_id, turn_correlation_id
from app.pool_hold import (
    HELD, WAITING as POOL_WAITING, PoolHolds, UnknownRoute, is_hold_ref, is_pool_trouble, ref_dict,
    refusal_is_terminal,
)
from app.reflect_graph import finish_detail as reflect_finish_detail
from app.self_sense_graph import finish_detail as self_sense_finish_detail
from orion.durable_admission.store import PostgresAdmissionStore
from orion.gpu_pool.client import DURABLE_RUN_HOLDER_PREFIX, durable_run_holder
from orion.schemas.durable_run import DurableRunRequestV1, DurableRunStateV1, DURABLE_RUN_STATE_KIND
from orion.schemas.harness_finalize import HarnessRunCancelV1
from orion.schemas.resource_admission import RESOURCE_EVENT_CHANNEL, RESOURCE_EVENT_KIND, ResourceEventV1

logger = logging.getLogger(__name__)
TERMINAL = {"completed", "failed", "cancelled"}
DEFAULT_WORKFLOW = "curiosity.investigate"
SELF_SENSE_WORKFLOW = "self_sense_eval"
REFLECT_WORKFLOW = "self_study.reflect"
# The node of each admitted workflow that does GPU work under the hold. A restarted driver that
# finds one of these still pending fences the previous attempt and replays it under the same hold.
WORK_NODES = {DEFAULT_WORKFLOW: {"run_started", "harness_turn"}, SELF_SENSE_WORKFLOW: {"ask_questions"},
              REFLECT_WORKFLOW: {"llm_call"}}
# Pool events (for a durable-run holder) after which a waiting run should look at its hold now.
HINT_EVENTS = frozenset({"granted", "recalled", "aborted", "expired", "retried", "backlogged", "unavailable",
                         "dead_lettered", "released", "cancelled"})
# A request id already used for an ended hold is skipped (a checkpoint older than the pool's
# record); bounded so a broken pool cannot spin a run forever.
HOLD_SEQ_SKIP_MAX = 20
MAX_CONCURRENT_DRIVERS = 4
# Which pool refusals fail the run is decided in one place: ``pool_hold.refusal_is_terminal``.
# Any other unavailable (dead-lettered after expiries during a durable-runs outage, an abort) is
# the hold's own history, not the run's: end it and ask again under a new request id. A refusal
# that says the POOL is in trouble (version skew, config roll, unreachable) keeps the run waiting
# and retries with bounded backoff (``_pool_refused``).


def _without_backoff(hold: dict) -> dict:
    """The hold minus the pool-refusal backoff fields: the pool answered, the episode is over."""
    return {k: v for k, v in hold.items() if k not in ("retry_at", "refusals")}


class HoldLost(RuntimeError):
    """The pool no longer holds this run's hold at its generation: stop using the GPU now."""


class AdmissionRuntime:
    def __init__(self, settings, runner, pool, *, store=None, clock=None, holds=None):
        self.settings, self.runner, self.pool = settings, runner, pool
        self.store = store or PostgresAdmissionStore(pool)
        self.now = clock or (lambda: datetime.now(timezone.utc))
        self.holds = holds or PoolHolds(runner._bus, source=settings.service_name)
        if settings.lease_heartbeat_sec * 2 > self.holds.hold_ttl_sec:
            # Two missed beats must still land inside the pool's TTL, or a slow bus expires a
            # healthy run's hold mid-turn.
            raise ValueError(f"DURABLE_RUNS_LEASE_HEARTBEAT_SEC={settings.lease_heartbeat_sec} must be at most half "
                             f"the pool's hold_lease_ttl_sec={self.holds.hold_ttl_sec}")
        admission_deps = AdmissionDeps(
            self.register, self.lease, self.execute, self.release, self.event,
            now=self.now, max_attempts=settings.retry_max_attempts,
            retry_base_seconds=settings.retry_base_sec, retry_max_seconds=settings.retry_max_sec,
            guard=self.guard, keep_for_outreach=self.keep_for_outreach)
        self.deps = admission_deps
        # One compiled graph per workflow. Before 2026-09-22 every admitted run shared the
        # curiosity graph; before 4.5 an admitted reflect run still did.
        self.graphs = {
            DEFAULT_WORKFLOW: build_admitted_graph(runner._curiosity_deps(), admission_deps, runner._checkpointer),
            SELF_SENSE_WORKFLOW: build_admitted_self_sense_graph(
                runner._self_sense_deps(), admission_deps, runner._checkpointer),
            REFLECT_WORKFLOW: build_admitted_reflect_graph(runner._reflect_deps(), admission_deps, runner._checkpointer),
        }
        # Back-compat alias used by older tests that reach for `.graph`.
        self.graph = self.graphs[DEFAULT_WORKFLOW]
        self.active: dict[str, asyncio.Task] = {}
        self._wake = asyncio.Event()
        self._hints: set[str] = set()           # runs a pool event said something changed for
        self._checked: dict[str, float] = {}    # run -> monotonic time of its last waiting-hold read
        # Door-A: run -> {"lease": ref, "since": datetime, "beat": monotonic of last heartbeat}
        self.outreach: dict[str, dict[str, Any]] = {}
        self._outreach_loaded_at: float | None = None
        # Holds whose release RPC failed: lease_id -> (run_id, lease, reason). Retried every reconcile
        # until the pool confirms -- a retryable hold left to its TTL is re-queued and re-granted to
        # nobody (review finding, 4.5).
        self._pending_release: dict[str, tuple[str, dict | None, str]] = {}

    def _graph_for(self, workflow: str | None):
        return self.graphs.get(workflow or DEFAULT_WORKFLOW) or self.graphs[DEFAULT_WORKFLOW]

    def _finish_detail_for(self, workflow: str | None, state: dict):
        workflow = workflow or DEFAULT_WORKFLOW
        if workflow == SELF_SENSE_WORKFLOW:
            return self_sense_finish_detail(state)
        if workflow == REFLECT_WORKFLOW:
            return reflect_finish_detail(state)
        return finish_detail(state)

    @staticmethod
    def config(run_id):
        return {"configurable": {"thread_id": run_id}}

    async def submit(self, request: DurableRunRequestV1) -> dict:
        if request.admission is None:
            raise ValueError("resource admission is required on this endpoint")
        row = await self.store.submit(request.model_dump(mode="json"))
        self._wake.set()
        return {"run_id": request.run_id, "status": row.get("terminal") or "waiting_resource",
                "workflow_kind": request.workflow, "requested_resource": request.admission.resource}

    # --- the run's hold (AdmissionDeps) ------------------------------------------------------
    async def register(self, state) -> dict:
        """resource_request: make sure the run has a live pool hold request.

        The accepted request row, not the checkpoint's copy, is the demand (a corrected row must
        not be shadowed by a stale checkpoint). Re-asking with the hold's own request id is
        idempotent at the pool; a new id (``<run_id>:<seq+1>``) only when the previous hold ended.
        """
        run_id = state["run_id"]
        row = await self.store.get_run(run_id)
        if row is None:
            raise KeyError(run_id)
        if row.get("control"):
            raise RunControlPending(row["control"])
        admission = row["request"]["admission"]
        hold = dict(state.get("hold") or {})
        seq = int(state.get("hold_seq") or 0)
        if not hold.get("request_id"):
            seq += 1
        for _ in range(HOLD_SEQ_SKIP_MAX):
            request_id = hold.get("request_id") or f"{run_id}:{seq}"
            try:
                reply = await self.holds.acquire(run_id, request_id, admission, correlation_id=state.get("correlation_id"))
            except UnknownRoute as exc:
                return {"status": "failed", "last_error": f"gpu_pool_{exc}", "hold": None, "hold_seq": seq}
            except Exception as exc:  # noqa: BLE001 -- unreachable pool (RPC timeout, bus down): transient
                # Remember the id: the acquire may have landed, and re-asking with it is idempotent.
                return await self._pool_refused(state, admission, hold, request_id, seq, "unreachable",
                                                f"{type(exc).__name__}: {exc}")
            if reply.status == "ok":
                # This request id names a hold that already ended (the checkpoint predates its
                # release): never reuse it -- the pool would hand back the ended lease.
                hold, seq = {}, seq + 1
                continue
            break
        else:
            return {"status": "failed", "last_error": "gpu_pool_hold_request_ids_exhausted", "hold": None, "hold_seq": seq}
        if reply.status == "unknown_lease" or (reply.status == "unavailable" and not reply.lease_id):
            # Refused at the door: the pool created nothing. Fail the run only when the refusal is a
            # property of the run (see refusal_is_terminal); pool-side trouble -- version skew
            # (``invalid:*``, incident 2026-09-26), a config roll, an unknown answer -- waits.
            reason = reply.reason or reply.status
            work_class, _, _ = self.holds.placement(admission)
            if refusal_is_terminal(self.holds.cfg, reason, work_class):
                return {"status": "failed", "last_error": f"gpu_pool_unavailable:{reason}",
                        "hold": None, "hold_seq": seq}
            return await self._pool_refused(state, admission, hold, request_id, seq, reply.status, reason)
        if reply.status in POOL_WAITING:
            work_class, _, _ = self.holds.placement(admission)
            await self.store.record_event(run_id, "run.waiting_resource", {
                "requested_lane": admission.get("preferred_lane"), "lease_id": reply.lease_id,
                "work_class": work_class, "position": reply.position, "pool_status": reply.status},
                event_id=f"waiting:{reply.lease_id}")
        return {"status": "waiting_resource", "hold": {"request_id": request_id, "lease_id": reply.lease_id},
                "hold_seq": seq}

    async def _pool_refused(self, state, admission: dict, hold: dict, request_id: str, seq: int,
                            pool_status: str, reason: str) -> dict:
        """The pool could not take the run's hold request right now (unreachable, or a refusal that
        is about the pool, not the run). The run stays ``waiting_resource`` under the same request
        id and asks again after a bounded exponential backoff (``hold.retry_at``, read by
        ``_hold_ready``). Never silent: each refusal is a ``run.waiting_resource`` event carrying the
        pool's status and reason."""
        run_id = state["run_id"]
        refusals = int(hold.get("refusals") or 0) + 1
        delay = min(self.settings.pool_retry_max_sec,
                    self.settings.pool_retry_base_sec * 2 ** min(refusals - 1, 30))
        retry_at = (self.now() + timedelta(seconds=delay)).isoformat()
        reason = " ".join(str(reason).split())[:500]   # one line: pydantic errors are multi-line
        logger.warning("durable_hold_pool_refused run=%s request_id=%s pool_status=%s reason=%s refusals=%s "
                       "retry_in=%.1fs", run_id, request_id, pool_status, reason, refusals, delay)
        try:
            work_class = self.holds.placement(admission)[0]
        except UnknownRoute:
            work_class = None
        await self.store.record_event(run_id, "run.waiting_resource", {
            "requested_lane": admission.get("preferred_lane"), "lease_id": hold.get("lease_id"),
            "work_class": work_class, "pool_status": pool_status, "reason": reason, "transient": True,
            "refusals": refusals, "retry_at": retry_at},
            # retry_at makes each refusal unique: `refusals` restarts at 1 in a later skew episode
            # under the same request id, and a reused entry_id would be dropped (ON CONFLICT).
            event_id=f"pool_refused:{request_id}:{refusals}:{retry_at}")
        return {"status": "waiting_resource", "hold_seq": seq,
                "hold": {"request_id": request_id, "lease_id": hold.get("lease_id"),
                         "retry_at": retry_at, "refusals": refusals}}

    async def lease(self, state) -> tuple[str, dict]:
        """resource_wait: what the pool says about the run's hold right now."""
        hold = state.get("hold") or {}
        waiting = {"status": "waiting_resource", "lease": None}
        if not hold.get("lease_id"):
            return WAITING, waiting
        try:
            reply = await self.holds.status(hold["lease_id"])
        except Exception as exc:  # noqa: BLE001 -- pool unreachable: stay in line, the tick retries
            logger.warning("durable_hold_status_failed run=%s lease=%s err=%s", state["run_id"], hold["lease_id"], exc)
            return WAITING, waiting
        if is_pool_trouble(reply):
            # The pool could not read our request (version skew): it said nothing about the hold.
            # Never end the hold on that; resource_request re-asks (idempotent) and backs off.
            logger.warning("durable_hold_status_refused run=%s lease=%s reason=%s", state["run_id"],
                           hold["lease_id"], reply.reason)
            return WAITING, waiting
        run_id = state["run_id"]
        if reply.status == "granted" and reply.grant is not None:
            ref = ref_dict(reply, durable_run_holder(run_id))
            detail = {"lease": ref, "lane": ref["role"], "lease_id": ref["lease_id"], "generation": ref["generation"],
                      "role": ref["role"], "requested_lane": (state.get("admission") or {}).get("preferred_lane")}
            await self.store.record_event(run_id, "run.resource_granted", detail,
                                          event_id=f"granted:{ref['lease_id']}:{ref['generation']}")
            await self.store.record_event(run_id, "run.lane_assigned", detail,
                                          event_id=f"assigned:{ref['lease_id']}:{ref['generation']}")
            return GRANTED, {"status": "admitted", "lease": ref, "hold": _without_backoff(hold)}
        if reply.status in POOL_WAITING:
            self._checked[run_id] = time.monotonic()   # just read: the poll fallback starts from here
            return WAITING, waiting
        if reply.status == "recall":
            # Recalled before the run started: give the seat back now and queue afresh.
            await self._end_hold(state, hold["lease_id"], None, "recalled_before_start")
            return GONE, {**waiting, "hold": None}
        if reply.status in ("ok", "unknown_lease"):
            return GONE, {**waiting, "hold": None}
        # unavailable / dead-lettered: this hold will not be granted again. End it (the pool keeps
        # such a lease for operator replay; replaying it would grant a run that moved on).
        await self._end_hold(state, hold["lease_id"], None, f"refused:{reply.reason}")
        reason = reply.reason or reply.status
        work_class = None
        try:
            work_class = self.holds.placement(state.get("admission") or {})[0]
        except UnknownRoute:
            pass
        if not refusal_is_terminal(self.holds.cfg, reason, work_class or ""):
            # e.g. dead-lettered after repeated expiries while durable-runs was down: the run
            # resumes (spec Decision 1 rule 6) under a fresh request id. A pool-side refusal
            # (config roll) lands on the door again, where it backs off instead of failing.
            return GONE, {**waiting, "hold": None}
        error = "workflow_deadline" if reason == "deadline" else f"gpu_pool_unavailable:{reason}"
        return REFUSED, {"status": "failed", "last_error": error, "lease": None, "hold": None}

    async def _beat(self, lease: dict) -> str | None:
        """Heartbeat a granted hold; raise HoldLost when the pool no longer holds it at our
        generation. A failed RPC is tolerated: the TTL is at least two beats."""
        try:
            reply = await self.holds.heartbeat(lease["lease_id"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("durable_hold_heartbeat_failed lease=%s err=%s", lease["lease_id"], exc)
            return None
        if is_pool_trouble(reply):
            # Same as an unanswered beat: the pool could not read it (version skew). Like an RPC
            # failure, a skew that lasts the whole turn also hides a hold the pool expired meanwhile;
            # the next readable beat (or the tail's guard) raises HoldLost / records it lost.
            logger.warning("durable_hold_heartbeat_refused lease=%s reason=%s", lease["lease_id"], reply.reason)
            return None
        if reply.status in HELD and reply.grant is not None and reply.grant.generation == lease["generation"]:
            if reply.status == "recall":
                logger.info("durable_hold_recalled lease=%s recall_by=%s", lease["lease_id"], reply.recall_by)
            return reply.status
        raise HoldLost(f"gpu_hold_lost:{reply.status}" + (f":{reply.reason}" if reply.reason else ""))

    async def execute(self, state, node):
        lease = state.get("lease")
        if not is_hold_ref(lease):
            raise RuntimeError("gpu_hold_missing")
        row = await self.store.get_run(state["run_id"])
        if row.get("control"):
            raise RunControlPending(row["control"])
        if await self._beat(lease) == "recall":
            # Already being recalled: never start a long turn on it. Give it back now and queue
            # afresh; this is not a failed attempt.
            await self._end_hold(state, lease["lease_id"], lease, "recalled_before_start")
            raise HoldRecalled("gpu_hold_recalled_before_start")
        detail = {"lease": lease, "lane": lease["role"]}
        if (state.get("workflow") or DEFAULT_WORKFLOW) == DEFAULT_WORKFLOW:
            # Joinable to the pool's child leases (acceptance check 2: no un-attached agent lease
            # under a turn whose run holds a hold).
            detail["turn_correlation_id"] = turn_correlation_id(state)
        await self.event(state, "run.started", detail)
        work = asyncio.create_task(node(state))
        timeout = float(state["brief"]["timeout_sec"])
        deadline = state["admission"].get("deadline_at")
        if deadline:
            timeout = min(timeout, max(0, (datetime.fromisoformat(deadline)-self.now()).total_seconds()))
        try:
            # Timeout starts only after the hold is confirmed. Each heartbeat re-reads the operator
            # control row and asks the pool whether the hold is still ours at this generation.
            async with asyncio.timeout(timeout):
                while True:
                    done, _ = await asyncio.wait({work}, timeout=self.settings.lease_heartbeat_sec)
                    if done:
                        return await work
                    row = await self.store.get_run(state["run_id"])
                    if row.get("control"):
                        raise RuntimeError(f"run_control:{row['control']}")
                    await self._beat(lease)
        except TimeoutError as exc:
            if deadline and self.now() >= datetime.fromisoformat(deadline):
                raise WorkflowDeadline("workflow_deadline") from exc
            raise
        finally:
            if not work.done():
                # Cancel Hub before awaiting the local task's cleanup -- self-sense clears
                # `_inflight_turn_correlation_ids` in its finally, so this must run while those ids
                # are still stamped on `state`.
                await self._cancel_harness(state, "durable_attempt_stopped")
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)

    async def guard(self, state):
        """Node boundary: deadline and operator control, then the hold. Returns the lease while
        the pool still grants it; a recalled hold is released right here (the boundary its
        clawback grace waits for) and a lost one is ended; either way None. Never waits."""
        deadline = state["admission"].get("deadline_at")
        if deadline and self.now() >= datetime.fromisoformat(deadline):
            raise WorkflowDeadline("workflow_deadline")
        row = await self.store.get_run(state["run_id"])
        if row.get("control"):
            raise RunControlPending(row["control"])
        lease = state.get("lease")
        if not is_hold_ref(lease):
            return None
        try:
            reply = await self.holds.heartbeat(lease["lease_id"])
        except Exception as exc:  # noqa: BLE001 -- cannot tell; the tail needs no GPU, keep the ref
            logger.warning("durable_hold_heartbeat_failed lease=%s err=%s", lease["lease_id"], exc)
            return lease
        if is_pool_trouble(reply):
            logger.warning("durable_hold_heartbeat_refused lease=%s reason=%s", lease["lease_id"], reply.reason)
            return lease
        same = reply.grant is not None and reply.grant.generation == lease["generation"]
        if reply.status == "granted" and same:
            return lease
        if reply.status == "recall" and same:
            await self._end_hold(state, lease["lease_id"], lease, "recalled")
            return None
        await self._record_lost(state, lease, reply.status, reply.reason)
        await self._end_hold(state, lease["lease_id"], lease, "lost")
        return None

    async def release(self, state, reason, keep_requeued: bool = False) -> dict:
        """End the run's hold (state update clears ``lease`` and ``hold``). ``keep_requeued``: a
        hold the pool already sent back to its queue (lost heartbeat, recall past grace) is kept --
        same lease_id, same place in line -- and the run waits for it again."""
        hold = dict(state.get("hold") or {})
        lease = state.get("lease") if is_hold_ref(state.get("lease")) else None
        lease_id = hold.get("lease_id") or (lease or {}).get("lease_id")
        if not lease_id and hold.get("request_id"):
            lease_id = await self._lease_for_request(state, hold["request_id"])
        if not lease_id:
            return {"lease": None, "hold": None}
        if keep_requeued:
            try:
                reply = await self.holds.status(lease_id)
            except Exception:  # noqa: BLE001 -- unknown: hand it back rather than strand a slot
                reply = None
            if reply is not None and reply.status in POOL_WAITING and not is_pool_trouble(reply):
                if lease:
                    await self._record_lost(state, lease, reply.status, reply.reason)
                return {"lease": None, "hold": {**_without_backoff(hold), "lease_id": lease_id}}
        await self._end_hold(state, lease_id, lease, reason)
        return {"lease": None, "hold": None}

    async def _lease_for_request(self, state, request_id: str) -> str | None:
        """The acquire may have landed at the pool without its reply reaching us: re-asking with the
        same request id is idempotent and names the lease, so it can be ended."""
        row = await self.store.get_run(state["run_id"])
        try:
            reply = await self.holds.acquire(state["run_id"], request_id, row["request"]["admission"],
                                             correlation_id=state.get("correlation_id"))
        except Exception:  # noqa: BLE001
            return None
        return reply.lease_id

    async def _end_hold(self, state, lease_id: str, lease: dict | None, reason: str) -> bool:
        """Release (or cancel) a hold at the pool. A failed RPC is not the end of it: the hold is
        queued for retry on every reconcile until the pool confirms (``_retry_releases``)."""
        run_id = state["run_id"]
        self.outreach.pop(run_id, None)
        try:
            reply = await self.holds.release(lease_id, outcome="cancelled" if reason in ("cancel", "cancelled") else "ok",
                                             detail=reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("durable_hold_release_failed run=%s lease=%s reason=%s err=%s (will retry)",
                           run_id, lease_id, reason, exc)
            self._pending_release[lease_id] = (run_id, lease, reason)
            return False
        if is_pool_trouble(reply):
            # The pool could not read the release (version skew): not released. Retry it.
            logger.warning("durable_hold_release_refused run=%s lease=%s reason=%s pool_reason=%s (will retry)",
                           run_id, lease_id, reason, reply.reason)
            self._pending_release[lease_id] = (run_id, lease, reason)
            return False
        self._pending_release.pop(lease_id, None)
        await self.store.record_event(run_id, "resource.lease_released", {
            "lease_id": lease_id, "generation": (lease or {}).get("generation"), "lane": (lease or {}).get("role"),
            "reason": reason, "pool_status": reply.status}, event_id=f"released:{lease_id}")
        return True

    async def _retry_releases(self) -> None:
        for lease_id, (run_id, lease, reason) in list(self._pending_release.items()):
            await self._end_hold({"run_id": run_id}, lease_id, lease, reason)

    async def _record_lost(self, state, lease: dict, status: str, reason: str | None) -> None:
        await self.store.record_event(state["run_id"], "resource.lease_expired", {
            "lease_id": lease["lease_id"], "generation": lease["generation"], "lane": lease.get("role"),
            "pool_status": status, "reason": reason}, event_id=f"expired:{lease['lease_id']}:{lease['generation']}")

    async def event(self, state, name, detail):
        await self.store.record_event(state["run_id"], name, detail)

    # --- Door-A: the hold outlives the run until Hub has composed --------------------------------
    async def keep_for_outreach(self, state) -> None:
        self.outreach[state["run_id"]] = {"lease": dict(state["lease"]), "since": self.now(), "beat": time.monotonic()}

    async def _beat_outreach(self) -> None:
        due = self._outreach_loaded_at is None or \
            time.monotonic() - self._outreach_loaded_at >= self.settings.hold_status_poll_sec
        if due:
            # A restarted process adopts the Door-A holds its predecessor left, so they are neither
            # stranded nor (after a lost heartbeat) re-queued and granted to a finished run.
            for row in await self.store.outreach_holds_pending(self.settings.outreach_hold_max_sec):
                detail = row["detail"] or {}
                if is_hold_ref(detail) and detail["lease_id"] not in self._pending_release:
                    self.outreach.setdefault(row["run_id"], {
                        "lease": {k: detail[k] for k in ("lease_id", "generation", "role", "holder")},
                        "since": row["generated_at"], "beat": 0.0})
            self._outreach_loaded_at = time.monotonic()
        for run_id, entry in list(self.outreach.items()):
            lease, state = entry["lease"], {"run_id": run_id}
            if (self.now() - entry["since"]).total_seconds() >= self.settings.outreach_hold_max_sec:
                logger.warning("durable_outreach_hold_timeout run=%s lease=%s", run_id, lease["lease_id"])
                await self._end_hold(state, lease["lease_id"], lease, "outreach_timeout")
                continue
            if time.monotonic() - entry["beat"] < self.settings.lease_heartbeat_sec:
                continue
            entry["beat"] = time.monotonic()
            try:
                await self._beat(lease)
            except HoldLost as exc:
                logger.warning("durable_outreach_hold_lost run=%s lease=%s %s", run_id, lease["lease_id"], exc)
                await self._record_lost(state, lease, str(exc), None)
                # Re-queued holds would be granted to a finished run: end it.
                await self._end_hold(state, lease["lease_id"], lease, "lost")

    async def release_outreach(self, run_id: str) -> dict:
        """Hub finished composing (``/runs/{id}/release-outreach-lease``): release the Door-A hold."""
        row = await self.store.get_run(run_id)
        if row is None:
            raise KeyError(run_id)
        if row.get("terminal") != "completed":
            raise ValueError("no door-a outreach lease hold on this run")
        entry = self.outreach.get(run_id)
        lease = (entry or {}).get("lease")
        if lease is None:
            workflow = row["request"].get("workflow") or DEFAULT_WORKFLOW
            snap = await self._graph_for(workflow).aget_state(self.config(run_id))
            lease = (snap.values or {}).get("lease")
        if not is_hold_ref(lease) or any(event.get("event") == "resource.lease_released"
                                         and (event.get("detail") or {}).get("lease_id") == lease["lease_id"]
                                         for event in await self.store.history(run_id)):
            self.outreach.pop(run_id, None)
            return {"released": False, "run_id": run_id, "reason": "already_released"}
        await self._end_hold({"run_id": run_id}, lease["lease_id"], lease, "outreach_done")
        self._wake.set()
        return {"released": True, "run_id": run_id}

    # --- pool events -----------------------------------------------------------------------------
    async def on_pool_event(self, payload: dict[str, Any]) -> None:
        """A pool lifecycle event: wake the run it names (hints only -- the graph always re-reads
        the pool's own answer, so a stale or duplicate event can never fake a grant)."""
        holder = str(payload.get("holder") or "")
        if not holder.startswith(DURABLE_RUN_HOLDER_PREFIX) or payload.get("event") not in HINT_EVENTS:
            return
        run_id = holder[len(DURABLE_RUN_HOLDER_PREFIX):]
        self._hints.add(run_id)
        if run_id in self.outreach:
            self.outreach[run_id]["beat"] = 0.0
        self._wake.set()

    async def _hold_ready(self, run_id: str, state: dict) -> bool:
        """Should a run interrupted in ``resource_wait`` be resumed now? Yes on a pool event for it;
        otherwise one ``status`` read per DURABLE_RUNS_HOLD_STATUS_POLL_SEC (the missed-event
        fallback), resuming only when the hold is no longer simply waiting in line."""
        hold = state.get("hold") or {}
        lease_id = hold.get("lease_id")
        retry_at = hold.get("retry_at")
        backing_off = bool(retry_at) and self.now() < datetime.fromisoformat(retry_at)
        if run_id in self._hints:
            # The pool only publishes events for a hold it created, so a hint means the pool is
            # answering -- even mid-backoff (e.g. an acquire that timed out but landed and was
            # granted: don't leave that seat idle until retry_at).
            self._hints.discard(run_id)
            self._checked[run_id] = time.monotonic()
            return True
        if backing_off:
            return False   # the pool refused the hold request; wait out the backoff (_pool_refused)
        if retry_at:
            self._checked[run_id] = time.monotonic()
            return True    # backoff over: ask the pool again now, whatever the poll interval says
        last = self._checked.get(run_id)
        if last is not None and time.monotonic() - last < self.settings.hold_status_poll_sec:
            return False
        self._checked[run_id] = time.monotonic()
        if not lease_id:
            return True  # no hold at the pool yet (acquire never answered): ask again
        try:
            reply = await self.holds.status(lease_id)
        except Exception:  # noqa: BLE001
            return False
        return reply.status not in POOL_WAITING

    async def _cancel_harness(self, state, reason):
        # Curiosity: hold-derived turn id. Self-sense: each question is a fresh uuid4 -- cancel
        # those from answers + any still in-flight.
        ids: list[str] = []
        try:
            ids.append(turn_correlation_id(state))
        except Exception:  # noqa: BLE001 -- state may lack lease/run_id
            pass
        meta = state.get("harness_turn_meta") if isinstance(state.get("harness_turn_meta"), dict) else {}
        if isinstance(meta.get("turn_correlation_id"), str):
            ids.append(meta["turn_correlation_id"])
        answers = state.get("answers") if isinstance(state.get("answers"), dict) else {}
        for answer in answers.values():
            if not isinstance(answer, dict):
                continue
            corr = answer.get("correlation_id")
            if isinstance(corr, str) and corr:
                ids.append(corr)
        inflight = state.get("_inflight_turn_correlation_ids")
        if isinstance(inflight, list):
            ids.extend(c for c in inflight if isinstance(c, str) and c)
        seen: set[str] = set()
        for correlation_id in ids:
            if not correlation_id or correlation_id in seen:
                continue
            seen.add(correlation_id)
            payload = HarnessRunCancelV1(correlation_id=correlation_id, reason=reason)
            await self.runner._publish(
                "orion:harness:run:cancel", "harness.run.cancel.v1", payload,
                self.runner._corr_for_admission(correlation_id),
            )

    @asynccontextmanager
    async def claim(self, run_id):
        # Session advisory locks serialize graph writers across service replicas.
        # Released immediately at an interrupt; no connection waits in a queue.
        async with self.pool.connection() as conn:
            cursor = await conn.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired", ("orion:durable:"+run_id,))
            acquired = (await cursor.fetchone())["acquired"]
            try:
                yield acquired
            finally:
                if acquired:
                    await conn.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", ("orion:durable:"+run_id,))

    async def _recover(self, graph, cfg, workflow: str, state: dict, snap) -> None:
        """This driver took a fresh session lock on a run whose checkpoint still names a lease: its
        predecessor died (or was stopped) mid-run and may still have a turn running in Hub.

        A pool hold keeps its lease_id and generation across the restart (spec Decision 1 rule 6),
        so the old turn is fenced explicitly -- harness cancel for its identity, and ``turn_fence``
        gives the replay a new one -- and the work node is replayed under the same hold once the
        pool confirms it. A legacy (pre-cutover) durable lease has no pool hold: it is dropped and
        the run asks the pool afresh."""
        next_node = snap.next[0]
        held = is_hold_ref(state.get("lease"))
        if next_node in WORK_NODES.get(workflow, WORK_NODES[DEFAULT_WORKFLOW]):
            if held:
                await self._cancel_harness(state, "worker_recovery")
            update = {"lease": None, "turn_fence": int(state.get("turn_fence") or 0) + (1 if held else 0)}
            if not held:
                update["hold"] = None
            if workflow == DEFAULT_WORKFLOW:
                # Record the fenced attempt's turn identity before the lease goes: Hub may still
                # finish that turn and write its harness_turn_trace row under this id.
                await graph.aupdate_state(cfg, {**update, "status": "retrying", "retry_node": None,
                                                **failed_turn_meta(state)}, as_node="retry_wait")
            else:
                await graph.aupdate_state(cfg, {**update, "status": "waiting_resource"}, as_node="resource_request")
            return
        released = await self.release(state, "worker_recovery") if held else {"lease": None, "hold": None}
        await graph.aupdate_state(cfg, released)

    async def _drive(self, row):
        run_id = row["run_id"]
        async with self.claim(run_id) as acquired:
            if not acquired:
                return
            await self.store.touch(run_id)
            row = await self.store.get_run(run_id)
            if row.get("terminal"):
                return
            request = row["request"]
            workflow = request.get("workflow") or DEFAULT_WORKFLOW
            graph = self._graph_for(workflow)
            cfg = self.config(run_id)
            snap = await graph.aget_state(cfg)
            if not snap.values:
                state = {
                    "run_id": run_id,
                    "correlation_id": request["correlation_id"],
                    "brief": request["brief"],
                    "admission": request["admission"],
                    "requested_at": request["requested_at"],
                    "attempt": 0,
                    "status": "queued",
                    "workflow": workflow,
                }
                # Persist initial checkpoint before dispatch. Inbox recovers a
                # death before this point; graph recovers a death after it.
                await graph.aupdate_state(cfg, state, as_node=START)
                snap = await graph.aget_state(cfg)
            state = dict(snap.values)
            if row.get("control") == "cancelled":
                released = await self.release(state, "cancelled")
                await graph.aupdate_state(cfg, {**released, "status": "cancelled"}, as_node="finish")
                await self._terminal(run_id, "cancelled", state, workflow=workflow)
                return
            if row.get("control") == "paused":
                await self.release(state, "paused")
                return
            if snap.next and state.get("lease"):
                await self._recover(graph, cfg, workflow, state, snap)
                snap = await graph.aget_state(cfg)
                state = dict(snap.values)
            deadline = state["admission"].get("deadline_at")
            if deadline and self.now() >= datetime.fromisoformat(deadline):
                released = await self.release(state, "deadline")
                await graph.aupdate_state(cfg, {**released, "status": "failed", "last_error": "workflow_deadline"},
                                          as_node="failed")
                await self._terminal(run_id, "failed", {**state, "last_error": "workflow_deadline"}, workflow=workflow)
                return
            if not snap.next:
                await self._terminal(run_id, state.get("status", "failed"), state, workflow=workflow)
                return
            try:
                resume = None
                if any(t.interrupts for t in snap.tasks):
                    if snap.next == ("resource_wait",) and not await self._hold_ready(run_id, state):
                        return
                    if snap.next == ("retry_wait",) and self.now() < datetime.fromisoformat(state["retry_at"]):
                        return
                    resume = Command(resume=True)
                    await self.event(state, "run.resumed", {"node": snap.next[0]})
                async for update in graph.astream(resume, cfg, stream_mode="updates", durability="sync"):
                    snap = await graph.aget_state(cfg)
                    state = dict(snap.values)
                    for node, delta in update.items():
                        if node == "__interrupt__":
                            continue
                        status = state.get("status", "running")
                        if status in TERMINAL or node == "retry_wait":
                            continue  # Atomic terminal projection owns terminal lifecycle facts.
                        checkpoint = snap.config["configurable"].get("checkpoint_id", "")
                        await self.store.record_event(run_id, "run."+status, {"node": node},
                            event_id=f"{run_id}:{checkpoint}:{node}:{status}")
                    if state.get("status") in TERMINAL and not snap.next:
                        await self._terminal(run_id, state["status"], state, workflow=workflow)
            except asyncio.CancelledError:
                raise
            except RunControlPending:
                return
            except WorkflowDeadline:
                released = await self.release(state, "workflow_deadline")
                await graph.aupdate_state(cfg, {**released, "status": "failed", "last_error": "workflow_deadline"},
                                          as_node="failed")
                await self._terminal(run_id, "failed", {**state, "last_error": "workflow_deadline"}, workflow=workflow)
            except Exception as exc:
                # The checkpoint retains the failing node. Reconciliation is
                # allowed to retry persistence/transport, never an empty result
                # -- and never forever: N failures at the same checkpoint (no
                # progress in between) fail the run with the error attached.
                logger.exception("durable_checkpoint_resume_failed run=%s", run_id)
                await self._resume_failed(run_id, graph, cfg, snap, state, exc, workflow)

    async def _resume_failed(self, run_id, graph, cfg, snap, state, exc, workflow):
        checkpoint_id = snap.config["configurable"].get("checkpoint_id", "")
        error = f"{type(exc).__name__}: {exc}"[:400]
        await self.event(state, "run.checkpoint_resume_failed",
                         {"node": list(snap.next), "checkpoint_id": checkpoint_id, "error": error})
        if not snap.next:
            # The graph already finished; only its projection failed. Never
            # rewrite a finished graph -- the next tick retries the projection.
            return
        failures, first_at, now = await self.store.resume_failures_since_progress(run_id)
        if failures < self.settings.resume_max_failures or first_at is None or (
                (now - first_at).total_seconds() < self.settings.resume_min_failure_span_sec):
            return
        last_error = f"checkpoint_resume_failed: {error} (x{failures} since last progress, at node {','.join(snap.next)})"
        logger.error("durable_checkpoint_resume_abandoned run=%s failures=%s error=%s", run_id, failures, error)
        released = await self.release(state, "checkpoint_resume_failed")
        try:
            await graph.aupdate_state(cfg, {**released, "status": "failed", "last_error": last_error}, as_node="failed")
        except Exception:  # noqa: BLE001 -- the durable row must still go terminal
            logger.exception("durable_checkpoint_abandon_graph_update_failed run=%s", run_id)
        await self._terminal(run_id, "failed", {**state, "last_error": last_error}, workflow=workflow)

    async def _terminal(self, run_id, status, state, *, workflow: str | None = None):
        wf = workflow or state.get("workflow") or DEFAULT_WORKFLOW
        if status == "completed":
            detail = self._finish_detail_for(wf, state)
        else:
            # Only what the graph recorded -- never re-derived here, since by
            # now the lease is cleared and a fresh derivation would name the
            # run's lineage, not the turn that actually failed.
            detail = {"error": state.get("last_error")}
            corr = recorded_turn_correlation_id(state)
            if corr:
                detail["turn_correlation_id"] = corr
        actual = await self.store.finish_projection(run_id, status, detail)
        if actual is not None and actual != status:
            await self._graph_for(wf).aupdate_state(self.config(run_id), {"status": actual}, as_node="finish")
        if actual is not None and actual != "completed" and run_id in self.outreach:
            # finish kept a Door-A hold, but the run did not end completed (a cancel won the race):
            # Hub will never compose for it and release_outreach refuses it -- end it here.
            entry = self.outreach[run_id]
            await self._end_hold({"run_id": run_id}, entry["lease"]["lease_id"], entry["lease"], f"terminal_{actual}")
        if actual is not None:
            self._hints.discard(run_id)
            self._checked.pop(run_id, None)
        self._wake.set()

    async def reconcile(self):
        try:
            await self._retry_releases()
        except Exception:  # noqa: BLE001
            logger.exception("durable_hold_release_retry_failed")
        try:
            await self._beat_outreach()
        except Exception:  # noqa: BLE001 -- Door-A upkeep must not stop the run loop
            logger.exception("durable_outreach_hold_upkeep_failed")
        for row in await self.store.list_pending():
            run_id = row["run_id"]
            if run_id in self.active:
                if row.get("control"):
                    self.active[run_id].cancel()
                continue
            if row.get("control") == "paused":
                # The pausing replica released the hold (control()); re-driving a paused run every
                # tick would only repeat pool RPCs. Resume clears control and it is driven again.
                continue
            if len(self.active) >= MAX_CONCURRENT_DRIVERS:
                break
            task = asyncio.create_task(self._drive(row), name=f"admitted-{run_id}")
            self.active[run_id] = task
            def finished(t, key=run_id):
                self.active.pop(key, None)
                if not t.cancelled() and t.exception():
                    logger.error("durable_driver_failed run=%s error=%s", key, t.exception())
            task.add_done_callback(finished)
        for raw in await self.store.pending_outbox():
            event = ResourceEventV1.model_validate(raw)
            if event.event == "run.completed" and event.entry_id == f"{event.run_id}:terminal:completed":
                row = await self.store.get_run(event.run_id)
                workflow = ((row or {}).get("request") or {}).get("workflow") or DEFAULT_WORKFLOW
                completion = DurableRunStateV1(entry_id=event.entry_id+":state", run_id=event.run_id,
                    workflow=workflow, thread_id=event.thread_id, node="finish", status="completed",
                    correlation_id=event.correlation_id, generated_at=event.generated_at, detail=event.detail)
                if not await self.runner._publish(self.settings.state_channel, DURABLE_RUN_STATE_KIND, completion,
                        self.runner._corr_for_admission(event.correlation_id)):
                    continue
            if await self.runner._publish(RESOURCE_EVENT_CHANNEL, RESOURCE_EVENT_KIND, event,
                    self.runner._corr_for_admission(event.correlation_id)):
                await self.store.ack_outbox(event.entry_id)

    async def wakeup(self, event):
        # Another replica's lifecycle fact: a hint to look again, never a grant.
        if event.event in {"run.resource_granted", "resource.lease_released", "resource.lease_expired"}:
            self._wake.set()

    async def run(self, stop):
        while not stop.is_set():
            self._wake.clear()
            try:
                await self.reconcile()
            except Exception:
                logger.exception("durable_admission_reconcile_failed")
            try:
                await asyncio.wait_for(self._wake.wait(), self.settings.admission_tick_sec)
            except TimeoutError:
                pass

    async def control(self, run_id, action):
        row = await self.store.get_run(run_id)
        if not row:
            raise KeyError(run_id)
        if row.get("terminal") or row.get("control") == "cancelled":
            return await self.status(run_id)
        await self.store.set_control(run_id, {"pause": "paused", "cancel": "cancelled", "resume": None}[action])
        if action in {"pause", "cancel"}:
            if run_id in self.active:
                self.active[run_id].cancel()
                await asyncio.gather(self.active[run_id], return_exceptions=True)
            workflow = row["request"].get("workflow") or DEFAULT_WORKFLOW
            snap = await self._graph_for(workflow).aget_state(self.config(run_id))
            if snap.values:
                state = {**dict(snap.values), "run_id": run_id}
                await self.release(state, action)
                if action == "cancel" and not (state.get("hold") or {}).get("request_id"):
                    # The driver may have been cancelled mid-acquire: the hold can exist at the pool
                    # under the next request id without ever reaching the checkpoint.
                    probe = f"{run_id}:{int(state.get('hold_seq') or 0) + 1}"
                    await self.release({**state, "hold": {"request_id": probe, "lease_id": None}}, action)
        await self.store.record_event(run_id, "run."+{"pause": "paused", "cancel": "cancelled", "resume": "resumed"}[action], {})
        self._wake.set()
        return await self.status(run_id)

    async def status(self, run_id):
        row = await self.store.get_run(run_id)
        if not row:
            raise KeyError(run_id)
        workflow = row["request"].get("workflow") or DEFAULT_WORKFLOW
        snap = await self._graph_for(workflow).aget_state(self.config(run_id))
        values = snap.values or {}
        lease = values.get("lease") if is_hold_ref(values.get("lease")) else None
        hold = values.get("hold") or None
        pool = None
        if (hold or {}).get("lease_id"):
            try:
                pool = (await self.holds.status(hold["lease_id"])).model_dump(mode="json")
            except Exception as exc:  # noqa: BLE001
                pool = {"status": "unreachable", "reason": f"{type(exc).__name__}: {exc}"[:200]}
        history = await self.store.history(run_id)
        first_grant = await self.store.first_event_at(run_id, "run.resource_granted")
        wait_end = first_grant or (row["updated_at"] if row.get("terminal") else self.now())
        return {"run_id": run_id, "thread_id": run_id, "workflow_kind": workflow,
                "status": row.get("terminal") or row.get("control") or values.get("status", "waiting_resource"),
                "requested_resource": row["request"]["admission"]["resource"], "lease": lease, "hold": hold,
                "pool": pool, "next": list(snap.next), "created_at": row["created_at"], "history": history,
                "queue_wait_seconds": max(0, (wait_end-row["created_at"]).total_seconds())}

    async def close(self):
        tasks = list(self.active.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
