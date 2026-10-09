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
    GONE, GRANTED, REFUSED, WAITING, AdmissionDeps, HoldPreempted, HoldRecalled, RunControlPending,
    WorkflowDeadline, build_admitted_graph,
)
from app.admitted_reflect_graph import build_admitted_reflect_graph
from app.compactor_digest_graph import build_compactor_digest_graph, finish_detail as compactor_digest_finish_detail
from app.journal_compose_graph import build_journal_compose_graph, finish_detail as journal_compose_finish_detail
from app.episode_distill_graph import build_episode_distill_graph, finish_detail as episode_distill_finish_detail
from orion.schemas.memory_episode import MEMORY_EPISODE_DISTILL_WORKFLOW
from orion.schemas.journal_compose_run import JOURNAL_COMPOSE_WORKFLOW
from app.admitted_self_sense_graph import build_admitted_self_sense_graph
from app.graph import failed_turn_meta, finish_detail, recorded_turn_correlation_id, turn_correlation_id, urgent_detail
from app.pool_hold import (
    HELD, URGENT_PREEMPT, WAITING as POOL_WAITING, PoolHolds, UnknownRoute, is_hold_ref, is_pool_trouble,
    hold_placement, ref_dict, refusal_is_terminal,
)
from app.reflect_graph import finish_detail as reflect_finish_detail
from app.orion_day_graph import (
    OrionDayDeps, build_orion_day_graph, finish_detail as orion_day_finish_detail, slim_brief as orion_day_slim_brief,
)
from app.orion_day_store import persist_letter as persist_orion_day_letter
from app.reading_graph import build_reading_graph, finish_detail as reading_finish_detail
from app.reverie_visual_graph import (
    RETRY_WINDOW_EXPIRED, abandon_request, build_reverie_visual_graph,
    finish_detail as reverie_visual_finish_detail, send_abandon,
    terminal_detail as reverie_visual_terminal_detail,
)
from orion.schemas.compactor_digest_run import COMPACTOR_DIGEST_WORKFLOW
from orion.schemas.reading_turn import READING_WORKFLOW
from orion.schemas.reverie_visual_run import REVERIE_VISUAL_WORKFLOW
from orion.schemas.orion_day import ORION_DAY_WORKFLOW, OrionDayRunBriefV1
from app.self_sense_graph import finish_detail as self_sense_finish_detail
from orion.durable_runs.registry_store import (
    ABANDON_ACKED_EVENT,
    ABANDON_GIVE_UP_SEC,
    ABANDON_PENDING_EVENT,
    DurableRunRegistryStore,
)
from orion.gpu_pool.client import DURABLE_RUN_HOLDER_PREFIX, durable_run_holder
from orion.schemas.gpu_pool import GpuLeaseReplyV1
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
              REFLECT_WORKFLOW: {"llm_call"}, READING_WORKFLOW: {"reading_turn"},
              # Only generate holds the diffusion hold; it releases it before its result is
              # checkpointed, so a restart after generate resumes at caption with no lease.
              REVERIE_VISUAL_WORKFLOW: {"generate"},
              # One LLM call per digest run; finalize lets the hold go before it calls cortex-orch.
              COMPACTOR_DIGEST_WORKFLOW: {"digest"},
              # Only compose holds the GPU; publish lets the hold go before it sends the write.
              JOURNAL_COMPOSE_WORKFLOW: {"compose"},
              # Both LLM calls run under the hold; a restart replays the first one without a
              # checkpointed text (a finished note is never regenerated). persist needs no GPU.
              ORION_DAY_WORKFLOW: {"write_note", "write_carry_forward"}}
# The DurableRunStateV1.node each admitted terminal is published under (the graph node it ends at).
TERMINAL_STATE_NODE = {"completed": "finish", "failed": "failed", "cancelled": "finish"}
# Pool events (for a durable-run holder) after which a waiting run should look at its hold now.
HINT_EVENTS = frozenset({"granted", "recalled", "aborted", "expired", "retried", "backlogged", "unavailable",
                         "dead_lettered", "released", "cancelled"})
# A request id already used for an ended hold is skipped (a checkpoint older than the pool's
# record); bounded so a broken pool cannot spin a run forever.
HOLD_SEQ_SKIP_MAX = 20
# A hold recalled for urgent work before its step started is not released (that forfeits its place in
# line): the driver polls the pool every PREEMPT_POLL_SEC until the pool's abort re-queues it in place,
# for at most urgent_preempt_grace_sec + PREEMPT_REQUEUE_MARGIN_SEC from first sight (local clock).
PREEMPT_POLL_SEC = 1.0
PREEMPT_REQUEUE_MARGIN_SEC = 3.0
# Background drivers at once. Urgent drivers are outside it, capped at urgent_max_concurrent.
MAX_CONCURRENT_DRIVERS = 4
# "system" runs (memory.episode_distill, 2026-10-02) are also outside it, capped here: a system run
# must not wait behind four background turns before it even asks the pool, where it then ranks
# above background. One at a time: the distiller is the only system producer (~1-2 episodes/day).
MAX_CONCURRENT_SYSTEM_DRIVERS = 1
# Which pool refusals fail the run is decided in one place: ``pool_hold.refusal_is_terminal``.
# Any other unavailable (dead-lettered after expiries during a durable-runs outage, an abort) is
# the hold's own history, not the run's: end it and ask again under a new request id. A refusal
# that says the POOL is in trouble (version skew, config roll, unreachable) keeps the run waiting
# and retries with bounded backoff (``_pool_refused``).


def _priority(row: dict) -> str | None:
    """A pending run's admitted priority (ResourceRequirementV1.priority on the stored request)."""
    return (((row.get("request") or {}).get("admission")) or {}).get("priority")


def _without_backoff(hold: dict) -> dict:
    """The hold minus the pool-refusal backoff fields: the pool answered, the episode is over."""
    return {k: v for k, v in hold.items() if k not in ("retry_at", "refusals")}


from app.admitted_graph import HoldLost

# release() reasons of a pool take-back (HoldLost / HoldPreempted): the node goes straight back to
# resource_wait, so a hold the pool already re-granted is kept for it.
TAKEBACK_RELEASE_REASONS = frozenset({HoldLost.release_reason, URGENT_PREEMPT})


class AdmissionRuntime:
    def __init__(self, settings, runner, pool, *, store=None, clock=None, holds=None):
        self.settings, self.runner, self.pool = settings, runner, pool
        self.store = store or DurableRunRegistryStore(pool)
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
            guard=self.guard, keep_for_outreach=self.keep_for_outreach, requeued=self.requeued,
            max_takebacks=settings.hold_max_takebacks)
        self.deps = admission_deps
        # One compiled graph per workflow. Before 2026-09-22 every admitted run shared the
        # curiosity graph; before 4.5 an admitted reflect run still did.
        self.graphs = {
            DEFAULT_WORKFLOW: build_admitted_graph(runner._curiosity_deps(), admission_deps, runner._checkpointer),
            SELF_SENSE_WORKFLOW: build_admitted_self_sense_graph(
                runner._self_sense_deps(), admission_deps, runner._checkpointer),
            REFLECT_WORKFLOW: build_admitted_reflect_graph(runner._reflect_deps(), admission_deps, runner._checkpointer),
            READING_WORKFLOW: build_reading_graph(lambda request: runner._run_reading_turn(request), admission_deps, runner._checkpointer),
            REVERIE_VISUAL_WORKFLOW: build_reverie_visual_graph(self._reverie_step, admission_deps, runner._checkpointer),
            # Late-bound (like reading above): the runner's method is read per call.
            COMPACTOR_DIGEST_WORKFLOW: build_compactor_digest_graph(
                lambda payload, **kw: runner._cortex_orch_rpc(payload, **kw), admission_deps, runner._checkpointer),
            # Late-bound (like reading/reverie above): the runner's methods are read per call.
            JOURNAL_COMPOSE_WORKFLOW: build_journal_compose_graph(
                lambda brief, **kw: runner._compose_journal(brief, **kw),
                lambda write: runner._publish_journal_write(write), admission_deps, runner._checkpointer),
            # Memory episode distiller (shadow, 2026-10-02). Late-bound like journal.compose.
            MEMORY_EPISODE_DISTILL_WORKFLOW: build_episode_distill_graph(
                lambda brief: self._load_episode(brief),
                lambda prompt, **kw: runner._call_memory_distill_llm(prompt, **kw),
                lambda **kw: self._persist_episode(**kw), admission_deps, runner._checkpointer),
            # Bound lazily (like reading's run_turn): resolved on the runner at call time.
            ORION_DAY_WORKFLOW: build_orion_day_graph(OrionDayDeps(
                call_verb_text=lambda *args, **kwargs: runner._call_verb_text(*args, **kwargs),
                persist_letter=lambda row: persist_orion_day_letter(self.pool, row),
                publish_journal=lambda entry: runner._publish_journal(entry),
                load_brief=self._orion_day_brief,
            ), admission_deps, runner._checkpointer),
        }
        # Back-compat alias used by older tests that reach for `.graph`.
        self.graph = self.graphs[DEFAULT_WORKFLOW]
        self.active: dict[str, asyncio.Task] = {}
        self._urgent_drivers: set[str] = set()   # the runs in ``active`` driven as urgent
        self._system_drivers: set[str] = set()   # the runs in ``active`` driven as system priority
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
        # reverie.visual runs that ended without completing and whose abandon thought has not
        # confirmed: run -> {"attempt_id", "visual_request", "reason", "failures", "due" (monotonic)}.
        # A cache of the store's run.abandon_pending records (reloaded every
        # DURABLE_RUNS_HOLD_STATUS_POLL_SEC, so a restart picks them up); retried from reconcile.
        self._abandons: dict[str, dict[str, Any]] = {}
        self._abandons_loaded_at: float | None = None
        self._abandoning: dict[str, asyncio.Task] = {}

    def _graph_for(self, workflow: str | None):
        return self.graphs.get(workflow or DEFAULT_WORKFLOW) or self.graphs[DEFAULT_WORKFLOW]

    def _finish_detail_for(self, workflow: str | None, state: dict):
        workflow = workflow or DEFAULT_WORKFLOW
        if workflow == SELF_SENSE_WORKFLOW:
            return self_sense_finish_detail(state)
        if workflow == REFLECT_WORKFLOW:
            return reflect_finish_detail(state)
        if workflow == READING_WORKFLOW:
            return reading_finish_detail(state)
        if workflow == REVERIE_VISUAL_WORKFLOW:
            return reverie_visual_finish_detail(state)
        if workflow == COMPACTOR_DIGEST_WORKFLOW:
            return compactor_digest_finish_detail(state)
        if workflow == JOURNAL_COMPOSE_WORKFLOW:
            return journal_compose_finish_detail(state)
        if workflow == ORION_DAY_WORKFLOW:
            return orion_day_finish_detail(state)
        if workflow == MEMORY_EPISODE_DISTILL_WORKFLOW:
            return episode_distill_finish_detail(state)
        return finish_detail(state)

    async def _load_episode(self, brief) -> dict:
        """memory.episode_distill load_episode: the turns' FULL text from chat_history_log (never a
        preview -- the validator checks quotes against exactly this), plus candidate referent keys."""
        from orion.memory.episode.distill import LOAD_TURNS_SQL, turns_from_rows, turns_to_state
        from orion.memory.episode.store import candidate_referent_keys

        async with self.pool.connection() as conn:
            rows = await (await conn.execute(LOAD_TURNS_SQL, (list(brief.turn_ids),))).fetchall()
        turns = turns_from_rows([dict(r) for r in rows])
        return {"turns": turns_to_state(turns), "candidate_referents": await candidate_referent_keys(self.pool)}

    async def _persist_episode(self, **kwargs) -> dict:
        from orion.memory.episode.store import persist_episode

        return await persist_episode(self.pool, **kwargs, referent_policy=self._referent_policy())

    def _referent_policy(self):
        """None when MEMORY_REFERENTS_ENABLED=false (the referent step is skipped)."""
        if not getattr(self.settings, "memory_referents_enabled", False):
            return None
        from orion.memory.referents.resolve import ReferentPolicy

        return ReferentPolicy(
            grounding_auto_accept=bool(self.settings.memory_alias_grounding_auto_accept),
            cooccurrence_auto_accept=bool(self.settings.memory_cooccurrence_auto_accept),
        )

    async def _orion_day_brief(self, state) -> OrionDayRunBriefV1:
        """The full orion_day.letter brief from the accepted request row (the checkpoint keeps
        only ``slim_brief``)."""
        row = await self.store.get_run(state["run_id"])
        if row is None:
            raise KeyError(state["run_id"])
        return OrionDayRunBriefV1.model_validate(row["request"]["brief"])

    @staticmethod
    def _checkpoint_brief(workflow: str, brief: dict) -> dict:
        """What the run's checkpoint carries of its brief. orion_day.letter keeps a slim copy: its
        full brief (material + digest) stays once in durable_admission_runs.request."""
        return orion_day_slim_brief(brief) if workflow == ORION_DAY_WORKFLOW else brief

    def _reverie_step(self, request, budget_sec=None):
        return self.runner._run_reverie_visual_step(request, budget_sec)

    @staticmethod
    def _deadline_error(workflow: str | None) -> str:
        return RETRY_WINDOW_EXPIRED if workflow == REVERIE_VISUAL_WORKFLOW else "workflow_deadline"

    @staticmethod
    def _terminal_detail_for(workflow: str, status: str, state: dict) -> dict:
        """failed/cancelled detail. Only what the graph recorded -- never re-derived here, since by
        now the lease is cleared and a fresh derivation would name the run's lineage, not the turn
        that actually failed."""
        if workflow == REVERIE_VISUAL_WORKFLOW:
            return reverie_visual_terminal_detail(state, status)
        # Urgent runs must end in a report, so their failed/cancelled facts say so (Hub keys on it).
        urgent = urgent_detail(state)
        if status == "cancelled":
            # An operator cancel carries no reason of its own (control() records only the action).
            return {"error": "cancelled", "urgent": urgent} if urgent is not None else {}
        detail = {"error": state.get("last_error")}
        corr = recorded_turn_correlation_id(state)
        if corr:
            detail["turn_correlation_id"] = corr
        if urgent is not None:
            detail["urgent"] = urgent
        return detail

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
            if reply.reason == URGENT_PREEMPT and await self._await_requeue(hold["lease_id"], reply):
                # Paused for an urgent run and back in line in its original place: wait for it again.
                await self._record_preempted(state, hold["lease_id"], reply.grant.generation if reply.grant else None,
                                             reply.grant.role if reply.grant else None, reply)
                return WAITING, waiting
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

    async def _beat(self, lease: dict, admission: dict | None = None) -> GpuLeaseReplyV1 | None:
        """Heartbeat a granted hold: the pool's granted/recall reply (None when unanswered). Raise
        HoldLost when the pool no longer holds it at our generation (``_taken_back``). A failed RPC
        is tolerated: the TTL is at least two beats."""
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
                logger.info("durable_hold_recalled lease=%s recall_by=%s reason=%s", lease["lease_id"],
                            reply.recall_by, reply.reason)
            return reply
        raise self._taken_back(lease, reply, admission)

    def _taken_back(self, lease: dict, reply: GpuLeaseReplyV1, admission: dict | None = None) -> Exception:
        """What a mid-node pool answer that is no longer "yours at this generation" means for the node.

        * re-queued for an urgent run -> HoldPreempted;
        * re-queued for any other reason (a recall past its grace: max_hold, owner reclaim, unlend;
          a lost heartbeat), or already re-granted at a newer generation -> HoldLost: same hold,
          the run waits for it again. Not an attempt;
        * ended by the pool (dead-lettered, released, unknown) -> HoldLost as well unless the pool's
          reason is a property of the run itself (``refusal_is_terminal``: deadline, a class nothing
          serves, ...): the run asks afresh, as ``lease`` does for the same answer before a node.
          A terminal reason is a plain error: the node's own failure handling decides.
        """
        detail = f"{reply.status}" + (f":{reply.reason}" if reply.reason else "")
        if reply.status in POOL_WAITING and reply.reason == URGENT_PREEMPT:
            return HoldPreempted(f"gpu_hold_preempted:{lease['lease_id']}")
        if reply.status in POOL_WAITING or reply.status in HELD:
            return HoldLost(f"gpu_hold_lost:{detail}")
        reason = str(reply.reason or reply.status)
        if reason == "deadline":
            return WorkflowDeadline("workflow_deadline")
        work_class = ""
        try:
            work_class = hold_placement(self.holds.cfg, admission or {})[0]
        except UnknownRoute:
            pass
        if refusal_is_terminal(self.holds.cfg, reason, work_class):
            return RuntimeError(f"gpu_pool_unavailable:{reason}")
        return HoldLost(f"gpu_hold_lost:{detail}")

    async def execute(self, state, node):
        lease = state.get("lease")
        if not is_hold_ref(lease):
            raise RuntimeError("gpu_hold_missing")
        row = await self.store.get_run(state["run_id"])
        if row.get("control"):
            raise RunControlPending(row["control"])
        beat = await self._beat(lease, state.get("admission"))
        if beat is not None and beat.status == "recall":
            # Already being recalled: never start a long turn on it. Not a failed attempt either way.
            if beat.reason == URGENT_PREEMPT and await self._await_requeue(lease["lease_id"], beat):
                raise HoldPreempted(f"gpu_hold_preempted_before_start:{lease['lease_id']}")
            # Any other recall (or a pool that never re-queued it): give it back now, queue afresh.
            await self._end_hold(state, lease["lease_id"], lease, "recalled_before_start")
            raise HoldRecalled("gpu_hold_recalled_before_start")
        detail = {"lease": lease, "lane": lease["role"]}
        if (state.get("workflow") or DEFAULT_WORKFLOW) in {DEFAULT_WORKFLOW, READING_WORKFLOW}:
            # Joinable to the pool's child leases (acceptance check 2: no un-attached agent lease
            # under a turn whose run holds a hold).
            detail["turn_correlation_id"] = turn_correlation_id(state)
        elif state.get("workflow") == REVERIE_VISUAL_WORKFLOW and state.get("step_correlation_id"):
            detail["step_correlation_id"] = state["step_correlation_id"]
        await self.event(state, "run.started", detail)
        work = asyncio.create_task(node(state))
        timeout = float(state["brief"]["timeout_sec"])
        deadline = state["admission"].get("deadline_at")
        if deadline:
            timeout = min(timeout, max(0, (datetime.fromisoformat(deadline)-self.now()).total_seconds()))
        try:
            # Timeout starts only after the hold is confirmed. Each heartbeat re-reads the operator
            # control row and asks the pool whether the hold is still ours at this generation.
            # Paused for urgent work: this run's in-flight call keeps the slot the urgent run waits for
            # until the harness is cancelled, so beat every PREEMPT_POLL_SEC to see the re-queue soon.
            # While urgent is enabled, beat within the grace so the recall is seen before the abort.
            steady = self.settings.lease_heartbeat_sec
            defaults = self.holds.cfg.defaults
            if int(defaults.urgent_max_concurrent) > 0:
                steady = min(steady, float(defaults.urgent_preempt_grace_sec))
            wait = steady
            async with asyncio.timeout(timeout):
                while True:
                    done, _ = await asyncio.wait({work}, timeout=wait)
                    if done:
                        break
                    row = await self.store.get_run(state["run_id"])
                    if row.get("control"):
                        raise RuntimeError(f"run_control:{row['control']}")
                    beat = await self._beat(lease, state.get("admission"))
                    if beat is not None:
                        wait = min(PREEMPT_POLL_SEC, steady) \
                            if beat.status == "recall" and beat.reason == URGENT_PREEMPT \
                            else steady
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
        # Outside the turn's time budget: a slow preempt check must not replace the work's own error.
        try:
            return await self._settled(work, lease, state.get("admission"))
        except TimeoutError as exc:
            if deadline and self.now() >= datetime.fromisoformat(deadline):
                raise WorkflowDeadline("workflow_deadline") from exc
            raise

    def _requeue_wait_sec(self, recall: GpuLeaseReplyV1) -> float:
        """How long to wait for the re-queue, from first sight. recall_by is the pool's clock: it
        only shortens the local grace when it is sane (in the future, sooner than the grace)."""
        grace = float(self.holds.cfg.defaults.urgent_preempt_grace_sec)
        wait = grace
        if recall.recall_by is not None:
            left = (recall.recall_by - self.now()).total_seconds()
            if 0 < left < grace:
                wait = left
        return wait + PREEMPT_REQUEUE_MARGIN_SEC

    async def _await_requeue(self, lease_id: str, recall: GpuLeaseReplyV1) -> bool:
        """A hold recalled for urgent work before its step started: wait (bounded) for the pool's
        abort to put it back in line in its original place. True once the pool says queued; False if
        it said something else (a re-queue for another reason included) or the wait ran out -- the
        caller then releases it as before."""
        end = time.monotonic() + self._requeue_wait_sec(recall)
        while True:
            try:
                reply = await asyncio.wait_for(self.holds.status(lease_id), max(0.1, end - time.monotonic()))
            except Exception as exc:  # noqa: BLE001 -- timeout / RPC failure: not re-queued yet
                logger.warning("durable_hold_status_failed lease=%s err=%s", lease_id, exc)
                reply = None
            if reply is not None and not is_pool_trouble(reply):
                if reply.status in POOL_WAITING:
                    return reply.reason == URGENT_PREEMPT
                if reply.status != "recall":
                    return False
            left = end - time.monotonic()
            if left <= 0:
                logger.warning("durable_hold_urgent_requeue_timeout lease=%s", lease_id)
                return False
            await asyncio.sleep(min(PREEMPT_POLL_SEC, left))

    async def _record_preempted(self, state, lease_id: str, generation: int | None, role: str | None,
                                reply: GpuLeaseReplyV1) -> None:
        """``run.preempted``: the pool paused this run's hold for an urgent run and kept its place."""
        await self.store.record_event(state["run_id"], "run.preempted", {
            "lease_id": lease_id, "generation": generation, "lane": role, "reason": URGENT_PREEMPT,
            "pool_status": reply.status, "position": reply.position},
            event_id=f"preempted:{lease_id}:{generation}")

    async def _settled(self, work: asyncio.Task, lease: dict, admission: dict | None = None):
        """The finished work's result. A work failure is checked against the pool once: a turn whose
        hold the pool took back (urgent pause, recall past its grace) often fails on its own before
        the next heartbeat sees it (its next LLM call cannot attach to the aborted hold) -- that is
        the pool's doing, not an attempt."""
        try:
            return work.result()
        except (HoldLost, RunControlPending, WorkflowDeadline):
            raise
        except Exception as exc:
            reply = await self._requeued_reply(lease)
            if reply is not None:
                raise self._taken_back(lease, reply, admission) from exc
            raise

    async def _requeued_reply(self, lease: dict) -> GpuLeaseReplyV1 | None:
        """The pool's status reply when it re-queued this hold (still ours, but no longer granted at
        this generation); None when it still grants it, ended it, or cannot be read."""
        try:
            reply = await self.holds.status(lease["lease_id"])
        except Exception as exc:  # noqa: BLE001 -- cannot tell: the failure stands as it is
            logger.warning("durable_hold_status_failed lease=%s err=%s", lease["lease_id"], exc)
            return None
        if is_pool_trouble(reply):
            return None
        if reply.status in POOL_WAITING:
            return reply
        if reply.status in HELD and reply.grant is not None and reply.grant.generation != lease["generation"]:
            return reply
        return None

    async def requeued(self, state) -> str | None:
        """AdmissionDeps.requeued: for a node whose failed turn is a returned result -- the release
        reason when the pool took the hold back, else None."""
        lease = state.get("lease")
        if not is_hold_ref(lease):
            return None
        reply = await self._requeued_reply(lease)
        if reply is None:
            return None
        return getattr(self._taken_back(lease, reply, state.get("admission")), "release_reason",
                       HoldLost.release_reason)

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
        # Also a hold re-queued for urgent work (queued, urgent_preempt): the tail needs no GPU, and a
        # kept re-queued hold would be granted to a run that no longer uses it.
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
            # Also a hold the pool already re-granted at a newer generation (re-queued and granted
            # again before this node noticed), but only on a take-back, which goes straight back to
            # resource_wait: ending it would lose the grant resource_wait is about to read. After a
            # failed attempt a backoff follows with nothing heartbeating it, so it is handed back.
            regranted = reason in TAKEBACK_RELEASE_REASONS and reply is not None \
                and reply.status in HELD and reply.grant is not None \
                and lease is not None and reply.grant.generation != lease["generation"]
            if reply is not None and (reply.status in POOL_WAITING or regranted) and not is_pool_trouble(reply):
                if lease and reply.status in POOL_WAITING and reply.reason == URGENT_PREEMPT:
                    await self._record_preempted(state, lease_id, lease["generation"], lease.get("role"), reply)
                elif lease:
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
            except (HoldLost, WorkflowDeadline, RuntimeError) as exc:  # a take-back or a refusal: gone either way
                logger.warning("durable_outreach_hold_lost run=%s lease=%s %s", run_id, lease["lease_id"], exc)
                await self._record_lost(state, lease, type(exc).__name__, str(exc)[:200])
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
        if state.get("workflow") == COMPACTOR_DIGEST_WORKFLOW:
            # No harness turn: each digest call is a plain cortex-orch verb RPC; a replay re-asks.
            return
        if state.get("workflow") in (REVERIE_VISUAL_WORKFLOW, ORION_DAY_WORKFLOW):
            # No harness turn: generate runs in orion-thought, whose replay is idempotent (the
            # recorded artifact, or a generate_in_flight retry); orion_day.letter's calls are plain
            # cortex verbs, never a harness run, so there is nothing to cancel by turn id.
            return
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
                    "brief": self._checkpoint_brief(workflow, request["brief"]),
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
                error = self._deadline_error(workflow)
                await graph.aupdate_state(cfg, {**released, "status": "failed", "last_error": error},
                                          as_node="failed")
                await self._terminal(run_id, "failed", {**state, "last_error": error}, workflow=workflow)
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
                error = self._deadline_error(workflow)
                await graph.aupdate_state(cfg, {**released, "status": "failed", "last_error": error},
                                          as_node="failed")
                await self._terminal(run_id, "failed", {**state, "last_error": error}, workflow=workflow)
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
            detail = self._terminal_detail_for(wf, status, state)
        abandon = None
        if wf == REVERIE_VISUAL_WORKFLOW:
            if status != "completed":
                abandon = await self._record_reverie_abandon(run_id, state)
            # A cancel that wins the race against this projection still says which dispatch it was.
            actual = await self.store.finish_projection(
                run_id, status, detail, cancelled_detail=self._terminal_detail_for(wf, "cancelled", state))
        elif (cancelled := self._terminal_detail_for(wf, "cancelled", state)):
            # Urgent runs: a cancel winning the race must still name the incident.
            actual = await self.store.finish_projection(run_id, status, detail, cancelled_detail=cancelled)
        else:
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
        if abandon is not None and actual in ("failed", "cancelled"):
            # Last, after every terminal fact: the run is terminal whether or not thought answers.
            # One try now, in the background (never holds this run's claim or driver slot); a
            # failed one stays pending and reconcile retries it (_retry_abandons).
            self._abandons[run_id] = {**abandon, "failures": 0, "due": 0.0, "pending_since": time.time()}
            self._spawn_abandon(run_id)

    async def _record_reverie_abandon(self, run_id: str, state: dict) -> dict | None:
        """A reverie.visual run is about to end without completing (graph ``failed``, run deadline
        noticed by the driver, operator cancel, resume-failure bound). Before the terminal projection,
        durably record that thought must still be told to abandon the attempt, so a crash anywhere
        after the projection still sends it (``run.abandon_pending``, read back by reconcile).
        ``attempt_id`` may be unknown: thought resolves the attempt by dispatch_id. Skipped while
        paused (the projection will not go terminal) or once completed. Never raises: the terminal
        transition must not wait on it."""
        row = None
        try:
            row = await self.store.get_run(run_id)
        except Exception:  # noqa: BLE001
            logger.exception("reverie_visual_abandon_row_read_failed run=%s", run_id)
        if row is not None and (row.get("terminal") == "completed" or row.get("control") == "paused"):
            return None
        brief = state.get("brief") or ((row or {}).get("request") or {}).get("brief") or {}
        entry = {"attempt_id": state.get("attempt_id"), "reason": state.get("last_error") or "terminal"}
        try:
            await self.store.record_event(run_id, ABANDON_PENDING_EVENT, entry, event_id=f"abandon_pending:{run_id}")
        except Exception:  # noqa: BLE001 -- still tried (and retried) from memory; not durable
            logger.exception("reverie_visual_abandon_record_failed run=%s", run_id)
        return {**entry, "visual_request": brief.get("visual_request")}

    def _spawn_abandon(self, run_id: str) -> asyncio.Task:
        """At most one abandon RPC in flight per run, off the reconcile loop's critical path."""
        task = self._abandoning.get(run_id)
        if task is None:
            task = asyncio.create_task(self._send_reverie_abandon(run_id), name=f"abandon-{run_id}")
            self._abandoning[run_id] = task
            task.add_done_callback(lambda t, key=run_id: self._abandoning.pop(key, None))
        return task

    async def _send_reverie_abandon(self, run_id: str) -> bool:
        """One abandon try. Thought's ``done``/``terminal`` answer is recorded as ``run.abandon_acked``
        (clears the pending record); anything else backs off base * 2^(n-1), capped at
        DURABLE_RUNS_RETRY_MAX_SEC, and stays pending. Never raises."""
        entry = self._abandons.get(run_id)
        if entry is None:
            return False
        ok, req = False, None
        try:
            req = abandon_request(run_id, entry["visual_request"], entry.get("attempt_id"))
            ok = await send_abandon(self._reverie_step, req, reason=entry["reason"])
            if ok:
                await self.store.record_event(run_id, ABANDON_ACKED_EVENT, {
                    "attempt_id": req.attempt_id, "dispatch_id": req.visual_request.dispatch_id,
                    "reason": entry["reason"], "tries": int(entry["failures"]) + 1},
                    event_id=f"abandon_acked:{run_id}")
        except Exception:  # noqa: BLE001 -- unbuildable request or the ack write failed: retry later
            logger.exception("reverie_visual_abandon_try_failed run=%s", run_id)
            ok = False
        if ok:
            self._abandons.pop(run_id, None)
            return True
        failures = int(entry["failures"]) + 1
        if time.time() - float(entry.get("pending_since") or time.time()) > ABANDON_GIVE_UP_SEC:
            # Thought's max-age sweep has released the attempt by now; stop asking.
            self._abandons.pop(run_id, None)
            logger.error("reverie_visual_abandon_given_up run=%s dispatch=%s failures=%s", run_id,
                         req.visual_request.dispatch_id if req else None, failures)
            return False
        delay = min(self.settings.retry_max_sec, self.settings.retry_base_sec * 2 ** min(failures - 1, 30))
        entry.update(failures=failures, due=time.monotonic() + delay)
        logger.warning("reverie_visual_abandon_pending run=%s dispatch=%s failures=%s retry_in=%.1fs", run_id,
                       req.visual_request.dispatch_id if req else None, failures, delay)
        return False

    async def _retry_abandons(self) -> None:
        """Reconcile: re-send every due pending abandon. The store's pending records are re-read
        every DURABLE_RUNS_HOLD_STATUS_POLL_SEC, so a restarted process adopts its predecessor's."""
        if self._abandons_loaded_at is None or \
                time.monotonic() - self._abandons_loaded_at >= self.settings.hold_status_poll_sec:
            for row in await self.store.abandons_pending():
                detail = row.get("detail") or {}
                brief = (row.get("request") or {}).get("brief") or {}
                generated_at = row.get("generated_at")
                self._abandons.setdefault(row["run_id"], {
                    "attempt_id": detail.get("attempt_id"), "reason": detail.get("reason") or "terminal",
                    "visual_request": brief.get("visual_request"), "failures": 0, "due": 0.0,
                    "pending_since": generated_at.timestamp() if isinstance(generated_at, datetime) else time.time()})
            self._abandons_loaded_at = time.monotonic()
        now = time.monotonic()
        for run_id, entry in list(self._abandons.items()):
            if run_id not in self._abandoning and now >= entry["due"]:
                self._spawn_abandon(run_id)

    async def reconcile(self):
        try:
            await self._retry_releases()
        except Exception:  # noqa: BLE001
            logger.exception("durable_hold_release_retry_failed")
        try:
            await self._retry_abandons()
        except Exception:  # noqa: BLE001 -- abandon upkeep must not stop the run loop
            logger.exception("reverie_visual_abandon_retry_failed")
        try:
            await self._beat_outreach()
        except Exception:  # noqa: BLE001 -- Door-A upkeep must not stop the run loop
            logger.exception("durable_outreach_hold_upkeep_failed")
        rows = await self.store.list_pending()
        urgent = {row["run_id"] for row in rows if _priority(row) == "urgent"}
        urgent_cap = int(self.holds.cfg.defaults.urgent_max_concurrent) if urgent else 0
        if urgent_cap <= 0:
            urgent = set()   # rollback switch: the pool treats urgent as background, and so do we
        system = {row["run_id"] for row in rows if _priority(row) == "system"} - urgent
        # Urgent first, then system (stable otherwise): an urgent run must not wait behind long
        # background turns before it even asks the pool, so it is exempt from
        # MAX_CONCURRENT_DRIVERS, capped instead at the pool's own urgent_max_concurrent. A system
        # run is exempt the same way, capped at MAX_CONCURRENT_SYSTEM_DRIVERS.
        for row in sorted(rows, key=lambda r: (r["run_id"] not in urgent, r["run_id"] not in system)):
            run_id = row["run_id"]
            if run_id in self.active:
                if row.get("control"):
                    self.active[run_id].cancel()
                continue
            if row.get("control") == "paused":
                # The pausing replica released the hold (control()); re-driving a paused run every
                # tick would only repeat pool RPCs. Resume clears control and it is driven again.
                continue
            if run_id in urgent:
                if len(self._urgent_drivers & self.active.keys()) >= urgent_cap:
                    continue
            elif run_id in system:
                if len(self._system_drivers & self.active.keys()) >= MAX_CONCURRENT_SYSTEM_DRIVERS:
                    continue
            elif len(self.active.keys() - self._urgent_drivers - self._system_drivers) >= MAX_CONCURRENT_DRIVERS:
                break
            task = asyncio.create_task(self._drive(row), name=f"admitted-{run_id}")
            self.active[run_id] = task
            if run_id in urgent:
                self._urgent_drivers.add(run_id)
            elif run_id in system:
                self._system_drivers.add(run_id)
            def finished(t, key=run_id):
                self.active.pop(key, None)
                self._urgent_drivers.discard(key)
                self._system_drivers.discard(key)
                if not t.cancelled() and t.exception():
                    logger.error("durable_driver_failed run=%s error=%s", key, t.exception())
            task.add_done_callback(finished)
        await self._publish_outbox()

    async def _publish_outbox(self) -> None:
        for raw in await self.store.pending_outbox():
            event = ResourceEventV1.model_validate(raw)
            terminal = event.event.removeprefix("run.")
            if terminal in TERMINAL_STATE_NODE and event.entry_id == f"{event.run_id}:terminal:{terminal}":
                # Every admitted terminal (completed, failed, cancelled) is also a DurableRunStateV1:
                # waiters (cortex-exec reflect, dispatch settlement) must see failures end the run.
                # At-least-once: acked only after both publishes; sql-writer dedupes on entry_id.
                row = await self.store.get_run(event.run_id)
                workflow = ((row or {}).get("request") or {}).get("workflow") or DEFAULT_WORKFLOW
                state_event = DurableRunStateV1(entry_id=event.entry_id+":state", run_id=event.run_id,
                    workflow=workflow, thread_id=event.thread_id, node=TERMINAL_STATE_NODE[terminal],
                    status=terminal, correlation_id=event.correlation_id, generated_at=event.generated_at,
                    detail=event.detail)
                if not await self.runner._publish(self.settings.state_channel, DURABLE_RUN_STATE_KIND, state_event,
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
                "queue_wait_seconds": max(0, (wait_end-row["created_at"]).total_seconds()),
                **({"reading_result": values.get("result"), "error": values.get("last_error"),
                    "work_started": await self.store.first_event_at(run_id, "run.started") is not None}
                   if workflow == READING_WORKFLOW else {}),
                **({"reverie_visual": {
                    "attempt_id": values.get("attempt_id"), "chain_id": values.get("chain_id"),
                    "outcome": values.get("outcome"), "reason": values.get("reason"),
                    "retries": int(values.get("retries") or 0), "retry_at": values.get("retry_at"),
                    "retry_node": values.get("retry_node"), "artifact_sha256": values.get("artifact_sha256"),
                    "generate_elapsed_sec": values.get("generate_elapsed_sec"),
                    "visual_elapsed_sec": values.get("visual_elapsed_sec"),
                    "started_at": values.get("started_at"), "finished_at": values.get("finished_at"),
                    "deadline_at": (row["request"]["admission"] or {}).get("deadline_at")},
                    "error": values.get("last_error"),
                    "work_started": await self.store.first_event_at(run_id, "run.started") is not None}
                   if workflow == REVERIE_VISUAL_WORKFLOW else {}),
                **({"orion_day": {
                    "letter_date": (values.get("brief") or {}).get("letter_date"),
                    "note_ready": bool(values.get("note_md")),
                    "carry_forward_ready": bool(values.get("carry_forward_md")),
                    "persisted": bool(values.get("persisted")),
                    "persist_outcome": values.get("persist_outcome"),
                    "llm_attempts": dict(values.get("llm_attempts") or {}),
                    "retry_at": values.get("retry_at"),
                    "deadline_at": (row["request"]["admission"] or {}).get("deadline_at")},
                    "error": values.get("last_error")}
                   if workflow == ORION_DAY_WORKFLOW else {})}

    async def close(self):
        # Pending abandons are durable (run.abandon_pending): the next process retries them.
        tasks = list(self.active.values()) + list(self._abandoning.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
