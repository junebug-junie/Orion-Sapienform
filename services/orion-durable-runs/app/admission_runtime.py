"""Drive admitted LangGraph threads; the broker only owns resource capacity."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
from langgraph.graph import START
from langgraph.types import Command

from app.admitted_graph import AdmissionDeps, RunControlPending, WorkflowDeadline, build_admitted_graph
from app.graph import finish_detail, turn_correlation_id
from orion.durable_admission.broker import ResourceBroker
from orion.durable_admission.capacity import PostgresCapacityStore
from orion.durable_admission.store import PostgresAdmissionStore, SubmissionConflict
from orion.schemas.durable_run import DurableRunRequestV1, DurableRunStateV1, DURABLE_RUN_STATE_KIND
from orion.schemas.harness_finalize import HarnessRunCancelV1
from orion.schemas.resource_admission import RESOURCE_EVENT_CHANNEL, RESOURCE_EVENT_KIND, ResourceEventV1

logger = logging.getLogger(__name__)
TERMINAL = {"completed", "failed", "cancelled"}


class AdmissionRuntime:
    def __init__(self, settings, runner, pool, *, store=None, broker=None, clock=None):
        self.settings, self.runner, self.pool = settings, runner, pool
        self.store = store or PostgresAdmissionStore(pool)
        self.now = clock or (lambda: datetime.now(timezone.utc))
        self.broker = broker or ResourceBroker(self.store, lanes={}, lease_seconds=settings.lease_seconds,
            widen_after_seconds=settings.widening_after_sec, hysteresis_seconds=settings.widening_hysteresis_sec,
            widening_enabled=settings.widening_enabled, shadow=settings.admission_shadow,
            capacity=PostgresCapacityStore(self.store, ttl_seconds=settings.lease_seconds) if settings.capacity_enabled else None)
        self.graph = build_admitted_graph(runner._curiosity_deps(), AdmissionDeps(
            self.register, self.store.get_lease, self.execute, self.release, self.event,
            now=self.now, max_attempts=settings.retry_max_attempts,
            retry_base_seconds=settings.retry_base_sec, retry_max_seconds=settings.retry_max_sec,
            guard=self.guard), runner._checkpointer)
        self.active: dict[str, asyncio.Task] = {}
        self._wake = asyncio.Event()
        self._catalog_at = 0.0
        from app.elastic_runtime import ElasticRuntime
        self.elastic = ElasticRuntime(self) if getattr(settings,"elastic_enabled",False) else None

    @staticmethod
    def config(run_id):
        return {"configurable": {"thread_id": run_id}}

    async def submit(self, request: DurableRunRequestV1) -> dict:
        if request.admission is None:
            raise ValueError("resource admission is required on this endpoint")
        derive_alternatives = not request.admission.alternatives
        if derive_alternatives:
            # The real Curiosity producer names its preferred logical lane.
            # Expand only explicit operator declarations, frozen at acceptance
            # so a duplicate receipt cannot mutate the durable demand.
            existing = await self.store.get_run(request.run_id)
            if existing:
                alternatives = existing["request"]["admission"]["alternatives"]
            else:
                policy = json.loads(self.settings.lane_policy_json or "{}")
                alternatives = sorted(lane for lane, declaration in policy.items()
                    if lane != request.admission.preferred_lane
                    and request.admission.preferred_lane in declaration.get("compatible_with", []))
            request = request.model_copy(update={"admission": request.admission.model_copy(
                update={"alternatives": alternatives})})
        try:
            row = await self.store.submit(request.model_dump(mode="json"))
        except SubmissionConflict:
            if not derive_alternatives:
                raise
            # Two replicas can read an empty inbox before either commits,
            # while using different rolling policy revisions. The winner's
            # derived candidates are immutable; all caller-supplied fields
            # still undergo the store's full conflict check on this retry.
            existing = await self.store.get_run(request.run_id)
            if existing is None:
                raise
            request = request.model_copy(update={"admission": request.admission.model_copy(update={
                "alternatives": existing["request"]["admission"]["alternatives"]})})
            row = await self.store.submit(request.model_dump(mode="json"))
        if not row.get("terminal") and not row.get("control") and not await self.store.get_demand(request.run_id):
            await self.store.register_demand(request.run_id, request.admission.model_dump(mode="json"))
        self._wake.set()
        return {"run_id": request.run_id, "status": row.get("terminal") or "waiting_resource",
                "workflow_kind": request.workflow, "requested_resource": request.admission.resource}

    async def register(self, state):
        await self.store.register_demand(state["run_id"], state["admission"])

    async def release(self, run_id, reason):
        lease = await self.store.get_lease(run_id)
        if lease:
            await self.store.release(lease, reason)
        await self.store.suspend_demand(run_id)
        self._wake.set()

    async def event(self, state, name, detail):
        await self.store.record_event(state["run_id"], name, detail)

    async def guard(self, state):
        deadline = state["admission"].get("deadline_at")
        if deadline and self.now() >= datetime.fromisoformat(deadline):
            raise WorkflowDeadline("workflow_deadline")
        row = await self.store.get_run(state["run_id"])
        if row.get("control"):
            raise RunControlPending(row["control"])
        lease = await self.store.get_lease(state["run_id"])
        if lease and await self.store.validate(lease):
            await self.store.renew(lease, self.settings.lease_seconds)
            return lease
        await self.register(state)
        return None

    async def _cancel_harness(self, state, reason):
        correlation_id = turn_correlation_id(state)
        payload = HarnessRunCancelV1(correlation_id=correlation_id, reason=reason)
        await self.runner._publish("orion:harness:run:cancel", "harness.run.cancel.v1", payload,
                                  self.runner._corr_for_admission(correlation_id))

    async def execute(self, state, node):
        lease = state.get("lease")
        if not lease or not await self.store.validate(lease):
            raise RuntimeError("resource_lease_lost")
        row = await self.store.get_run(state["run_id"])
        if row.get("control"):
            raise RunControlPending(row["control"])
        await self.event(state, "run.started", {"lease": lease})
        work = asyncio.create_task(node(state))
        timeout = float(state["brief"]["timeout_sec"])
        deadline = state["admission"].get("deadline_at")
        if deadline:
            timeout = min(timeout, max(0, (datetime.fromisoformat(deadline)-self.now()).total_seconds()))
        try:
            # Timeout starts only after resource validation. Each heartbeat
            # checks the authoritative control/lease row; old grants cannot
            # extend a newer generation or accept a stale response.
            async with asyncio.timeout(timeout):
                while True:
                    done, _ = await asyncio.wait({work}, timeout=self.settings.lease_heartbeat_sec)
                    if done:
                        result = await work
                        if not await self.store.validate(lease):
                            raise RuntimeError("resource_lease_lost")
                        return result
                    row = await self.store.get_run(state["run_id"])
                    if row.get("control") or not await self.store.renew(lease, self.settings.lease_seconds):
                        raise RuntimeError("resource_lease_lost")
        except TimeoutError as exc:
            if deadline and self.now() >= datetime.fromisoformat(deadline):
                raise WorkflowDeadline("workflow_deadline") from exc
            raise
        finally:
            if not work.done():
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                await self._cancel_harness(state, "durable_attempt_stopped")

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

    async def _drive(self, row):
        run_id = row["run_id"]
        if self.settings.admission_shadow:
            return  # Broker persists would-be decisions without driving cognition.
        async with self.claim(run_id) as acquired:
            if not acquired:
                return
            await self.store.touch(run_id)
            row = await self.store.get_run(run_id)
            if row.get("terminal"):
                return
            cfg = self.config(run_id)
            snap = await self.graph.aget_state(cfg)
            request = row["request"]
            if not snap.values:
                state = {"run_id": run_id, "correlation_id": request["correlation_id"],
                         "brief": request["brief"], "admission": request["admission"],
                         "requested_at": request["requested_at"], "attempt": 0, "status": "queued"}
                # Persist initial checkpoint before dispatch. Inbox recovers a
                # death before this point; graph recovers a death after it.
                await self.graph.aupdate_state(cfg, state, as_node=START)
                snap = await self.graph.aget_state(cfg)
            state = dict(snap.values)
            if row.get("control") == "cancelled":
                await self.release(run_id, "cancelled")
                await self.graph.aupdate_state(cfg, {"status": "cancelled", "lease": None}, as_node="finish")
                await self._terminal(run_id, "cancelled", state)
                return
            if row.get("control") == "paused":
                await self.release(run_id, "paused")
                return
            if snap.next and state.get("lease"):
                # This driver acquired a fresh session lock for an unfinished
                # inference attempt. Its predecessor may still be running in
                # Hub after a broken connection. Fence that generation before
                # replaying the expensive node under a new grant.
                await self.release(run_id, "worker_recovery")
                if snap.next[0] in {"harness_turn", "run_started"}:
                    await self.graph.aupdate_state(cfg, {"lease": None, "status": "retrying", "retry_node": None}, as_node="retry_wait")
                else:
                    await self.graph.aupdate_state(cfg, {"lease": None})
                snap = await self.graph.aget_state(cfg)
                state = dict(snap.values)
            deadline = state["admission"].get("deadline_at")
            if deadline and self.now() >= datetime.fromisoformat(deadline):
                await self.release(run_id, "deadline")
                await self.graph.aupdate_state(cfg, {"status": "failed", "last_error": "workflow_deadline"}, as_node="failed")
                await self._terminal(run_id, "failed", {**state, "last_error": "workflow_deadline"})
                return
            if not snap.next:
                await self._terminal(run_id, state.get("status", "failed"), state)
                return
            resume = None
            if any(t.interrupts for t in snap.tasks):
                if snap.next == ("resource_wait",) and not await self.store.get_lease(run_id):
                    # Expiry withdraws the old grant. Re-register this graph's
                    # still-pending acquisition, keeping its original age.
                    await self.register(state)
                    return
                if snap.next == ("retry_wait",) and self.now() < datetime.fromisoformat(state["retry_at"]):
                    return
                resume = Command(resume=True)
                await self.event(state, "run.resumed", {"node": snap.next[0]})
            try:
                async for update in self.graph.astream(resume, cfg, stream_mode="updates", durability="sync"):
                    snap = await self.graph.aget_state(cfg)
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
                        await self._terminal(run_id, state["status"], state)
            except asyncio.CancelledError:
                raise
            except RunControlPending:
                return
            except WorkflowDeadline:
                await self.graph.aupdate_state(cfg, {"status": "failed", "last_error": "workflow_deadline"}, as_node="failed")
                await self._terminal(run_id, "failed", {**state, "last_error": "workflow_deadline"})
            except Exception:
                # The checkpoint retains the failing node. Reconciliation is
                # allowed to retry persistence/transport, never an empty result.
                logger.exception("durable_checkpoint_resume_failed run=%s", run_id)
                await self.event(state, "run.checkpoint_resume_failed", {"node": list(snap.next)})

    async def _terminal(self, run_id, status, state):
        detail = finish_detail(state) if status == "completed" else {"error": state.get("last_error")}
        actual = await self.store.finish_projection(run_id, status, detail)
        if actual is not None and actual != status:
            await self.graph.aupdate_state(self.config(run_id), {"status": actual}, as_node="finish")
        self._wake.set()

    async def refresh_lanes(self):
        policy = json.loads(self.settings.lane_policy_json or "{}")
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(self.settings.gateway_url.rstrip("/")+"/routes")
            response.raise_for_status()
            routes = response.json()["routes"]
            async def occupancy(upstream):
                try:
                    result = await client.get(upstream.rstrip("/")+"/slots")
                    result.raise_for_status()
                    slots = result.json()
                    if not isinstance(slots, list) or not slots or not all(
                        isinstance(slot, dict) and isinstance(slot.get("is_processing"), bool) for slot in slots
                    ):
                        return upstream, None
                    return upstream, any(slot["is_processing"] for slot in slots)
                except (httpx.HTTPError, ValueError):
                    return upstream, None
            occupancy_by_backend = dict(await asyncio.gather(*(
                occupancy(upstream) for upstream in {r["upstream"] for r in routes if r.get("upstream")}
            )))
        lanes = {}
        for route in routes:
            lane = route["id"]
            declaration = policy.get(lane, {})
            capabilities = dict(declaration.get("capabilities", {}))
            capabilities.pop("minimum_context_tokens", None)
            capabilities.pop("context_tokens", None)
            if route.get("n_ctx") is not None:
                capabilities["context_tokens"] = route["n_ctx"]
            if route.get("vision") is not None:
                capabilities["vision"] = route["vision"]
            lanes[lane] = {**declaration, "backend_key": route.get("upstream"),
                "healthy": route.get("status") == "up",
                "configured": bool(route.get("upstream")), "capabilities": capabilities,
                "quality_tier": declaration.get("quality_tier", 0),
                "external_busy": occupancy_by_backend.get(route.get("upstream"))}
        burst = lanes.get("agent-burst")
        if burst:
            preferred = next((r for r in routes if r["id"] == "agent"), {})
            actual = next((r for r in routes if r["id"] == "agent-burst"), {})
            same_model = preferred.get("model") and burst.get("activation_model") == preferred.get("model")
            burst["activatable"] = bool(burst.get("activatable") and self.elastic and same_model
                and burst["backend_key"].rstrip("/") == self.settings.elastic_backend.rstrip("/"))
            # A stopped route has no measured context. Activation may use the
            # audited same-model contract; assignment requires live facts again.
            burst["activation_capabilities"] = {**burst["capabilities"],
                "context_tokens":preferred.get("n_ctx"),"vision":preferred.get("vision")} if same_model else {}
            burst["healthy"] = bool(burst["healthy"] and same_model and actual.get("model") == preferred.get("model")
                and actual.get("n_ctx") == preferred.get("n_ctx") and actual.get("vision") == preferred.get("vision"))
        self.broker.lanes = lanes

    async def reconcile(self):
        if self.elastic:
            await self.elastic.tick()
        await self.broker.tick()
        for row in await self.store.list_pending():
            run_id = row["run_id"]
            if run_id in self.active:
                if row.get("control"):
                    self.active[run_id].cancel()
                continue
            if len(self.active) >= 4:
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
                completion = DurableRunStateV1(entry_id=event.entry_id+":state", run_id=event.run_id,
                    workflow="curiosity.investigate", thread_id=event.thread_id, node="finish", status="completed",
                    correlation_id=event.correlation_id, generated_at=event.generated_at, detail=event.detail)
                if not await self.runner._publish(self.settings.state_channel, DURABLE_RUN_STATE_KIND, completion,
                        self.runner._corr_for_admission(event.correlation_id)):
                    continue
            if await self.runner._publish(RESOURCE_EVENT_CHANNEL, RESOURCE_EVENT_KIND, event,
                    self.runner._corr_for_admission(event.correlation_id)):
                await self.store.ack_outbox(event.entry_id)

    async def wakeup(self, event):
        # Grants are hints; the graph rereads the authoritative fenced row.
        if event.event in {"run.resource_granted", "resource.lease_released", "resource.lease_expired"}:
            self._wake.set()

    async def run(self, stop):
        while not stop.is_set():
            self._wake.clear()
            try:
                await self.refresh_lanes()
            except Exception:
                self.broker.lanes = {}  # stale route health cannot grant capacity
                logger.warning("durable_route_catalog_unavailable", exc_info=True)
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
            await self.release(run_id, action)
            if run_id in self.active:
                self.active[run_id].cancel()
                await asyncio.gather(self.active[run_id], return_exceptions=True)
        elif action == "resume":
            await self.store.register_demand(run_id, row["request"]["admission"])
        await self.store.record_event(run_id, "run."+{"pause": "paused", "cancel": "cancelled", "resume": "resumed"}[action], {})
        self._wake.set()
        return await self.status(run_id)

    async def status(self, run_id):
        row = await self.store.get_run(run_id)
        if not row:
            raise KeyError(run_id)
        snap = await self.graph.aget_state(self.config(run_id))
        lease = await self.store.get_lease(run_id)
        history = await self.store.history(run_id)
        demand = await self.store.get_demand(run_id)
        first_grant = await self.store.first_granted_at(run_id)
        wait_end = first_grant or (row["updated_at"] if row.get("terminal") else self.now())
        return {"run_id": run_id, "thread_id": run_id, "workflow_kind": row["request"]["workflow"],
                "status": row.get("terminal") or row.get("control") or snap.values.get("status", "waiting_resource"),
                "requested_resource": row["request"]["admission"]["resource"], "lease": lease,
                "next": list(snap.next), "created_at": row["created_at"], "history": history,
                "admission": (demand or {}).get("decision", {}),
                "queue_wait_seconds": max(0, (wait_end-row["created_at"]).total_seconds())}

    async def close(self):
        if self.elastic:
            await self.elastic.close()
        tasks = list(self.active.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
