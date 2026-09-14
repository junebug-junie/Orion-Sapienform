"""Fixed GPU2 actuation adapter within existing admission reconciliation.

No inference, workflow execution, or queue ownership here. SQL owns intent;
controller owns containers; Gateway discovery owns model readiness.
"""
import asyncio
import logging
import math
from datetime import timedelta
import httpx
from orion.autonomy.thermal_gate import thermal_state
from orion.durable_admission.elastic import ElasticStore, LANE, SLOT
from orion.reverie.baseline import load_baseline_policy
from orion.schemas.reverie_visual import VisualActivityV1

logger = logging.getLogger(__name__)


class ElasticRuntime:
    def __init__(self, runtime):
        self.runtime = runtime
        self.settings = runtime.settings
        self.store = ElasticStore(runtime.store,self.settings.elastic_backend)
        self.job = None
        self.initialized = False
        self.thermal = "hot"  # conservative rearm across restart

    async def environment(self):
        s = self.settings
        if not s.elastic_thermal_enabled:
            return {"eligible": False, "reason": "thermal_eligibility_disabled"}
        try:
            policy = load_baseline_policy()
            async with httpx.AsyncClient(timeout=3) as client:
                cabinet = await client.get(s.elastic_cabinet_url)
                cabinet.raise_for_status()
                raw = cabinet.json()
                temp = float(raw["snapshot"]["frame"]["environment"]["temp_c"])
                age = float(raw["age_sec"])
                if not math.isfinite(temp) or not math.isfinite(age) or age < 0:
                    raise ValueError("invalid_thermal_reading")
                verdict = thermal_state(temp_c=temp,age_sec=age,previous_state=self.thermal)
                if not verdict.degraded:
                    self.thermal = verdict.state
                result = {"eligible": False, "thermal_state": verdict.state, "reason": verdict.reason}
                if verdict.degraded or not verdict.allows_gpu_work:
                    return result
                activity_response = await client.get(policy.thought_url.rstrip("/")+"/visual-chain/activity")
                activity_response.raise_for_status()
                activity = VisualActivityV1.model_validate(activity_response.json())
                now = self.runtime.now()
                age = (now-activity.observed_at).total_seconds()
                if activity.history_status != "ok" or not 0 <= age <= policy.freshness_sec:
                    return {**result,"reason":"visual_activity_unavailable"}
                urgent = policy.enabled and (not activity.last_success_at or
                    activity.last_success_at+timedelta(seconds=policy.interval_sec) <= now)
                if urgent or activity.active_attempt_id:
                    return {**result,"reason":"visual_baseline_urgent","baseline_urgent":True}
                status = await client.get(s.elastic_controller_url.rstrip("/")+"/v1/gpu-slots/circe-gpu2/status")
                status.raise_for_status()
                slot = status.json()
                result.update(diffusion_state=slot.get("active"), controller_state=slot.get("state"))
                if not slot.get("enabled") or slot.get("active") not in {"diffusion","agent-burst"}:
                    return {**result,"reason":"gpu2_not_physically_eligible"}
                return {**result,"eligible":True,"reason":"eligible"}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return {"eligible":False,"reason":"eligibility_unavailable"}

    async def tick(self):
        s = self.settings
        if not self.initialized:
            if not s.capacity_enabled or not s.admission_enabled:
                raise ValueError("elastic_requires_admission_and_capacity")
            await self.store.initialize()
            self.runtime.broker.elastic = self.store
            self.initialized = True
        env = await self.environment()
        broker = self.runtime.broker
        broker.elastic_environment = {**env,"checked_at":self.runtime.now().isoformat()}
        broker.elastic_shadow = s.elastic_shadow
        broker.elastic_budget = {"diffusion_drain_seconds":s.elastic_drain_budget,
            "controller_transition_seconds":s.elastic_transition_budget,"cold_model_load_seconds":s.elastic_cold_budget}
        async with self.store.store.transaction() as conn:
            now = await self.store.store.now(conn)
            row = await self.store.row(conn)
            if (row["state"] in {"requested", "failed"} and row["desired_target"] == LANE
                    and row["requested_at"] and (now-row["requested_at"]).total_seconds() >= s.elastic_max_borrow
                    and s.elastic_restoration and not s.elastic_shadow):
                await self.store.intent(conn,"diffusion",row["run_id"],{"reason":"activation_window_expired"})
            elif row["state"] == "ready":
                expired = (now-row["requested_at"]).total_seconds() >= s.elastic_max_borrow
                keep = env.get("eligible") and not expired and s.elastic_assignments and not s.elastic_shadow
                waiting = await (await conn.execute("SELECT 1 FROM durable_resource_demands d JOIN durable_admission_runs r USING(run_id) WHERE d.status='pending' AND r.terminal IS NULL AND r.control IS NULL AND d.decision->'eligible_lanes' ? 'agent-burst' AND NOT (d.decision->'suppressed' ? 'agent-burst') LIMIT 1")).fetchone()
                lease = await (await conn.execute("SELECT 1 FROM durable_resource_leases WHERE backend_key=%s AND status='active' AND expires_at>%s LIMIT 1", (self.store.backend,now))).fetchone()
                permit = await (await conn.execute("SELECT 1 FROM durable_gateway_permits WHERE backend_key=%s AND status='active' AND expires_at>%s LIMIT 1", (self.store.backend,now))).fetchone()
                if keep and (waiting or lease or permit or not row["admission_observed"]):
                    await conn.execute("UPDATE durable_elastic_slot SET idle_since=NULL WHERE slot=%s",(SLOT,))
                else:
                    # Closure forbids only new leases. Existing lease owners
                    # retain their remaining sequential FCC requests.
                    idle = row["idle_since"] or now
                    grace_done = (now-idle).total_seconds() >= s.elastic_idle_grace
                    await conn.execute("UPDATE durable_elastic_slot SET admissions_open=%s,idle_since=%s WHERE slot=%s",
                        (bool(keep and not grace_done and row["admissions_open"]),idle,SLOT))
                    residency_done = row["resident_at"] and (now-row["resident_at"]).total_seconds() >= s.elastic_min_residency
                    if s.elastic_restoration and not s.elastic_shadow and not lease and not permit and residency_done and (not keep or grace_done):
                        await self.store.intent(conn,"diffusion",row["run_id"],{"reason":"maximum_borrow" if expired else env.get("reason") if not keep else "idle_grace"})
            elif row["state"] == "idle" and row["last_restored_at"] and (now-row["last_restored_at"]).total_seconds() < s.elastic_min_residency:
                broker.elastic_environment = {**env,"eligible":False,"reason":"diffusion_min_residency"}
        row = await self.store.snapshot()
        if row.get("state") in {"requested","restoring","failed"} and not s.elastic_shadow:
            if self.job is None or self.job.done():
                self.job = asyncio.create_task(self.actuate(),name="gpu2-intent-reconcile")

    async def actuate(self):
        # Existing per-key advisory claim pattern, separate from any graph run.
        async with self.runtime.claim("elastic:circe-gpu2") as acquired:
            if not acquired:
                return
            row = await self.store.snapshot()
            if not row.get("can_transition"):
                return
            if row["desired_target"] == LANE:
                env = await self.environment()
                if not env.get("eligible"):
                    # Cold request may have been accepted by controller before
                    # restart. Close and restore, never leave a stranded loan.
                    if self.settings.elastic_restoration:
                        async with self.store.store.transaction() as conn:
                            await self.store.intent(conn,"diffusion",row["run_id"],env)
                    return
            try:
                if row["run_id"]:
                    await self.store.store.record_event(row["run_id"],"resource.elastic_started",
                        {"operation_id":row["operation_id"]},event_id=row["operation_id"]+":started")
                async with httpx.AsyncClient(timeout=1200) as client:
                    response = await client.post(self.settings.elastic_controller_url.rstrip("/")+"/v1/gpu-slots/activate",
                        json={"slot":SLOT,"target":row["desired_target"],"operation_id":row["operation_id"],"generation":row["generation"]})
                    result = response.json()
                    if result.get("status") == "busy":
                        return
                healthy = row["desired_target"] == "diffusion"
                if row["desired_target"] == LANE and result.get("status") in {"success", "noop"}:
                    await self.runtime.refresh_lanes()
                    meta = self.runtime.broker.lanes.get(LANE,{})
                    healthy = meta.get("healthy") is True and meta.get("external_busy") is False
                await self.store.complete(row["operation_id"],result,healthy=healthy,assignments=self.settings.elastic_assignments)
            except Exception as exc:
                logger.error("gpu2_intent_reconcile_failed operation=%s error_type=%s",row["operation_id"],type(exc).__name__)
                await self.store.complete(row["operation_id"],{"status":"failed","error":type(exc).__name__},healthy=False,assignments=False)
            finally:
                self.runtime._wake.set()

    async def close(self):
        if self.job:
            self.job.cancel()
            await asyncio.gather(self.job,return_exceptions=True)
