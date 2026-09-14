"""Durable fixed-slot intent, sharing the existing capacity transaction lock."""
from psycopg.types.json import Jsonb
from .policy import satisfies

SLOT = "circe-gpu2"
LANE = "agent-burst"


def activation_decision(requirement, meta, *, waited, threshold, widening,
                        preferred_start, queued_ahead, budget, hysteresis, environment, retained_burst=False):
    """Declaration-based estimate, never a lease or a learned signal."""
    cost = sum(budget.values()) + queued_ahead + float(meta.get("switching_cost_seconds", 0))
    reason = None
    if not widening:
        reason = "widening_disabled"
    elif not requirement.get("allow_elastic_activation"):
        reason = "run_elastic_disabled"
    elif requirement.get("operator_override") or requirement.get("pinned_lane"):
        reason = "operator_pin"
    elif waited < threshold:
        reason = "wait_threshold"
    elif requirement.get("preferred_lane") != "agent" or LANE not in requirement.get("alternatives", []):
        reason = "not_candidate"
    elif not meta.get("configured") or not meta.get("backend_key") or not meta.get("activatable"):
        reason = "not_activatable"
    elif "agent" not in meta.get("compatible_with", []):
        reason = "compatibility_not_declared"
    elif not satisfies(meta.get("activation_capabilities", {}), requirement.get("requirements", {})):
        reason = "hard_requirements_incompatible"
    elif meta.get("quality_drop", 0) > meta.get("max_quality_drop", 1):
        reason = "quality_drop"
    elif not environment.get("eligible"):
        reason = environment.get("reason", "physical_eligibility_unknown")
    elif not retained_burst and (preferred_start is None or cost + hysteresis >= preferred_start):
        reason = "cold_start_hysteresis"
    return {"preferred_lane": "agent", "elastic_candidate": LANE,
            "threshold_elapsed": waited >= threshold, "predicted_preferred_start": preferred_start,
            "predicted_burst_start": cost, "budget_kind": "conservative_declaration",
            "retained_lane_reactivation": retained_burst, "budget": budget, "suppression_reason": reason, "environment": environment}


class ElasticStore:
    def __init__(self, store, backend):
        self.store, self.backend = store, backend.rstrip("/")

    async def initialize(self):
        # Production never applies schema here. Missing migration fails startup.
        async with self.store.transaction() as conn:
            await conn.execute("INSERT INTO durable_elastic_slot(slot,backend_key) VALUES (%s,%s) ON CONFLICT DO NOTHING", (SLOT,self.backend))
            row = await self.row(conn)
            if row["backend_key"] != self.backend:
                raise ValueError("elastic_backend_change_requires_operator_drain")

    async def row(self, conn):
        return await (await conn.execute("SELECT * FROM durable_elastic_slot WHERE slot=%s", (SLOT,))).fetchone()

    async def snapshot(self):
        async with self.store.transaction() as conn:
            row = await self.row(conn)
            if row is None:
                return {"enabled": False, "can_transition": False}
            now = await self.store.now(conn)
            leases = await (await conn.execute("SELECT 1 FROM durable_resource_leases WHERE backend_key=%s AND status='active' AND expires_at>%s LIMIT 1", (self.backend,now))).fetchone()
            permits = await (await conn.execute("SELECT 1 FROM durable_gateway_permits WHERE backend_key=%s AND status='active' AND expires_at>%s LIMIT 1", (self.backend,now))).fetchone()
            return {**row, "enabled": True, "can_transition": not row["admissions_open"] and not leases and not permits,
                    "active_lease": bool(leases), "active_permit": bool(permits)}

    async def intent(self, conn, target, run_id, detail):
        now = await self.store.now(conn)
        row = await self.row(conn)
        generation = row["generation"]+1
        operation = f"gpu2:{generation}:{target}"
        await conn.execute("UPDATE durable_elastic_slot SET generation=%s,operation_id=%s,run_id=%s,desired_target=%s,state=%s,admissions_open=false,requested_at=%s,idle_since=NULL,detail=%s WHERE slot=%s",
            (generation,operation,run_id,target,"requested" if target == LANE else "restoring",now,Jsonb(detail),SLOT))
        if run_id:
            await self.store._event(conn,run_id,"resource.elastic_requested",{"operation_id":operation,"target":target,**detail},event_id=operation,now=now)
        return await self.row(conn)

    async def complete(self, operation, result, *, healthy, assignments):
        async with self.store.transaction() as conn:
            row = await self.row(conn)
            if row["operation_id"] != operation:
                return
            now = await self.store.now(conn)
            success = result.get("status") in {"success","noop"} and healthy
            burst = row["desired_target"] == LANE
            restored = not success and burst and result.get("restored") is True
            state = "ready" if success and burst else "idle" if success or restored else "failed"
            if restored:
                await conn.execute("UPDATE durable_elastic_slot SET desired_target='diffusion',last_restored_at=%s WHERE slot=%s", (now,SLOT))
            # The broker still checks real Gateway health/capabilities/occupancy.
            await conn.execute("UPDATE durable_elastic_slot SET state=%s,admission_observed=false,admissions_open=%s,resident_at=CASE WHEN %s THEN %s ELSE resident_at END,last_restored_at=CASE WHEN %s THEN %s ELSE last_restored_at END,detail=%s WHERE slot=%s",
                (state,success and burst and assignments,success and burst,now,success and not burst,now,Jsonb(result),SLOT))
            if row["run_id"]:
                await self.store._event(conn,row["run_id"],"resource.elastic_completed" if success else "resource.elastic_failed",
                    {"operation_id":operation,"target":row["desired_target"],**result},event_id=operation+(":completed" if success else ":failed"),now=now)
