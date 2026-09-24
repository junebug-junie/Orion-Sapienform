"""Thin atomic capacity broker. Never calls Gateway or executes a workflow."""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import uuid4

from psycopg.types.json import Jsonb

from .policy import decide_lane, widen_alternatives
from .store import PostgresAdmissionStore


class ResourceBroker:
    def __init__(self, store: PostgresAdmissionStore, lanes: dict[str, dict[str, Any]], *, lease_seconds: float = 300,
                 widen_after_seconds: float = 1200, hysteresis_seconds: float = 120,
                 widening_enabled: bool = False, shadow: bool = False, capacity=None):
        if lease_seconds <= 0 or widen_after_seconds < 0 or hysteresis_seconds < 0:
            raise ValueError("invalid admission timing")
        self.store, self.lanes = store, lanes
        self.lease_seconds = lease_seconds
        self.widen_after_seconds = widen_after_seconds
        self.hysteresis_seconds = hysteresis_seconds
        self.widening_enabled, self.shadow = widening_enabled, shadow
        self.capacity = capacity
        self.elastic = None
        self.elastic_environment = {"eligible": False, "reason": "not_checked"}
        self.elastic_budget = {}

    async def tick(self) -> list[dict[str, Any]]:
        # Route aliases must contend for the same physical reservation even
        # when one configured base URL includes its optional trailing slash.
        self.lanes = {key: {**meta, "backend_key": str(meta.get("backend_key") or "").rstrip("/")}
                      for key, meta in self.lanes.items()}
        grants: list[dict[str, Any]] = []
        async with self.store.transaction() as conn:
            now = await self.store.now(conn)
            elastic_row = await self.elastic.row(conn) if self.elastic else None
            for lane, meta in self.lanes.items():
                if lane == "agent-burst" or (elastic_row and meta["backend_key"] == elastic_row["backend_key"]):
                    meta["healthy"] = bool(meta.get("healthy") and lane == "agent-burst" and
                        elastic_row and elastic_row["admissions_open"])

            await self.store._expire(conn, now)
            active = await (await conn.execute("SELECT l.*,r.request FROM durable_resource_leases l JOIN durable_admission_runs r USING(run_id) WHERE l.status='active'")).fetchall()
            remaining = {row["backend_key"]: max(self.lease_seconds,
                float(row["request"]["brief"]["timeout_sec"]) - (now-row["granted_at"]).total_seconds()) for row in active}
            if self.capacity is not None:
                for permit in await self.capacity.active(conn, now):
                    backend = permit["backend_key"]
                    remaining[backend] = max(remaining.get(backend, 0), self.lease_seconds,
                                             (permit["deadline_at"]-now).total_seconds())
            for meta in self.lanes.values():
                if meta.get("external_busy", False) is not False:
                    key = meta.get("backend_key")
                    remaining[key] = max(remaining.get(key, 0), float(meta.get("busy_budget_seconds", 3600)))
            pending = await (await conn.execute("SELECT d.*,r.request,(SELECT l.lane FROM durable_resource_leases l WHERE l.run_id=d.run_id ORDER BY l.generation LIMIT 1) AS first_assigned_lane FROM durable_resource_demands d JOIN durable_admission_runs r USING(run_id) WHERE d.status='pending' AND r.control IS NULL AND r.terminal IS NULL ORDER BY d.created_at,d.demand_id FOR UPDATE OF d")).fetchall()
            if elastic_row and elastic_row["admissions_open"]:
                await conn.execute("UPDATE durable_elastic_slot SET admission_observed=true WHERE slot='circe-gpu2'")
            ahead: dict[str, float] = {}
            for demand in pending:
                # Policy-additive: a lane declared compatible after this demand was frozen
                # still counts (see widen_alternatives). The stored row is not rewritten.
                widened = widen_alternatives(demand["requirement"], self.lanes)
                requirement = widened
                retained_lane = demand.get("first_assigned_lane")
                retain_assignment = bool(retained_lane and not requirement.get("operator_override")
                                         and not requirement.get("pinned_lane"))
                if retain_assignment:
                    # Retrying a study is not a portable workflow phase. Its
                    # first grant remains authoritative even after a waiting
                    # tick replaces decision.assigned_lane with None.
                    requirement = {**requirement, "pinned_lane": retained_lane}
                decision = decide_lane(requirement, self.lanes,
                                       waited_seconds=(now - demand["created_at"]).total_seconds(),
                                       active_remaining=remaining, queued_ahead=ahead,
                                       lease_seconds=self.lease_seconds,
                                       widen_after_seconds=self.widen_after_seconds,
                                       hysteresis_seconds=self.hysteresis_seconds,
                                       widening_enabled=self.widening_enabled)
                detail = decision.detail()
                if self.elastic and retained_lane in {None,"agent-burst"}:
                    from .elastic import activation_decision
                    meta = self.lanes.get("agent-burst", {})
                    elastic_detail = activation_decision(widened, meta,
                        retained_burst=retained_lane == "agent-burst",
                        waited=(now-demand["created_at"]).total_seconds(), threshold=self.widen_after_seconds,
                        widening=self.widening_enabled, preferred_start=decision.estimates.get("agent"),
                        queued_ahead=ahead.get(meta.get("backend_key"),0), budget=self.elastic_budget,
                        hysteresis=self.hysteresis_seconds, environment=self.elastic_environment)
                    detail["elastic"] = elastic_detail
                    if (elastic_detail["suppression_reason"] is None and not decision.assigned_lane
                            and elastic_row and elastic_row["state"] == "idle"
                            and not self.shadow and not getattr(self, "elastic_shadow", True)):
                        elastic_row = await self.elastic.intent(conn,"agent-burst",demand["run_id"],elastic_detail)

                if retain_assignment:
                    detail["retained_assigned_lane"] = retained_lane
                    detail["suppressed"] = {
                        lane: "run_assignment_locked" if reason == "experimental_pin" else reason
                        for lane, reason in detail["suppressed"].items()
                    }
                    if requirement["preferred_lane"] != retained_lane:
                        detail["suppressed"][requirement["preferred_lane"]] = "run_assignment_locked"
                    if detail["reason"] == "experimental_pin":
                        detail["reason"] = "run_assignment_locked"
                detail["shadow"] = self.shadow
                if self.capacity is not None:
                    # Reserve drain priority only on lanes actually selectable
                    # now. Hysteresis-suppressed candidates cannot block native
                    # traffic, and shadow evaluation reserves no capacity.
                    detail["eligible_backend_keys"] = [] if self.shadow else sorted({
                        self.lanes[lane]["backend_key"] for lane in decision.eligible_lanes
                        if lane not in decision.suppressed
                        and isinstance(self.lanes[lane].get("external_busy", False), bool)})
                previous = demand["decision"] or {}
                if detail != previous:
                    await conn.execute("UPDATE durable_resource_demands SET decision=%s WHERE demand_id=%s", (Jsonb(detail), demand["demand_id"]))
                    # Snapshot estimates can move each tick; events are transitions,
                    # not one new telemetry row per floating-point clock value.
                    if set(decision.eligible_lanes) - set(previous.get("eligible_lanes", [requirement["preferred_lane"]])):
                        await self.store._event(conn, demand["run_id"], "run.resource_eligibility_expanded", detail, now=now)
                    if detail["suppressed"] != previous.get("suppressed", {}):
                        await self.store._event(conn, demand["run_id"], "run.lane_swap_suppressed", detail, now=now)
                lane = decision.assigned_lane
                if lane and not self.shadow:
                    backend = self.lanes[lane]["backend_key"]
                    lease = await (await conn.execute("INSERT INTO durable_resource_leases(lease_id,demand_id,run_id,resource_key,lane,backend_key,granted_at,expires_at,heartbeat_at,status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'active') RETURNING *",
                                                    (uuid4().hex, demand["demand_id"], demand["run_id"], f"llm.route.{lane}", lane, backend, now, now + timedelta(seconds=self.lease_seconds), now))).fetchone()
                    await conn.execute("UPDATE durable_resource_demands SET status='granted' WHERE demand_id=%s", (demand["demand_id"],))
                    payload = {key: value.isoformat() if hasattr(value, "isoformat") else value for key, value in lease.items()}
                    await self.store._event(conn, demand["run_id"], "run.resource_granted", {"lease": payload, **detail}, event_id=f"granted:{lease['lease_id']}", now=now)
                    await self.store._event(conn, demand["run_id"], "run.lane_assigned", {"lease": payload, **detail}, event_id=f"assigned:{lease['lease_id']}", now=now)
                    remaining[backend] = float(demand["request"]["brief"]["timeout_sec"])
                    grants.append(lease)
                else:
                    # FIFO reservations are shared across aliases. A newly widened
                    # run keeps its original age and cannot overtake older native
                    # work. Suppressed alternatives do not hoard idle capacity.
                    for backend in {self.lanes[lane]["backend_key"] for lane in decision.eligible_lanes if lane not in decision.suppressed}:
                        ahead[backend] = ahead.get(backend, 0) + float(demand["request"]["brief"]["timeout_sec"])
        return grants
