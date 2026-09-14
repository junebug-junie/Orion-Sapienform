"""Postgres inbox, resource facts and transactional lifecycle outbox.

The `terminal` field is solely a projection written by the graph driver. This
store neither advances a workflow nor schedules its retries. A short global
transaction lock intentionally serializes the small background admission queue;
no transaction/connection is held while a run waits for capacity.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from orion.schemas.resource_admission import ResourceRequirementV1, ResourceLeaseV1, ResourceEventV1

ADMISSION_LOCK = 741853219


class SubmissionConflict(ValueError):
    """An existing run id was reused for a different immutable request."""


class PostgresAdmissionStore:
    def __init__(self, pool: Any, *, clock: Callable[[], datetime] | None = None):
        self.pool = pool
        self.clock = clock  # Test-only fake clock; production uses the DB clock.

    async def setup(self) -> None:
        directory = Path(__file__).resolve().parents[2] / "services/orion-sql-db"
        async with self.pool.connection() as conn:
            async with conn.transaction():
                for name in ("manual_migration_durable_resource_admission_v1.sql", "manual_migration_gateway_capacity_v1.sql", "manual_migration_gpu2_elastic_v1.sql"):
                    await conn.execute((directory / name).read_text(), prepare=False)

    async def now(self, conn: Any) -> datetime:
        if self.clock:
            return self.clock()
        row = await (await conn.execute("SELECT clock_timestamp() AS now")).fetchone()
        return row["now"]

    @asynccontextmanager
    async def transaction(self):
        async with self.pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SELECT pg_advisory_xact_lock(%s)", (ADMISSION_LOCK,))
                yield conn

    async def _event(self, conn: Any, run_id: str, event: str, detail: dict[str, Any], *, event_id: str | None = None, now: datetime | None = None):
        row = await (await conn.execute("SELECT request FROM durable_admission_runs WHERE run_id=%s", (run_id,))).fetchone()
        if row is None:
            raise KeyError(run_id)
        entry_id = event_id or uuid4().hex
        occurred = now or await self.now(conn)
        payload = {"schema_version": "durable.resource.event.v1", "entry_id": entry_id,
                   "event": event, "run_id": run_id, "thread_id": run_id,
                   "correlation_id": row["request"]["correlation_id"],
                   "generated_at": occurred.isoformat(), "detail": {"session_id": row["request"].get("brief", {}).get("session_id"), **detail}}
        payload = ResourceEventV1.model_validate(payload).model_dump(mode="json")
        await conn.execute("INSERT INTO durable_resource_events(entry_id,run_id,event,generated_at,payload) VALUES (%s,%s,%s,%s,%s) ON CONFLICT(entry_id) DO NOTHING",
                           (entry_id, run_id, event, occurred, Jsonb(payload)))
        return payload

    async def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        run_id = request["run_id"]
        async with self.transaction() as conn:
            now = await self.now(conn)
            inserted = await (await conn.execute("INSERT INTO durable_admission_runs(run_id,request,created_at,updated_at) VALUES (%s,%s,%s,%s) ON CONFLICT(run_id) DO NOTHING RETURNING *",
                                                (run_id, Jsonb(request), now, now))).fetchone()
            if inserted:
                await self._event(conn, run_id, "run.accepted", {}, event_id=f"accepted:{run_id}", now=now)
                return inserted
            row = await (await conn.execute("SELECT * FROM durable_admission_runs WHERE run_id=%s", (run_id,))).fetchone()
            def comparable(value):
                result = {k: v for k, v in value.items() if k != "requested_at"}
                if result.get("admission") is not None:
                    result["admission"] = {"allow_elastic_activation": False, **result["admission"]}
                return result
            if comparable(row["request"]) != comparable(request):
                raise SubmissionConflict("run_id already exists with a different request")
            return row

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            return await (await conn.execute("SELECT * FROM durable_admission_runs WHERE run_id=%s", (run_id,))).fetchone()

    async def list_pending(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            return await (await conn.execute("SELECT * FROM durable_admission_runs WHERE terminal IS NULL ORDER BY updated_at,run_id LIMIT %s", (limit,))).fetchall()

    async def set_control(self, run_id: str, control: str | None) -> None:
        if control not in {None, "paused", "cancelled"}:
            raise ValueError("unsupported control")
        async with self.transaction() as conn:
            await conn.execute("UPDATE durable_admission_runs SET control=%s, updated_at=%s WHERE run_id=%s AND terminal IS NULL AND control IS DISTINCT FROM 'cancelled'", (control, await self.now(conn), run_id))
            if control:
                await conn.execute("UPDATE durable_resource_demands SET status='suspended' WHERE run_id=%s AND status='pending'", (run_id,))

    async def touch(self, run_id: str) -> None:
        """Fair reconciliation rotation; does not change original queue age."""
        async with self.pool.connection() as conn:
            await conn.execute("UPDATE durable_admission_runs SET updated_at=clock_timestamp() WHERE run_id=%s", (run_id,))

    async def mark_terminal(self, run_id: str, status: str) -> None:
        if status not in {"completed", "failed", "cancelled", "abandoned"}:
            raise ValueError("not a terminal graph projection")
        async with self.transaction() as conn:
            await conn.execute("UPDATE durable_admission_runs SET terminal=CASE WHEN control='cancelled' THEN 'cancelled' ELSE %s END,updated_at=%s WHERE run_id=%s AND terminal IS NULL AND control IS DISTINCT FROM 'paused'", (status, await self.now(conn), run_id))
            leases = await (await conn.execute("SELECT * FROM durable_resource_leases WHERE run_id=%s AND status='active'", (run_id,))).fetchall()
            for lease in leases:
                await self._release(conn, lease, status)
            await conn.execute("UPDATE durable_resource_demands SET status='withdrawn' WHERE run_id=%s", (run_id,))

    async def finish_projection(self, run_id: str, status: str, detail: dict[str, Any]) -> str | None:
        """Atomically publish graph terminal facts and release resources.

        The shared transaction lock linearizes an operator control against
        completion; a cancelled run can never acquire a completed outbox event.
        """
        if status not in {"completed", "failed", "cancelled"}:
            raise ValueError("invalid terminal graph status")
        async with self.transaction() as conn:
            row = await (await conn.execute("SELECT * FROM durable_admission_runs WHERE run_id=%s", (run_id,))).fetchone()
            if row["terminal"]:
                return row["terminal"]
            if row["control"] == "paused":
                return None
            if row["control"] == "cancelled":
                status, detail = "cancelled", {}
            leases = await (await conn.execute("SELECT * FROM durable_resource_leases WHERE run_id=%s AND status='active'", (run_id,))).fetchall()
            for lease in leases:
                await self._release(conn, lease, status)
            await self._event(conn, run_id, "run."+status, detail, event_id=f"{run_id}:terminal:{status}")
            await conn.execute("UPDATE durable_admission_runs SET terminal=%s,updated_at=%s WHERE run_id=%s", (status, await self.now(conn), run_id))
            await conn.execute("UPDATE durable_resource_demands SET status='withdrawn' WHERE run_id=%s", (run_id,))
            return status

    async def register_demand(self, run_id: str, requirement: dict[str, Any], step: str = "harness_turn") -> dict[str, Any]:
        requirement = ResourceRequirementV1.model_validate(requirement).model_dump(mode="json")
        demand_id = f"{run_id}:{step}:{requirement['resource']}"
        async with self.transaction() as conn:
            row = await (await conn.execute("SELECT * FROM durable_admission_runs WHERE run_id=%s", (run_id,))).fetchone()
            if not row:
                raise KeyError(run_id)
            if row["terminal"] or row["control"]:
                raise ValueError("run cannot request capacity while controlled or terminal")
            existing = await (await conn.execute("SELECT * FROM durable_resource_demands WHERE run_id=%s", (run_id,))).fetchone()
            if existing and (existing["demand_id"] != demand_id or existing["requirement"] != requirement):
                raise SubmissionConflict("run demand is immutable")
            now = await self.now(conn)
            demand = await (await conn.execute("INSERT INTO durable_resource_demands(demand_id,run_id,requirement,created_at,status) VALUES (%s,%s,%s,%s,'pending') ON CONFLICT(run_id) DO UPDATE SET status=CASE WHEN durable_resource_demands.status='suspended' THEN 'pending' ELSE durable_resource_demands.status END RETURNING *",
                                              (demand_id, run_id, Jsonb(requirement), row["created_at"]))).fetchone()
            if not existing:
                await self._event(conn, run_id, "run.waiting_resource", {"demand_id": demand_id, "requested_lane": requirement["preferred_lane"]}, now=now)
            return demand

    async def get_demand(self, run_id: str) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            return await (await conn.execute("SELECT * FROM durable_resource_demands WHERE run_id=%s", (run_id,))).fetchone()

    async def suspend_demand(self, run_id: str) -> None:
        async with self.transaction() as conn:
            await conn.execute("UPDATE durable_resource_demands SET status='suspended' WHERE run_id=%s AND status != 'withdrawn'", (run_id,))

    async def get_lease(self, run_id: str) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            return await (await conn.execute("SELECT * FROM durable_resource_leases WHERE run_id=%s AND status='active' AND expires_at>%s", (run_id, await self.now(conn)))).fetchone()

    async def first_granted_at(self, run_id: str) -> datetime | None:
        """Initial admission timestamp; retries never restart queue timing."""
        async with self.pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT granted_at FROM durable_resource_leases WHERE run_id=%s ORDER BY generation LIMIT 1",
                (run_id,),
            )).fetchone()
            return row["granted_at"] if row else None

    @staticmethod
    def _identity(lease: dict[str, Any]) -> tuple:
        return (lease["lease_id"], lease["run_id"], lease["generation"], lease["resource_key"], lease["lane"], lease["backend_key"], lease["demand_id"])

    async def validate(self, lease: dict[str, Any]) -> bool:
        try:
            identity = self._identity(lease)
        except KeyError:
            return False
        async with self.pool.connection() as conn:
            row = await (await conn.execute("SELECT 1 FROM durable_resource_leases l JOIN durable_admission_runs r USING(run_id) WHERE lease_id=%s AND l.run_id=%s AND generation=%s AND resource_key=%s AND lane=%s AND backend_key=%s AND demand_id=%s AND status='active' AND expires_at>%s AND r.control IS NULL AND r.terminal IS NULL", (*identity, await self.now(conn)))).fetchone()
            return row is not None

    async def renew(self, lease: dict[str, Any], ttl_seconds: float) -> dict[str, Any] | None:
        if ttl_seconds <= 0:
            raise ValueError("lease TTL must be positive")
        async with self.transaction() as conn:
            now = await self.now(conn)
            return await (await conn.execute("UPDATE durable_resource_leases l SET heartbeat_at=%s,expires_at=GREATEST(expires_at,%s) FROM durable_admission_runs r WHERE l.run_id=r.run_id AND lease_id=%s AND l.run_id=%s AND generation=%s AND resource_key=%s AND lane=%s AND backend_key=%s AND demand_id=%s AND l.status='active' AND l.expires_at>%s AND r.control IS NULL AND r.terminal IS NULL RETURNING l.*",
                                            (now, now + timedelta(seconds=ttl_seconds), *self._identity(lease), now))).fetchone()

    async def _release(self, conn: Any, lease: dict[str, Any], reason: str) -> bool:
        row = await (await conn.execute("UPDATE durable_resource_leases SET status='released' WHERE lease_id=%s AND run_id=%s AND generation=%s AND resource_key=%s AND lane=%s AND backend_key=%s AND demand_id=%s AND status='active' RETURNING *", self._identity(lease))).fetchone()
        if row:
            await conn.execute("UPDATE durable_resource_demands SET status='suspended' WHERE demand_id=%s AND status='granted'", (row["demand_id"],))
            await self._event(conn, row["run_id"], "resource.lease_released", {"lease_id": row["lease_id"], "generation": row["generation"], "lane": row["lane"], "reason": reason}, event_id=f"released:{row['lease_id']}")
        return row is not None

    async def release(self, lease: dict[str, Any], reason: str = "released") -> bool:
        async with self.transaction() as conn:
            return await self._release(conn, lease, reason)

    async def _expire(self, conn: Any, now: datetime) -> list[dict[str, Any]]:
        rows = await (await conn.execute("UPDATE durable_resource_leases SET status='expired' WHERE status='active' AND expires_at<=%s RETURNING *", (now,))).fetchall()
        for row in rows:
            await conn.execute("UPDATE durable_resource_demands SET status='suspended' WHERE demand_id=%s AND status='granted'", (row["demand_id"],))
            await self._event(conn, row["run_id"], "resource.lease_expired", {"lease_id": row["lease_id"], "generation": row["generation"], "lane": row["lane"]}, event_id=f"expired:{row['lease_id']}", now=now)
        return rows

    async def expire(self) -> list[dict[str, Any]]:
        async with self.transaction() as conn:
            return await self._expire(conn, await self.now(conn))

    async def record_event(self, run_id: str, event: str, detail: dict[str, Any], event_id: str | None = None) -> dict[str, Any]:
        async with self.transaction() as conn:
            return await self._event(conn, run_id, event, detail, event_id=event_id)

    async def history(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            rows = await (await conn.execute("SELECT payload FROM durable_resource_events WHERE run_id=%s ORDER BY generated_at DESC,entry_id DESC LIMIT %s", (run_id, limit))).fetchall()
            return [row["payload"] for row in reversed(rows)]

    async def pending_outbox(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            rows = await (await conn.execute("SELECT payload FROM durable_resource_events WHERE published_at IS NULL ORDER BY generated_at,entry_id LIMIT %s", (limit,))).fetchall()
            return [row["payload"] for row in rows]

    async def ack_outbox(self, event_id: str) -> None:
        async with self.pool.connection() as conn:
            await conn.execute("UPDATE durable_resource_events SET published_at=clock_timestamp() WHERE entry_id=%s AND published_at IS NULL", (event_id,))

    async def queue_snapshot(self) -> dict[str, Any]:
        async with self.pool.connection() as conn:
            now = await self.now(conn)
            queued = await (await conn.execute("SELECT requirement->>'preferred_lane' AS lane,count(*) AS depth,EXTRACT(EPOCH FROM %s-min(created_at))::double precision AS oldest_age_seconds FROM durable_resource_demands WHERE status='pending' GROUP BY 1", (now,))).fetchall()
            active = await (await conn.execute("SELECT lane,count(*) AS active_leases FROM durable_resource_leases WHERE status='active' AND expires_at>%s GROUP BY lane", (now,))).fetchall()
            counts = await (await conn.execute("SELECT event,count(*) AS count FROM durable_resource_events GROUP BY event ORDER BY event")).fetchall()
            suppressions = await (await conn.execute("SELECT reason,count(*) AS count FROM durable_resource_events CROSS JOIN LATERAL jsonb_each_text(payload->'detail'->'suppressed') AS reasons(lane,reason) WHERE event='run.lane_swap_suppressed' GROUP BY reason")).fetchall()
            waits = await (await conn.execute("SELECT DISTINCT ON (l.run_id) EXTRACT(EPOCH FROM l.granted_at-r.created_at)::double precision AS seconds,l.lane,d.requirement->>'preferred_lane' AS preferred FROM durable_resource_leases l JOIN durable_admission_runs r USING(run_id) JOIN durable_resource_demands d USING(demand_id) ORDER BY l.run_id,l.generation")).fetchall()
            buckets = [0, 1, 10, 60, 300, 1200, 3600]
            histogram = {str(bound): sum(w["seconds"] <= bound for w in waits) for bound in buckets}
            histogram["+Inf"] = len(waits)
            return {"queued": queued, "active": active, "lifecycle_counts": {r["event"]: r["count"] for r in counts},
                    "widening_suppression_reasons": {r["reason"]: r["count"] for r in suppressions},
                    "wait_duration_seconds": {"cumulative_buckets": histogram, "count": len(waits),
                                              "sum": sum(w["seconds"] for w in waits)},
                    "alternative_assignments": sum(w["lane"] != w["preferred"] for w in waits)}
