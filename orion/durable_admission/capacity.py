"""Request permits sharing the durable broker's transaction and physical keys."""
from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from psycopg.types.json import Jsonb

from orion.schemas.resource_admission import CapacityAcquireV1, CapacityPermitV1, CapacityTokenV1
from .store import PostgresAdmissionStore, SubmissionConflict


class PostgresCapacityStore:
    def __init__(self, store: PostgresAdmissionStore, *, ttl_seconds: float = 90,
                 reserve_waiting: bool = True):
        if not 5 <= ttl_seconds <= 3600:
            raise ValueError("capacity TTL must be between 5 and 3600 seconds")
        self.store, self.ttl_seconds = store, ttl_seconds
        self.reserve_waiting = reserve_waiting

    @staticmethod
    def public(row):
        return CapacityPermitV1.model_validate({key: row[key] for key in CapacityPermitV1.model_fields}).model_dump(mode="json")

    async def expire(self, conn, now):
        await conn.execute("UPDATE durable_gateway_permits SET status='expired' WHERE status='active' AND expires_at<=%s", (now,))

    async def active(self, conn, now):
        await self.expire(conn, now)
        return await (await conn.execute("SELECT * FROM durable_gateway_permits WHERE status='active'")).fetchall()

    async def acquire(self, request: CapacityAcquireV1):
        data = request.model_dump(mode="json")
        backend = request.backend_key.rstrip("/")
        data["backend_key"] = backend
        async with self.store.transaction() as conn:
            now = await self.store.now(conn)
            await self.expire(conn, now)
            old = await (await conn.execute("SELECT * FROM durable_gateway_permits WHERE request_id=%s", (request.request_id,))).fetchone()
            if old:
                if old["request"] != data:
                    raise SubmissionConflict("request_id already has a different capacity request")
                if old["status"] != "active":
                    return {"acquired": False, "reason": "request_finished"}
                # Even an idempotent acknowledgement must revalidate its owner.
            await self.store._expire(conn, now)
            held = await (await conn.execute("SELECT l.*,r.control,r.terminal FROM durable_resource_leases l JOIN durable_admission_runs r USING(run_id) WHERE l.backend_key=%s AND l.status='active'", (backend,))).fetchone()
            if request.lease:
                token = request.lease
                if not held or held["control"] or held["terminal"] or token.status != "active" or any(held[key] != getattr(token, key) for key in
                        ("lease_id", "run_id", "generation", "resource_key", "lane", "backend_key", "demand_id")) or token.lane != request.lane:
                    return {"acquired": False, "reason": "resource_lease_stale"}
            elif held:
                return {"acquired": False, "reason": "durable_lease_active"}
            if old:
                return {"acquired": True, "reason": "duplicate", "permit": self.public(old)}

            active = await (await conn.execute("SELECT * FROM durable_gateway_permits WHERE backend_key=%s AND status='active'", (backend,))).fetchall()
            if request.lease and active:
                return {"acquired": False, "reason": "owner_request_active"}
            if not request.lease:
                # A revoked owner can still have uncancellable upstream work.
                if any(row["lease_id"] for row in active):
                    return {"acquired": False, "reason": "capacity_full"}
                limit = min([request.max_inflight, *(row["max_inflight"] for row in active)])
                if len(active) >= limit:
                    return {"acquired": False, "reason": "capacity_full"}
                # Once the broker has advertised an eligible reservation,
                # let existing calls drain instead of letting fresh traffic
                # continuously steal the small grant window between polls.
                # A disabled/shadow admission driver cannot honor old drain
                # decisions persisted by a previous deployment. Existing live
                # leases above still fence capacity until released/expired.
                waiting = None
                if self.reserve_waiting:
                    waiting = await (await conn.execute("SELECT 1 FROM durable_resource_demands d JOIN durable_admission_runs r USING(run_id) WHERE d.status='pending' AND r.control IS NULL AND r.terminal IS NULL AND d.decision->'eligible_backend_keys' ? %s LIMIT 1", (backend,))).fetchone()
                if waiting:
                    return {"acquired": False, "reason": "durable_waiting"}
            row = await (await conn.execute(
                "INSERT INTO durable_gateway_permits(request_id,permit_id,request,correlation_id,lane,backend_key,lease_id,generation,max_inflight,deadline_at,granted_at,heartbeat_at,expires_at,status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'active') RETURNING *",
                (request.request_id, uuid4().hex, Jsonb(data), request.correlation_id, request.lane, backend,
                 request.lease.lease_id if request.lease else None, request.lease.generation if request.lease else None,
                 request.max_inflight, now+timedelta(seconds=request.budget_sec), now, now,
                 now+timedelta(seconds=self.ttl_seconds)))).fetchone()
            return {"acquired": True, "reason": "acquired", "permit": self.public(row)}

    async def renew(self, token: CapacityTokenV1):
        async with self.store.transaction() as conn:
            now = await self.store.now(conn)
            await self.expire(conn, now)
            row = await (await conn.execute("UPDATE durable_gateway_permits SET heartbeat_at=%s,expires_at=%s WHERE request_id=%s AND permit_id=%s AND status='active' RETURNING *", (now, now+timedelta(seconds=self.ttl_seconds), token.request_id, token.permit_id))).fetchone()
            # Retain occupancy even after the owning lease is revoked: the
            # Gateway may be waiting for an uncancellable executor thread.
            return {"valid": row is not None, "permit": self.public(row) if row else None}

    async def release(self, token: CapacityTokenV1):
        async with self.store.transaction() as conn:
            row = await (await conn.execute("UPDATE durable_gateway_permits SET status='released' WHERE request_id=%s AND permit_id=%s AND status='active' RETURNING request_id", (token.request_id, token.permit_id))).fetchone()
            return {"released": row is not None}

    async def snapshot(self):
        async with self.store.pool.connection() as conn:
            now = await self.store.now(conn)
            rows = await (await conn.execute("SELECT * FROM durable_gateway_permits WHERE status='active' AND expires_at>%s ORDER BY backend_key,granted_at", (now,))).fetchall()
            return {"active_permits": [self.public(row) for row in rows]}
