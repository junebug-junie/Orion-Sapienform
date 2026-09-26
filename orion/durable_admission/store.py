"""Postgres inbox (run registry) and transactional lifecycle outbox for durable runs.

Stage 4.5 (GPU pool cutover, docs/superpowers/specs/2026-09-25-gpu-pool-stage4-durable-runs-and-actuation.md):
the GPU pool is the only scheduler. A run's GPU is a pool *hold* (orion.gpu_pool.client), whose id
lives in the run's LangGraph checkpoint; this store no longer writes demands, grants or leases.

``durable_resource_demands`` / ``durable_resource_leases`` / ``durable_elastic_slot`` are FROZEN:
nothing here inserts into them. They stay because ``capacity.py`` (world-model and visual-chain
GPU permits, deleted in stage 5) still joins them, and ``_expire`` below is its only use of the
lease half. The `terminal` field is solely a projection written by the graph driver.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from orion.schemas.resource_admission import ResourceEventV1

ADMISSION_LOCK = 741853219
# ResourceRequirementV1 fields only the deleted broker read (spec: accepted and ignored until
# producers stop sending them in PR 4.6, because the model is extra="forbid").
IGNORED_ADMISSION_FIELDS = frozenset({"allow_elastic_activation", "alternatives", "pinned_lane", "operator_override"})


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
                    # Stage 4.5: the broker's lane-choice fields are accepted and ignored (the pool
                    # places the run), so they cannot make a duplicate receipt a conflict. A row
                    # accepted before the cutover carries broker-derived alternatives.
                    result["admission"] = {k: v for k, v in result["admission"].items()
                                           if k not in IGNORED_ADMISSION_FIELDS}
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

    async def touch(self, run_id: str) -> None:
        """Fair reconciliation rotation; does not change original queue age."""
        async with self.pool.connection() as conn:
            await conn.execute("UPDATE durable_admission_runs SET updated_at=clock_timestamp() WHERE run_id=%s", (run_id,))

    async def finish_projection(self, run_id: str, status: str, detail: dict[str, Any]) -> str | None:
        """Atomically publish the graph's terminal fact.

        The shared transaction lock linearizes an operator control against completion; a cancelled
        run can never acquire a completed outbox event. The run's GPU pool hold is released by the
        graph node that ends the run (or kept for Door-A outreach), never here: this store has no
        grant of its own to release since stage 4.5.
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
            await self._event(conn, run_id, "run."+status, detail, event_id=f"{run_id}:terminal:{status}")
            await conn.execute("UPDATE durable_admission_runs SET terminal=%s,updated_at=%s WHERE run_id=%s", (status, await self.now(conn), run_id))
            return status

    async def first_event_at(self, run_id: str, event: str) -> datetime | None:
        """When ``event`` first happened for this run (e.g. the first pool grant: queue wait ends)."""
        async with self.pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT min(generated_at) AS at FROM durable_resource_events WHERE run_id=%s AND event=%s",
                (run_id, event))).fetchone()
            return row["at"] if row else None

    async def outreach_holds_pending(self, max_age_seconds: float) -> list[dict[str, Any]]:
        """Door-A holds a previous process left for Hub: completed runs whose ``run.outreach_pending``
        names a pool hold with no ``resource.lease_released`` for it yet (restart adoption)."""
        async with self.pool.connection() as conn:
            now = await self.now(conn)
            rows = await (await conn.execute(
                "SELECT e.run_id, e.generated_at, e.payload->'detail' AS detail FROM durable_resource_events e "
                "JOIN durable_admission_runs r USING(run_id) WHERE e.event='run.outreach_pending' "
                "AND r.terminal='completed' AND e.generated_at > %s AND e.payload->'detail' ? 'holder' "
                "AND NOT EXISTS (SELECT 1 FROM durable_resource_events x WHERE x.run_id=e.run_id "
                "AND x.event='resource.lease_released' "
                "AND x.payload->'detail'->>'lease_id' = e.payload->'detail'->>'lease_id')",
                (now - timedelta(seconds=max_age_seconds),))).fetchall()
            return list(rows)

    async def _expire(self, conn: Any, now: datetime) -> list[dict[str, Any]]:
        # Legacy (frozen) durable leases only; capacity.py's last use of this table. Stage 5 deletes it.
        rows = await (await conn.execute("UPDATE durable_resource_leases SET status='expired' WHERE status='active' AND expires_at<=%s RETURNING *", (now,))).fetchall()
        for row in rows:
            await conn.execute("UPDATE durable_resource_demands SET status='suspended' WHERE demand_id=%s AND status='granted'", (row["demand_id"],))
            await self._event(conn, row["run_id"], "resource.lease_expired", {"lease_id": row["lease_id"], "generation": row["generation"], "lane": row["lane"]}, event_id=f"expired:{row['lease_id']}", now=now)
        return rows

    async def record_event(self, run_id: str, event: str, detail: dict[str, Any], event_id: str | None = None) -> dict[str, Any]:
        async with self.transaction() as conn:
            return await self._event(conn, run_id, event, detail, event_id=event_id)

    # Nodes that only wait for or re-request capacity. Their completion is not
    # progress: a grant -> fail -> worker_recovery -> re-request cycle must
    # keep counting toward the resume-failure bound.
    WAIT_NODES = ("resource_request", "resource_wait", "retry_wait")

    async def resume_failures_since_progress(self, run_id: str) -> tuple[int, datetime | None, datetime]:
        """(count, first failure time, now) of resume failures since the run's
        last real node progress. Pre-2026-09-25 failure rows (no checkpoint_id)
        are not counted."""
        async with self.pool.connection() as conn:
            now = await self.now(conn)
            row = await (await conn.execute(
                "WITH progress AS (SELECT max(generated_at) AS at FROM durable_resource_events "
                "WHERE run_id=%s AND payload->'detail' ? 'node' "
                "AND event NOT IN ('run.resumed','run.checkpoint_resume_failed') "
                "AND NOT (payload->'detail'->>'node' = ANY(%s))) "
                "SELECT count(*) AS n, min(generated_at) AS first_at FROM durable_resource_events, progress "
                "WHERE run_id=%s AND event='run.checkpoint_resume_failed' AND payload->'detail' ? 'checkpoint_id' "
                "AND (progress.at IS NULL OR generated_at > progress.at)",
                (run_id, list(self.WAIT_NODES), run_id),
            )).fetchone()
            return int(row["n"]), row["first_at"], now

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

