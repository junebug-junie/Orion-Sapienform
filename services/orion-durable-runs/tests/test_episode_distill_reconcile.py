"""Reconciler for memory.episode_distill (review 2026-10-02): lost close events and failed runs are
resubmitted as NEW durable attempts, bounded and idempotent. No side queue."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.episode_distill_reconcile import next_attempt_run_id, reconcile_once

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
SETTINGS = SimpleNamespace(memory_episode_distill_route="memory_distill", memory_episode_distill_timeout_sec=600.0,
                           memory_episode_distill_max_tokens=4096, memory_episode_distill_deadline_hours=20.0,
                           memory_episode_distill_max_attempts=3)


def _a(run_id, terminal, age_min):
    return {"run_id": run_id, "terminal": terminal, "updated_at": NOW - timedelta(minutes=age_min)}


@pytest.mark.parametrize("attempts, expected", [
    ([], "memdistill-e"),                                                            # lost close event
    ([_a("memdistill-e", None, 5)], None),                                           # in flight
    ([_a("memdistill-e", "completed", 90)], None),                                   # done
    ([_a("memdistill-e", "failed", 30)], None),                                      # failed, too recent
    ([_a("memdistill-e", "failed", 90)], "memdistill-e-a2"),                         # new attempt id
    ([_a("memdistill-e", "failed", 300), _a("memdistill-e-a2", "abandoned", 90)], "memdistill-e-a3"),
    ([_a("memdistill-e", "failed", 300), _a("memdistill-e-a2", "failed", 200),
      _a("memdistill-e-a3", "failed", 90)], None),                                   # max attempts
])
def test_next_attempt(attempts, expected):
    assert next_attempt_run_id("e", attempts, now=NOW, max_attempts=3) == expected


ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
SQL = Path(__file__).resolve().parents[2] / "orion-sql-db"


@pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")
def test_reconcile_pass_against_postgres_is_bounded_and_idempotent():
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    from psycopg_pool import AsyncConnectionPool

    async def run():
        name = f"recon_{uuid.uuid4().hex[:10]}"
        async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
            await admin.execute(f'CREATE DATABASE "{name}"')
        pool = AsyncConnectionPool(conninfo=ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}", min_size=1, max_size=2,
                                   open=False, kwargs={"autocommit": True, "row_factory": dict_row})
        await pool.open()
        try:
            async with pool.connection() as conn:
                for f in ("manual_migration_memory_consolidation_v1.sql", "manual_migration_memory_episode_v1.sql",
                          "manual_migration_episode_memory_v1.sql"):
                    await conn.execute((SQL / f).read_text())
                await conn.execute("CREATE TABLE durable_admission_runs (run_id text PRIMARY KEY, request jsonb, "
                                   "terminal text, updated_at timestamptz)")

                async def episode(eid, closed_min_ago, status="closed"):
                    turns = [{"correlation_id": f"{eid}-t1", "at": (NOW - timedelta(hours=5)).isoformat()}]
                    await conn.execute(
                        "INSERT INTO memory_episode_shadow (episode_id, status, episode_status, turns, started_at, "
                        "last_turn_at, closed_at, close_reason, juniper_turn_count) VALUES "
                        "(%s, 'closed', %s, %s, %s, %s, %s, 'v2:phase_long_gap', 1)",
                        (eid, status, Jsonb(turns), NOW - timedelta(hours=5), NOW - timedelta(hours=5),
                         NOW - timedelta(minutes=closed_min_ago)))

                await episode("lost", 60)                 # no run at all -> base id
                await episode("failed", 180)              # failed 2 h ago -> a2
                await episode("busy", 120)                # in flight -> nothing
                await episode("done", 300)                # has episode_distill_run -> nothing
                await episode("fresh", 5)                 # inside the grace period -> nothing
                await episode("skipped", 60, status="skipped")
                await conn.execute("INSERT INTO durable_admission_runs VALUES ('memdistill-failed', '{}', 'failed', %s)",
                                   (NOW - timedelta(hours=2),))
                await conn.execute("INSERT INTO durable_admission_runs VALUES ('memdistill-busy', '{}', NULL, %s)",
                                   (NOW - timedelta(minutes=10),))
                await conn.execute("INSERT INTO episode_distill_run (episode_id, run_id, memories_kept, "
                                   "memories_rejected, questions_kept, downgrades) VALUES ('done', 'memdistill-done', 1, 0, 0, 0)")

            submitted = []

            async def submit(request):
                async with pool.connection() as c:
                    inserted = await (await c.execute(
                        "INSERT INTO durable_admission_runs VALUES (%s, %s, NULL, %s) ON CONFLICT DO NOTHING RETURNING run_id",
                        (request.run_id, Jsonb(request.model_dump(mode="json")), NOW))).fetchone()
                if inserted is None:
                    raise ValueError("run_id already exists")
                assert request.admission.priority == "system" and request.brief.turn_ids
                submitted.append(request.run_id)

            first = await reconcile_once(pool, submit, SETTINGS, now=NOW)
            assert sorted(first) == ["memdistill-failed-a2", "memdistill-lost"]
            assert await reconcile_once(pool, submit, SETTINGS, now=NOW) == []      # idempotent: now in flight
            assert sorted(submitted) == ["memdistill-failed-a2", "memdistill-lost"]
        finally:
            await pool.close()
            async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
                await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')

    asyncio.run(run())
