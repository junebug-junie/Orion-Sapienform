"""Self-inquiry outcome rows against a REAL Postgres: the value model's history never reads them.

Runs only when ``ORION_TEST_PG_URI`` points at a throwaway database; the test applies
``services/orion-sql-db/manual_migration_curiosity_spend_v1.sql`` (and the replica-plan migration, for
``arm_propensities``) itself into a fresh schema, e.g.:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55511:5432 postgres:16-alpine
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55511/postgres pytest tests/test_curiosity_self_inquiry_outcomes_pg.py
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")
SQL_DIR = Path(__file__).resolve().parents[2] / "orion-sql-db"


def _tested(pid: str, before: float, after: float) -> dict:
    return {"kind": "tested", "prior_id": pid, "before": before, "after": after, "nats": 0.1}


@needs_pg
def test_the_history_read_skips_self_inquiry_outcomes_and_keeps_undecided_ones() -> None:
    import asyncpg

    from scripts.curiosity_offer_decisions import load_prior_test_history

    schema = f"t_{uuid4().hex[:10]}"

    async def run() -> list:
        admin = await asyncpg.connect(PG)
        await admin.execute(f"CREATE SCHEMA {schema}")
        await admin.close()
        pool = await asyncpg.create_pool(PG, min_size=1, max_size=2,
                                         server_settings={"search_path": schema})
        try:
            async with pool.acquire() as conn:
                for f in ("manual_migration_curiosity_spend_v1.sql", "manual_migration_curiosity_replica_plan_v1.sql"):
                    await conn.execute((SQL_DIR / f).read_text())
                await conn.execute(
                    "INSERT INTO curiosity_offer_decisions (run_id, arm, value_arm_propensity) VALUES "
                    "('inv1', 'value_order', 0.33), ('si1', 'self_inquiry', 1.0)")
                for rid, pid in (("inv1", "p_inv"), ("si1", "p_self"), ("old1", "p_old")):
                    await conn.execute(
                        "INSERT INTO curiosity_run_outcomes (run_id, turn_ok, realized_nats, per_prior) "
                        "VALUES ($1, true, 0.1, $2::jsonb)", rid, json.dumps([_tested(pid, 0.5, 0.7)]))
            return await load_prior_test_history(pool, days=30)
        finally:
            await pool.close()
            admin = await asyncpg.connect(PG)
            await admin.execute(f"DROP SCHEMA {schema} CASCADE")
            await admin.close()

    history = asyncio.run(run())
    assert sorted(r.prior_id for r in history) == ["p_inv", "p_old"]
