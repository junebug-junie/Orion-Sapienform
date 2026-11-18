"""Stage 1 migrations: apply, GIN CONCURRENTLY outside a transaction, roll back, re-apply (review 2026-10-02)."""
from __future__ import annotations

import asyncio
import re
import os
import uuid
from pathlib import Path

import pytest

SQL = Path(__file__).resolve().parents[2] / "orion-sql-db"
ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")


def test_the_transactional_migration_builds_no_blocking_gin_index():
    assert "USING GIN" not in (SQL / "manual_migration_memory_episode_v1.sql").read_text()
    assert "CREATE INDEX CONCURRENTLY" in (SQL / "manual_migration_memory_episode_v1_gin.sql").read_text()


@pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")
def test_apply_rollback_reapply():
    import psycopg

    async def run():
        name = f"mig_{uuid.uuid4().hex[:10]}"
        async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
            await admin.execute(f'CREATE DATABASE "{name}"')
        dsn = ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}"
        try:
            async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
                async def run_file(f):   # one statement at a time, like plain `psql -f` (autocommit)
                    code = "\n".join(re.sub(r"--.*$", "", l) for l in (SQL / f).read_text().splitlines())
                    for stmt in [x for x in code.split(";") if x.strip()]:
                        await conn.execute(stmt)

                async def tables():
                    rows = await (await conn.execute(
                        "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1")).fetchall()
                    return [r[0] for r in rows]

                await run_file("manual_migration_memory_consolidation_v1.sql")
                for _ in range(2):
                    await run_file("manual_migration_memory_episode_v1.sql")
                    await run_file("manual_migration_memory_episode_v1_gin.sql")
                    await run_file("manual_migration_episode_memory_v1.sql")
                    assert {"memory_episode_shadow", "episode_memory", "episode_memory_evidence"} <= set(await tables())
                    idx = await (await conn.execute("SELECT 1 FROM pg_indexes WHERE indexname='idx_mcw_turns_gin'")).fetchone()
                    assert idx is not None
                    await run_file("manual_migration_episode_memory_v1_rollback.sql")
                    await run_file("manual_migration_memory_episode_v1_rollback.sql")
                    assert set(await tables()) == {"memory_consolidation_windows", "memory_graph_suggest_drafts"}
                    cols = await (await conn.execute(
                        "SELECT column_name FROM information_schema.columns WHERE table_name='memory_consolidation_windows'"
                    )).fetchall()
                    assert "close_reason" not in {c[0] for c in cols}
        finally:
            async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
                await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')

    asyncio.run(run())
