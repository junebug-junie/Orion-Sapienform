"""The "Orion is asking" resolve bridge end to end, against a disposable Postgres.

The REAL Hub route (scripts/ask_routes.py, asyncpg, real SQL) closes a memory confirmation card and
writes the attention_loop_outcome row in one transaction; then the memory side
(orion.memory.episode.confirmation, what orion-memory-consolidation runs) applies it, once from the
published event and once with the bus dead, through the table catch-up alone.

Set ORION_MEMORY_EPISODE_TEST_DATABASE_URL (admin DSN of a throwaway server). Synthetic rows only.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HUB_ROOT.parents[1]
for p in (str(REPO_ROOT), str(HUB_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from orion.memory.episode import confirmation as c  # noqa: E402
from orion.schemas.attention_salience import AttentionLoopOutcomeV1  # noqa: E402
from scripts import ask_routes  # noqa: E402

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
SQL_DIR = REPO_ROOT / "services" / "orion-sql-db"
MIGRATIONS = (
    "manual_migration_episode_memory_v1.sql",
    "manual_migration_walkway_camera_v1.sql",
    "manual_migration_attention_loop_outcome.sql",
    "manual_migration_memory_confirmation_v1.sql",
)
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")


def _run(body, monkeypatch, *, bus_alive: bool):
    published: list[AttentionLoopOutcomeV1] = []

    async def _pub_outcome(o):
        if bus_alive:
            published.append(o)
        return bus_alive

    async def _pub_answered(_e):
        return bus_alive

    monkeypatch.setattr(ask_routes, "_publish_loop_outcome", _pub_outcome)
    monkeypatch.setattr(ask_routes, "_publish_answered", _pub_answered)
    monkeypatch.setattr(ask_routes, "_confirmation_loop_enabled", lambda: True)

    async def _go():
        import asyncpg
        import httpx
        from fastapi import FastAPI

        name = f"askres_{uuid.uuid4().hex[:10]}"
        admin = await asyncpg.connect(ADMIN_DSN)
        await admin.execute(f'CREATE DATABASE "{name}"')
        await admin.close()
        pool = await asyncpg.create_pool(dsn=ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}", min_size=1, max_size=3)
        try:
            for f in MIGRATIONS:
                await pool.execute((SQL_DIR / f).read_text())
            app = FastAPI()
            app.include_router(ask_routes.router)
            app.state.memory_pg_pool = pool
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://hub") as client:
                await body(pool, client, published)
        finally:
            await pool.close()
            admin = await asyncpg.connect(ADMIN_DSN)
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            await admin.close()

    asyncio.run(_go())


async def _seed(pool) -> tuple[str, str]:
    mid = str(uuid.uuid4())
    await pool.execute(
        """INSERT INTO episode_memory (memory_id, episode_id, purpose, voice, channel, statement, stakes, stakes_reason,
               confirmation_state, strength, half_life_days, last_reinforced_at)
           VALUES ($1, 'ep', 'about_juniper', 'juniper_said', 'chat', 'Juniper told me her sister feels distant.',
                   'high', 'family_relationships', 'pending_confirmation', 0.9, 180, now())""",
        uuid.UUID(mid))
    await c.run_tick(pool, now=datetime.now(timezone.utc))
    return mid, c.ask_id_for(c.loop_id_for(mid))


@pytest.mark.parametrize("resolution,note,state", [
    ("confirmed", "", "confirmed"),
    ("revised", "Juniper told me her brother feels distant.", "corrected"),
    ("rejected", "", "rejected"),
])
def test_resolve_then_bus_consumer_closes_the_memory(monkeypatch, resolution, note, state):
    async def body(pool, client, published):
        mid, ask_id = await _seed(pool)
        listed = (await client.get("/api/asks?status=open")).json()["asks"]
        assert [a["ask_id"] for a in listed] == [ask_id]
        assert listed[0]["memory_statement"] == "Juniper told me her sister feels distant."

        r = await client.post(f"/api/asks/{ask_id}/resolve", json={"resolution": resolution, "note": note})
        assert r.status_code == 200, r.text
        card = await pool.fetchrow("SELECT * FROM orion_ask WHERE ask_id = $1", ask_id)
        outcome = await pool.fetchrow("SELECT * FROM attention_loop_outcome")
        assert card["status"] == c.RESOLUTION_ASK_STATUS[resolution]
        assert outcome["outcome_id"] == c.outcome_id_for(ask_id) == r.json()["outcome_id"]
        assert outcome["loop_id"] == c.loop_id_for(mid) and outcome["verdict"] == c.RESOLUTION_VERDICT[resolution]
        assert outcome["note"] == note

        # The bus consumer's path: apply straight from the published event.
        assert len(published) == 1
        ev = published[0]
        async with pool.acquire() as conn:
            got = await c.apply_outcome(conn, c.OutcomeToApply(
                outcome_id=ev.outcome_id, loop_id=ev.loop_id, verdict=ev.verdict, note=ev.note,
                features=ev.features_at_close, actor=ev.actor))
        assert got == resolution
        m = await pool.fetchrow("SELECT confirmation_state FROM episode_memory WHERE memory_id = $1", uuid.UUID(mid))
        assert m["confirmation_state"] == state
        # The whole closure joins on outcome_id (spec section 5 "Trace").
        assert await pool.fetchval(
            """SELECT count(*) FROM attention_loop_outcome o
               JOIN orion_ask a ON a.ask_id = o.features_at_close->>'ask_id'
               JOIN episode_memory_event e ON e.outcome_id = o.outcome_id""") >= 1
        # The catch-up has nothing left to do.
        async with pool.acquire() as conn:
            assert await c.pending_outcomes(conn) == []
        assert (await client.post(f"/api/asks/{ask_id}/resolve", json={"resolution": "confirmed"})).status_code == 409

    _run(body, monkeypatch, bus_alive=True)


def test_dead_bus_still_reaches_the_same_end_state_through_the_table(monkeypatch):
    async def body(pool, client, published):
        mid, ask_id = await _seed(pool)
        r = await client.post(f"/api/asks/{ask_id}/resolve", json={"resolution": "confirmed"})
        assert r.status_code == 200 and r.json()["published_outcome"] is False and published == []
        summary = await c.run_tick(pool, now=datetime.now(timezone.utc))
        assert summary["applied"] == 1
        m = await pool.fetchrow("SELECT confirmation_state, voice FROM episode_memory WHERE memory_id = $1",
                                uuid.UUID(mid))
        assert (m["confirmation_state"], m["voice"]) == ("confirmed", "worked_out_together")

    _run(body, monkeypatch, bus_alive=False)


def test_legacy_answer_and_dismiss_cannot_orphan_a_memory_card(monkeypatch):
    async def body(pool, client, published):
        _mid, ask_id = await _seed(pool)
        assert (await client.post(f"/api/asks/{ask_id}/answer", json={"answer": "yes"})).status_code == 409
        assert (await client.post(f"/api/asks/{ask_id}/dismiss")).status_code == 409
        assert await pool.fetchval("SELECT status FROM orion_ask WHERE ask_id = $1", ask_id) == "open"
        assert await pool.fetchval("SELECT count(*) FROM attention_loop_outcome") == 0

    _run(body, monkeypatch, bus_alive=True)
