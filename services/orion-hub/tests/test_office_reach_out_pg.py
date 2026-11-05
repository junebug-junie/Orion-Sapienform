"""Office reach-out against a REAL Postgres: the claim column, the restart rule, sql-writer's rebuild.

Runs only when ``ORION_TEST_PG_URI`` points at a throwaway database with
manual_migration_vision_office_session_v1.sql, manual_migration_vision_office_session_reach_out_v1.sql and
manual_migration_endogenous_outreach_decisions_v1.sql applied (want_ledger is not needed: the want producer
is off; the decision-log write is stubbed, its read is real), e.g.:

    docker run -d --rm --name pg -e POSTGRES_PASSWORD=t -p 127.0.0.1:55498:5432 postgres:16-alpine
    for f in vision_office_session_v1 vision_office_session_reach_out_v1 endogenous_outreach_decisions_v1; do
      docker exec -i pg psql -U postgres < ../orion-sql-db/manual_migration_$f.sql; done
    ORION_TEST_PG_URI=postgresql://postgres:t@127.0.0.1:55498/postgres pytest tests/test_office_reach_out_pg.py
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

PG = os.getenv("ORION_TEST_PG_URI", "")
needs_pg = pytest.mark.skipif(not PG, reason="ORION_TEST_PG_URI not set (throwaway Postgres only)")

T0 = datetime(2026, 11, 5, 1, 24, 7, tzinfo=timezone.utc)
NOW = T0 + timedelta(minutes=20)
SID = "cam0-pgtest"
TIMELINE = [{"at": (T0 + timedelta(seconds=40 * k)).isoformat(), "text": t, "repeats": 1, "event_id": f"e{k}"}
            for k, t in enumerate(["sitting at a desk, opening a box", "reaching into a cabinet",
                                   "sitting at a desk, unpacking a plastic bag and handling a piece of equipment",
                                   "standing in a closet, reaching into a storage bin"])]


async def _reset(pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM vision_office_session")
        await conn.execute(
            "INSERT INTO vision_office_session (session_id, stream_id, status, started_at, last_seen_at, ended_at, "
            "closed_at, who_text, who, timeline, label) VALUES ($1, 'cam0', 'closed', $2, $3, $3, $3, "
            "'Juniper (probably)', $4::jsonb, $5::jsonb, 'Juniper (probably) in the office: x')",
            SID, T0, T0 + timedelta(minutes=7),
            json.dumps({"people": [{"name": "Juniper", "basis": "prior", "individual_ids": []}], "unknown": 0}),
            json.dumps(TIMELINE))


def _outreach(monkeypatch, delivered, *, busy=False):
    from scripts import endogenous_outreach as eo
    from scripts import endogenous_outreach_decisions as decisions_mod

    monkeypatch.setattr(decisions_mod, "record_decision", lambda result, **kw: None)
    monkeypatch.setattr(eo, "_seconds_since_juniper_spoke", lambda: None)

    async def fake_generate(self, prompt, session_id, correlation_id, retrieval_query=None):
        if busy:
            self.register_connection("c1", asyncio.Queue(), {})
            self.note_busy("c1")
        return "Looked like you were unpacking equipment at 18:25. What are you building?", {}

    async def fake_deliver(self, **kw):
        delivered.append(kw)

    monkeypatch.setattr(eo.EndogenousOutreach, "_generate", fake_generate)
    monkeypatch.setattr(eo.EndogenousOutreach, "_deliver", fake_deliver)
    return eo.EndogenousOutreach(
        enabled=True, tick_interval_sec=300.0, min_cooldown_sec=0.0, daily_cap=0, quiet_start_hour=-1,
        quiet_end_hour=-1, timeout_sec=5.0, agent_lane_timeout_sec=5.0, notify_channel="orion:notify:in_app",
        fallback_session_id="orion_outreach", trigger_evaluator=lambda: None, timezone_name="UTC")


def _loop(pool, outreach):
    from scripts.office_reach_out import OfficeReachOut

    return OfficeReachOut(reach_out_enabled=True, event_wants_enabled=False, pool_provider=lambda: pool,
                          outreach_provider=lambda: outreach, timezone_name="America/Denver", clock=lambda: NOW)


@needs_pg
def test_one_message_per_visit_survives_a_restart(monkeypatch):
    import asyncpg

    async def run():
        pool = await asyncpg.create_pool(PG, min_size=1, max_size=2)
        try:
            await _reset(pool)
            delivered: list = []
            out = _outreach(monkeypatch, delivered)
            assert [r["reason"] for r in (await _loop(pool, out).tick())["results"]] == ["sent"]
            row = await pool.fetchrow("SELECT reached_out_at, reach_out_decision_id FROM vision_office_session "
                                      "WHERE session_id = $1", SID)
            assert row["reached_out_at"] == NOW and row["reach_out_decision_id"]
            # A restarted Hub (a new loop, empty memory) finds the mark and sends nothing.
            again = await _loop(pool, _outreach(monkeypatch, delivered)).tick()
            assert again["pending"] == 0 and len(delivered) == 1
        finally:
            await pool.close()

    asyncio.run(run())


@needs_pg
def test_a_refused_delivery_releases_the_visit_for_a_later_tick(monkeypatch):
    import asyncpg

    async def run():
        pool = await asyncpg.create_pool(PG, min_size=1, max_size=2)
        try:
            await _reset(pool)
            delivered: list = []
            first = await _loop(pool, _outreach(monkeypatch, delivered, busy=True)).tick()
            assert [r["reason"] for r in first["results"]] == ["turn_in_flight"] and delivered == []
            assert await pool.fetchval("SELECT reached_out_at FROM vision_office_session WHERE session_id = $1",
                                       SID) is None
            later = await _loop(pool, _outreach(monkeypatch, delivered)).tick()
            assert [r["reason"] for r in later["results"]] == ["sent"] and len(delivered) == 1
        finally:
            await pool.close()

    asyncio.run(run())


@needs_pg
def test_a_claim_is_taken_once_and_released_only_by_its_owner():
    import asyncpg

    from scripts.office_reach_out import CLAIM_SQL, RELEASE_SQL

    async def run():
        pool = await asyncpg.create_pool(PG, min_size=1, max_size=2)
        try:
            await _reset(pool)
            assert len(await pool.fetch(CLAIM_SQL, SID, NOW, "d1")) == 1
            assert await pool.fetch(CLAIM_SQL, SID, NOW, "d2") == []        # a second claimant gets nothing
            await pool.execute(RELEASE_SQL, SID, "d2")                       # not its claim: no-op
            assert await pool.fetchval("SELECT reach_out_decision_id FROM vision_office_session") == "d1"
            await pool.execute(RELEASE_SQL, SID, "d1")
            assert await pool.fetchval("SELECT reached_out_at FROM vision_office_session") is None
        finally:
            await pool.close()

    asyncio.run(run())


@needs_pg
def test_failed_attempts_in_the_decision_log_cap_a_restarted_hub(monkeypatch):
    import asyncpg

    async def run():
        pool = await asyncpg.create_pool(PG, min_size=1, max_size=2)
        try:
            await _reset(pool)
            await pool.execute("DELETE FROM endogenous_outreach_decisions")
            for k in range(3):
                await pool.execute(
                    "INSERT INTO endogenous_outreach_decisions (decision_id, decided_at, outreach, reason, result_json) "
                    "VALUES ($1, $2, false, 'orion_passed', $3::jsonb)", f"d{k}", NOW - timedelta(minutes=5),
                    json.dumps({"source": "office_reach_out", "office_session_id": SID}))
            delivered: list = []
            summary = await _loop(pool, _outreach(monkeypatch, delivered)).tick()
            assert summary["pending"] == 0 and delivered == []
            # Another source's rows for the same id do not count.
            await pool.execute("UPDATE endogenous_outreach_decisions SET result_json = "
                               "jsonb_set(result_json, '{source}', '\"reading_finding_outreach\"')")
            summary = await _loop(pool, _outreach(monkeypatch, delivered)).tick()
            assert summary["pending"] == 1 and len(delivered) == 1
        finally:
            await pool.execute("DELETE FROM endogenous_outreach_decisions")
            await pool.close()

    asyncio.run(run())
