"""The memory confirmation loop against a disposable Postgres (asyncpg, the real SQL).

Set ORION_MEMORY_EPISODE_TEST_DATABASE_URL (admin DSN of a throwaway server). Each test creates and
drops its own database and applies the real migrations: episode_memory, orion_ask (walkway camera),
attention_loop_outcome, and the confirmation indexes. Synthetic rows only.

Covers: card creation from a high-stakes memory, the 5-card cap and the queue behind it, the 7-day
expiry (silence is never a yes), each answer's effect on the memory, the catch-up read (an outcome
whose bus event was lost is still applied), idempotency and the bus/catch-up race.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from orion.memory.episode import confirmation as c

ADMIN_DSN = os.environ.get("ORION_MEMORY_EPISODE_TEST_DATABASE_URL")
SQL_DIR = Path(__file__).resolve().parents[4] / "services" / "orion-sql-db"
MIGRATIONS = (
    "manual_migration_episode_memory_v1.sql",
    "manual_migration_walkway_camera_v1.sql",
    "manual_migration_attention_loop_outcome.sql",
    "manual_migration_memory_confirmation_v1.sql",
)
pytestmark = pytest.mark.skipif(not ADMIN_DSN, reason="ORION_MEMORY_EPISODE_TEST_DATABASE_URL not set")

T0 = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)


def run(coro_fn):
    async def _go():
        import asyncpg

        name = f"memconf_{uuid.uuid4().hex[:10]}"
        admin = await asyncpg.connect(ADMIN_DSN)
        await admin.execute(f'CREATE DATABASE "{name}"')
        await admin.close()
        pool = await asyncpg.create_pool(dsn=ADMIN_DSN.rsplit("/", 1)[0] + f"/{name}", min_size=1, max_size=4)
        try:
            for f in MIGRATIONS:
                await pool.execute((SQL_DIR / f).read_text())
            await coro_fn(pool)
        finally:
            await pool.close()
            admin = await asyncpg.connect(ADMIN_DSN)
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            await admin.close()

    asyncio.run(_go())


async def _memory(pool, *, stakes="high", reason="family_relationships", state=None, voice="juniper_said",
                  channel="chat", created=T0, statement=None, half_life=180.0) -> str:
    mid = str(uuid.uuid4())
    state = state or ("pending_confirmation" if stakes == "high" else "auto")
    await pool.execute(
        """INSERT INTO episode_memory (memory_id, episode_id, purpose, voice, channel, statement, stakes, stakes_reason,
               confirmation_state, strength, half_life_days, last_reinforced_at, created_at, updated_at)
           VALUES ($1, 'ep-1', 'about_juniper', $2, $3, $4, $5, $6, $7, 0.9, $8, $9, $9, $9)""",
        uuid.UUID(mid), voice, channel, statement or f"Juniper told me about thing {mid[:6]}.", stakes, reason,
        state, half_life, created)
    await pool.execute(
        "INSERT INTO episode_memory_evidence VALUES ($1, 'chat_prompt', 'corr-1', 'a quote here', 'sha', true)",
        uuid.UUID(mid))
    await pool.execute("INSERT INTO episode_memory_referent VALUES ($1, 'person:sister', 'about')", uuid.UUID(mid))
    return mid


async def _mem(pool, mid):
    return await pool.fetchrow("SELECT * FROM episode_memory WHERE memory_id = $1", uuid.UUID(mid))


async def _answer(pool, mid, resolution, note="", *, now=T0):
    """What the Hub's resolve route writes: the card update + the outcome row, one transaction."""
    loop = c.loop_id_for(mid)
    ask_id = c.ask_id_for(loop)
    oid = c.outcome_id_for(ask_id)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("UPDATE orion_ask SET status=$2, answer=$3, answered_at=$4 WHERE ask_id=$1",
                               ask_id, c.RESOLUTION_ASK_STATUS[resolution], resolution, now)
            await conn.execute(
                """INSERT INTO attention_loop_outcome (outcome_id, loop_id, theme_key, verdict, actor, note,
                       features_at_close, created_at) VALUES ($1, $2, $2, $3, 'juniper', $4, $5::jsonb, $6)""",
                oid, loop, c.RESOLUTION_VERDICT[resolution], note,
                json.dumps(c.outcome_features(resolution=resolution, ask_id=ask_id, memory_id=mid)), now)
    return oid


async def _events(pool, mid):
    return [r["op"] for r in await pool.fetch(
        "SELECT op FROM episode_memory_event WHERE memory_id = $1 ORDER BY created_at, op", uuid.UUID(mid))]


def test_high_stakes_memory_gets_one_card_and_low_gets_none():
    async def body(pool):
        high = await _memory(pool, reason="health")
        low = await _memory(pool, stakes="low", reason="none")
        summary = await c.run_tick(pool, now=T0)
        assert summary == {"expired": 0, "applied": 0, "opened": 1}
        ask = await pool.fetchrow("SELECT * FROM orion_ask")
        assert ask["source_kind"] == "memory_confirmation"
        assert ask["source_ref"] == c.loop_id_for(high) and ask["ask_id"] == c.ask_id_for(c.loop_id_for(high))
        assert ask["status"] == "open" and ask["asked_of"] == "juniper"
        assert ask["expires_at"] == T0 + timedelta(days=7)
        assert c.WHY_BY_REASON["health"] in ask["question"] and "You told me something" in ask["question"]
        assert json.loads(ask["evidence_refs"]) == [f"memory:{high}", "chat_prompt:corr-1"]
        m = await _mem(pool, high)
        assert m["confirmation_state"] == "pending_confirmation" and m["confirmation_loop_id"] == c.loop_id_for(high)
        assert await _events(pool, high) == ["confirm_asked"]
        assert (await _mem(pool, low))["confirmation_loop_id"] is None
        # Idempotent: a second tick opens nothing new.
        assert (await c.run_tick(pool, now=T0))["opened"] == 0
        assert await pool.fetchval("SELECT count(*) FROM orion_ask") == 1
    run(body)


def test_cap_of_five_open_cards_and_the_queue_behind_it():
    async def body(pool):
        mids = [await _memory(pool, created=T0 + timedelta(minutes=i)) for i in range(8)]
        assert (await c.run_tick(pool, now=T0))["opened"] == 5
        asked = {r["source_ref"] for r in await pool.fetch("SELECT source_ref FROM orion_ask WHERE status='open'")}
        assert asked == {c.loop_id_for(m) for m in mids[:5]}  # oldest first
        assert (await c.run_tick(pool, now=T0))["opened"] == 0
        # An open vision ask does not take a memory slot; an open_question card does.
        await pool.execute("INSERT INTO orion_ask (ask_id, question, source_kind, source_ref) "
                           "VALUES ('v', 'q', 'vision_individual', 'ind')")
        await _answer(pool, mids[0], "confirmed")
        summary = await c.run_tick(pool, now=T0)
        assert summary == {"expired": 0, "applied": 1, "opened": 1}
        assert await pool.fetchval("SELECT count(*) FROM orion_ask WHERE status='open' "
                                   "AND source_kind='memory_confirmation'") == 5
        assert (await _mem(pool, mids[5]))["confirmation_loop_id"] == c.loop_id_for(mids[5])
        assert (await _mem(pool, mids[6]))["confirmation_loop_id"] is None
    run(body)


def test_concurrent_openers_never_exceed_the_cap():
    async def body(pool):
        for i in range(12):
            await _memory(pool, created=T0 + timedelta(minutes=i))

        async def opener():
            async with pool.acquire() as conn:
                return await c.open_cards(conn, now=T0)

        results = await asyncio.gather(opener(), opener(), opener())
        assert sum(len(r) for r in results) == 5
        assert await pool.fetchval("SELECT count(*) FROM orion_ask WHERE status='open'") == 5
    run(body)


def test_seven_day_expiry_marks_unconfirmed_never_confirmed_and_frees_the_slot():
    async def body(pool):
        mids = [await _memory(pool, created=T0 + timedelta(minutes=i)) for i in range(6)]
        await c.run_tick(pool, now=T0)
        assert (await c.run_tick(pool, now=T0 + timedelta(days=6, hours=23)))["expired"] == 0
        summary = await c.run_tick(pool, now=T0 + timedelta(days=7))
        assert summary["expired"] == 5 and summary["opened"] == 1
        for m in mids[:5]:
            row = await _mem(pool, m)
            assert row["confirmation_state"] == "unconfirmed"
            assert row["voice"] == "juniper_said" and row["strength"] == pytest.approx(0.9)  # not a yes
            assert await _events(pool, m) == ["confirm_asked", "ask_expired"]
        assert await pool.fetchval("SELECT count(*) FROM orion_ask WHERE status='expired'") == 5
        assert await pool.fetchval("SELECT count(*) FROM attention_loop_outcome") == 0  # expiry is not an answer
        # Never re-asked: an expired memory keeps its loop id, so it does not re-enter the queue.
        assert (await c.run_tick(pool, now=T0 + timedelta(days=8)))["opened"] == 0
    run(body)


def test_expiry_done_by_another_sweeper_still_moves_the_memory():
    """orion-sql-writer expires every open ask past expires_at; the memory must follow."""
    async def body(pool):
        mid = await _memory(pool)
        await c.run_tick(pool, now=T0)
        await pool.execute("UPDATE orion_ask SET status='expired'")
        assert (await c.run_tick(pool, now=T0 + timedelta(days=1)))["expired"] == 1
        assert (await _mem(pool, mid))["confirmation_state"] == "unconfirmed"
    run(body)


def test_confirm_relabels_voice_reinforces_and_records_the_outcome():
    async def body(pool):
        mid = await _memory(pool)
        await c.run_tick(pool, now=T0)
        oid = await _answer(pool, mid, "confirmed", "yes, that's right")
        assert (await c.run_tick(pool, now=T0 + timedelta(minutes=1)))["applied"] == 1
        m = await _mem(pool, mid)
        assert (m["confirmation_state"], m["voice"], m["status"]) == ("confirmed", "worked_out_together", "active")
        assert m["strength"] == pytest.approx(1.0) and m["half_life_days"] == pytest.approx(360.0)
        assert m["reinforcement_count"] == 1
        ev = await pool.fetchrow("SELECT * FROM episode_memory_event WHERE memory_id=$1 AND op='confirmed'",
                                 uuid.UUID(mid))
        assert ev["outcome_id"] == oid and ev["actor"] == "juniper"
        assert json.loads(ev["evidence"])["prior_voice"] == "juniper_said"
        quotes = {r["source_kind"]: r["quote"] for r in await pool.fetch(
            "SELECT source_kind, quote FROM episode_memory_evidence WHERE memory_id=$1", uuid.UUID(mid))}
        assert quotes["juniper_confirmation"] == "yes, that's right"
    run(body)


def test_revise_supersedes_the_original_and_confirms_her_wording():
    async def body(pool):
        mid = await _memory(pool, statement="Juniper told me her sister feels distant.")
        await c.run_tick(pool, now=T0)
        oid = await _answer(pool, mid, "revised", "  Juniper told me her brother feels distant, not her sister. ")
        await c.run_tick(pool, now=T0 + timedelta(minutes=1))
        old = await _mem(pool, mid)
        assert (old["confirmation_state"], old["status"]) == ("corrected", "superseded")
        assert old["statement"] == "Juniper told me her sister feels distant."  # the original is kept
        new = await pool.fetchrow("SELECT * FROM episode_memory WHERE supersedes_memory_id=$1", uuid.UUID(mid))
        assert new["statement"] == "Juniper told me her brother feels distant, not her sister."
        assert (new["confirmation_state"], new["voice"], new["status"]) == ("confirmed", "worked_out_together", "active")
        assert new["prompt_version"] == "juniper_revision" and new["stakes_reason"] == "family_relationships"
        assert await pool.fetchval("SELECT referent_key FROM episode_memory_referent WHERE memory_id=$1",
                                   new["memory_id"]) == "person:sister"
        kinds = {r["source_kind"] for r in await pool.fetch(
            "SELECT source_kind FROM episode_memory_evidence WHERE memory_id=$1", new["memory_id"])}
        assert kinds == {"chat_prompt", "juniper_revision"}
        assert await _events(pool, mid) == ["confirm_asked", "revised"]
        ops = {r["op"]: r["outcome_id"] for r in await pool.fetch(
            "SELECT op, outcome_id FROM episode_memory_event WHERE memory_id=$1", new["memory_id"])}
        assert ops == {"created": oid}
        # The revised memory is confirmed, so it is never asked about again.
        assert (await c.run_tick(pool, now=T0 + timedelta(minutes=2)))["opened"] == 0
    run(body)


def test_reject_excludes_the_memory_from_recall():
    async def body(pool):
        keep = await _memory(pool)
        drop = await _memory(pool, created=T0 + timedelta(minutes=1))
        await c.run_tick(pool, now=T0)
        await _answer(pool, drop, "rejected")
        await c.run_tick(pool, now=T0 + timedelta(minutes=1))
        m = await _mem(pool, drop)
        assert (m["confirmation_state"], m["status"]) == ("rejected", "rejected")
        recallable = {str(r["memory_id"]) for r in await pool.fetch(
            f"SELECT memory_id FROM episode_memory WHERE {c.RECALLABLE_WHERE}")}
        assert keep in recallable and drop not in recallable
        assert await _events(pool, drop) == ["confirm_asked", "rejected"]
    run(body)


def test_catch_up_applies_an_outcome_whose_bus_event_was_lost():
    """Kill the bus: nothing calls the bus handler. The table read alone reaches the same state."""
    async def body(pool):
        mid = await _memory(pool)
        await c.run_tick(pool, now=T0)
        await _answer(pool, mid, "confirmed")
        async with pool.acquire() as conn:
            pending = await c.pending_outcomes(conn)
        assert [p.loop_id for p in pending] == [c.loop_id_for(mid)]
        await c.run_tick(pool, now=T0 + timedelta(minutes=1))
        assert (await _mem(pool, mid))["confirmation_state"] == "confirmed"
        async with pool.acquire() as conn:
            assert await c.pending_outcomes(conn) == []
    run(body)


def test_bus_and_catch_up_racing_on_one_outcome_apply_it_once():
    async def body(pool):
        mid = await _memory(pool)
        await c.run_tick(pool, now=T0)
        await _answer(pool, mid, "confirmed")
        async with pool.acquire() as conn:
            outcome = (await c.pending_outcomes(conn))[0]

        async def apply():
            async with pool.acquire() as conn:
                return await c.apply_outcome(conn, outcome, now=T0)

        results = sorted(await asyncio.gather(apply(), apply(), apply()))
        assert results == ["already_applied", "already_applied", "confirmed"]
        m = await _mem(pool, mid)
        assert m["reinforcement_count"] == 1
        assert await _events(pool, mid) == ["confirm_asked", "confirmed"]
    run(body)


def test_first_answer_wins_and_bad_outcomes_are_logged_once():
    async def body(pool):
        mid = await _memory(pool)
        await c.run_tick(pool, now=T0)
        await _answer(pool, mid, "confirmed")
        await c.run_tick(pool, now=T0)
        late = c.OutcomeToApply(outcome_id="late", loop_id=c.loop_id_for(mid), verdict="dismissed", note="",
                                features={"resolution": "rejected"})
        orphan = c.OutcomeToApply(outcome_id="orphan", loop_id=c.loop_id_for(str(uuid.uuid4())), verdict="resolved",
                                  note="", features={"resolution": "confirmed"})
        async with pool.acquire() as conn:
            assert await c.apply_outcome(conn, late) == "already_resolved"
            assert await c.apply_outcome(conn, orphan) == "orphaned"
            assert await c.apply_outcome(conn, orphan) == "already_applied"
            other = c.OutcomeToApply(outcome_id="x", loop_id="open-loop-1", verdict="resolved", note="", features={})
            assert await c.apply_outcome(conn, other) == "not_memory"
        assert (await _mem(pool, mid))["confirmation_state"] == "confirmed"
    run(body)


def test_revise_without_a_note_is_invalid_and_leaves_the_memory_pending():
    async def body(pool):
        mid = await _memory(pool)
        await c.run_tick(pool, now=T0)
        await _answer(pool, mid, "revised", "")
        await c.run_tick(pool, now=T0)
        assert (await _mem(pool, mid))["confirmation_state"] == "pending_confirmation"
        assert await _events(pool, mid) == ["confirm_asked", "outcome_invalid"]
    run(body)


def test_reverie_memory_card_is_never_framed_as_something_juniper_said():
    async def body(pool):
        await _memory(pool, voice="orion_thought", channel="reverie", reason="orion_relationship")
        await c.run_tick(pool, now=T0)
        q = await pool.fetchval("SELECT question FROM orion_ask")
        frame = q.split("“")[0]
        assert "You told me something" not in frame and "not from anything you told me" in frame
        assert c.WHY_BY_REASON["orion_relationship"] in q
    run(body)
