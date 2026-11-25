"""The self-question door against a real Postgres: the migration (twice), the mint upsert's door rules from every
state, the judge write's guards, demotion, the offer queries with the real focus slot, and the introspect filter.
Skipped without ORION_TEST_PG_URI (a throwaway database: the run creates and drops its own schema). CI runs it in
.github/workflows/self-question-door-tests.yml."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

DSN = os.getenv("ORION_TEST_PG_URI")
pytestmark = pytest.mark.skipif(not DSN, reason="ORION_TEST_PG_URI (throwaway database) required")

REPO = Path(__file__).resolve().parents[3]
MIGRATIONS = [REPO / p for p in (
    "scripts/sql/2026-09-18_curiosity_self_questions.sql",
    "services/orion-sql-db/manual_migration_curiosity_self_question_door_v1.sql",
    "services/orion-sql-db/manual_migration_curiosity_self_question_door_v1.sql",  # twice: changes nothing
    "services/orion-sql-db/manual_migration_curiosity_focus_request_v1.sql",
    "services/orion-sql-db/manual_migration_curiosity_focus_request_refs_v1.sql",
    "services/orion-sql-db/manual_migration_curiosity_focus_request_refused_v1.sql")]
NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_the_door_round_trips_in_real_sql() -> None:
    asyncpg = pytest.importorskip("asyncpg")
    from orion.curiosity import focus
    from orion.curiosity.self_question_pool import SELECT_ALL_SQL, UPSERT_MINT_SQL, UPSERT_SEED_SQL
    from scripts import self_question_door as door
    from scripts.curiosity_finding_register import JudgeResult
    from scripts.curiosity_introspect_listener import SELF_QUESTIONS_SQL

    schema = "self_question_door_test_" + uuid4().hex[:10]
    admin = await asyncpg.connect(DSN)
    await admin.execute(f'CREATE SCHEMA "{schema}"')
    try:
        pool = await asyncpg.create_pool(DSN, min_size=1, max_size=3, server_settings={"search_path": schema})
        try:
            async with pool.acquire() as conn:
                for m in MIGRATIONS:
                    await conn.execute(m.read_text())

                async def row(qid):
                    return dict(await conn.fetchrow("SELECT * FROM curiosity_self_questions WHERE question_id = $1", qid))

                # Juniper's seed row: no door. A mint on her id never rewrites it.
                await conn.execute(UPSERT_SEED_SQL, "lived.who_matters", "Who matters?", "lived", True, "juniper")
                await conn.execute(UPSERT_MINT_SQL, "lived.who_matters", "Orion's words", "lived", False)
                j = await row("lived.who_matters")
                assert (j["text"], j["minted_by"], j["door"]) == ("Who matters?", "juniper", None)

                # A new mint is held. A pre-door row (door NULL) re-minted with the same text is held too.
                await conn.execute(UPSERT_MINT_SQL, "lived.orion.a", "Who do I miss?", "lived", False)
                assert (await row("lived.orion.a"))["door"] == "held"
                await conn.execute("INSERT INTO curiosity_self_questions (question_id, text, family, minted_by) "
                                   "VALUES ('lived.orion.old', 'old', 'lived', 'orion')")
                await conn.execute(UPSERT_MINT_SQL, "lived.orion.old", "old", "lived", False)
                assert (await row("lived.orion.old"))["door"] == "held"

                # From each judged state: the same text keeps the door; new text is held with attempts reset;
                # demoted stays demoted whatever the text.
                for state in ("admitted", "routed", "unjudged", "demoted"):
                    qid = f"lived.orion.{state}"
                    await conn.execute(UPSERT_MINT_SQL, qid, "same", "lived", False)
                    await conn.execute("UPDATE curiosity_self_questions SET door = $2, door_attempts = 2 "
                                       "WHERE question_id = $1", qid, state)
                    await conn.execute(UPSERT_MINT_SQL, qid, "same", "lived", False)
                    r = await row(qid)
                    assert (r["door"], r["door_attempts"]) == (state, 2)
                    await conn.execute(UPSERT_MINT_SQL, qid, "new words", "lived", False)
                    r = await row(qid)
                    want = ("demoted", 2) if state == "demoted" else ("held", 0)
                    assert (r["door"], r["door_attempts"], r["text"]) == (*want, "new words")

                # The judge write: guarded on door = 'held' and on the text that was judged.
                args = ("admitted", "about who Orion misses", NOW, "qwen3.8-flash-next-iq3_s", "sha", 0)
                assert await conn.execute(door.SET_DOOR_SQL, "lived.orion.a", *args, "other words") == "UPDATE 0"
                assert await conn.execute(door.SET_DOOR_SQL, "lived.who_matters", *args, "Who matters?") == "UPDATE 0"
                assert await conn.execute(door.SET_DOOR_SQL, "lived.orion.a", *args, "Who do I miss?") == "UPDATE 1"
                a = await row("lived.orion.a")
                assert a["door"] == "admitted" and a["door_judge_model"] == "qwen3.8-flash-next-iq3_s"

                # Demotion: only an admitted Orion row.
                assert await conn.fetchval(door.DEMOTE_SQL, "lived.who_matters", "r1", "x") is None
                assert await conn.fetchval(door.DEMOTE_SQL, "lived.orion.a", "r1", "finding operational") == "lived.orion.a"
                a = await row("lived.orion.a")
                assert (a["door"], a["door_demoted_by_run"]) == ("demoted", "r1")

                # Held rows, least recently tried first.
                held = [r["question_id"] for r in await conn.fetch(door.HELD_SQL, 20)]
                assert "lived.orion.old" in held and "lived.who_matters" not in held

                # Pick and introspect readers: the door column is read; non-admitted Orion rows are not offered.
                assert {r["question_id"] for r in await conn.fetch(SELECT_ALL_SQL)} >= {"lived.orion.a", "lived.who_matters"}
                await conn.execute("UPDATE curiosity_self_questions SET door = 'admitted' WHERE question_id = 'lived.orion.admitted'")
                await conn.execute("UPDATE curiosity_self_questions SET door = 'routed' WHERE question_id = 'lived.orion.routed'")
                listed = {r["question_id"] for r in await conn.fetch(SELF_QUESTIONS_SQL, None, 50)}
                assert "lived.who_matters" in listed and "lived.orion.admitted" in listed
                assert not {"lived.orion.a", "lived.orion.old", "lived.orion.routed", "lived.orion.unjudged"} & listed

            # The loop against the real focus slot: one offer, spaced; Prime replaces it; re-offered as #2.
            async def judge(question, answer):
                # Held rows are admitted here, so the only question to route is the demoted lived.orion.a.
                return JudgeResult("philosophical", "about who Orion misses", None, "qwen3.8-flash-next-iq3_s")

            async with pool.acquire() as conn:
                # Only the demoted lived.orion.a is left to route.
                await conn.execute("UPDATE curiosity_self_questions SET door = 'unjudged' "
                                   "WHERE door IN ('routed', 'demoted') AND question_id <> 'lived.orion.a'")
            loop = door.SelfQuestionDoor(enabled=True, pool_provider=lambda: pool, judge=judge, clock=lambda: NOW,
                                         offer_spacing_sec=3600.0)
            first = await loop.tick()
            assert first["offered"] == "lived.orion.a"
            pending = await focus.read_pending(pool)
            assert pending.chat_correlation_id == f"{focus.DOOR_CORRELATION_PREFIX}lived.orion.a"
            await focus.queue_focus(pool, question="What do I want?", why="w", origin="self",
                                    chat_correlation_id="prime:t1", keep_pending=True)
            assert (await focus.read_pending(pool)).chat_correlation_id == "prime:t1"
            assert (await loop.tick())["offered"] is None          # the slot is Prime's now
            async with pool.acquire() as conn:
                await conn.execute("UPDATE curiosity_focus_request SET status = 'cleared' WHERE status = 'pending'")
            assert (await loop.tick())["offered"] is None          # spaced: an offer was made within the hour
            loop.offer_spacing_sec = 0.0
            assert (await loop.tick())["offered"] == "lived.orion.a"
            assert (await focus.read_pending(pool)).chat_correlation_id == f"{focus.DOOR_CORRELATION_PREFIX}lived.orion.a#2"
            # A run takes it: never offered again.
            assert await focus.take_focus(pool, run_id="run7") is not None
            assert (await loop.tick())["offered"] is None
        finally:
            await pool.close()
    finally:
        await admin.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await admin.close()
