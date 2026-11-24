"""The day dream's SQL against a real Postgres (app/dream_day_store.py): every ref kind resolves (and a missing id
does not), persist writes memories + the dream_day row once, a replay of the same run writes nothing twice, and a
second run for the same day gets ``already_written`` and writes nothing.

Set ORION_ADMISSION_TEST_DSN to a disposable database (CI provides one). Each test makes its own schema.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

DSN = os.getenv("ORION_ADMISSION_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

MIGRATIONS = ("manual_migration_episode_memory_v1.sql", "manual_migration_dream_day.sql",
              "manual_migration_dream_day_grounding.sql")
# Only the columns the resolve queries read (live types, 2026-11-18).
REF_TABLES = """
CREATE TABLE temporal_self_event (event_id text PRIMARY KEY, source_kind text, source_ref text, label text,
    verdict text, occurred_at timestamptz NOT NULL DEFAULT '2026-11-17 19:00+00', payload_json jsonb NOT NULL DEFAULT '{}'::jsonb);
CREATE TABLE substrate_durable_run_state (entry_id text PRIMARY KEY, created_at timestamptz, run_id varchar,
    workflow varchar, node varchar, status varchar, detail json);
CREATE TABLE vision_events (event_id varchar PRIMARY KEY, stream_id text, narrative text);
CREATE TABLE grammar_acts (act_id varchar PRIMARY KEY, actor varchar, verb varchar, object_ref text, outcome_note text);
CREATE TABLE dream_cycle (cycle_id text PRIMARY KEY, trigger text, status text, hypothesis_count int, cycle_json jsonb,
    started_at timestamptz NOT NULL DEFAULT '2026-11-18 01:00+00', no_link_count int);
CREATE TABLE dream_hypothesis (hypothesis_id text PRIMARY KEY, claim text, why text);
CREATE TABLE harness_turn_trace (correlation_id varchar PRIMARY KEY, workflow text, created_at timestamp,
    run_artifact jsonb);
CREATE TABLE chat_history_log (id varchar PRIMARY KEY, correlation_id varchar, prompt text, response text);
INSERT INTO temporal_self_event (event_id, source_kind, source_ref, label, verdict, payload_json) VALUES
    ('reverie_chain:r1', 'reverie_chain', 'r1', 'an open loop', NULL, '{"duration_sec": 3588.8}'),
    ('curiosity_run:run1', 'curiosity_run', 'run1', NULL, NULL, '{"arm": "self_inquiry", "n_moved": 0}'),
    ('curiosity_run:old1', 'curiosity_run', 'old1', NULL, NULL, '{"arm": "value_order", "n_moved": 2}'),
    ('vision_percept:v1', 'vision_percept', 'v1', NULL, NULL, '{"entities": ["chair", "desk"]}'),
    ('chat_turn:chat-1', 'chat_turn', 'chat-1', NULL, NULL, '{"speaker": "juniper"}');
-- The run's finish rows: the older one first, then the one the resolver must take (newest), then another run's.
INSERT INTO substrate_durable_run_state VALUES
    ('e1', '2026-11-17 18:00+00', 'run1', 'curiosity.investigate', 'finish', 'completed',
     '{"finding_text": "an older finding"}'),
    ('e2', '2026-11-17 19:00+00', 'run1', 'curiosity.investigate', 'finish', 'completed',
     json_build_object('finding_text', 'my answers name 1.257 real records on average ' || repeat('x', 5000))),
    ('e3', '2026-11-17 19:00+00', 'run1', 'reading.turn', 'finish', 'completed', '{"finding_text": "not this run"}');
INSERT INTO vision_events VALUES ('v1', 'cam0', 'A scene with a desk and a chair is observed.');
INSERT INTO grammar_acts VALUES ('act_1', 'hub.world_pulse_read', 'look', 'reading:x', 'stage1_completed');
INSERT INTO dream_cycle VALUES ('dc-1', 'pressure', 'completed', 0, '{"note": "pairs dream=3"}', '2026-11-18 01:00+00', 4);
INSERT INTO dream_hypothesis VALUES ('h-1', 'two things link', 'same hour');
INSERT INTO harness_turn_trace VALUES ('c-1', 'reminiscence_otherwise', '2026-11-17 10:00',
    '{"final_text": "saved rm:2026-11-17:otherwise"}'),
    ('c-rm', 'reminiscence_what_it_was', '2026-11-21 13:00',
     '{"final_text": "reminisce what_it_was saved rm:2026-11-21:what_it_was:ffc6f3f855"}');
INSERT INTO chat_history_log VALUES ('chat-1', 'corr-1', 'hello', 'hi');
"""


def with_db(fn):
    from psycopg import AsyncConnection
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    async def go():
        schema = "dream_day_test_" + uuid4().hex[:10]
        async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
        try:
            async with AsyncConnectionPool(DSN, min_size=1, max_size=2, open=False, kwargs={
                    "autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                    "options": f"-c search_path={schema}"}) as pool:
                async with pool.connection() as conn:
                    for name in MIGRATIONS:
                        await conn.execute((REPO_ROOT / "services" / "orion-sql-db" / name).read_text(), prepare=False)
                    await conn.execute(REF_TABLES, prepare=False)
                await fn(pool)
        finally:
            async with await AsyncConnection.connect(DSN, autocommit=True) as conn:
                await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')

    asyncio.run(go())


def test_every_ref_kind_resolves_and_persist_writes_once():
    from app.dream_day_store import persist_dream_day, resolve_refs
    from orion.memory.dream_day import build_memories, check_answer, parse_answer

    refs = ["experience:reverie_chain:r1", "act:act_1", "dream_cycle:dc-1", "dream_hypothesis:h-1",
            "harness_turn_trace:c-1", "chat_turn:chat-1", "chat_turn:corr-1", "act:act_missing", "chat_turn:x';--"]
    day = date(2026, 11, 17)

    async def body(pool):
        resolved = await resolve_refs(pool, refs)
        assert set(resolved) == set(refs[:7])
        # The short text (the stored evidence quote) is unchanged; the grounding judge reads the record's detail too
        # (orion/memory/dream_day.py RESOLVE_SQL).
        assert resolved["experience:reverie_chain:r1"] == "reverie_chain: an open loop"
        assert resolved["chat_turn:chat-1"] == "Juniper: hello / Orion: hi"
        full = await resolve_refs(pool, refs, full=True)
        assert set(full) == set(resolved)
        assert full["experience:reverie_chain:r1"].startswith("reverie_chain: an open loop ")
        assert "3588.8" in full["experience:reverie_chain:r1"]
        assert full["harness_turn_trace:c-1"].endswith("reply: saved rm:2026-11-17:otherwise")
        assert full["act:act_1"] == resolved["act:act_1"]
        assert full["chat_turn:chat-1"].endswith("full turn: Juniper: hello / Orion: hi")
        text = ('{"memories": [{"statement": "I read a paper and my reverie kept an open loop.", '
                '"refs": ["act:act_1", "experience:reverie_chain:r1"]}], "plan": [], '
                '"day": {"text": "A quiet day.", "refs": ["dream_cycle:dc-1"]}}')
        check = check_answer(parse_answer(text), resolved, [])
        result = build_memories(check, resolved, day)
        now = datetime(2026, 11, 18, 8, 30, tzinfo=timezone.utc)

        def fields(run_id):
            return {"day": day, "run_id": run_id, "items_dropped": 0, "refs_dropped": 0, "plan_rejected": 0,
                    "memories_over_cap": 0, "kept": 0, "dropped": 0, "new": 0, "plan": [], "day_text": check.day_text,
                    "day_refs": check.day_refs, "material": {}, "read": {"unread": {}}, "now_shown_id": "s",
                    "turn_correlation_id": "t", "created_at": now, "memories_unsupported": 1,
                    "memories_unjudged": 0,
                    "grounding": {"enabled": True, "unsupported": 1, "verdicts": [
                        {"statement": "the camera saw the kids", "status": "unsupported",
                         "unsupported_claims": ["the kids"]}]}}

        first = await persist_dream_day(pool, fields("dream-day-2026-11-17"), result, "memory_distill")
        assert first["outcome"] == "written" and first["memories_written"] == 1 and first["produced"]
        replay = await persist_dream_day(pool, fields("dream-day-2026-11-17"), result, "memory_distill")
        assert replay["outcome"] == "written" and replay["memory_ids"] == first["memory_ids"]
        other = await persist_dream_day(pool, fields("dream-day-2026-11-17-a2"), result, "memory_distill")
        assert other["outcome"] == "already_written" and other["run_id"] == "dream-day-2026-11-17"
        async with pool.connection() as conn:
            mem = await (await conn.execute("SELECT channel, voice, count(*) AS n FROM episode_memory "
                                            "GROUP BY 1, 2")).fetchall()
            ev = await (await conn.execute("SELECT source_kind, source_id FROM episode_memory_evidence "
                                           "ORDER BY source_id")).fetchall()
            row = await (await conn.execute("SELECT run_id, produced, memories_written, day_text, memories_unsupported, "
                                            "grounding->'verdicts'->0->'unsupported_claims' AS claims "
                                            "FROM dream_day")).fetchall()
        assert [dict(r) for r in mem] == [{"channel": "dream", "voice": "orion_thought", "n": 1}]
        assert [r["source_id"] for r in ev] == ["act:act_1", "experience:reverie_chain:r1"]
        assert [dict(r) for r in row] == [{"run_id": "dream-day-2026-11-17", "produced": True, "memories_written": 1,
                                           "day_text": "A quiet day.", "memories_unsupported": 1,
                                           "claims": ["the kids"]}]

    with_db(body)


def test_the_judge_reads_each_cited_kinds_real_content_capped(tmp_path):
    """PR dream-day-citable-facts: the kinds dream memories actually cite carry their real words for the judge
    (a curiosity run's finding, a camera's caption, a chat turn's words, the time), each detail capped; the short
    text a memory stores is unchanged; a reminiscence receipt reads as its saved note (#3739); a stored night's
    ``dream_day_read:<day>`` resolves to the count lines its row kept."""
    import json

    from app.dream_day_store import resolve_refs
    from orion.memory.dream_day import EXPERIENCE_DETAIL_CHARS

    from app import dream_day_receipts   # the ledger's file name, from the one module allowed to name it

    (tmp_path / dream_day_receipts.rm.LEDGER_FILE).write_text(json.dumps({
        "schema_version": "reminiscence_hop.v1", "hop_id": "rm:2026-11-21:what_it_was:ffc6f3f855",
        "episode": "2026-11-21", "hop": "what_it_was", "period_start": "2025-11-11", "period_end": "2025-11-17",
        "note": "[pr_text] That week had 27 commits and no merged PR.", "claims": [],
        "written_at": "2026-11-21T13:00:57+00:00"}) + "\n")
    refs = ["experience:curiosity_run:run1", "experience:curiosity_run:old1", "experience:vision_percept:v1",
            "experience:chat_turn:chat-1", "dream_cycle:dc-1", "harness_turn_trace:c-rm", "harness_turn_trace:c-1",
            "dream_day_read:2026-11-16", "dream_day_read:2026-11-15"]

    async def body(pool):
        async with pool.connection() as conn:
            await conn.execute(
                "INSERT INTO dream_day (day, run_id, status, produced, note, memories_written, memory_ids, items_dropped,"
                " refs_dropped, memories_rejected, plan_rejected, kept, dropped, new, plan, day_refs, material, read,"
                " created_at) VALUES ('2026-11-16', 'dream-day-2026-11-16', 'written', false, 'x', 0, '{}', 0, 0, 0, 0,"
                " 0, 0, 0, '[]', '{}', '{}', %s, now()), ('2026-11-15', 'dream-day-2026-11-15', 'written', false, 'x',"
                " 0, '{}', 0, 0, 0, 0, 0, 0, 0, '[]', '{}', '{}', '{\"unread\": {}}', now())",
                (json.dumps({"unread": {}, "totals": "body: 226 experiences (gpu_wait 75)",
                             "totals_read_at": "2026-11-17T08:40:00Z"}),))
        short = await resolve_refs(pool, refs)
        full = await resolve_refs(pool, refs, full=True, self_dir=str(tmp_path))
        # A stored night with no totals (an older row) does not resolve: no totals ref without a stored copy.
        assert "dream_day_read:2026-11-15" not in full and "dream_day_read:2026-11-15" not in short
        assert set(full) == set(short) == set(refs) - {"dream_day_read:2026-11-15"}
        # The stored evidence quote (short text) never carries the detail.
        assert short["experience:curiosity_run:run1"] == "curiosity_run: "
        assert short["harness_turn_trace:c-rm"].startswith("harness turn reminiscence_what_it_was at ")
        run = full["experience:curiosity_run:run1"]
        assert "at 2026-11-17 12:00 Denver" in run and '"arm": "self_inquiry"' in run
        assert "what the run found: my answers name 1.257 real records" in run and "older finding" not in run
        assert "not this run" not in run
        # Capped in SQL: the 5,000-character finding never floods the judge.
        assert len(run) <= len("curiosity_run: ") + 1 + EXPERIENCE_DETAIL_CHARS
        assert "what the run found" not in full["experience:curiosity_run:old1"]   # no finish row: the payload alone
        assert "camera cam0 (the office camera) saw: A scene with a desk and a chair" in full["experience:vision_percept:v1"]
        assert "the turn: Juniper: hello / Orion: hi" in full["experience:chat_turn:chat-1"]
        assert "started 2026-11-17 18:00 Denver; 4 pair(s) declined" in full["dream_cycle:dc-1"]
        rm = full["harness_turn_trace:c-rm"]
        assert "reminisce what_it_was saved rm:2026-11-21:what_it_was:ffc6f3f855" in rm
        assert "the week 2025-11-11..2025-11-17: [pr_text] That week had 27 commits and no merged PR." in rm
        assert "the reminiscence note" not in full["harness_turn_trace:c-1"]   # not a receipt: unchanged
        # Without the ledger dir (the stored-quote path, or no mount), the receipt stays the receipt.
        no_dir = await resolve_refs(pool, ["harness_turn_trace:c-rm"], full=True)
        assert "27 commits" not in no_dir["harness_turn_trace:c-rm"]
        assert full["dream_day_read:2026-11-16"].endswith("body: 226 experiences (gpu_wait 75)")
        assert short["dream_day_read:2026-11-16"] == ("dig dream-day 2026-11-16 (my read of that day, at "
                                                      "2026-11-17T08:40:00Z)")

    with_db(body)
