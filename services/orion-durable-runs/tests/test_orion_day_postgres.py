"""orion_day.letter end to end on real Postgres: real checkpoints, the real run registry, the REAL
GPU pool in process, the real orion_day_letter migration and the real store.

The run holds an agent hold for both LLM calls (each call carries it), the hold is released
before persist, the letter row lands once with the note and carry-forward in separate columns,
the checkpoint carries only the slim brief, and a second run for the same day writes nothing."""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_admission_runtime_postgres import DSN, InProcessPool, runtime, with_database
from test_orion_day_graph import CARRY, NOTE, make_brief
from orion.orion_day.brief import build_orion_day_request
from orion.schemas.orion_day import (
    ORION_DAY_CARRY_FORWARD_VERB, ORION_DAY_NOTE_VERB, OrionDayLetterSourcesV1, orion_day_journal_entry_id,
)

pytestmark = pytest.mark.skipif(not DSN, reason="isolated ORION_ADMISSION_TEST_DSN required")

MIGRATION = Path(__file__).resolve().parents[2] / "orion-sql-db" / "manual_migration_orion_day_letter_v1.sql"


def attach_letter_io(rt, gpu):
    """The runner's two real seams for this workflow, recorded."""
    rt.runner.verb_calls = []
    rt.runner.journals = []

    async def call_verb_text(verb, metadata, llm_route, *, gpu_lease, timeout_sec, user_text):
        [hold] = [h for h in gpu.leases(holder=f"durable-runs:{rt._current}") if h["status"] == "granted"]
        rt.runner.verb_calls.append((verb, llm_route, gpu_lease, hold["lease_id"], set(metadata["orion_day_input"])))
        return NOTE if verb == ORION_DAY_NOTE_VERB else CARRY

    async def publish_journal(entry):
        rt.runner.journals.append(entry)
        return entry.entry_id

    rt.runner._call_verb_text = call_verb_text
    rt.runner._publish_journal = publish_journal


async def migrate(pool):
    async with pool.connection() as conn:
        await conn.execute(MIGRATION.read_text(), prepare=False)


def request_for(attempt: int):
    brief = make_brief()
    return build_orion_day_request(brief, attempt=attempt,
                                   deadline_at=datetime.now(timezone.utc) + timedelta(hours=2))


def test_letter_run_holds_the_agent_card_for_both_calls_and_persists_once():
    async def scenario(pool, saver, store):
        await migrate(pool)
        await migrate(pool)  # idempotent
        rt = runtime(pool, saver, store)
        await rt.gpu.boot()
        attach_letter_io(rt, rt.gpu)
        req = request_for(1)
        rt._current = req.run_id
        await rt.submit(req)
        await rt._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"

        # Both calls under the run's own hold, on the agent route; the note call never sees a note.
        assert [c[0] for c in rt.runner.verb_calls] == [ORION_DAY_NOTE_VERB, ORION_DAY_CARRY_FORWARD_VERB]
        for verb, route, ref, hold_id, keys in rt.runner.verb_calls:
            assert route == "agent" and ref.lease_id == hold_id and ref.role == "agent"
        assert "note_md" not in rt.runner.verb_calls[0][4] and "note_md" in rt.runner.verb_calls[1][4]
        [hold] = rt.gpu.leases(holder=f"durable-runs:{req.run_id}")
        assert hold["status"] == "released"

        async with pool.connection() as conn:
            row = await (await conn.execute("SELECT * FROM orion_day_letter WHERE letter_date=%s",
                                            (date(2026, 9, 29),))).fetchone()
        assert row["run_id"] == req.run_id
        assert row["note_md"] == NOTE.strip() and row["carry_forward_md"] == CARRY
        assert row["journal_entry_id"] == orion_day_journal_entry_id("2026-09-29")
        assert row["material"]["letter_date"] == "2026-09-29"
        OrionDayLetterSourcesV1.model_validate(row["sources"])
        [entry] = rt.runner.journals
        assert entry.body == NOTE.strip() and entry.trigger_kind == "orion_day_letter"

        # The checkpoint keeps the slim brief; the full brief lives once in the request row.
        snap = await rt._graph_for("orion_day.letter").aget_state(rt.config(req.run_id))
        assert "material" not in snap.values["brief"] and "llm_view" not in snap.values["brief"]
        stored = (await store.get_run(req.run_id))["request"]["brief"]
        assert "material" in stored and "llm_view" in stored

        completed = next(e for e in await store.history(req.run_id) if e["event"] == "run.completed")
        assert completed["detail"]["line"] == "orion_day" and completed["detail"]["persisted"] is True
        status = await rt.status(req.run_id)
        assert status["orion_day"]["persisted"] is True and status["orion_day"]["letter_date"] == "2026-09-29"

        # A second attempt for the same day: completes, writes nothing, journals nothing.
        again = request_for(2)
        rt._current = again.run_id
        await rt.submit(again)
        await rt._drive(await store.get_run(again.run_id))
        assert (await store.get_run(again.run_id))["terminal"] == "completed"
        done = next(e for e in await store.history(again.run_id) if e["event"] == "run.completed")
        assert done["detail"]["persist_outcome"] == "already_written"
        assert done["detail"]["existing_run_id"] == req.run_id
        # It republishes the day's entry from the stored row: identical to the first, so a first run
        # whose journal never landed is healed and a healthy one is untouched.
        first, second = rt.runner.journals
        assert second.model_dump() == first.model_dump()
        async with pool.connection() as conn:
            count = await (await conn.execute("SELECT count(*) AS n FROM orion_day_letter")).fetchone()
        assert count["n"] == 1
        await rt.close()
    asyncio.run(with_database(scenario))


class Crash(BaseException):
    """A process death mid-node: escapes every except Exception, leaves the checkpoint behind."""


def test_restart_mid_carry_forward_resumes_under_the_same_hold_without_rewriting_the_note():
    """AdmissionRuntime._recover with WORK_NODES[orion_day.letter]: a driver dies inside
    write_carry_forward while holding the hold; the next driver fences it, waits for the same
    hold, and replays ONLY the carry-forward."""
    async def scenario(pool, saver, store):
        await migrate(pool)
        gpu = await InProcessPool().boot()
        rt = runtime(pool, saver, store, gpu=gpu)
        attach_letter_io(rt, gpu)
        real = rt.runner._call_verb_text

        async def dies_in_carry_forward(verb, *args, **kwargs):
            if verb == ORION_DAY_CARRY_FORWARD_VERB:
                raise Crash()
            return await real(verb, *args, **kwargs)

        rt.runner._call_verb_text = dies_in_carry_forward
        req = request_for(1)
        rt._current = req.run_id
        await rt.submit(req)
        with pytest.raises(Crash):
            await rt._drive(await store.get_run(req.run_id))
        snap = await rt._graph_for("orion_day.letter").aget_state(rt.config(req.run_id))
        assert snap.next == ("write_carry_forward",) and snap.values["lease"]
        [hold] = gpu.leases(holder=f"durable-runs:{req.run_id}")
        assert hold["status"] == "granted"

        # The next process: a fresh runtime on the same database and pool.
        rt2 = runtime(pool, saver, store, gpu=gpu)
        attach_letter_io(rt2, gpu)
        rt2._current = req.run_id
        await rt2._drive(await store.get_run(req.run_id))
        assert (await store.get_run(req.run_id))["terminal"] == "completed"
        assert [c[0] for c in rt2.runner.verb_calls] == [ORION_DAY_CARRY_FORWARD_VERB]  # no second note
        final = await rt2._graph_for("orion_day.letter").aget_state(rt2.config(req.run_id))
        assert final.values["llm_attempts"]["write_note"] == 1
        assert final.values["turn_fence"] == 1  # the dead driver's attempt was fenced
        async with pool.connection() as conn:
            row = await (await conn.execute("SELECT note_md FROM orion_day_letter")).fetchone()
        assert row["note_md"] == NOTE.strip()
        await rt.close()
        await rt2.close()
    asyncio.run(with_database(scenario))


def test_migration_refuses_a_merged_note():
    async def scenario(pool, saver, store):
        await migrate(pool)
        from psycopg.errors import CheckViolation
        from psycopg.types.json import Jsonb

        async with pool.connection() as conn:
            with pytest.raises(CheckViolation):
                await conn.execute(
                    "INSERT INTO orion_day_letter (letter_date, run_id, window_start, window_end, note_md, "
                    "carry_forward_md, material) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (date(2026, 9, 28), "r", datetime(2026, 9, 28, 6, tzinfo=timezone.utc),
                     datetime(2026, 9, 29, 6, tzinfo=timezone.utc), "same", "same", Jsonb({})))
    asyncio.run(with_database(scenario))
