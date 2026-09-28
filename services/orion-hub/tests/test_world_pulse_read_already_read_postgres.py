"""A URL Orion already read is passed on, never read again (disposable cluster).

Live 2026-09-27 the NVIDIA Rubin page finished Stage 1 three times: the ingress
only folded requests onto reads still in flight.

RUN_READING_POSTGRES=1 python -m pytest services/orion-hub/tests/test_world_pulse_read_already_read_postgres.py -q
"""
import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from orion.schemas.reading import DurableReadingReceiptV1, SourceFetchEvidenceV1
from orion.schemas.reading_turn import ReadingRunBriefV1
from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1
from orion.world_pulse_read import operator as op
from orion.world_pulse_read import queue
from orion.world_pulse_read.durable import bind_turn
from test_reading_postgres import db, local_pg, request  # noqa: F401 -- fixture reuse

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reading_dns")]

URL = "https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer"


def _run(coro):
    asyncio.run(coro)


async def _read(conn, url=URL, *, stage2_done=False):
    """Enqueue, claim and finish Stage 1 (optionally Stage 2) for ``url``."""
    made = await queue.enqueue_reading(conn, request(url))
    seed = await queue.claim_next_seed(conn)
    assert seed.seed_id == made["seed_id"]
    handoff = WorldPulseReadHandoffV1(
        seed_ref=seed, what_i_learned="Rubin pairs a new GPU with a new CPU.",
        trace_id=str(uuid4()), created_at=datetime.now(timezone.utc),
        read_evidence=[SourceFetchEvidenceV1(tool_name="WebFetch", url=url, content_chars=900)],
    )
    await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)
    if stage2_done:
        await conn.execute(
            "UPDATE world_pulse_read_seed SET stage2_status='done', stage2_completed_at=now() "
            "WHERE seed_id=$1", seed.seed_id,
        )
    return seed.seed_id


async def _legacy_row(conn, *, status="pending", stage2_status="pending"):
    """A same-URL row that predates the guard: enqueue elsewhere, then move it."""
    made = await queue.enqueue_reading(conn, request(f"https://example.org/{uuid4().hex}"))
    await conn.execute(
        "UPDATE world_pulse_read_seed SET url=$2, status=$3, stage2_status=$4 WHERE seed_id=$1",
        made["seed_id"], URL, status, stage2_status,
    )
    return made["seed_id"]


async def _row(conn, seed_id):
    return await conn.fetchrow("SELECT * FROM world_pulse_read_seed WHERE seed_id=$1", seed_id)


def test_asking_for_an_already_read_url_is_blocked_and_says_so(local_pg):
    async def run():
        conn, _ = await db(local_pg)
        first = await _read(conn)
        again = await queue.enqueue_reading(conn, request(URL))
        # The receipt is the earlier read's, flagged so the caller can tell Juniper.
        assert again["duplicate"] == queue.ALREADY_READ
        assert again["duplicate_of"] == first
        assert again["status"] != "queued"
        assert again["summary"] == "Rubin pairs a new GPU with a new CPU."
        DurableReadingReceiptV1.model_validate(again)
        own = await _row(conn, again["seed_id"])
        assert (own["status"], own["stage2_status"], own["last_error"]) == (
            "skipped", "skipped", queue.ALREADY_READ,
        )
        # Nothing new to claim: the URL is not read a second time.
        assert await queue.claim_next_seed(conn) is None
        await conn.close()
    _run(run())


def test_in_flight_and_fresh_urls_are_not_marked_already_read(local_pg):
    async def run():
        conn, _ = await db(local_pg)
        fresh = await queue.enqueue_reading(conn, request(URL))
        assert fresh["duplicate"] is None
        joined = await queue.enqueue_reading(conn, request(URL))
        assert joined["duplicate"] == "already_queued"
        assert joined["duplicate_of"] == fresh["seed_id"]
        await conn.close()
    _run(run())


def test_sweep_passes_on_waiting_rows_for_an_already_read_url(local_pg):
    async def run():
        conn, _ = await db(local_pg)
        first = await _read(conn)
        waiting = await _legacy_row(conn)
        bound = await _legacy_row(conn)
        await bind_turn(conn, ReadingRunBriefV1(
            seed_id=bound, stage=1, prompt="Saved prompt", session_id="reading", timeout_sec=900,
        ), str(uuid4()))
        other = await queue.enqueue_reading(conn, request("https://example.org/different"))

        joined = await queue.enqueue_reading(conn, request("https://example.org/joined"))
        await conn.execute(
            "UPDATE world_pulse_read_seed SET url=$2, status='skipped', stage2_status='skipped', "
            "duplicate_of=$3 WHERE seed_id=$1", joined["seed_id"], URL, waiting,
        )

        assert await queue.skip_already_read_stage1(conn) == 1
        row = await _row(conn, waiting)
        assert (row["status"], row["stage2_status"], row["last_error"], row["duplicate_of"]) == (
            "skipped", "skipped", queue.ALREADY_READ, first,
        )
        assert row["completed_at"] is not None
        # A request that had joined the passed-on row now follows the real read.
        status = await queue.reading_status(conn, joined["request_id"])
        assert (status["duplicate_of"], status["duplicate"]) == (first, queue.ALREADY_READ)
        assert status["summary"] == "Rubin pairs a new GPU with a new CPU."
        own = await queue.reading_status(conn, (await _row(conn, waiting))["request_id"])
        assert own["duplicate"] == queue.ALREADY_READ
        # A row with an open run is left to it; other URLs are untouched.
        assert (await _row(conn, bound))["status"] == "pending"
        assert (await _row(conn, other["seed_id"]))["status"] == "pending"
        assert await queue.skip_already_read_stage1(conn) == 0
        await conn.close()
    _run(run())


def test_stage2_sweep_needs_a_finished_follow_up_for_the_url(local_pg):
    async def run():
        conn, _ = await db(local_pg)
        first = await _read(conn)
        waiting = await _legacy_row(conn, status="done")
        # The earlier read's follow-up hasn't finished: nothing to pass on yet.
        assert await queue.skip_already_read_stage2(conn) == 0
        await conn.execute("UPDATE world_pulse_read_seed SET stage2_status='done' WHERE seed_id=$1", first)
        assert await queue.skip_already_read_stage2(conn) == 1
        row = await _row(conn, waiting)
        assert (row["status"], row["stage2_status"], row["stage2_error"]) == ("done", "skipped", queue.ALREADY_READ)
        await conn.close()
    _run(run())


def test_retry_refuses_to_read_an_already_read_url_again(local_pg):
    async def run():
        conn, _ = await db(local_pg)
        await _read(conn, stage2_done=True)
        again = await queue.enqueue_reading(conn, request(URL))
        await conn.execute(
            "UPDATE world_pulse_read_seed SET duplicate_of=NULL WHERE seed_id=$1", again["seed_id"],
        )
        with pytest.raises(op.OperatorActionError) as err:
            await op.retry_read(conn, again["seed_id"], stage=1)
        assert err.value.code == queue.ALREADY_READ

        second = await _legacy_row(conn, status="done", stage2_status="skipped")
        await conn.execute(
            "UPDATE world_pulse_read_seed SET handoff_json=(SELECT handoff_json FROM world_pulse_read_seed "
            "WHERE status='done' AND stage2_status='done' LIMIT 1) WHERE seed_id=$1", second,
        )
        with pytest.raises(op.OperatorActionError) as err:
            await op.retry_read(conn, second, stage=2)
        assert err.value.code == queue.ALREADY_READ
        await conn.close()
    _run(run())


def test_stage2_retry_refuses_while_another_follow_up_for_the_url_runs(local_pg):
    async def run():
        conn, _ = await db(local_pg)
        first = await _read(conn)
        second = await _legacy_row(conn, status="done", stage2_status="failed")
        await conn.execute(
            "UPDATE world_pulse_read_seed SET handoff_json=(SELECT handoff_json FROM world_pulse_read_seed "
            "WHERE seed_id=$2) WHERE seed_id=$1", second, first,
        )
        with pytest.raises(op.OperatorActionError) as err:
            await op.retry_read(conn, second, stage=2)
        assert err.value.code == "url_already_active"
        await conn.close()
    _run(run())


def test_done_without_fetch_evidence_does_not_block_a_new_read(local_pg):
    async def run():
        conn, _ = await db(local_pg)
        old = await _legacy_row(conn, status="done", stage2_status="failed")
        fresh = await queue.enqueue_reading(conn, request(URL))
        assert fresh["duplicate"] is None and fresh["duplicate_of"] is None
        assert fresh["status"] == "queued"
        assert await queue.skip_already_read_stage1(conn) == 0
        assert (await _row(conn, old))["status"] == "done"
        await conn.close()
    _run(run())
