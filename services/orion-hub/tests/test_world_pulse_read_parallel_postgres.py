"""Reading runs up to max_in_flight turns per stage at once (Juniper 2026-10-14: "it should
load balance across 3 GPUs!"). Before, a tick claimed ONE seed, and CLAIM_SQL puts a seed
with a submitted durable turn first, so the in-flight read was re-claimed every tick and no
second turn was ever submitted: the pool never saw two agent holds to spread.

Real SQL (claims, bindings, exclusion); the durable turn is a stand-in that binds and
reports "still running" (ReadingPending) until the test lets a seed finish.

RUN_READING_POSTGRES=1 python -m pytest services/orion-hub/tests/test_world_pulse_read_parallel_postgres.py -q
"""
import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.reading import SourceFetchEvidenceV1
from orion.schemas.reading_turn import ReadingRunBriefV1
from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1
from orion.world_pulse_read import queue
from orion.world_pulse_read.durable import ReadingPending, bind_turn
from scripts import world_pulse_read_pipeline as s1
from scripts import world_pulse_read_stage2 as s2
from test_reading_postgres import db, local_pg, request  # noqa: F401 -- fixture reuse

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reading_dns")]


def _handoff(seed):
    return WorldPulseReadHandoffV1(
        seed_ref=seed, what_i_learned="A source-backed finding.", trace_id=str(uuid4()),
        created_at=datetime.now(timezone.utc),
        read_evidence=[SourceFetchEvidenceV1(tool_name="WebFetch", url=seed.url, content_chars=500)])


def _pipe(cls, conn, monkeypatch, module, max_in_flight, stage):
    monkeypatch.setattr(module, f"wallet_{'a' if stage == 1 else 'b'}_block_reason", lambda _: None)
    pipe = cls(enabled=True, tick_interval_sec=60, timeout_sec=900, session_id="reading",
               pool_provider=lambda: None, source_ref=ServiceRef(name="orion-hub"), max_in_flight=max_in_flight)

    async def with_conn(callback):
        return await callback(conn)
    pipe._with_conn = with_conn
    pipe._bus = SimpleNamespace(redis=None, publish=AsyncMock())
    return pipe


async def _bound(conn, stage):
    return await conn.fetchval("SELECT count(DISTINCT seed_id) FROM reading_durable_turn "
                               "WHERE stage=$1 AND consumed_at IS NULL", stage)


@pytest.mark.parametrize("max_in_flight,expect", [(1, 1), (3, 3)])
def test_stage1_keeps_max_in_flight_turns_submitted_and_polls_them_first(local_pg, monkeypatch, max_in_flight, expect):
    async def run():
        conn, _ = await db(local_pg)
        for i in range(4):
            await queue.enqueue_reading(conn, request(f"https://example.org/read-{i}"))
        pipe = _pipe(s1.WorldPulseReadPipeline, conn, monkeypatch, s1, max_in_flight, 1)
        pipe._maybe_enqueue_recent = AsyncMock()
        done, polled = set(), []

        async def read(seed):
            polled.append(seed.seed_id)
            await bind_turn(conn, ReadingRunBriefV1(seed_id=seed.seed_id, stage=1, prompt="p",
                                                    session_id="reading", timeout_sec=900), str(uuid4()))
            if seed.seed_id in done:
                return _handoff(seed)
            raise ReadingPending("durable:running")
        pipe._stage1_read = read

        assert await pipe.tick() == "waiting_resource"
        assert await _bound(conn, 1) == expect
        first = list(polled)
        # Next tick: the same in-flight seeds are polled, no new turn starts.
        polled.clear()
        await pipe.tick()
        assert polled == first and await _bound(conn, 1) == expect
        # Every claim was released back to pending (no claim held between ticks).
        assert await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed WHERE status='claimed'") == 0
        # One finishes: it is recorded done; the slot it frees is filled on the next tick.
        done.add(first[0])
        polled.clear()
        await pipe.tick()
        assert await conn.fetchval("SELECT status FROM world_pulse_read_seed WHERE seed_id=$1", first[0]) == "done"
        polled.clear()
        await pipe.tick()
        started = await conn.fetchval(
            "SELECT count(DISTINCT seed_id) FROM reading_durable_turn d WHERE stage=1 AND EXISTS ("
            "SELECT 1 FROM world_pulse_read_seed s WHERE s.seed_id=d.seed_id AND s.status='pending')")
        assert started == min(expect, 3)  # in flight never exceeds max_in_flight
        assert len(polled) == len(set(polled)) <= max_in_flight  # each seed at most once per tick
        await conn.close()
    asyncio.run(run())


def test_stage1_priority_order_is_kept_for_new_starts(local_pg, monkeypatch):
    async def run():
        conn, _ = await db(local_pg)
        low = await queue.enqueue_reading(conn, request("https://example.org/low"))
        await conn.execute("UPDATE world_pulse_read_seed SET priority=10 WHERE seed_id=$1", low["seed_id"])
        highs = [await queue.enqueue_reading(conn, request(f"https://example.org/high-{i}")) for i in range(2)]
        pipe = _pipe(s1.WorldPulseReadPipeline, conn, monkeypatch, s1, 2, 1)
        pipe._maybe_enqueue_recent = AsyncMock()
        polled = []

        async def read(seed):
            polled.append(seed.seed_id)
            await bind_turn(conn, ReadingRunBriefV1(seed_id=seed.seed_id, stage=1, prompt="p",
                                                    session_id="reading", timeout_sec=900), str(uuid4()))
            raise ReadingPending("durable:running")
        pipe._stage1_read = read
        await pipe.tick()
        assert polled == [h["seed_id"] for h in highs]  # the priority-10 row waits
        await conn.close()
    asyncio.run(run())


def test_stage2_runs_max_in_flight_turns_at_once(local_pg, monkeypatch):
    async def run():
        conn, _ = await db(local_pg)
        for i in range(4):
            await queue.enqueue_reading(conn, request(f"https://example.org/s2-{i}"))
            seed = await queue.claim_next_seed(conn)
            await queue.mark_seed_done(conn, seed.seed_id, trace_id=str(uuid4()), handoff=_handoff(seed))
        pipe = _pipe(s2.WorldPulseReadStage2Pipeline, conn, monkeypatch, s2, 3, 2)
        polled = []

        async def stage2(handoff):
            sid = handoff.seed_ref.seed_id
            polled.append(sid)
            await bind_turn(conn, ReadingRunBriefV1(seed_id=sid, stage=2, prompt="p",
                                                    session_id="reading", timeout_sec=900), str(uuid4()))
            raise ReadingPending("durable:running")
        pipe._stage2_pass = stage2
        assert await pipe.tick() == "waiting_resource"
        assert await _bound(conn, 2) == 3 and len(set(polled)) == 3
        polled.clear()
        await pipe.tick()
        assert await _bound(conn, 2) == 3 and len(set(polled)) == 3  # same three, no fourth
        await conn.close()
    asyncio.run(run())


def test_settings_ship_three_in_flight():
    from pathlib import Path

    from app.settings import Settings

    assert Settings.model_fields["HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT"].default == 3
    env = (Path(__file__).resolve().parents[1] / ".env_example").read_text()
    assert "HUB_WORLD_PULSE_READ_MAX_IN_FLIGHT=3" in env
