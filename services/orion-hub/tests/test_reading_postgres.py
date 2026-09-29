"""Opt-in real PostgreSQL tests in a disposable local cluster; no production DSN.

RUN_READING_POSTGRES=1 python -m pytest services/orion-hub/tests/test_reading_postgres.py -q
Models, DNS and bus publication are mocked. SQL, commits, locks and migrations are real.
"""
import asyncio
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

from orion.schemas.reading import ReadingRequestedV1
from orion.schemas.world_pulse_read import WorldPulseReadHandoffV1, WorldPulseReadSeedV1, WorldPulseReadStage2ResultV1
from orion.world_pulse_read import queue
from orion.world_pulse_read.events import REQUESTED_CHANNEL
from orion.world_pulse_read.journal import journal_entry
from scripts.world_pulse_read_stage2 import WorldPulseReadStage2Pipeline
from orion.core.bus.bus_schemas import ServiceRef

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("reading_dns")]
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def local_pg(tmp_path_factory):
    if os.environ.get("RUN_READING_POSTGRES") != "1":
        pytest.skip("opt-in disposable PostgreSQL integration")
    candidates = sorted(Path("/usr/lib/postgresql").glob("*/bin/initdb"), reverse=True)
    initdb = shutil.which("initdb") or (str(candidates[0]) if candidates else None)
    if not initdb:
        pytest.fail("initdb unavailable for requested integration")
    pg = Path(initdb).parent
    work = tmp_path_factory.mktemp("reading-pg")
    data, sock = work / "data", work / "socket"
    sock.mkdir()
    subprocess.run([str(pg / "initdb"), "-D", str(data), "-A", "trust", "-U", "reading_test", "--no-locale"], check=True, capture_output=True)
    subprocess.run([str(pg / "pg_ctl"), "-D", str(data), "-l", str(work / "postgres.log"), "-o", f"-k {sock} -h '' -p 16479", "-w", "start"], check=True, capture_output=True)
    try:
        yield {"host": str(sock), "port": 16479, "user": "reading_test", "database": "postgres"}
    finally:
        subprocess.run([str(pg / "pg_ctl"), "-D", str(data), "-m", "fast", "-w", "stop"], check=True, capture_output=True)


async def db(local_pg):
    conn = await asyncpg.connect(**local_pg)
    schema = "reading_" + uuid4().hex
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}"')
    await queue.ensure_seed_queue_schema(conn)
    await conn.execute("CREATE TABLE journal_entries (entry_id text PRIMARY KEY, source_ref text, body text)")
    return conn, schema


def request(url="https://example.org/article", **kwargs):
    return ReadingRequestedV1(url=url, requested_by="juniper", invocation_context="unified_chat", why_now="A new context", **kwargs)


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("reason", ["outside_window", "refund_backoff", "disabled"])
@pytest.mark.parametrize("active", [False, True, "consumed", "other_stage"])
def test_admission_gates_only_allow_existing_bindings(local_pg, monkeypatch, stage, reason, active):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from orion.schemas.reading import SourceFetchEvidenceV1
    from orion.schemas.reading_turn import ReadingRunBriefV1
    from orion.world_pulse_read.durable import bind_turn, ReadingPending
    from scripts import world_pulse_read_pipeline as s1
    from scripts import world_pulse_read_stage2 as s2

    async def run():
        conn, _ = await db(local_pg)
        await queue.enqueue_reading(conn, request())
        seed = await queue.claim_next_seed(conn)
        handoff = WorldPulseReadHandoffV1(
            seed_ref=seed, what_i_learned="A source-backed finding.",
            trace_id=str(uuid4()), created_at=datetime.now(timezone.utc),
            read_evidence=[SourceFetchEvidenceV1(
                tool_name="WebFetch", url=seed.url, content_chars=500,
            )],
        )
        if stage == 2:
            await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)
        else:
            await conn.execute("UPDATE world_pulse_read_seed SET status='pending', claimed_at=NULL")
        if active:
            binding = await bind_turn(conn, ReadingRunBriefV1(
                seed_id=seed.seed_id, stage=3-stage if active == "other_stage" else stage, prompt="Saved prompt",
                session_id="reading", timeout_sec=900,
            ), str(uuid4()))
            if active == "consumed":
                await conn.execute("UPDATE reading_durable_turn SET consumed_at=now() WHERE run_id=$1", binding.run_id)
        module = s1 if stage == 1 else s2
        cls = s1.WorldPulseReadPipeline if stage == 1 else s2.WorldPulseReadStage2Pipeline
        monkeypatch.setattr(module, f"wallet_{'a' if stage == 1 else 'b'}_block_reason", lambda _: reason)
        pipe = cls(enabled=reason != "disabled", tick_interval_sec=60, timeout_sec=900, session_id="reading",
            pool_provider=lambda: None, source_ref=ServiceRef(name="orion-hub"))
        async def with_conn(callback):
            return await callback(conn)
        pipe._with_conn = with_conn
        pipe._bus = SimpleNamespace(redis=None, publish=AsyncMock())
        if stage == 1:
            pipe._maybe_enqueue_recent = AsyncMock()
        work = AsyncMock(side_effect=ReadingPending("existing run"))
        setattr(pipe, "_stage1_read" if stage == 1 else "_stage2_pass", work)
        expected_poll = active is True and reason != "disabled"
        assert await pipe.tick() == ("waiting_resource" if expected_poll else reason)
        assert work.await_count == int(expected_poll)
        row = await conn.fetchrow("SELECT attempts, stage2_attempts FROM world_pulse_read_seed")
        assert tuple(row) == (0, 0)
        await conn.close()
    asyncio.run(run())


def test_migration_is_additive_and_rerunnable(local_pg):
    async def run():
        conn = await asyncpg.connect(**local_pg)
        schema = "migration_" + uuid4().hex
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        for name in ("world_pulse_read_seed_queue", "world_pulse_read_stage2"):
            await conn.execute((ROOT / f"services/orion-sql-db/manual_migration_{name}_v1.sql").read_text())
        await conn.execute("INSERT INTO world_pulse_read_seed(seed_id,kind,run_id,url) VALUES('legacy','finding','r','https://example.org/old')")
        migration = (ROOT / "services/orion-sql-db/manual_migration_general_reading_v1.sql").read_text()
        await conn.execute(migration)
        await conn.execute(migration)
        retry_migration = (ROOT / "services/orion-sql-db/manual_migration_world_pulse_read_retry_v1.sql").read_text()
        await conn.execute(retry_migration)
        await conn.execute(retry_migration)
        from orion.world_pulse_read.durable import READING_DURABLE_SQL
        await conn.execute(READING_DURABLE_SQL)
        legacy = await queue.claim_next_seed(conn)
        assert legacy.seed_id == "legacy"
        legacy_status = await queue.reading_status(conn, url="https://example.org/old")
        assert legacy_status["status"] == "started"
        assert legacy_status["request_id"] is None
        assert legacy_status["seed_id"] == "legacy"
        assert legacy_status["matched_request_count"] == 1
        from orion.schemas.reading import ReadingStatusReceiptV1

        ReadingStatusReceiptV1.model_validate(legacy_status)
        assert await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed") == 1
        await conn.close()
    asyncio.run(run())


def test_reading_durable_binding_survives_restart_and_retires_atomically(local_pg):
    from orion.world_pulse_read.durable import bind_turn, consume_turn, release_claim
    from orion.schemas.reading_turn import ReadingRunBriefV1

    async def run():
        conn, schema = await db(local_pg)
        req = request()
        await queue.enqueue_reading(conn, req)
        seed = await queue.claim_next_seed(conn)
        brief = ReadingRunBriefV1(seed_id=seed.seed_id, stage=1, prompt="Original prompt",
                                  session_id="reading", timeout_sec=900)
        first = await bind_turn(conn, brief, str(uuid4()))
        await release_claim(conn, seed.seed_id, 1)
        await conn.close()
        conn = await asyncpg.connect(**local_pg)
        await conn.execute(f'SET search_path TO "{schema}"')
        replay = await bind_turn(conn, brief.model_copy(update={"prompt": "Changed prompt"}), str(uuid4()))
        assert replay == first
        assert await conn.fetchval("SELECT attempts FROM world_pulse_read_seed") == 0
        async with conn.transaction():
            await queue.mark_seed_failed(conn, seed.seed_id,
                error="turn_deferred:stance_react_timeout", max_attempts=3)
            await consume_turn(conn, first.run_id)
        retry = await bind_turn(conn, brief, str(uuid4()))
        assert retry.run_id != first.run_id
        assert await conn.fetchval("SELECT attempts FROM world_pulse_read_seed") == 0
        await conn.close()
    asyncio.run(run())


@pytest.mark.parametrize("cancelled", [False, True])
def test_worker_wait_or_cancel_does_not_charge_or_spend_attempt(local_pg, cancelled):
    from unittest.mock import AsyncMock
    from types import SimpleNamespace
    from fakeredis.aioredis import FakeRedis
    from scripts.world_pulse_read_pipeline import WorldPulseReadPipeline
    from orion.world_pulse_read.durable import ReadingPending, ReadingCancelled
    from orion.world_pulse_read.wallet_a import read_wallet_a_state

    async def run():
        conn, _ = await db(local_pg)
        await queue.enqueue_reading(conn, request())
        redis = FakeRedis()
        pipe = WorldPulseReadPipeline(enabled=True, tick_interval_sec=60, timeout_sec=900, session_id="reading", pool_provider=lambda: None,
            source_ref=ServiceRef(name="orion-hub"))
        async def with_conn(callback):
            return await callback(conn)
        pipe._with_conn = with_conn
        pipe._bus = SimpleNamespace(redis=redis, publish=AsyncMock())
        pipe._maybe_enqueue_recent = AsyncMock()
        pipe._generate = AsyncMock(side_effect=ReadingCancelled("run") if cancelled else ReadingPending("queued"))
        assert await pipe.tick(force=True) == ("cancelled" if cancelled else "waiting_resource")
        row = await conn.fetchrow("SELECT status, attempts FROM world_pulse_read_seed")
        assert row["attempts"] == 0
        assert row["status"] == ("skipped" if cancelled else "pending")
        _, count = await read_wallet_a_state(redis, now=datetime.now(timezone.utc), timezone_name="UTC")
        assert count == 0
        await redis.aclose()
        await conn.close()
    asyncio.run(run())


def test_url_status_selects_latest_alias_and_preserves_earlier_failure(local_pg):
    async def run():
        conn, _ = await db(local_pg)
        url = "https://arxiv.org/abs/2310.19279"
        old, current, alias = request(url), request(url), request(url)
        await queue.enqueue_reading(conn, old)
        await conn.execute(
            "UPDATE world_pulse_read_seed SET status='done', stage2_status='failed' WHERE request_id=$1",
            old.request_id,
        )
        await queue.enqueue_reading(conn, current)
        await queue.enqueue_reading(conn, alias)
        before = await conn.fetch("SELECT * FROM world_pulse_read_seed ORDER BY seed_id")
        async with conn.transaction(readonly=True):
            latest = await queue.reading_status(conn, url=url + "#abstract")
            assert latest["request_id"] == str(alias.request_id)
            # `old` was marked done without fetch evidence, so it is not a read.
            assert latest["duplicate_of"] == "reading:" + str(current.request_id)
            assert latest["duplicate"] == "already_queued"
            assert latest["status"] == "queued"
            assert latest["queue_position"] == 1
            assert latest["matched_request_count"] == 3
            assert latest["selection"] == "latest_request"
            assert (await queue.reading_status(conn, old.request_id))["status"] == "failed"
            missing = await queue.reading_status(conn, url=url + "v2")
            assert missing["status"] == "not_found"
            assert missing["request_id"] is None
        assert await conn.fetch("SELECT * FROM world_pulse_read_seed ORDER BY seed_id") == before
        await conn.close()
    asyncio.run(run())


def test_concurrent_submissions_alias_active_work_and_allow_reread_without_evidence(local_pg):
    async def run():
        conn, schema = await db(local_pg)
        other = await asyncpg.connect(**local_pg)
        await other.execute(f'SET search_path TO "{schema}"')
        a, b = request(), request()
        receipts = await asyncio.gather(queue.enqueue_reading(conn, a), queue.enqueue_reading(other, b))
        assert all(r["status"] == "queued" for r in receipts)
        assert await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed WHERE status='pending'") == 1
        assert await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed WHERE duplicate_of IS NOT NULL") == 1
        assert {r["request"]["request_id"] for r in receipts} == {str(a.request_id), str(b.request_id)}
        await queue.enqueue_reading(conn, a)
        assert await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed") == 2
        await conn.execute("UPDATE world_pulse_read_seed SET status='done', stage2_status='done' WHERE duplicate_of IS NULL")
        # Done without fetch evidence (pre-2026-09-25 rows) is not a read.
        assert (await queue.enqueue_reading(conn, request()))["status"] == "queued"
        assert await conn.fetchval("SELECT count(*) FROM world_pulse_read_seed WHERE status='pending'") == 1
        await other.close()
        await conn.close()
    asyncio.run(run())


def test_event_sees_committed_request_from_another_connection(local_pg):
    async def run():
        conn, schema = await db(local_pg)
        other = await asyncpg.connect(**local_pg)
        await other.execute(f'SET search_path TO "{schema}"')
        seen = []
        class Bus:
            async def publish(self, channel, envelope):
                assert channel == REQUESTED_CHANNEL
                seen.append(await other.fetchval("SELECT count(*) FROM world_pulse_read_seed WHERE request_id=$1", uuid_from(envelope.payload["request_id"])))
        assert (await queue.enqueue_reading(conn, request(), bus=Bus()))["status"] == "queued"
        assert seen == [1]
        async with conn.transaction():
            with pytest.raises(RuntimeError, match="own its commit"):
                await queue.enqueue_reading(conn, request("https://example.org/second"))
        await other.close()
        await conn.close()
    asyncio.run(run())


def uuid_from(value):
    from uuid import UUID
    return UUID(value)


def test_reading_status_reports_queue_position_only_while_queued(local_pg):
    """Repro of a live incident: asked how far back a queued read was, Orion
    had no way to say anything but "queued" -- no position, no sense of
    whether that meant seconds or days. queue_position/queue_depth answer
    that; both must be null again once the row leaves the pending queue."""
    async def run():
        conn, schema = await db(local_pg)
        ahead = [request(f"https://example.org/ahead-{i}") for i in range(3)]
        for req in ahead:
            await queue.enqueue_reading(conn, req)
        target = request("https://example.org/target")
        await queue.enqueue_reading(conn, target)
        behind = request("https://example.org/behind")
        await queue.enqueue_reading(conn, behind)

        status = await queue.reading_status(conn, target.request_id)
        assert status["status"] == "queued"
        assert status["queue_position"] == 4
        assert status["queue_depth"] == 5

        seed = await queue.claim_next_seed(conn)
        assert seed.request.request_id == ahead[0].request_id
        handoff = WorldPulseReadHandoffV1(
            seed_ref=seed, what_i_learned="noted", trace_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
        )
        await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)

        after_one_done = await queue.reading_status(conn, target.request_id)
        assert after_one_done["queue_position"] == 3
        assert after_one_done["queue_depth"] == 4

        for expected in (ahead[1], ahead[2]):
            claimed = await queue.claim_next_seed(conn)
            assert claimed.request.request_id == expected.request_id
            handoff = WorldPulseReadHandoffV1(
                seed_ref=claimed, what_i_learned="noted", trace_id=str(uuid4()),
                created_at=datetime.now(timezone.utc),
            )
            await queue.mark_seed_done(conn, claimed.seed_id, trace_id=handoff.trace_id, handoff=handoff)

        # Claim target itself (now the oldest pending seed) but leave it
        # claimed, not done, so status flips to "started" without target
        # ever completing.
        claimed_target = await queue.claim_next_seed(conn)
        assert claimed_target.request.request_id == target.request_id

        claimed_status = await queue.reading_status(conn, target.request_id)
        assert claimed_status["status"] == "started"
        assert claimed_status["queue_position"] is None
        assert claimed_status["queue_depth"] is None
        await conn.close()
    asyncio.run(run())


def test_stage2_lineage_global_roundtrip_cap_and_durable_result(local_pg):
    async def run():
        conn, schema = await db(local_pg)
        parent = request(parent_run_id="real-run", parent_trace_id="real-trace")
        await queue.enqueue_reading(conn, parent)
        seed = await queue.claim_next_seed(conn)
        handoff = WorldPulseReadHandoffV1(seed_ref=seed, what_i_learned="The source proposes a testable hypothesis.", trace_id=str(uuid4()), created_at=datetime.now(timezone.utc))
        await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)
        claim = await queue.claim_next_stage2_seed(conn)
        assert claim.seed.request == seed.request
        child = request("https://example.org/follow", parent_run_id=parent.parent_run_id, parent_trace_id=parent.parent_trace_id, root_request_id=parent.request_id, parent_request_id=parent.request_id)
        await queue.enqueue_reading(conn, child, max_round_trips=1)
        with pytest.raises(ValueError, match="round_trip_cap"):
            await queue.enqueue_reading(conn, request("https://example.org/third", root_request_id=parent.request_id, parent_request_id=child.request_id), max_round_trips=1)
        result = WorldPulseReadStage2ResultV1(summary="This remains a source-attributed candidate.", trace_id=str(uuid4()), created_at=datetime.now(timezone.utc), request=parent)
        await queue.mark_stage2_done(conn, seed.seed_id, stage2_trace_id=result.trace_id, result=result)
        status = await queue.reading_status(conn, parent.request_id)
        assert status["status"] == "landing_pending"
        assert status["summary"] == result.summary
        assert await queue.confirm_landings(conn) == []
        for entry in (journal_entry(handoff), journal_entry(handoff, result)):
            await conn.execute("INSERT INTO journal_entries VALUES($1,$2,$3)", entry.entry_id, entry.source_ref, entry.body)
        assert len(await queue.confirm_landings(conn)) == 1
        assert (await queue.reading_status(conn, parent.request_id))["status"] == "completed"
        await conn.close()
    asyncio.run(run())


def test_new_arrivals_cannot_starve_older_retries_at_same_priority(local_pg):
    """Both workers preserve FIFO across failures and fresh feed arrivals."""
    async def run():
        conn, schema = await db(local_pg)

        # Stage 1: the retried seed is enqueued FIRST (older created_at), then
        # fails once (transient) so it returns to pending with attempts=1.
        # A fresh same-priority item must not jump ahead of this retry.
        retried_req = request("https://example.org/retried-first")
        await queue.enqueue_reading(conn, retried_req)
        retried_seed = await queue.claim_next_seed(conn)
        outcome = await queue.mark_seed_failed(
            conn, retried_seed.seed_id, error="turn_deferred:x", max_attempts=3
        )
        assert outcome.retry_scheduled

        fresh_req = request("https://example.org/fresh-second")
        await queue.enqueue_reading(conn, fresh_req)

        claimed = await queue.claim_next_seed(conn)
        assert claimed.seed_id == retried_seed.seed_id

        # The fresh item remains claimable once the older retry is running.
        claimed_again = await queue.claim_next_seed(conn)
        assert claimed_again.url == str(fresh_req.url)

        # Stage 2 uses the same FIFO contract.
        for url in ("https://example.org/s2-retried-first", "https://example.org/s2-fresh-second"):
            req = request(url)
            await queue.enqueue_reading(conn, req)
            seed = await queue.claim_next_seed(conn)
            handoff = WorldPulseReadHandoffV1(
                seed_ref=seed, what_i_learned="A finding.", trace_id=str(uuid4()),
                created_at=datetime.now(timezone.utc),
            )
            await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)

        s2_retried = await queue.claim_next_stage2_seed(conn)
        assert s2_retried.seed.url == "https://example.org/s2-retried-first"
        s2_outcome = await queue.mark_stage2_failed(
            conn, s2_retried.seed.seed_id, error="stage2_turn_timeout", max_attempts=3
        )
        assert s2_outcome.retry_scheduled

        s2_claimed = await queue.claim_next_stage2_seed(conn)
        assert s2_claimed.seed.url == "https://example.org/s2-retried-first"

        await conn.close()
    asyncio.run(run())


@pytest.mark.parametrize("stage2", [False, True])
def test_capacity_outage_does_not_exhaust_reading_attempts(local_pg, stage2):
    async def run():
        conn, _ = await db(local_pg)
        req = request()
        await queue.enqueue_reading(conn, req)
        seed = await queue.claim_next_seed(conn)
        if stage2:
            handoff = WorldPulseReadHandoffV1(
                seed_ref=seed, what_i_learned="A finding.", trace_id=str(uuid4()),
                created_at=datetime.now(timezone.utc),
            )
            await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)
        fail = queue.mark_stage2_failed if stage2 else queue.mark_seed_failed
        claim = queue.claim_next_stage2_seed if stage2 else queue.claim_next_seed
        for _ in range(5):
            if stage2 or _ > 0:
                assert await claim(conn) is not None
            outcome = await fail(conn, seed.seed_id,
                                 error="turn_deferred:stance_react_failed: gpu_pool_unavailable:deadline",
                                 max_attempts=3)
            assert outcome.status == "pending"
            assert outcome.attempts == 0
        # Once the reader actually runs, failures still exhaust the budget.
        for attempt in range(1, 4):
            assert await claim(conn) is not None
            outcome = await fail(conn, seed.seed_id, error="turn_error:fcc_stream_stalled", max_attempts=3)
            assert outcome.attempts == attempt
            assert outcome.status == ("pending" if attempt < 3 else "failed")
        assert await claim(conn) is None
        await conn.close()
    asyncio.run(run())


def test_live_verifier_reads_direct_alias_and_missing_sources_without_writes(local_pg):
    from orion.world_pulse_read.verify import inspect_reading

    async def run():
        conn, _ = await db(local_pg)
        original, alias = request(), request()
        await queue.enqueue_reading(conn, original)
        direct = await inspect_reading(conn, str(original.url))
        assert direct["request_id"] == str(original.request_id)
        assert direct["verified_complete"] is False
        await queue.enqueue_reading(conn, alias)
        before = await conn.fetch("SELECT * FROM world_pulse_read_seed ORDER BY seed_id")
        by_alias = await inspect_reading(conn, str(alias.url))
        assert by_alias["request_id"] == str(alias.request_id)
        assert by_alias["seed_id"] == "reading:" + str(original.request_id)
        assert by_alias["stage1_status"] == "pending"
        assert by_alias["gaps"] == direct["gaps"]
        missing = await inspect_reading(conn, "https://example.org/absent")
        assert missing["gaps"] == ["not_found"]
        assert await conn.fetch("SELECT * FROM world_pulse_read_seed ORDER BY seed_id") == before
        await conn.close()
    asyncio.run(run())


def test_stage1_and_stage2_retry_lifecycle_fail_pending_claim_exhaust(local_pg):
    """Real SQL exercise of the whole retry lifecycle on both stages: a
    transient failure returns the row to `pending` and clears the claim, a
    later tick can claim it again exactly like a fresh seed, and the Nth
    transient failure (attempts hits max_attempts) goes terminal -- not just
    the arithmetic, the actual claim -> fail -> reclaim -> claim cycle."""
    async def run():
        conn, schema = await db(local_pg)
        max_attempts = 3

        # --- Stage 1: transient failure retries, then exhausts. ---
        req = request()
        await queue.enqueue_reading(conn, req)
        for attempt in range(1, max_attempts + 1):
            seed = await queue.claim_next_seed(conn)
            assert seed is not None, f"attempt {attempt}: seed not claimable"
            outcome = await queue.mark_seed_failed(
                conn, seed.seed_id, error="turn_deferred:stance_react_failed: x",
                max_attempts=max_attempts,
            )
            assert outcome.attempts == attempt
            expect_pending = attempt < max_attempts
            assert outcome.retry_scheduled is expect_pending
            row = await conn.fetchrow("SELECT * FROM world_pulse_read_seed WHERE seed_id = $1", seed.seed_id)
            assert row["status"] == ("pending" if expect_pending else "failed")
            assert row["attempts"] == attempt
            if expect_pending:
                assert row["claimed_at"] is None
        # Exhausted: no longer claimable.
        assert await queue.claim_next_seed(conn) is None

        # --- Stage 2: same lifecycle, non-transient stays terminal on try 1. ---
        req2 = request("https://example.org/stage2-retry")
        await queue.enqueue_reading(conn, req2)
        seed2 = await queue.claim_next_seed(conn)
        handoff = WorldPulseReadHandoffV1(seed_ref=seed2, what_i_learned="A finding.", trace_id=str(uuid4()), created_at=datetime.now(timezone.utc))
        await queue.mark_seed_done(conn, seed2.seed_id, trace_id=handoff.trace_id, handoff=handoff)

        claim = await queue.claim_next_stage2_seed(conn)
        assert claim is not None
        outcome = await queue.mark_stage2_failed(
            conn, seed2.seed_id, error="1 validation error for WorldPulseReadStage2ResultV1",
            max_attempts=max_attempts,
        )
        assert outcome.status == "failed"
        assert outcome.attempts == 1
        assert await queue.claim_next_stage2_seed(conn) is None  # non-transient: no retry

        # A fresh transient Stage 2 failure DOES retry and is reclaimable.
        req3 = request("https://example.org/stage2-retry-2")
        await queue.enqueue_reading(conn, req3)
        seed3 = await queue.claim_next_seed(conn)
        handoff3 = WorldPulseReadHandoffV1(seed_ref=seed3, what_i_learned="Another finding.", trace_id=str(uuid4()), created_at=datetime.now(timezone.utc))
        await queue.mark_seed_done(conn, seed3.seed_id, trace_id=handoff3.trace_id, handoff=handoff3)
        claim3 = await queue.claim_next_stage2_seed(conn)
        outcome3 = await queue.mark_stage2_failed(
            conn, seed3.seed_id, error="stage2_turn_timeout", max_attempts=max_attempts,
        )
        assert outcome3.retry_scheduled
        assert outcome3.attempts == 1
        reclaimed = await queue.claim_next_stage2_seed(conn)
        assert reclaimed is not None and reclaimed.seed.seed_id == seed3.seed_id

        retries = await queue.count_retry_state(conn, max_attempts=max_attempts)
        assert retries["stage1_exhausted"] == 1
        # seed2's failure was non-transient and terminal on attempt 1 -- a bad
        # seed, not a burned-out retry budget, so it is NOT counted as
        # "exhausted" (that label is reserved for attempts >= max_attempts).
        assert retries["stage2_exhausted"] == 0
        await conn.close()
    asyncio.run(run())


def test_missing_journals_replay_saved_artifacts_without_model_or_wallet(local_pg):
    async def run():
        conn, schema = await db(local_pg)
        req = request()
        await queue.enqueue_reading(conn, req)
        seed = await queue.claim_next_seed(conn)
        handoff = WorldPulseReadHandoffV1(seed_ref=seed, what_i_learned="Source-attributed learning.", trace_id=str(uuid4()), created_at=datetime.now(timezone.utc))
        result = WorldPulseReadStage2ResultV1(summary="A candidate with a remaining question.", round_trips=2, trace_id=str(uuid4()), created_at=datetime.now(timezone.utc), request=req)
        await queue.mark_seed_done(conn, seed.seed_id, trace_id=handoff.trace_id, handoff=handoff)
        await queue.mark_stage2_done(conn, seed.seed_id, stage2_trace_id=result.trace_id, result=result)
        class Bus:
            async def publish(self, channel, envelope):
                p = envelope.payload
                await conn.execute("INSERT INTO journal_entries VALUES($1,$2,$3) ON CONFLICT DO NOTHING", p["entry_id"], p["source_ref"], p["body"])
        pipe = object.__new__(WorldPulseReadStage2Pipeline)
        pipe._bus, pipe._source_ref = Bus(), ServiceRef(name="orion-hub")
        async def with_conn(fn):
            return await fn(conn)
        pipe._with_conn = with_conn
        assert (await queue.reading_status(conn, req.request_id))["status"] == "landing_pending"
        await pipe._repair_journal_landings()
        await pipe._repair_journal_landings()
        assert await conn.fetchval("SELECT count(*) FROM journal_entries") == 2
        stage2_body = await conn.fetchval("SELECT body FROM journal_entries WHERE source_ref LIKE 'world_pulse_read_stage2:%'")
        assert "round_trips=2" in stage2_body
        assert len(await queue.confirm_landings(conn)) == 1
        assert (await queue.reading_status(conn, req.request_id))["status"] == "completed"
        await conn.close()
    asyncio.run(run())


def test_stale_digest_items_skipped_by_real_sql(local_pg):
    """Only never-claimed `pending` digest_item rows older than the cutoff
    move; findings, readings, claimed rows and fresh digest items stay."""
    async def run():
        conn, _schema = await db(local_pg)
        rows = [
            ("digest_item:old", "digest_item", "pending", "6 days"),
            ("digest_item:fresh", "digest_item", "pending", "1 day"),
            ("digest_item:claimed-old", "digest_item", "claimed", "9 days"),
            ("finding:old", "finding", "pending", "18 days"),
        ]
        for seed_id, kind, status, age in rows:
            await conn.execute(
                "INSERT INTO world_pulse_read_seed(seed_id, kind, run_id, url, status, priority, created_at) "
                "VALUES($1, $2, 'r', $3, $4, 10, now() - $5::text::interval)",
                seed_id, kind, f"https://example.org/{seed_id}", status, age,
            )
        assert await queue.skip_stale_digest_items(conn, max_age_sec=0) == 0
        assert await queue.skip_stale_digest_items(conn, max_age_sec=5 * 86400) == 1
        got = {r["seed_id"]: (r["status"], r["last_error"]) for r in await conn.fetch(
            "SELECT seed_id, status, last_error FROM world_pulse_read_seed")}
        assert got == {
            "digest_item:old": ("skipped", "stale_digest_item"),
            "digest_item:fresh": ("pending", None),
            "digest_item:claimed-old": ("claimed", None),
            "finding:old": ("pending", None),
        }
        # Idempotent on the next tick.
        assert await queue.skip_stale_digest_items(conn, max_age_sec=5 * 86400) == 0
        await conn.close()
    asyncio.run(run())


def test_stale_sweep_keeps_a_digest_item_a_request_is_aliased_to(local_pg):
    """A finding / reading request for a URL already pending as a digest item
    is stored as an alias of that row; sweeping the row would drop the request."""
    async def run():
        conn, _schema = await db(local_pg)
        await conn.execute(
            "INSERT INTO world_pulse_read_seed(seed_id, kind, run_id, url, status, priority, created_at) "
            "VALUES('digest_item:shared', 'digest_item', 'r', 'https://example.org/shared', 'pending', 10, now() - interval '9 days'),"
            "      ('digest_item:alone', 'digest_item', 'r', 'https://example.org/alone', 'pending', 10, now() - interval '9 days')"
        )
        await conn.execute(
            "INSERT INTO world_pulse_read_seed(seed_id, kind, run_id, url, status, stage2_status, priority, duplicate_of) "
            "VALUES('reading:alias', 'reading', 'r', 'https://example.org/shared', 'skipped', 'skipped', 0, 'digest_item:shared')"
        )
        assert await queue.skip_stale_digest_items(conn, max_age_sec=5 * 86400) == 1
        got = dict(await conn.fetch("SELECT seed_id, status FROM world_pulse_read_seed WHERE kind = 'digest_item'"))
        assert got == {"digest_item:shared": "pending", "digest_item:alone": "skipped"}
        await conn.close()
    asyncio.run(run())


def test_reading_results_introspection_is_read_only_and_distinguishes_states(local_pg):
    from orion.schemas.introspect import DEFAULT_TEXT_CAP, URL_CAP
    from orion.world_pulse_read.introspect import reading_results

    async def run():
        conn, _ = await db(local_pg)
        # Production journal_entries (sql-writer) has these columns; the shared
        # helper's minimal table does not.
        await conn.execute(
            "ALTER TABLE journal_entries ADD COLUMN created_at timestamptz NOT NULL DEFAULT now(), "
            "ADD COLUMN title text"
        )
        done = request("https://example.org/done")
        hollow = request("https://example.org/hollow")
        queued = request("https://example.org/queued")
        failed = request("https://example.org/failed")
        for r in (done, hollow, queued, failed):
            await queue.enqueue_reading(conn, r)
        mark_done = """UPDATE world_pulse_read_seed
               SET status='done', stage2_status='done',
                   handoff_json=$2::jsonb, stage2_result_json=$3::jsonb,
                   handoff_at=now(), stage2_completed_at=now(), landing_at=now(),
                   trace_id='t1', stage2_trace_id='t2'
               WHERE request_id=$1"""
        await conn.execute(
            mark_done,
            done.request_id,
            json.dumps({"what_i_learned": "stage one note", "read_evidence": [
                {"tool_name": "WebFetch", "url": "https://example.org/done", "content_chars": 500},
            ]}),
            json.dumps({"summary": "S" * (DEFAULT_TEXT_CAP + 300)}),
        )
        # Pre-2026-09-25 shape: marked done with no tool-trace read of the source.
        await conn.execute(
            mark_done,
            hollow.request_id,
            json.dumps({"what_i_learned": "metadata-only guess"}),
            json.dumps({"summary": "built on an unread handoff"}),
        )
        await conn.execute(
            "INSERT INTO journal_entries (entry_id, source_ref, body) "
            "VALUES ('j1', 'world_pulse_read_stage2:t2', 'Journal body about the source')"
        )
        await conn.execute(
            "UPDATE world_pulse_read_seed SET status='failed', last_error='boom', handoff_json=$2::jsonb "
            "WHERE request_id=$1",
            failed.request_id, json.dumps({"what_i_learned": "rejected handoff"}),
        )
        long_url = "https://example.org/" + "p" * 700
        await queue.enqueue_reading(conn, request(long_url))
        before = await conn.fetch("SELECT * FROM world_pulse_read_seed ORDER BY seed_id")

        recent = await reading_results(conn)
        assert recent.ok and recent.total_available == 1
        [item] = recent.items
        assert item.kind == "reading_result" and item.epistemic_status == "unsettled"
        assert item.extra["url"] == "https://example.org/done"
        assert item.extra["learned"] is True
        assert item.extra["source_read"] is True
        assert item.truncated and len(item.text) == DEFAULT_TEXT_CAP
        assert "journal_excerpt" not in item.extra
        assert "url_truncated" not in item.extra

        unread = await reading_results(conn, url="https://example.org/hollow")
        assert unread.items[0].extra["reading_status"] == "completed"
        assert unread.items[0].extra["source_read"] is False
        assert unread.items[0].extra["learned"] is False
        assert unread.items[0].text == ""

        by_url = await reading_results(conn, url="https://EXAMPLE.org/done#x")
        assert by_url.total_available == 1
        assert by_url.items[0].extra["journal_excerpt"] == "Journal body about the source"
        assert by_url.items[0].extra["request_id"] == str(done.request_id)

        clipped = (await reading_results(conn, url=long_url)).items[0].extra
        assert len(clipped["url"]) == URL_CAP and clipped["url_truncated"] is True

        pending = await reading_results(conn, request_id=queued.request_id)
        assert pending.items[0].text == ""
        assert pending.items[0].extra["learned"] is False
        assert pending.items[0].extra["reading_status"] == "queued"

        rejected = await reading_results(conn, request_id=failed.request_id)
        assert rejected.items[0].extra["reading_status"] == "failed"
        assert rejected.items[0].extra["learned"] is False
        assert rejected.items[0].text == ""

        missing = await reading_results(conn, url="https://example.org/never")
        assert missing.ok and missing.items == [] and missing.total_available == 0

        future = await reading_results(conn, since=datetime(2999, 1, 1, tzinfo=timezone.utc))
        assert future.ok and future.items == [] and future.total_available == 0

        assert await conn.fetch("SELECT * FROM world_pulse_read_seed ORDER BY seed_id") == before
        await conn.close()

    asyncio.run(run())


def test_reading_search_index_and_query_use_real_sql_gate(local_pg):
    import httpx

    from orion.world_pulse_read.search import (
        ReadingSearchConfig, index_missing_readings, search_readings, verified_rows,
    )

    cfg = ReadingSearchConfig(
        chroma_url="http://chroma.test", embed_url="http://embed.test/embedding",
        collection="orion_reading_results", min_similarity=0.6,
    )

    async def run():
        conn, _ = await db(local_pg)
        done, hollow, queued, dup = (
            request(f"https://example.org/{n}") for n in ("done", "hollow", "queued", "dup")
        )
        for r in (done, hollow, queued, dup):
            await queue.enqueue_reading(conn, r)
        mark = """UPDATE world_pulse_read_seed SET status='done', stage2_status='done',
                  handoff_json=$2::jsonb, stage2_result_json=$3::jsonb, handoff_at=now(),
                  stage2_completed_at=now(), landing_at=now() WHERE request_id=$1"""
        await conn.execute(mark, done.request_id, json.dumps({"read_evidence": [
            {"tool_name": "WebFetch", "url": "https://example.org/done", "content_chars": 500}]}),
            json.dumps({"summary": "GPU supply is tight"}))
        await conn.execute(mark, hollow.request_id, json.dumps({"what_i_learned": "guess"}),
                           json.dumps({"summary": "built on nothing"}))
        # A genuinely read, finished duplicate passes the Python gate; only SQL stops it.
        await conn.execute(mark, dup.request_id, json.dumps({"read_evidence": [
            {"tool_name": "WebFetch", "url": "https://example.org/dup", "content_chars": 500}]}),
            json.dumps({"summary": "duplicate of done"}))
        seeds = {r["url"]: r["seed_id"] for r in await conn.fetch("SELECT url, seed_id FROM world_pulse_read_seed")}
        await conn.execute(
            "UPDATE world_pulse_read_seed SET duplicate_of=$1 WHERE seed_id=$2",
            seeds["https://example.org/done"], seeds["https://example.org/dup"],
        )

        def handler(req: httpx.Request) -> httpx.Response:
            if req.url.host == "embed.test":
                return httpx.Response(200, json={"doc_id": "q", "embedding": [1.0, 0.0], "embedding_dim": 2})
            if req.url.path.endswith("/orion_reading_results"):
                return httpx.Response(200, json={"id": "cid", "metadata": None})
            if req.url.path.endswith("/get"):
                return httpx.Response(200, json={"ids": [], "metadatas": []})
            ids = [seeds[f"https://example.org/{n}"] for n in ("done", "hollow", "queued", "dup")]
            return httpx.Response(200, json={"ids": [ids], "distances": [[0.2] * len(ids)]})

        published = []

        class Bus:
            async def publish(self, channel, envelope):
                published.append(envelope.payload["doc_id"])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            rows = await verified_rows(conn)
            result = await index_missing_readings(rows, cfg, client=client, bus=Bus(), source=ServiceRef(name="orion-hub"))
            assert (result.indexed, result.pending) == (1, 0)
            assert published == [seeds["https://example.org/done"]]
            found = await search_readings(conn, cfg, client=client, query="graphics cards", limit=5)
            assert found.total_available == 1
            assert found.items[0].extra["url"] == "https://example.org/done"
            assert found.items[0].extra["similarity"] == 0.9
            assert found.items[0].text == "GPU supply is tight"
            future = await search_readings(conn, cfg, client=client, query="gpus", limit=5,
                                           since=datetime(2999, 1, 1, tzinfo=timezone.utc))
            assert future.items == [] and future.total_available == 0
        await conn.close()

    asyncio.run(run())


def test_document_capture_and_path_lookup_on_real_postgres(local_pg, tmp_path):
    from orion.world_pulse_read.documents import DocumentPolicy, DocumentSourceError

    doc = tmp_path / "spec_100%_done.md"
    doc.write_text("# Spec\n\nv1 body\n")
    policy = DocumentPolicy.from_values(roots=str(tmp_path), extensions=None, max_bytes=4096)

    async def run():
        conn, _ = await db(local_pg)
        first = await queue.enqueue_reading(conn, request(url=f"file://{doc}"), documents=policy)
        row = await conn.fetchrow("SELECT url FROM world_pulse_read_seed WHERE seed_id=$1", first["seed_id"])
        sha = row["url"].rsplit("=", 1)[1]
        snap = await conn.fetchrow("SELECT content, content_chars, first_source FROM reading_document_snapshot WHERE sha256=$1", sha)
        assert snap["content"] == "# Spec\n\nv1 body\n" and snap["first_source"] == row["url"]
        doc.write_text("# Spec\n\nv2 body\n")
        second = await queue.enqueue_reading(conn, request(url=f"file://{doc}"), documents=policy)
        assert second["duplicate"] is None
        # `%` in the path must not act as a LIKE wildcard; split_part is exact.
        status = await queue.reading_status(conn, url=str(doc))
        assert status["request_id"] == second["request_id"] and status["matched_request_count"] == 2
        assert (await queue.reading_status(conn, url=str(tmp_path / "spec_1.md")))["status"] == "not_found"
        # A pinned source must name a stored snapshot; it is never read from disk.
        with pytest.raises(DocumentSourceError, match="document_snapshot_missing"):
            await queue.enqueue_reading(conn, request(url=f"file://{doc}?sha256={'e' * 64}"), documents=policy)
        # A real snapshot hash cannot vouch for a path it was not captured from.
        other = tmp_path / "other.md"
        other.write_text("other")
        with pytest.raises(DocumentSourceError, match="document_snapshot_missing"):
            await queue.enqueue_reading(conn, request(url=f"file://{other}?sha256={sha}"), documents=policy)
        # The exact captured ref is accepted again (folded onto the first read).
        again = await queue.enqueue_reading(conn, request(url=row["url"]), documents=policy)
        assert again["duplicate_of"] == first["seed_id"]
        await conn.close()

    asyncio.run(run())
