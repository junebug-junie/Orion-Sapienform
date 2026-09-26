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
            assert latest["duplicate_of"] == "reading:" + str(current.request_id)
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


def test_concurrent_submissions_alias_active_work_and_allow_later_reread(local_pg):
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
