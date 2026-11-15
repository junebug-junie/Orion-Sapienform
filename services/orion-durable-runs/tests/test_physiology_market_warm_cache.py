"""Issue #3382: a bid never waits on the market's history refresh or weights replay while a snapshot exists.

Live 2026-11-15: the 300 s history cache refreshed INLINE on the bid path (acts + uses + event history, ~1.3 s,
then the 30-day weights replay), so 14/14 camera-ask bids missed sql-writer's 0.5 s client budget. Now the last
snapshot is served stale-while-revalidate with ONE background refresh, and the weights are recomputed in the
background too. Fakes only: a pool whose reads sleep."""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from app import physiology_market as pm  # noqa: E402
from tests.test_physiology_bid import NOW, _Redis, _acts, _market, _state, _want  # noqa: E402

TTL = 300.0
FAST = 0.25         # a bid served from a snapshot must answer well inside the 0.5 s client budget


class _SlowConn:
    def __init__(self, pool):
        self.pool = pool

    async def execute(self, sql, params=None):
        if "grammar_acts" in sql:
            self.pool.act_reads += 1
        await asyncio.sleep(self.pool.delay)
        if self.pool.fail:
            raise RuntimeError("db down")

        class _C:
            def __init__(self, rows):
                self.rows = rows

            async def fetchall(self):
                return self.rows

        if "want_ledger" in sql:
            return _C([])
        return _C(self.pool.acts if "grammar_acts" in sql else [])


class _SlowPool:
    def __init__(self, delay=0.0, fail=False):
        self.delay, self.fail, self.act_reads, self.acts = delay, fail, 0, _acts()

    def connection(self):
        pool = self

        class _CM:
            async def __aenter__(self):
                return _SlowConn(pool)

            async def __aexit__(self, *a):
                return False

        return _CM()


def _warm_market(pool):
    m = _market(_state().model_dump_json(), pool, [])
    return m


async def _seed(m, pool):
    """A first snapshot at NOW, read fast."""
    delay, pool.delay = pool.delay, 0.0
    d = await m.bid(_want(100), NOW)
    assert d.outcome == "funded"
    pool.delay = delay
    return m._cache


def test_expired_cache_and_slow_reader_serves_the_old_snapshot_and_refreshes_once():
    async def go():
        pool = _SlowPool(delay=1.0)
        m = _warm_market(pool)
        old = await _seed(m, pool)
        reads = pool.act_reads
        later = NOW + timedelta(seconds=TTL + 30)
        t0 = time.monotonic()
        ds = await asyncio.gather(*(m.bid(_want(i), later + timedelta(milliseconds=i)) for i in range(5)))
        elapsed = time.monotonic() - t0
        assert elapsed < FAST, elapsed                     # nobody waited on the 1 s read
        assert all(d.reason != "physiology_unavailable" for d in ds), [(d.reason, d.why) for d in ds]
        assert m._cache is old and m.counts["history_served_stale"] == 5
        await m._refresh_task                              # the one background refresh
        assert pool.act_reads == reads + 1                 # exactly one refresh, not one per bid
        assert m._cache is not old and m._cache.at == later
        assert m._weights[0][0] == later                   # weights came with the new snapshot
        # The next bid is fresh: no read, no refresh.
        await m.bid(_want(9), later + timedelta(seconds=5))
        assert pool.act_reads == reads + 1

    asyncio.run(go())


def test_a_failed_refresh_keeps_the_last_snapshot_and_its_weights():
    async def go():
        pool = _SlowPool()
        m = _warm_market(pool)
        old = await _seed(m, pool)
        old_w = m._weights
        pool.fail = True
        later = NOW + timedelta(seconds=TTL + 30)
        d = await m.bid(_want(1), later)
        assert d.reason != "physiology_unavailable"
        await asyncio.gather(m._refresh_task, return_exceptions=True)
        await asyncio.sleep(0)                             # the done callback runs
        assert m._cache is old and m._weights is old_w
        assert m.counts["history_refresh_failed"] == 1
        # Bids right after a failure do not hammer the database: retries are spaced by RETRY_SEC.
        reads = pool.act_reads
        await m.bid(_want(2), later + timedelta(seconds=5))
        assert pool.act_reads == reads and m._cache is old
        await m.bid(_want(3), later + timedelta(seconds=pm.RETRY_SEC + 1))
        await asyncio.gather(m._refresh_task, return_exceptions=True)
        assert pool.act_reads == reads + 1

    asyncio.run(go())


def test_a_snapshot_past_the_staleness_bound_is_not_served():
    async def go():
        pool = _SlowPool()
        m = _warm_market(pool)
        await _seed(m, pool)
        pool.fail = True
        too_old = NOW + timedelta(seconds=TTL * pm.MAX_STALE_TTLS + 1)
        d = await m.bid(_want(1), too_old)
        assert d.outcome == "deferred" and d.reason == "physiology_unavailable"

    asyncio.run(go())


def test_no_snapshot_waits_once_and_a_failure_is_unavailable():
    async def go():
        pool = _SlowPool(delay=0.2)
        m = _warm_market(pool)
        ds = await asyncio.gather(*(m.bid(_want(i), NOW) for i in range(3)))
        assert all(d.reason != "physiology_unavailable" for d in ds), [(d.reason, d.why) for d in ds]
        assert pool.act_reads == 1                         # concurrent first bids share one read

        bad = _SlowPool(fail=True)
        m2 = _warm_market(bad)
        d = await m2.bid(_want(1), NOW)
        assert d.outcome == "deferred" and d.reason == "physiology_unavailable"

    asyncio.run(go())


def test_a_new_day_keeps_the_last_weights_and_replays_in_the_background():
    async def go():
        pool = _SlowPool()
        m = _warm_market(pool)
        await _seed(m, pool)
        old_w = m._weights
        real = m._compute_weights
        calls = []

        def slow(hist, now, choices):
            calls.append(now)
            time.sleep(1.0)                                # the 30-day replay, in its worker thread
            return real(hist, now, choices)

        m._compute_weights = slow
        # Same snapshot, but the UTC day moved (a fresh snapshot still cached: no history read).
        m._cache.at = NOW.replace(hour=23, minute=59, second=0, microsecond=0)
        next_day = m._cache.at + timedelta(minutes=2)
        fresh_state = _state(next_day).model_dump_json()
        m._redis_getter = lambda: _Redis(fresh_state)
        t0 = time.monotonic()
        d = await m.bid(_want(1), next_day)
        assert time.monotonic() - t0 < FAST and d.reason != "physiology_unavailable", (d.reason, d.why)
        assert m._weights is old_w                         # served the last weights meanwhile
        await m.bid(_want(2), next_day + timedelta(seconds=1))
        await m._weights_task
        assert len(calls) == 1                             # one replay, not one per bid
        assert m._weights[0][1] == next_day.date()

    asyncio.run(go())


def test_run_refresher_warms_the_cache_before_any_bid():
    async def go():
        pool = _SlowPool()
        m = _warm_market(pool)
        stop = asyncio.Event()
        task = asyncio.create_task(m.run_refresher(stop))
        for _ in range(100):
            if m._cache is not None and m._weights is not None:
                break
            await asyncio.sleep(0.01)
        assert m._cache is not None and m._weights is not None
        stop.set()
        await asyncio.wait_for(task, 2.0)

    asyncio.run(go())
