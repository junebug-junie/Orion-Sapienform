"""Issue #3382: a bid never waits on the market's history refresh or weights replay while a snapshot exists.

Live 2026-11-15: the 300 s history cache refreshed INLINE on the bid path (acts + uses + event history, ~1.3 s,
then the 30-day weights replay), so 14/14 camera-ask bids missed sql-writer's 0.5 s client budget. Now the last
snapshot is served stale-while-revalidate with ONE background refresh, and the weights are recomputed in the
background too. Fakes only: a pool whose reads sleep."""

from __future__ import annotations

import asyncio
import threading
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
FAST = 0.4          # a bid served from a snapshot must answer well inside the 0.5 s client budget


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


def _late_market(pool):
    """A market whose body state is at LATE (23:59 UTC), so bids around midnight are priced."""
    m = _warm_market(pool)
    m._redis_getter = lambda: _Redis(_state(LATE).model_dump_json())
    return m


LATE = NOW.replace(hour=23, minute=59, second=0, microsecond=0)


def _gated_weights(m):
    """Make the weights replay block on a threading.Event, counting calls."""
    real, gate, calls = m._compute_weights, threading.Event(), []

    def gated(hist, now, choices):
        calls.append(now)
        gate.wait(5.0)
        return real(hist, now, choices)

    m._compute_weights = gated
    return gate, calls


def test_a_new_day_keeps_the_last_weights_and_replays_in_the_background():
    async def go():
        pool = _SlowPool()
        m = _late_market(pool)
        await m.bid(_want(100), LATE)                      # snapshot and weights both at 23:59
        old_w, snap = m._weights, m._cache
        assert old_w[0][0] == snap.at == LATE
        gate, calls = _gated_weights(m)
        next_day = LATE + timedelta(minutes=2)             # same snapshot (fresh), new UTC day
        t0 = time.monotonic()
        d = await m.bid(_want(1), next_day)
        assert time.monotonic() - t0 < FAST and d.reason != "physiology_unavailable", (d.reason, d.why)
        assert m._weights is old_w and m._cache is snap    # served the last weights meanwhile
        await m.bid(_want(2), next_day + timedelta(seconds=1))
        gate.set()
        await m._weights_task
        assert len(calls) == 1                             # one replay, not one per bid
        assert m._weights[0] == (snap.at, next_day.date(), None) and m._cache is snap

    asyncio.run(go())


def test_a_late_weights_replay_never_overwrites_a_newer_snapshot_and_none_starts_during_a_refresh():
    async def go():
        pool = _SlowPool()
        m = _late_market(pool)
        await m.bid(_want(100), LATE)
        snap, w0 = m._cache, m._weights
        gate, calls = _gated_weights(m)
        next_day = LATE + timedelta(minutes=2)
        # The replay for snap starts on a new day; a refresh swaps in first, the stale replay finishes after.
        await m.bid(_want(2), next_day)
        wtask = m._weights_task
        fresh = pm._Cached(at=next_day + timedelta(minutes=1), acts=snap.acts, uses=snap.uses, events=snap.events)
        newer = ((fresh.at, fresh.at.date(), None), *w0[1:])
        m._cache, m._weights = fresh, newer
        gate.set()
        await wtask
        assert m._weights is newer                         # the late replay was for an older snapshot

        # While a refresh is in flight, a key change does not start a separate replay.
        pool.delay = 0.3
        n = len(calls)
        later = fresh.at + timedelta(seconds=TTL + 1)
        await m.bid(_want(3), later)                       # stale: kicks the refresh
        assert m._refresh_task is not None and not m._refresh_task.done()
        m._weights = (("other",), *m._weights[1:])         # e.g. a new personality read
        m._weights_task, m._weights_last_attempt = None, None
        await m.bid(_want(4), later + timedelta(seconds=1))
        assert m._weights_task is None
        await m._refresh_task
        assert len(calls) == n + 1                         # only the refresh's own replay

    asyncio.run(go())


def test_a_bid_older_than_the_snapshot_is_served_fresh_without_a_refresh():
    async def go():
        pool = _SlowPool()
        m = _warm_market(pool)
        await _seed(m, pool)
        snap, reads = m._cache, pool.act_reads
        d = await m.bid(_want(1), NOW - timedelta(seconds=2))   # a caller's clock read before the refresher's
        assert d.reason != "physiology_unavailable"
        assert m._cache is snap and pool.act_reads == reads and m._refresh_task.done()
        assert m.counts["history_served_stale"] == 0

    asyncio.run(go())


def test_a_hung_refresh_is_abandoned_so_the_next_one_can_run(monkeypatch):
    async def go():
        monkeypatch.setattr(pm, "REFRESH_TIMEOUT_SEC", 0.2)
        pool = _SlowPool()
        m = _warm_market(pool)
        await _seed(m, pool)
        pool.delay = 10.0
        later = NOW + timedelta(seconds=TTL + 1)
        await m.bid(_want(1), later)
        await asyncio.gather(m._refresh_task, return_exceptions=True)
        await asyncio.sleep(0)
        assert m.counts["history_refresh_failed"] == 1 and m._refresh_task.done()

    asyncio.run(go())


def test_run_refresher_warms_the_cache_recovers_after_a_failed_boot_and_stops():
    async def go():
        pool = _SlowPool(fail=True)
        m = _warm_market(pool)
        m._ttl = 2.0                                       # refresh every 1.6 s
        stop = asyncio.Event()
        task = asyncio.create_task(m.run_refresher(stop))
        await asyncio.sleep(0.1)
        assert m._cache is None and m.counts["history_refresh_failed"] == 1   # boot refresh failed
        pool.fail = False
        for _ in range(300):
            if m._cache is not None and m._weights is not None:
                break
            await asyncio.sleep(0.01)
        assert m._cache is not None and m._weights is not None   # the next cycle recovered
        stop.set()
        await asyncio.wait_for(task, 2.0)

    asyncio.run(go())


def test_aclose_cancels_an_in_flight_refresh():
    async def go():
        pool = _SlowPool()
        m = _warm_market(pool)
        await _seed(m, pool)
        pool.delay = 10.0
        await m.bid(_want(1), NOW + timedelta(seconds=TTL + 1))
        t = m._refresh_task
        await asyncio.wait_for(m.aclose(), 1.0)
        assert t.done() and m.counts["history_refresh_failed"] == 0

    asyncio.run(go())
