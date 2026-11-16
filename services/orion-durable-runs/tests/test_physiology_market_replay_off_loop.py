"""Issue #3382, second cause: the 30-day weights replay must not slow the bids.

Live 2026-11-16 07:00Z: a personality read changed the weights key, the first bid of a camera-ask burst started
the replay in a thread, and its ~3.3 s of pure-Python CPU work held the GIL: bids that take ~0.25 s took
0.45-0.70 s and missed sql-writer's 0.5 s budget. Now the replay runs in a worker process and only the refresher
starts one. A bid reads the appetite of its own family only."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from app import physiology_market as pm  # noqa: E402
from app import weights_worker as ww  # noqa: E402
from orion.physiology.choice import build_context, decide_priced  # noqa: E402
from tests.test_physiology_bid import FAMS, NOW, _Pool, _market, _state, _want  # noqa: E402

BUSY_SEC = 1.5

# A CPU-bound replay: ~BUSY_SEC of pure Python, then the real one. It touches a marker file when it starts, so
# the test knows it is running. Written to its own module under tmp_path: the worker process imports it by name,
# and a name under ``tests`` is ambiguous there (the repo root has a ``tests`` package too).
BUSY_MODULE = """
import time
from pathlib import Path

from app import weights_worker as ww

MARKER = {marker!r}
BUSY_SEC = {busy!r}


def busy_replay(families, acts, uses, now, choices):
    Path(MARKER).touch()
    end, x = time.perf_counter() + BUSY_SEC, 0
    while time.perf_counter() < end:
        x += 1
    return ww.compute_weights(families, acts, uses, now, choices)
"""


def _busy_replay(tmp_path, monkeypatch):
    import importlib

    marker = tmp_path / "replay_started"
    name = f"busy_replay_{abs(hash(str(tmp_path)))}"
    (tmp_path / f"{name}.py").write_text(BUSY_MODULE.format(marker=str(marker), busy=BUSY_SEC))
    monkeypatch.syspath_prepend(str(tmp_path))     # the spawned worker gets this sys.path too
    return importlib.import_module(name).busy_replay, marker


def _heavy_acts(n=1200, family="durable.self_inquiry"):
    """Enough history that one bid costs real CPU (its appetite build), as live bids do (~0.2 s)."""
    out = []
    for i in range(n):
        closed = NOW - timedelta(minutes=30 + 30 * i)
        out.append({"act_id": f"h{i}", "actor": family, "opened_at": closed - timedelta(minutes=10),
                    "closed_at": closed, "outcome_value": 1.0, "outcome_source": "producer", "signal_id": None,
                    "cost_json": json.dumps({"seconds": 600.0, "gpu_slot_s_by_lane": {"agent": 300.0},
                                             "basis": {"wh": "x" * 400}})})
    return out


class _Personality:
    """The parts of app.personality_values.PersonalityValues the market reads."""

    def __init__(self):
        self.enabled, self.at, self.kicks = False, NOW, 0

    def snapshot_at(self):
        return self.at

    def choices(self):
        return None

    def kick(self, now):
        self.kicks += 1
        return None

    def health(self):
        return {}


def _bid_times(m, start, n):
    """Per bid: (wall seconds, this thread's CPU seconds). Waiting for the GIL is wall time with no CPU time,
    so wall / CPU measures contention and is immune to the host's speed changing between bids."""
    async def go():
        out = []
        for i in range(n):
            t0, c0 = time.perf_counter(), time.thread_time()
            d = await m.bid(_want(start + i), NOW + timedelta(minutes=2, milliseconds=i))
            out.append((time.perf_counter() - t0, time.thread_time() - c0))
            assert d.reason != "physiology_unavailable", (d.reason, d.why)
        return out
    return go()


HEAVY_ACTS = 6000      # bids cost ~50 ms of CPU each, so the GIL contention a thread replay causes is visible


def _slowdown(tmp_path, monkeypatch, factory) -> tuple[float, list, list]:
    """While a CPU-bound replay runs: the median bid's wall time over its own CPU time (1.0: the bid never
    waited for the GIL). ``factory``: the replay's executor (the worker process, or a thread for the control).
    Wall-time ratios against a baseline flaked on a shared host (bids moved 10 <-> 26 ms between runs)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    busy_replay, marker = _busy_replay(tmp_path, monkeypatch)

    async def go():
        m = _market(_state().model_dump_json(), _Pool(acts=_heavy_acts(HEAVY_ACTS)), [])
        m._executor_factory = factory
        try:
            await m.bid(_want(0), NOW)                     # boot: the first weights
            assert m._weights is not None and m._executor is not None
            base = await _bid_times(m, 100, 7)
            m._replay = busy_replay
            m.personality = _Personality()                 # a new personality read: the key moved
            m.personality.at = NOW + timedelta(seconds=1)
            rtask = asyncio.create_task(m.reconcile_weights(NOW + timedelta(minutes=2)))
            for _ in range(1000):                          # wait until the replay is inside its busy loop
                if marker.exists():
                    break
                await asyncio.sleep(0.01)
            assert marker.exists(), "the replay never started"
            during = await _bid_times(m, 200, 7)
            assert not rtask.done(), "the replay finished before the bids: they did not overlap"
            await rtask
            assert m._weights[0][2] == m.personality.at   # the refresher's replay landed
        finally:
            await m.aclose()
        return base, during

    base, during = asyncio.run(go())
    return statistics.median(w / max(c, 1e-6) for w, c in during), base, during


def test_a_cpu_bound_replay_in_the_worker_process_does_not_slow_the_bids(tmp_path, monkeypatch):
    if len(os.sched_getaffinity(0)) < 3:
        # With two cores the OS can put the busy worker on the bid loop's core: CPU sharing, not the GIL.
        pytest.skip("needs 3+ CPUs to separate GIL contention from core sharing")
    # Best of three: on a loaded host the OS sometimes shares a core between the worker and the loop (ratio ~2
    # for that run). GIL contention is systematic: the thread control below read 1.8-2.1 in 40 of 40 runs.
    seen = []
    for i in range(3):
        ratio, base, during = _slowdown(tmp_path / str(i), monkeypatch, ww.process_executor)
        seen.append(round(ratio, 2))
        if ratio < 1.3:
            return
    raise AssertionError(f"every attempt waited on the GIL: wall/CPU ratios {seen}")


def test_control_the_same_replay_in_a_thread_does_slow_the_bids(tmp_path, monkeypatch):
    """The control for the test above: proves it can fail. In a thread the busy replay shares the loop's GIL."""
    from concurrent.futures import ThreadPoolExecutor

    ratio, base, during = _slowdown(tmp_path, monkeypatch, lambda: ThreadPoolExecutor(1))
    assert ratio > 1.6, (ratio, base, during)


def test_a_personality_change_never_makes_a_bid_start_a_replay():
    async def go():
        m = _market(_state().model_dump_json(), _Pool(acts=_heavy_acts(50)), [])
        m.personality = _Personality()
        await m.bid(_want(0), NOW)
        w0 = m._weights
        calls = []
        real = m._compute_weights

        def counted(hist, now, choices):
            calls.append(now)
            return real(hist, now, choices)

        m._compute_weights = counted
        m.personality.at = NOW + timedelta(seconds=30)    # the personality refresh landed
        for i in range(3):
            await m.bid(_want(i + 1), NOW + timedelta(seconds=40 + i))
        assert calls == [] and m._weights is w0           # the bids kept the last weights
        assert m.personality.kicks == 0                   # and never started a personality read either
        await m.reconcile_weights(NOW + timedelta(seconds=45))
        assert len(calls) == 1 and m._weights[0][2] == m.personality.at

    asyncio.run(go())


def test_the_refresher_reads_personality_then_history_then_reconciles():
    async def go():
        m = _market(_state().model_dump_json(), _Pool(acts=_heavy_acts(50)), [])
        p = _Personality()
        order = []

        def kick(now):
            order.append("personality")
            p.kicks += 1

            async def read():
                p.at = now
            return asyncio.ensure_future(read())

        p.kick = kick
        m.personality = p
        real_refresh, real_rec = m._refresh, m.reconcile_weights

        async def refresh(now):
            order.append("history")
            return await real_refresh(now)

        async def reconcile(now):
            order.append("reconcile")
            return await real_rec(now)

        m._refresh, m.reconcile_weights = refresh, reconcile
        stop = asyncio.Event()
        task = asyncio.create_task(m.run_refresher(stop))
        for _ in range(200):
            if "reconcile" in order:
                break
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(task, 2.0)
        assert order[:3] == ["personality", "history", "reconcile"]
        assert m._weights[0][2] == p.at                   # the history's replay used the fresh personality

    asyncio.run(go())


def test_pack_and_run_packed_give_the_same_weights_without_act_costs():
    pool = _Pool(acts=_heavy_acts(300))
    acts = [pm.act_from_row(r) for r in pool.acts]
    assert all(a.cost for a in acts)
    direct = ww.compute_weights(FAMS, acts, [], NOW, None)
    packed = ww.pack(FAMS, acts, [], NOW, None)
    assert len(packed[1]) == -(-len(acts) // ww.CHUNK)  # chunked, not one big pickle
    assert ww.run_packed(ww.compute_weights, packed) == direct


def test_a_dead_worker_is_dropped_and_the_next_replay_starts_a_new_one():
    import concurrent.futures

    class _Broken(concurrent.futures.Executor):
        shut = False

        def submit(self, fn, *a, **k):
            raise concurrent.futures.process.BrokenProcessPool("worker died")

        def shutdown(self, wait=True, *, cancel_futures=False):
            _Broken.shut = True

    async def go():
        m = _market(_state().model_dump_json(), _Pool(acts=_heavy_acts(20)), [])
        made = []

        def factory():
            made.append(1)
            return _Broken() if len(made) == 1 else concurrent.futures.ThreadPoolExecutor(1)

        m._executor_factory = factory
        d = await m.bid(_want(0), NOW)                     # the boot refresh's replay hits the dead worker
        assert d.reason == "physiology_unavailable" and m._weights is None
        assert _Broken.shut and m._executor is None        # dropped, not reused
        d = await m.bid(_want(1), NOW + timedelta(seconds=1))
        assert d.reason != "physiology_unavailable" and len(made) == 2 and m._weights is not None
        await m.aclose()

    asyncio.run(go())


def test_a_bid_builds_only_its_own_familys_appetite_and_decides_the_same():
    from orion.physiology.choice import ReservationBook

    acts = [pm.act_from_row(r) for r in _heavy_acts(100)]
    st = _state()
    full = build_context(FAMS, acts, [], st, NOW)
    one = build_context(FAMS, acts, [], st, NOW, only_families=("durable.self_inquiry",))
    assert set(one.appetite) == {"durable.self_inquiry"} and len(full.appetite) > 1
    assert one.appetite["durable.self_inquiry"] == full.appetite["durable.self_inquiry"]
    want = _want(1)
    a = decide_priced(want, st, full, ReservationBook(), NOW)
    b = decide_priced(want, st, one, ReservationBook(), NOW)
    assert a == b


def test_boot_does_not_wait_on_a_slow_personality_read():
    """At boot the history snapshot comes first: a cold personality read (empty embedding cache) must not
    leave bids without a snapshot. Later cycles wait for it (bounded), so the history replay uses it."""
    async def go():
        m = _market(_state().model_dump_json(), _Pool(acts=_heavy_acts(50)), [])
        p = _Personality()
        gate = asyncio.Event()

        def kick(now):
            async def read():
                await gate.wait()
                p.at = now
            return asyncio.ensure_future(read())

        p.kick = kick
        m.personality = p
        stop = asyncio.Event()
        task = asyncio.create_task(m.run_refresher(stop))
        for _ in range(200):
            if m._cache is not None:
                break
            await asyncio.sleep(0.01)
        assert m._cache is not None and m._weights is not None      # the snapshot did not wait
        gate.set()
        stop.set()
        await asyncio.wait_for(task, 2.0)

    asyncio.run(go())


def test_a_stuck_worker_is_killed_and_the_next_replay_gets_a_new_one(monkeypatch):
    import concurrent.futures

    class _Stuck(concurrent.futures.Executor):
        killed = False

        def submit(self, fn, *a, **k):
            return concurrent.futures.Future()             # never completes

        def shutdown(self, wait=True, *, cancel_futures=False):
            _Stuck.killed = True

    async def go():
        monkeypatch.setattr(pm, "REFRESH_TIMEOUT_SEC", 0.2)
        m = _market(_state().model_dump_json(), _Pool(acts=_heavy_acts(20)), [])
        await m.bid(_want(0), NOW)                         # boot weights, in a thread (no factory yet)
        made = []

        def factory():
            made.append(1)
            return _Stuck() if len(made) == 1 else concurrent.futures.ThreadPoolExecutor(1)

        m._executor_factory = factory
        m._weights = ((NOW - timedelta(seconds=1), NOW.date(), None), *m._weights[1:])   # the key moved
        await m.reconcile_weights(NOW)                     # times out: kept the last weights
        assert _Stuck.killed and m._executor is None and m.counts["weights_refresh_failed"] == 1
        await m.reconcile_weights(NOW)                     # a new worker
        assert len(made) == 2 and m._weights[0][0] == m._cache.at
        await m.aclose()

    asyncio.run(go())


def test_no_choices_through_a_real_worker_means_no_holdback_weights_and_health_shows_the_key():
    async def go():
        m = _market(_state().model_dump_json(), _Pool(acts=_heavy_acts(50)), [])
        m._executor_factory = ww.process_executor
        try:
            await m.bid(_want(0), NOW)
            ctx = await m.context(_state(), NOW, family="durable.self_inquiry")
            assert ctx.weights_holdback is None and m._weights[1] is m._weights[2]
            h = m.health(NOW)
            assert h["weights_history_at"] == m._cache.at.isoformat() and h["weights_day"] == NOW.date().isoformat()
            assert h["weights_personality_at"] is None and h["weights_worker"] is True
        finally:
            await m.aclose()

    asyncio.run(go())


def test_an_unknown_or_non_bidding_family_still_has_no_appetite_and_decides_the_same():
    from orion.physiology.choice import ReservationBook

    acts = [pm.act_from_row(r) for r in _heavy_acts(30)]
    st = _state()
    full = build_context(FAMS, acts, [], st, NOW)
    non_bidding = [f for f, c in FAMS.families.items() if not c.bids]
    for fam in ["no.such_family", *non_bidding[:1]]:
        one = build_context(FAMS, acts, [], st, NOW, only_families=(fam,))
        assert one.appetite == {}
        want = _want(9, family=fam)
        a = decide_priced(want, st, full, ReservationBook(), NOW)
        b = decide_priced(want, st, one, ReservationBook(), NOW)
        assert a == b, fam


def test_kill_terminates_a_busy_worker_process(tmp_path, monkeypatch):
    busy_replay, marker = _busy_replay(tmp_path, monkeypatch)

    async def go():
        ex = ww.process_executor()
        fut = asyncio.get_running_loop().run_in_executor(ex, busy_replay, FAMS, [], [], NOW, None)
        for _ in range(1000):
            if marker.exists():
                break
            await asyncio.sleep(0.01)
        procs = list(ex._processes.values())
        assert procs and all(p.is_alive() for p in procs)
        ww.kill(ex)
        for _ in range(100):
            if not any(p.is_alive() for p in procs):
                break
            await asyncio.sleep(0.01)
        assert not any(p.is_alive() for p in procs)        # killed, well before BUSY_SEC ran out
        await asyncio.gather(fut, return_exceptions=True)

    asyncio.run(go())
