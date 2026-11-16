"""Issue #3382, second cause: the 30-day weights replay must not slow the bids.

Live 2026-11-16 07:00Z: a personality read changed the weights key, the first bid of a camera-ask burst started
the replay in a thread, and its ~3.3 s of pure-Python CPU work held the GIL: bids that take ~0.25 s took
0.45-0.70 s and missed sql-writer's 0.5 s budget. Now the replay runs in a worker process and only the refresher
starts one. A bid reads the appetite of its own family only."""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path

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
    async def go():
        out = []
        for i in range(n):
            t0 = time.perf_counter()
            d = await m.bid(_want(start + i), NOW + timedelta(minutes=2, milliseconds=i))
            out.append(time.perf_counter() - t0)
            assert d.reason != "physiology_unavailable", (d.reason, d.why)
        return out
    return go()


def test_a_cpu_bound_replay_in_the_worker_process_does_not_slow_the_bids(tmp_path, monkeypatch):
    busy_replay, marker = _busy_replay(tmp_path, monkeypatch)

    async def go():
        m = _market(_state().model_dump_json(), _Pool(acts=_heavy_acts()), [])
        m._executor_factory = ww.process_executor
        try:
            await m.bid(_want(0), NOW)                     # boot: the first weights, replayed in the worker
            assert m._weights is not None and m._executor is not None
            base = await _bid_times(m, 100, 5)
            m._replay = busy_replay
            m.personality = _Personality()                 # a new personality read: the key moved
            m.personality.at = NOW + timedelta(seconds=1)
            rtask = asyncio.create_task(m.reconcile_weights(NOW + timedelta(minutes=2)))
            for _ in range(1000):                          # wait until the worker is inside the busy replay
                if marker.exists():
                    break
                await asyncio.sleep(0.01)
            assert marker.exists(), "the replay never started in the worker"
            during = await _bid_times(m, 200, 5)
            assert not rtask.done(), "the replay finished before the bids: they did not overlap"
            await rtask
            assert m._weights[0][2] == m.personality.at   # the refresher's replay landed
        finally:
            await m.aclose()
        return base, during

    base, during = asyncio.run(go())
    # In a thread the busy replay shares the GIL with the loop and roughly doubles each bid; in the worker
    # process it does not touch the loop's GIL.
    bound = statistics.median(base) * 1.5 + 0.03
    assert max(during) < bound, (base, during)


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
