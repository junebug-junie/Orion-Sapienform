"""Switchback runner lifecycle (app/experiment_runner.py): guard, blocks + arms, heat abort, restart
safety, double-run refusal, and the end-to-end analysis on synthetic data with a planted effect."""
from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import pandas as pd
import pytest

from app.causal import CausalError, design_switchback
from app.effects import Effects
from app.experiment_runner import (
    GPU_SPEC,
    LOCK_KEY,
    RUNNER_SOURCE,
    ExperimentRunner,
    RunnerRefused,
    _ceil_to,
    _ts,
    override_key,
    recover,
    stream_of,
)
from app.series import parse_registry
from tests.helpers import T0, daily_series, points_of, spec_doc
from tests.test_engine import Clock, FakeStore, _engine, fast_zoo  # noqa: F401  (fixture)

ACTION = "vision.focus:walkway"
KEY = override_key("walkway")
GUARD_T = {"name": "experiment_guard", "value": 32.0, "window": "30min", "max_probability": 0.05}


class MemStore(FakeStore):
    def __init__(self, points):
        super().__init__(points)
        self.cov_raw = {}
        self.exps, self.ivs, self.effect_rows = {}, [], []

    def change_events(self):
        return []

    def insert_experiment(self, row):
        self.exps[row["experiment_id"]] = {**row, "status": "designed"}

    def experiment(self, eid):
        return self.exps.get(eid)

    def set_experiment_status(self, eid, status):
        self.exps[eid]["status"] = status

    def insert_intervention(self, row):
        row = {"intervention_id": f"iv{len(self.ivs)}", "arm": None, "experiment_id": None, **row}
        self.ivs.append(row)
        return row

    def interventions(self, action=None, experiment_id=None, limit=500):
        rows = [r for r in self.ivs if (action is None or r["action"] == action)
                and (experiment_id is None or r["experiment_id"] == experiment_id)]
        return sorted(rows, key=lambda r: r["started_at"], reverse=True)[:limit]

    def insert_effect(self, row):
        row = {"estimate_id": f"e{len(self.effect_rows)}", **row}
        self.effect_rows.append(row)
        return row


class FakeRedis:
    """SET NX/XX/EX, GET, DELETE, TTL against the test clock."""

    def __init__(self, clock):
        self.clock, self.data = clock, {}

    def _live(self, k):
        v = self.data.get(k)
        if v and v[1] is not None and v[1] <= self.clock():
            del self.data[k]
            return None
        return v

    async def set(self, k, value, nx=False, xx=False, ex=None):
        cur = self._live(k)
        if (nx and cur) or (xx and not cur):
            return None
        self.data[k] = (value, self.clock() + timedelta(seconds=ex) if ex else None)
        return True

    async def get(self, k):
        v = self._live(k)
        return v[0] if v else None

    async def delete(self, k):
        return 1 if self.data.pop(k, None) else 0

    def ttl(self, k):
        v = self._live(k)
        return None if not v else (v[1] - self.clock()).total_seconds() if v[1] else -1


class FakeProbe:
    """Router /healthz: +10 dispatches per snapshot, on the experiment tier while the key is live."""

    def __init__(self, redis):
        self.redis, self.counts = redis, {}

    def snapshot(self, stream_id):
        k = "triggered:experiment" if self.redis._live(KEY) else "baseline:-"
        self.counts[k] = self.counts.get(k, 0) + 10
        return {"tier_counts": dict(self.counts), "override_active": bool(self.redis._live(KEY)),
                "override_enabled": True}


async def _direct(fn, *a):
    return fn(*a)


def _world(effect=0.0, gpu_effect=0.0, seed=7, guard=None, hot_at=None, noise=0.05):
    """Five days of daily-rhythm cabinet data; the run starts on day 4. Returns the runner pieces."""
    reg = parse_registry(spec_doc(thresholds=[GUARD_T]))
    s = daily_series(5, amp=1.0, noise=noise, seed=seed)
    now = T0 + timedelta(days=4, minutes=1)
    clock = Clock(now)
    start_at = _ceil_to(now + timedelta(minutes=2), 5)
    design = design_switchback(start_at, 20, 12, seed)
    g = daily_series(5, amp=0.0, noise=3.0, seed=seed + 1) + 72   # GPU watts around 100
    for b in design["blocks"]:
        if b["arm"] == "on":
            m = (s.index >= b["start"]) & (s.index < b["end"])
            s[m] += effect
            g[m] += gpu_effect
    if hot_at is not None:
        s[s.index >= hot_at] = 33.0
    store = MemStore(points_of(s, jitter_sec=150))
    store.cov_raw = {GPU_SPEC.series_id: points_of(g, jitter_sec=150)}
    fx = Effects(_engine(reg, store, clock, []))
    redis = FakeRedis(clock)
    calls = {"guard": 0}

    def _guard(spec):
        calls["guard"] += 1
        return guard(clock()) if guard else {"safe": True, "probability": 0.0, "reason": "ok"}

    async def sleep(dt):
        clock.t = clock.t + timedelta(seconds=dt)
        await asyncio.sleep(0)

    runner = ExperimentRunner(fx, redis, FakeProbe(redis), override_ttl_sec=120, tick_sec=60,
                              guard_every_sec=300, sleep=sleep, to_thread=_direct, guard=_guard)
    return runner, fx, store, redis, clock, design, calls


async def _run(runner, **kw):
    out = await runner.start(ACTION, "test.temp", 20, 12, 10, kw.pop("seed", 7), 2)
    await runner._task
    return out


def test_only_vision_focus_actions_have_an_actuator():
    assert stream_of("vision.focus:walkway") == "walkway"
    with pytest.raises(CausalError):
        stream_of("gpu_burn")


def test_guard_refusal_starts_nothing():
    runner, fx, store, redis, *_ = _world(guard=lambda now: {"safe": False, "reason": "too likely to overheat"})
    with pytest.raises(RunnerRefused) as exc:
        asyncio.run(runner.start(ACTION, "test.temp", 20, 12, 10, 7, 2))
    assert exc.value.status == 409 and not store.exps and not redis.data


def test_full_run_logs_every_block_with_its_arm_and_finds_the_planted_effect():
    runner, fx, store, redis, clock, design, _ = _world(effect=0.6, gpu_effect=25.0)
    out = asyncio.run(_run(runner))
    eid = out["experiment"]["experiment_id"]
    logged = sorted((r for r in store.ivs if r["arm"]), key=lambda r: r["started_at"])
    assert [r["arm"] for r in logged] == [b["arm"] for b in design["blocks"]]
    assert all(r["experiment_id"] == eid and r["source"] == RUNNER_SOURCE for r in logged)
    on_notes = [json.loads(r["notes"]) for r in logged if r["arm"] == "on"]
    assert all(n["router_override_active_at_end"] and n["router_dispatch_delta"] == {"triggered:experiment": 10}
               for n in on_notes)
    st = runner.status()
    assert st["status"] == "completed" and not st["running"] and st["blocks_logged"] == 12
    a, m = st["result"]["analysis"], st["result"]["manipulation"]
    assert a["claim_kind"] == "experimental_effect" and a["ci_lo"] <= 0.6 <= a["ci_hi"] and a["p_value"] < 0.05
    assert store.exps[eid]["status"] == "analyzed"
    assert m["verdict"] == "tier_raised" and m["router"]["off"]["triggered_fraction"] == 0.0
    gpu = m["watts"][GPU_SPEC.series_id]
    assert gpu["ci_lo"] <= 25.0 <= gpu["ci_hi"] and gpu["p_value"] < 0.05
    assert "walkway" in st["result"]["summary"]
    assert KEY not in redis.data and LOCK_KEY not in redis.data


def test_zero_effect_reads_as_no_effect():
    runner, *_ = _world(effect=0.0, seed=11)
    asyncio.run(_run(runner, seed=11))
    a = runner.status()["result"]["analysis"]
    assert a["ci_lo"] <= 0.0 <= a["ci_hi"]


def test_key_is_held_with_a_ttl_and_lapses_if_the_runner_dies():
    runner, fx, store, redis, clock, design, _ = _world()
    first_on = next(b for b in design["blocks"] if b["arm"] == "on")

    async def go():
        await runner.start(ACTION, "test.temp", 20, 12, 10, 7, 2)
        while not redis._live(KEY):
            await asyncio.sleep(0)
        assert 0 < redis.ttl(KEY) <= 120 and clock() >= pd.Timestamp(first_on["start"])
        runner._task.cancel()   # stand-in for a crash: stop the loop ...
        try:
            await runner._task
        except asyncio.CancelledError:
            pass
    asyncio.run(go())
    # ... and even if nothing had cleaned up, an un-refreshed key expires by itself.
    redis.data[KEY] = ("on", clock() + timedelta(seconds=120))
    clock.t = clock.t + timedelta(seconds=121)
    assert redis._live(KEY) is None


def test_heat_crossing_the_hot_line_aborts_and_clears_focus():
    hot_at = T0 + timedelta(days=4, hours=1)
    runner, fx, store, redis, clock, design, _ = _world(hot_at=hot_at)
    out = asyncio.run(_run(runner))
    eid = out["experiment"]["experiment_id"]
    st = runner.status()
    assert st["status"] == "aborted" and "hot line" in st["abort_reason"] and st["result"] is None
    assert store.exps[eid]["status"] == "cancelled"
    abort = [r for r in store.ivs if r["arm"] is None]
    assert len(abort) == 1 and abort[0]["experiment_id"] == eid and "aborted" in json.loads(abort[0]["notes"])
    assert all(r["ended_at"] <= hot_at + timedelta(minutes=1) for r in store.ivs)
    assert KEY not in redis.data and LOCK_KEY not in redis.data


def test_guard_breach_mid_run_aborts():
    t_bad = T0 + timedelta(days=4, hours=2)
    runner, fx, store, redis, *_ = _world(
        guard=lambda now: {"safe": now < t_bad, "probability": 0.0 if now < t_bad else 0.2, "reason": "x"})
    asyncio.run(_run(runner))
    st = runner.status()
    assert st["status"] == "aborted" and "guard breached" in st["abort_reason"] and KEY not in redis.data


def test_guard_runs_before_every_on_block():
    runner, fx, store, redis, clock, design, calls = _world()
    asyncio.run(_run(runner))
    n_on = sum(b["arm"] == "on" for b in design["blocks"])
    assert calls["guard"] >= 1 + n_on   # the start check plus one per ON block (plus periodic ones)


def test_second_start_is_refused_while_running_and_when_another_process_holds_the_lock():
    runner, fx, store, redis, *_ = _world()

    async def go():
        await runner.start(ACTION, "test.temp", 20, 12, 10, 7, 2)
        with pytest.raises(RunnerRefused) as exc:
            await runner.start(ACTION, "test.temp", 20, 12, 10, 7, 2)
        assert exc.value.status == 409
        await runner.shutdown("test over")
    asyncio.run(go())
    assert runner.status()["status"] == "aborted" and KEY not in redis.data
    asyncio.run(redis.set(LOCK_KEY, "someone-else", ex=100))
    with pytest.raises(RunnerRefused, match="another runner"):
        asyncio.run(runner.start(ACTION, "test.temp", 20, 12, 10, 7, 2))
    assert len(store.exps) == 1


def test_lost_lock_aborts():
    runner, fx, store, redis, clock, *_ = _world()

    async def go():
        await runner.start(ACTION, "test.temp", 20, 12, 10, 7, 2)
        await asyncio.sleep(0)
        await redis.delete(LOCK_KEY)
        await runner._task
    asyncio.run(go())
    assert "lost the runner lock" in runner.status()["abort_reason"]


def test_operator_abort_records_and_cancels():
    runner, fx, store, redis, *_ = _world()

    async def go():
        out = await runner.start(ACTION, "test.temp", 20, 12, 10, 7, 2)
        for _ in range(50):
            await asyncio.sleep(0)
        assert await runner.shutdown("aborted by operator")
        assert not await runner.shutdown("again")
        return out["experiment"]["experiment_id"]
    eid = asyncio.run(go())
    assert store.exps[eid]["status"] == "cancelled" and runner.status()["abort_reason"] == "aborted by operator"


def test_recover_after_restart_cancels_the_half_done_run_and_clears_its_key():
    runner, fx, store, redis, clock, design, _ = _world()
    row = fx.design_experiment(ACTION, "test.temp", _ts(design["blocks"][0]["start"]), 20, 12, 10, 7)
    eid = row["experiment_id"]
    for b in row["design"]["blocks"][:3]:
        fx.log_intervention(ACTION, _ts(b["start"]), _ts(b["end"]), RUNNER_SOURCE, "{}", b["arm"], eid)
    asyncio.run(redis.set(LOCK_KEY, eid, ex=100))
    asyncio.run(redis.set(KEY, "on", ex=100))
    assert asyncio.run(recover(fx, redis, _direct)) == [eid]
    assert store.exps[eid]["status"] == "cancelled"
    assert any(r["arm"] is None and "interrupted" in r["notes"] for r in store.ivs)
    assert KEY not in redis.data and LOCK_KEY not in redis.data
    assert asyncio.run(recover(fx, redis, _direct)) == []   # idempotent


def test_recover_leaves_a_finished_but_unanalyzed_run_alone():
    runner, fx, store, redis, clock, design, _ = _world()
    row = fx.design_experiment(ACTION, "test.temp", _ts(design["blocks"][0]["start"]), 20, 12, 10, 7)
    for b in row["design"]["blocks"]:
        fx.log_intervention(ACTION, _ts(b["start"]), _ts(b["end"]), RUNNER_SOURCE, "{}", b["arm"], row["experiment_id"])
    assert asyncio.run(recover(fx, redis, _direct)) == []
    assert store.exps[row["experiment_id"]]["status"] == "designed"


def test_blocks_start_on_a_grain_boundary():
    runner, fx, store, redis, clock, design, _ = _world()
    out = asyncio.run(_run(runner))
    starts = [b["start"] for b in out["experiment"]["design"]["blocks"]]
    assert all(pd.Timestamp(s).timestamp() % 300 == 0 for s in starts)


# --- HTTP -------------------------------------------------------------------------------------------
def test_http_run_refusals_and_status(monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main
    runner, fx, store, redis, *_ = _world(guard=lambda now: {"safe": False, "reason": "too likely to overheat"})
    monkeypatch.setattr(main, "engine", fx.engine)
    monkeypatch.setattr(main, "runner", None)
    c = TestClient(main.app)   # no context manager: lifespan (bus, DB) does not run
    assert c.post("/experiments/run", json={"action": ACTION}).status_code == 503   # flag off
    monkeypatch.setattr(main, "runner", runner)
    assert c.post("/experiments/run", json={"action": "gpu_burn"}).status_code == 422
    r = c.post("/experiments/run", json={"action": ACTION, "series_id": "test.temp"})
    assert r.status_code == 409 and r.json()["detail"]["error"] == "guard refused"
    assert c.get("/experiments/runner").json()["running"] is False
    assert c.post("/experiments/runner/abort").status_code == 409
