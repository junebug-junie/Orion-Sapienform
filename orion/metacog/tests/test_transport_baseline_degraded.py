"""Degraded replies never teach the transport baseline a speed (2026-10-13).

Live shape being guarded: 2026-10-08, cortex-orch -> orion:cortex:exec:request:background.
A slow hop (~8-15 s real calls, a handful per hour) suddenly carried ~60 calls/hour that
were gpu_pool_unavailable refusals answered in ~0.3-1 s. Counted as successes they
dragged the floor 13.9 s -> 1.26 s; real calls then read ~6x "saturated" for days.
"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone

from orion.metacog.transport_baseline import (
    TransportBaselineConfig,
    fold_snapshot,
    load_state,
    new_state,
    state_key,
    state_to_dict,
)

T0 = datetime(2026, 10, 8, 0, 0, 0, tzinfo=timezone.utc)
WIN = 30.0
HOP = "orion:cortex:exec:request:background"
SERVICE, INSTANCE = "cortex-orch", "main"
CFG = TransportBaselineConfig()
KEY = state_key(SERVICE, INSTANCE, HOP)


def _stats(latencies_ms: list[float], *, timeouts: int = 0, degraded: int = 0) -> dict:
    logs = [math.log(x) for x in latencies_ms]
    out = {
        "success_count": len(latencies_ms),
        "timeout_count": timeouts,
        "log_ms_sum": sum(logs),
        "log_ms_sumsq": sum(v * v for v in logs),
        "max_ms": max(latencies_ms) if latencies_ms else None,
    }
    if degraded:
        out["degraded_count"] = degraded
    return out


def _snap(i: int, cl: dict) -> dict:
    start = T0 + timedelta(seconds=WIN * i)
    return {
        "service": SERVICE,
        "instance": INSTANCE,
        "window_start": start.isoformat(),
        "window_end": (start + timedelta(seconds=WIN)).isoformat(),
        "success_count": 0,
        "timeout_count": 0,
        "channel_latency": cl,
    }


def _fold(state, i: int, stats: dict):
    return fold_snapshot(state, _snap(i, {HOP: stats}), config=CFG)


def _warm_slow_hop(state, rng: random.Random, *, center_ms: float = 12000.0, windows: int = 40) -> int:
    for i in range(windows):
        _fold(state, i, _stats([center_ms * math.exp(rng.gauss(0, 0.15)) for _ in range(5)]))
    return windows


def _shed(state, rng: random.Random, start: int, *, windows: int, as_success: bool) -> tuple[list, list]:
    """One real ~9 s call every other window amid 30 fast refusals per window.
    ``as_success`` replays the pre-fix producer (refusals recorded as successes)."""
    events, obs = [], []
    for k in range(windows):
        real = [9000.0 * math.exp(rng.gauss(0, 0.1))] if k % 2 == 0 else []
        refusals = [rng.uniform(300.0, 1000.0) for _ in range(30)]
        stats = _stats(real + refusals) if as_success else _stats(real, degraded=len(refusals))
        res = _fold(state, start + k, stats)
        events.extend(res.events)
        obs.extend(res.observations)
    return events, obs


def test_a_burst_of_degraded_replies_cannot_drop_the_floor() -> None:
    rng = random.Random(8)
    state = new_state(CFG)
    i = _warm_slow_hop(state, rng)
    floor_before = math.exp(state.keys[KEY].floor)
    assert 10000.0 < floor_before < 14000.0

    _shed(state, rng, i, windows=600, as_success=False)  # 5 hours
    floor_after = math.exp(state.keys[KEY].floor)
    assert floor_after >= 0.95 * floor_before, (floor_before, floor_after)


def test_the_same_burst_counted_as_successes_reproduces_the_live_collapse() -> None:
    """Control: the test above bites. Without the exclusion the floor collapses ~10x,
    as it did live on 2026-10-08."""
    rng = random.Random(8)
    state = new_state(CFG)
    i = _warm_slow_hop(state, rng)
    floor_before = math.exp(state.keys[KEY].floor)
    _shed(state, rng, i, windows=600, as_success=True)
    assert math.exp(state.keys[KEY].floor) < 0.2 * floor_before


def test_after_the_shed_normal_traffic_reads_calm() -> None:
    rng = random.Random(9)
    state = new_state(CFG)
    i = _warm_slow_hop(state, rng)
    events, _ = _shed(state, rng, i, windows=600, as_success=False)
    i += 600
    obs = []
    for k in range(40):
        res = _fold(state, i + k, _stats([12000.0 * math.exp(rng.gauss(0, 0.15)) for _ in range(5)]))
        events.extend(res.events)
        obs.extend(o for o in res.observations if o.evaluated)
    ratios = [o.saturation_ratio for o in obs]
    assert 0.8 <= sorted(ratios)[len(ratios) // 2] <= 1.3, ratios
    assert not [e for e in events if e.condition in ("saturation", "regime_shift")]


def test_degraded_replies_open_no_condition_and_are_reported_per_window() -> None:
    rng = random.Random(3)
    state = new_state(CFG)
    i = _warm_slow_hop(state, rng)
    events = []
    for k in range(60):
        res = _fold(state, i + k, _stats([], degraded=20))
        events.extend(res.events)
        (ob,) = res.observations
        assert ob.degraded_count == 20 and ob.success_count == 0 and not ob.evaluated
        assert ob.calls_per_min == 40.0  # delivered calls still count as load
    assert events == []  # not a timeout, not zero_success: transport delivered


def test_a_degraded_only_window_breaks_the_traffic_time_chain() -> None:
    rng = random.Random(4)
    state = new_state(CFG)
    i = _warm_slow_hop(state, rng)
    ks = state.keys[KEY]
    _fold(state, i, _stats([11000.0]))
    _fold(state, i + 1, _stats([11000.0]))
    assert ks.active_s > 0
    _fold(state, i + 2, _stats([], degraded=5))
    assert ks.active_s == 0.0 and ks.last_success_ts == 0.0


def test_a_pool_with_degraded_replies_is_tainted_and_holds_the_floor_down() -> None:
    rng = random.Random(5)
    state = new_state(CFG)
    i = _warm_slow_hop(state, rng)
    ks = state.keys[KEY]
    fast_count, fast_mean, floor = ks.fast_count, ks.fast_mean, ks.floor
    # Real survivors read faster than normal (a lighter load while the pool sheds):
    res = _fold(state, i, _stats([4000.0] * 5, degraded=1))
    assert res.observations[0].evaluated
    assert ks.fast_count == fast_count and ks.fast_mean == fast_mean  # no fast fold
    assert ks.floor == floor  # held: not evidence of a new best speed
    assert ks.level_mean < fast_mean  # the level itself is unguarded and still moves
    # The very same window with no degraded reply moves the floor down.
    res = _fold(state, i + 1, _stats([4000.0] * 5))
    assert ks.floor < floor


def test_the_taint_survives_pooling_across_windows() -> None:
    rng = random.Random(6)
    state = new_state(CFG)
    i = _warm_slow_hop(state, rng)
    ks = state.keys[KEY]
    floor = ks.floor
    _fold(state, i, _stats([4000.0, 4000.0]))
    _fold(state, i + 1, _stats([], degraded=3))  # degraded-only window inside the pool
    assert ks.pend_degraded and ks.pend_tainted
    _fold(state, i + 2, _stats([4000.0] * 3))  # pool reaches min_calls
    assert ks.floor == floor
    assert not ks.pend_degraded  # reset with the pool


def test_snapshot_without_degraded_count_folds_exactly_as_before() -> None:
    rng = random.Random(7)
    a, b = new_state(CFG), new_state(CFG)
    for i in range(30):
        lat = [1000.0 * math.exp(rng.gauss(0, 0.3)) for _ in range(6)]
        with_zero = dict(_stats(lat), degraded_count=0)
        ra = fold_snapshot(a, _snap(i, {HOP: _stats(lat)}), config=CFG)
        rb = fold_snapshot(b, _snap(i, {HOP: with_zero}), config=CFG)
        assert ra.observations == rb.observations
    assert state_to_dict(a) == state_to_dict(b)


def test_pend_degraded_round_trips_and_an_older_checkpoint_loads_without_it() -> None:
    rng = random.Random(1)
    state = new_state(CFG)
    i = _warm_slow_hop(state, rng)
    _fold(state, i, _stats([9000.0], degraded=2))  # pool open, degraded seen
    dumped = json.loads(json.dumps(state_to_dict(state)))
    assert dumped["keys"][KEY]["pend_degraded"] is True
    loaded, reason = load_state(dumped, CFG)
    assert reason is None and loaded.keys[KEY].pend_degraded is True

    del dumped["keys"][KEY]["pend_degraded"]  # a checkpoint written before 2026-10-13
    loaded, reason = load_state(dumped, CFG)
    assert reason is None and loaded.keys[KEY].pend_degraded is False

    dumped["keys"][KEY]["pend_degraded"] = "yes"
    _, reason = load_state(dumped, CFG)
    assert reason and reason.startswith("malformed_state")
