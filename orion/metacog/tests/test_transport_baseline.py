"""Tests for orion/metacog/transport_baseline.py (spec 2026-09-24, A1-A4)."""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from orion.metacog.transport_baseline import (
    TransportBaselineConfig,
    fold_snapshot,
    is_excluded,
    load_state,
    new_state,
    state_key,
    state_to_dict,
)

T0 = datetime(2026, 9, 24, 0, 0, 0, tzinfo=timezone.utc)
WIN = 30.0
HOP = "orion:cortex:exec:request:chat"
CFG = TransportBaselineConfig()


def _stats(latencies_ms: list[float], timeouts: int = 0) -> dict:
    logs = [math.log(x) for x in latencies_ms]
    return {
        "success_count": len(latencies_ms),
        "timeout_count": timeouts,
        "log_ms_sum": sum(logs),
        "log_ms_sumsq": sum(v * v for v in logs),
        "max_ms": max(latencies_ms) if latencies_ms else None,
    }


def _snap(i: int, channel_latency: dict | None, *, service="cortex-exec", instance="chat") -> dict:
    start = T0 + timedelta(seconds=WIN * i)
    payload = {
        "service": service,
        "instance": instance,
        "window_start": start.isoformat(),
        "window_end": (start + timedelta(seconds=WIN)).isoformat(),
        "success_count": 0,
        "timeout_count": 0,
    }
    if channel_latency is not None:
        payload["channel_latency"] = channel_latency
    return payload


def _noisy(rng: random.Random, center_ms: float, n: int = 10, sigma: float = 0.3) -> list[float]:
    return [center_ms * math.exp(rng.gauss(0.0, sigma)) for _ in range(n)]


def _run(state, windows, *, cfg=CFG, exclude=()):
    """windows: iterable of (i, channel_latency). Returns (events, observations)."""
    events, obs = [], []
    for i, cl in windows:
        res = fold_snapshot(state, _snap(i, cl), config=cfg, exclude_labels=exclude)
        events.extend(res.events)
        obs.extend(res.observations)
    return events, obs


def _warm(state, rng, *, center=1000.0, n_windows=40, start=0, cfg=CFG):
    return _run(state, ((i, {HOP: _stats(_noisy(rng, center))}) for i in range(start, start + n_windows)), cfg=cfg)


# ---------------------------------------------------------------- math


def test_first_window_baseline_is_geometric_mean_from_log_sufficient_stats():
    state = new_state(CFG)
    lat = [100.0, 1000.0, 10000.0, 100.0, 1000.0]
    res = fold_snapshot(state, _snap(0, {HOP: _stats(lat)}), config=CFG)
    obs = res.observations[0]
    geo = math.exp(sum(math.log(x) for x in lat) / len(lat))
    assert math.isclose(obs.window_mean_ms, geo, rel_tol=1e-9)
    assert math.isclose(obs.baseline_ms, geo, rel_tol=1e-9)
    assert math.isclose(obs.floor_ms, geo, rel_tol=1e-9)
    assert obs.saturation_ratio == 1.0
    assert obs.z is None  # no baseline yet: not "measured, calm"


def test_snapshot_without_channel_latency_is_skipped():
    state = new_state(CFG)
    res = fold_snapshot(state, _snap(0, None), config=CFG)
    assert res.skipped_reason == "no_channel_latency"
    assert state.keys == {}


def test_state_is_keyed_by_service_instance_hop():
    state = new_state(CFG)
    fold_snapshot(state, _snap(0, {HOP: _stats([10.0] * 5)}, instance="chat"), config=CFG)
    fold_snapshot(state, _snap(0, {HOP: _stats([10.0] * 5)}, instance="background"), config=CFG)
    assert set(state.keys) == {
        state_key("cortex-exec", "chat", HOP),
        state_key("cortex-exec", "background", HOP),
    }


# ---------------------------------------------------------------- warm-up


def test_no_latency_condition_before_warmup():
    state = new_state(CFG)
    rng = random.Random(1)
    events, _ = _run(state, ((i, {HOP: _stats(_noisy(rng, 1000.0))}) for i in range(CFG.n_warm - 1)))
    events2, _ = _run(state, ((i, {HOP: _stats([50000.0] * 10)}) for i in range(CFG.n_warm - 1, CFG.n_warm + 1)))
    assert not [e for e in events + events2 if e.condition in ("spike", "saturation")]


def test_windows_below_min_calls_pool_until_min_calls():
    state = new_state(CFG)
    _, obs = _run(state, ((i, {HOP: _stats([1000.0])}) for i in range(10)))
    evaluated = [o.evaluated for o in obs]
    assert evaluated == [False, False, False, False, True] * 2


def test_sparse_pool_dropped_after_max_aggregate_windows():
    cfg = replace(CFG, max_aggregate_windows=3)
    state = new_state(cfg)
    # one call, then quiet windows with only timeouts elsewhere keep the key known
    _run(state, [(0, {HOP: _stats([1000.0])}), (1, {HOP: _stats([], timeouts=1)}), (2, {}), (3, {})], cfg=cfg)
    ks = state.keys[state_key("cortex-exec", "chat", HOP)]
    assert ks.pend_n == 0 and ks.pend_windows == 0


# ---------------------------------------------------------------- guard 1


def test_spike_needs_two_sustained_windows_and_does_not_update_fast():
    state = new_state(CFG)
    rng = random.Random(2)
    _warm(state, rng)
    ks = state.keys[state_key("cortex-exec", "chat", HOP)]
    before = (ks.fast_mean, ks.fast_var, ks.fast_count)
    ev1, _ = _run(state, [(40, {HOP: _stats([8000.0] * 10)})])
    assert not [e for e in ev1 if e.condition == "spike"]
    assert (ks.fast_mean, ks.fast_var, ks.fast_count) == before  # not learned
    ev2, _ = _run(state, [(41, {HOP: _stats([8000.0] * 10)})])
    spikes = [e for e in ev2 if e.condition == "spike"]
    assert [e.phase for e in spikes] == ["open"]
    assert spikes[0].z >= CFG.spike_z
    assert math.isclose(spikes[0].window_mean_ms, 8000.0)
    assert (ks.fast_mean, ks.fast_var, ks.fast_count) == before


def test_borderline_window_is_clipped_at_mean_plus_2_sigma():
    state = new_state(CFG)
    rng = random.Random(3)
    _warm(state, rng)
    ks = state.keys[state_key("cortex-exec", "chat", HOP)]
    sigma = math.sqrt(max(ks.fast_var, CFG.min_variance))
    mean0 = ks.fast_mean
    target = mean0 + 2.5 * sigma  # 2 <= z < 3
    _run(state, [(40, {HOP: _stats([math.exp(target)] * 10)})])
    clipped = mean0 + 2.0 * sigma
    assert math.isclose(ks.fast_mean, CFG.fast_alpha * clipped + (1 - CFG.fast_alpha) * mean0, rel_tol=1e-9)


def test_timeout_tainted_window_does_not_update_fast():
    state = new_state(CFG)
    rng = random.Random(4)
    _warm(state, rng)
    ks = state.keys[state_key("cortex-exec", "chat", HOP)]
    before = ks.fast_count
    _run(state, [(40, {HOP: _stats(_noisy(rng, 1000.0), timeouts=1)})])
    assert ks.fast_count == before


# ---------------------------------------------------------------- timeouts


def test_timeout_opens_without_minimum_escalates_and_closes_with_duration():
    state = new_state(CFG)
    ev, _ = _run(state, [(0, {HOP: _stats([], timeouts=1)})])
    assert [(e.condition, e.phase, e.timeout_count) for e in ev] == [("timeout", "open", 1)]
    ev, _ = _run(state, [(1, {HOP: _stats([2000.0], timeouts=1)})])
    assert ev == []  # same magnitude: no new row
    ev, _ = _run(state, [(2, {HOP: _stats([3000.0], timeouts=2)})])
    assert [(e.condition, e.phase) for e in ev] == [("timeout", "escalate")]
    # quiet windows: key absent from channel_latency still ages the episode
    quiet = int(CFG.close_quiet_s // WIN)
    ev, _ = _run(state, ((i, {}) for i in range(3, 3 + quiet + 1)))
    closes = [e for e in ev if e.phase == "close"]
    assert [(e.condition) for e in closes] == ["timeout"]
    assert closes[0].duration_s >= CFG.close_quiet_s
    assert closes[0].peak_ms == 3000.0


def test_intermittent_timeouts_inside_quiet_window_are_one_episode():
    state = new_state(CFG)
    ev, _ = _run(state, ((i, {HOP: _stats([100.0] * 5, timeouts=1 if i % 4 == 0 else 0)}) for i in range(40)))
    assert [e.phase for e in ev if e.condition == "timeout"] == ["open"]


def test_zero_success_requires_prior_successful_traffic():
    state = new_state(CFG)
    ev, _ = _run(state, [(0, {HOP: _stats([], timeouts=2)})])
    assert not [e for e in ev if e.condition == "zero_success"]
    state = new_state(CFG)
    rng = random.Random(5)
    _warm(state, rng, n_windows=5)
    ev, _ = _run(state, [(5, {HOP: _stats([], timeouts=3)})])
    zs = [e for e in ev if e.condition == "zero_success"]
    assert [e.phase for e in zs] == ["open"]
    assert zs[0].calls_per_min_usual and zs[0].calls_per_min_usual > 0


# ---------------------------------------------------------------- exclusion


def test_is_excluded_matches_health_label_verb_and_whole_key():
    labels = ["log_orion_metacognition"]
    assert is_excluded("orion:cortex:exec:request:background#log_orion_metacognition", labels)
    assert is_excluded("verb:log_orion_metacognition", labels)
    assert is_excluded("log_orion_metacognition", labels)
    assert not is_excluded("orion:cortex:exec:request:background#chat_general", labels)
    assert not is_excluded("orion:cortex:exec:request:background", labels)
    assert not is_excluded("anything", [])


def test_excluded_key_is_baselined_and_events_are_flagged():
    key = "orion:cortex:exec:request:background#log_orion_metacognition"
    state = new_state(CFG)
    ev, obs = _run(
        state, [(0, {key: _stats([15000.0] * 5, timeouts=1)})], exclude=["log_orion_metacognition"]
    )
    assert ev and all(e.excluded for e in ev)
    assert obs[0].excluded and obs[0].evaluated


# ---------------------------------------------------------------- persistence


def test_state_roundtrips_through_json_and_continues_identically():
    rng_a, rng_b = random.Random(6), random.Random(6)
    a = new_state(CFG)
    _warm(a, rng_a, n_windows=30)
    b = new_state(CFG)
    _warm(b, rng_b, n_windows=15)
    b, reason = load_state(json.loads(json.dumps(state_to_dict(b))), CFG)
    assert reason is None
    _warm(b, rng_b, n_windows=15, start=15)
    assert state_to_dict(a) == state_to_dict(b)


def test_config_fingerprint_mismatch_cold_starts():
    state = new_state(CFG)
    _warm(state, random.Random(7), n_windows=5)
    other = replace(CFG, spike_z=4.0)
    loaded, reason = load_state(state_to_dict(state), other)
    assert loaded.keys == {}
    assert reason and reason.startswith("config_fingerprint_mismatch")
    assert loaded.fingerprint == other.fingerprint()


def test_load_state_never_raises_on_garbage():
    for garbage in (None, "x", {"schema_version": 1}, {"schema_version": 1, "fingerprint": CFG.fingerprint(), "keys": {"k": {"bogus": 1}}}):
        st, reason = load_state(garbage, CFG)
        assert st.keys == {} and reason


def test_out_of_order_window_is_ignored():
    state = new_state(CFG)
    _run(state, [(5, {HOP: _stats([100.0] * 5)})])
    ks = state.keys[state_key("cortex-exec", "chat", HOP)]
    before = state_to_dict(state)
    _run(state, [(2, {HOP: _stats([9000.0] * 5, timeouts=3)})])
    assert state_to_dict(state) == before
    assert ks.fast_count == 1


# ---------------------------------------------------------------- acceptance 7


def _plateau_run(cfg, *, ramp_windows: int, plateau_hours: float, factor: float, seed: int):
    state = new_state(cfg)
    rng = random.Random(seed)
    _warm(state, rng, n_windows=120)
    base = math.log(1000.0)
    i = 120
    events, obs = [], []
    total = ramp_windows + int(plateau_hours * 3600 / WIN)
    for step in range(total):
        frac = min(1.0, (step + 1) / ramp_windows) if ramp_windows else 1.0
        center = math.exp(base + frac * math.log(factor))
        # busier too: load rises with the plateau (evidence only)
        n = 10 + int(20 * frac)
        ev, ob = _run(state, [(i, {HOP: _stats(_noisy(rng, center, n=n))})], cfg=cfg)
        events.extend(ev)
        obs.extend(ob)
        i += 1
    return state, events, obs


def test_acceptance_7_busy_plateau_is_saturation_then_exactly_one_regime_shift():
    _, events, obs = _plateau_run(CFG, ramp_windows=120, plateau_hours=12, factor=2.5, seed=11)
    assert not [e for e in events if e.condition == "spike"], "slow creep must not z-spike"
    sat = [e for e in events if e.condition == "saturation"]
    regime = [e for e in events if e.condition == "regime_shift"]
    assert sat and sat[0].phase == "open"
    assert len(regime) == 1
    # never reported as cleared before the new normal was stated
    assert not [e for e in sat if e.phase == "close"]
    open_idx = next(k for k, e in enumerate(events) if e.condition == "saturation")
    regime_idx = events.index(regime[0])
    assert open_idx < regime_idx
    assert regime[0].duration_s >= CFG.regime_after_s
    assert regime[0].duration_s < CFG.regime_after_s + 120
    assert regime[0].saturation_ratio >= CFG.saturation_close_ratio
    assert regime[0].floor_ms is not None and regime[0].floor_ms < 1300.0  # old normal stated
    # every observation between open and regime_shift shows saturation open
    open_ts_i = next(k for k, o in enumerate(obs) if "saturation" in o.open_conditions)
    regime_obs = [k for k, o in enumerate(obs) if "saturation" not in o.open_conditions and k > open_ts_i]
    first_gap = regime_obs[0]
    assert all("saturation" in o.open_conditions for o in obs[open_ts_i:first_gap])
    # after the stated shift the plateau is the new normal
    tail = obs[-20:]
    assert all(0.8 <= o.saturation_ratio <= 1.3 for o in tail if o.saturation_ratio is not None)
    # load carried as evidence
    assert sat[0].calls_per_min_usual is not None and sat[0].calls_per_min > 0


def test_step_change_is_not_hidden_by_guard_1():
    """A step change freezes `fast` (guard 1); saturation must still see it via
    the unguarded level and state it as a regime shift."""
    cfg = replace(CFG, regime_after_s=3600.0)
    _, events, obs = _plateau_run(cfg, ramp_windows=0, plateau_hours=3, factor=4.0, seed=12)
    conds = [(e.condition, e.phase) for e in events]
    assert ("spike", "open") in conds
    assert ("saturation", "open") in conds
    assert conds.count(("regime_shift", "open")) == 1
    # after the re-seed the spike closes too; nothing is left open
    assert ("spike", "close") in conds
    assert obs[-1].open_conditions == ()


# ---------------------------------------------------------------- rest state


def test_calm_returns_to_rest_after_incident():
    state = new_state(CFG)
    rng = random.Random(13)
    _warm(state, rng, n_windows=240)
    # incident: 10 minutes at 5x, with some timeouts
    inc, _ = _run(
        state,
        ((i, {HOP: _stats(_noisy(rng, 5000.0), timeouts=1 if i % 3 == 0 else 0)}) for i in range(240, 260)),
    )
    assert {"spike", "timeout"} <= {e.condition for e in inc if e.phase == "open"}
    # calm again for 2 hours
    after, obs = _run(state, ((i, {HOP: _stats(_noisy(rng, 1000.0))}) for i in range(260, 500)))
    closes = {e.condition for e in after if e.phase == "close"}
    assert {"spike", "timeout"} <= closes
    tail = obs[-60:]
    zs = sorted(o.z for o in tail)
    ratios = sorted(o.saturation_ratio for o in tail)
    assert abs(zs[len(zs) // 2]) <= 0.5
    assert 0.8 <= ratios[len(ratios) // 2] <= 1.3
    assert obs[-1].open_conditions == ()
    # the incident did not contaminate the baseline
    assert 800.0 <= obs[-1].baseline_ms <= 1250.0


def test_fully_calm_key_never_fires():
    state = new_state(CFG)
    events, obs = _warm(state, random.Random(14), n_windows=2000)
    assert events == []
    zs = sorted(o.z for o in obs[200:])
    assert abs(zs[len(zs) // 2]) < 0.3


def test_floor_is_frozen_upward_while_saturation_is_open():
    """Even with a floor that would otherwise absorb the plateau within hours,
    an open saturation episode cannot be healed by the floor creeping up to
    meet it -- only a stated regime_shift may move it."""
    cfg = replace(CFG, floor_half_life_up_s=1800.0)
    _, events, _ = _plateau_run(cfg, ramp_windows=0, plateau_hours=8, factor=4.0, seed=15)
    sat = [e for e in events if e.condition == "saturation"]
    assert sat and sat[0].phase == "open"
    assert not [e for e in sat if e.phase == "close"]
    assert len([e for e in events if e.condition == "regime_shift"]) == 1


def _ratio_median_after(state, rng, *, center, n_calls, windows, start=0):
    _, obs = _run(state, ((i, {HOP: _stats(_noisy(rng, center, n=n_calls))}) for i in range(start, start + windows)))
    rs = sorted(o.saturation_ratio for o in obs if o.evaluated)
    return rs[len(rs) // 2]


def test_sparse_calm_key_ratio_rests_at_one():
    """Regression (mesh eval, 2026-09-24): a floor following raw window means
    tracked the noise's lower envelope, so a calm pooled 1-call/window key read
    ratio ~1.4 forever. Floor-follows-level must rest near 1."""
    state = new_state(CFG)
    r = _ratio_median_after(state, random.Random(21), center=300.0, n_calls=1, windows=1500)
    assert 0.95 <= r <= 1.1


def test_unlucky_first_window_does_not_seed_the_floor():
    """Regression (mesh eval): a low first draw seeded floor and level; the level
    recovered, the slow-up floor did not (ratio stuck at 1.31)."""
    state = new_state(CFG)
    _run(state, [(0, {HOP: _stats([150.0] * 5)})])  # ~0.6x the true center
    r = _ratio_median_after(state, random.Random(22), center=250.0, n_calls=5, windows=600, start=1)
    assert 0.95 <= r <= 1.1
