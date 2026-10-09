"""Gate tests for EWMA-derived, static-floor-bounded transport thresholds."""
from __future__ import annotations

import math
import random
from pathlib import Path

import yaml

from orion.field import transport_thresholds as tt

REPO = Path(__file__).resolve().parents[1]
STATIC = {"watch_at": 0.25, "summarize_at": 0.5, "propose_at": 0.75}
CFG = tt.ThresholdConfig()
STEP = 30.0


class FakeRedis:
    store: dict = {}

    def hget(self, key, field):
        return self.store.get((key, field))

    def hset(self, key, field, value):
        self.store[(key, field)] = value


def _feed(values, *, start=None, cfg=CFG, state=None, t0=1_000_000.0):
    t = t0
    for v in values:
        state = tt.update_state(state, v, t, cfg)
        t += STEP
    return state, t


import pytest


@pytest.fixture(autouse=True)
def _fresh_cache():
    tt.clear_state_cache()
    yield
    tt.clear_state_cache()


def _calm(n, seed=1):
    rng = random.Random(seed)
    return [max(0.0, rng.gauss(0.046, 0.03)) for _ in range(n)]


def _eff(state, now, cfg=CFG, static=STATIC):
    return tt.effective_thresholds(static, state, cfg, now_ts=now, channel_id="bus_synaptic_pressure")


def test_cold_start_is_unknown_and_static():
    state, t = _feed(_calm(100))
    eff = _eff(state, t)
    assert eff["_meta"]["reason"] == "cold_start"
    assert eff["_meta"]["z_fast"] is None  # unknown, not 0.0
    for r in tt.RUNGS:
        assert eff[r]["source"] == "static" and eff[r]["value"] == STATIC[r]


def test_first_sample_has_no_z():
    s = tt.update_state(None, 0.1, 1.0, CFG)
    assert s.last_z_fast is None and s.n == 1


def test_no_state_is_static():
    eff = _eff(None, 1.0)
    assert eff["_meta"]["reason"] == "no_state"
    assert eff["watch_at"]["value"] == 0.25


def test_warm_calm_channel_triggers_earlier_but_never_later():
    state, t = _feed(_calm(CFG.min_samples + 500))
    eff = _eff(state, t)
    assert eff["_meta"]["reason"] == "derived"
    w = eff["watch_at"]
    assert w["source"] == "derived" and w["value"] < 0.25
    assert w["value"] >= tt.MAX_TIGHTEN_RATIO * 0.25  # bounded tightening
    assert w["window_sec"] == CFG.slow_half_life_sec and w["n_samples"] > CFG.min_samples
    assert eff["watch_at"]["value"] <= eff["summarize_at"]["value"] <= eff["propose_at"]["value"]
    for r in tt.RUNGS:
        assert eff[r]["value"] <= STATIC[r]  # floor


def test_chronic_hot_channel_never_reads_calm():
    # calm for 3 days, then hot at 0.6 for 6 days: baseline adapts, z -> ~0.
    state, t = _feed(_calm(CFG.min_samples + 2000))
    state, t = _feed([0.6] * (6 * 2880), state=state, t0=t)
    eff = _eff(state, t)
    assert abs(state.last_z_fast) < 0.5  # baseline absorbed the heat (z alone would say calm)
    assert eff["_meta"]["chronic_hot"] is True
    # ...but the threshold is still at or below static, so 0.6 still trips every rung it did before.
    for r in ("watch_at", "summarize_at"):
        assert eff[r]["value"] <= STATIC[r]
        assert 0.6 >= eff[r]["value"]


def test_floor_holds_for_random_histories():
    rng = random.Random(7)
    for trial in range(20):
        level = rng.choice([0.0, 0.05, 0.3, 0.9])
        vals = [min(1.0, max(0.0, rng.gauss(level, rng.choice([0.0, 0.02, 0.2])))) for _ in range(CFG.min_samples + 300)]
        state, t = _feed(vals)
        eff = _eff(state, t)
        for r in tt.RUNGS:
            assert eff[r]["value"] <= STATIC[r], (trial, r)


def test_null_static_rung_stays_null():
    state, t = _feed(_calm(CFG.min_samples + 100))
    eff = _eff(state, t, static={"watch_at": 0.5, "summarize_at": 0.75, "propose_at": None})
    assert eff["propose_at"]["value"] is None


def test_stale_state_falls_back_to_static():
    state, t = _feed(_calm(CFG.min_samples + 100))
    eff = _eff(state, t + CFG.max_state_age_sec + 1)
    assert eff["_meta"]["reason"] == "stale_state"
    assert eff["watch_at"]["source"] == "static"


def test_flag_off_is_pure_static():
    state, t = _feed(_calm(CFG.min_samples + 500))
    off = tt.ThresholdConfig(enabled=False)
    eff = _eff(state, t, cfg=off)
    for r in tt.RUNGS:
        assert eff[r]["value"] == STATIC[r] and eff[r]["source"] == "static"


def test_unwired_channels_are_static():
    state, t = _feed(_calm(CFG.min_samples + 500))
    for ch in ("contract_pressure", "observer_failure_pressure", "transport_reliability_pressure"):
        eff = tt.effective_thresholds(STATIC, state, CFG, now_ts=t, channel_id=ch)
        assert eff["_meta"]["reason"] == "channel_not_wired"
        assert eff["watch_at"]["value"] == 0.25


def test_outage_gap_does_not_reset_baseline():
    state, t = _feed(_calm(500))
    before = state.slow_ewma
    after = tt.update_state(state, 0.9, t + 86400, CFG)  # a day-long gap, one hot sample
    assert abs(after.slow_ewma - before) < 0.05


def test_non_finite_sample_ignored():
    s, _ = _feed([0.1, 0.1])
    assert tt.update_state(s, math.nan, 5e6, CFG) == s


def test_env_flag_parsing():
    assert tt.ThresholdConfig.from_env({}).enabled is True
    assert tt.ThresholdConfig.from_env({"TRANSPORT_THRESHOLDS_DERIVED_ENABLED": "false"}).enabled is False
    assert tt.ThresholdConfig.from_env({"TRANSPORT_THRESHOLDS_MIN_SAMPLES": "500"}).min_samples == 500
    # garbage is clamped, cold start cannot be disabled by a bad value
    bad = tt.ThresholdConfig.from_env({
        "TRANSPORT_THRESHOLDS_MIN_SAMPLES": "0", "TRANSPORT_THRESHOLDS_K_WATCH": "-3",
        "TRANSPORT_THRESHOLDS_SLOW_HALF_LIFE_SEC": "0",
    })
    assert bad.min_samples >= 100 and bad.k_watch >= 2.0 and bad.slow_half_life_sec >= 3600.0


def test_redis_roundtrip_producer_to_reader(monkeypatch):
    FakeRedis.store = {}
    monkeypatch.setattr(tt, "_client", lambda url: FakeRedis())
    cfg = tt.ThresholdConfig(min_samples=50)
    t = 2_000_000.0
    for v in _calm(200):
        tt.record_sample("bus_synaptic_pressure", v, "redis://x", cfg, now_ts=t)
        t += STEP
    eff = tt.fetch_effective_thresholds("bus_synaptic_pressure", STATIC, "redis://x", cfg, now_ts=t)
    assert eff["watch_at"]["source"] == "derived"
    off = tt.fetch_effective_thresholds(
        "bus_synaptic_pressure", STATIC, "redis://x", tt.ThresholdConfig(enabled=False), now_ts=t
    )
    assert off["watch_at"]["value"] == 0.25


def test_redis_failure_degrades_to_static(monkeypatch):
    def boom(url):
        raise ConnectionError("down")

    monkeypatch.setattr(tt, "_client", boom)
    eff = tt.fetch_effective_thresholds("bus_synaptic_pressure", STATIC, "redis://x", CFG)
    assert eff["watch_at"]["source"] == "static" and eff["_meta"]["reason"] == "no_state"
    assert tt.record_sample("bus_synaptic_pressure", 0.1, "redis://x", CFG) is None


def test_scale_weight_matches_topology_edge():
    topo = yaml.safe_load((REPO / "config/field/orion_field_topology.v1.yaml").read_text())
    edges = [
        e for e in topo["edges"]
        if e.get("source_id") == "node:substrate.bus_synaptic" and e.get("target_id") == "capability:transport"
    ]
    assert edges and float(edges[0]["weight"]) == tt.DERIVED_CHANNELS["bus_synaptic_pressure"]


def test_wired_channels_exist_in_policy():
    policy = yaml.safe_load((REPO / "config/substrate-lattice/transport_lattice_policy.v1.yaml").read_text())
    assert set(tt.DERIVED_CHANNELS) <= set(policy["channels"])


def test_derived_value_is_independent_of_the_first_sample():
    """Review finding: with a seeded EWMA the first reading kept ~71% weight at
    24 h, so a 0.0 first sample gave a hair-trigger and a 0.5 one gave static."""
    outs = []
    for first in (0.0, 0.5):
        state, t = _feed([first] + _calm(CFG.min_samples + 20))
        eff = _eff(state, t)
        assert eff["_meta"]["reason"] == "derived"
        outs.append((eff["watch_at"]["value"], state.slow_ewma))
    assert abs(outs[0][0] - outs[1][0]) < 0.01
    assert abs(outs[0][1] - outs[1][1]) < 0.01


def test_cold_start_needs_elapsed_time_not_just_sample_count():
    # 3000 samples 1 s apart: count is enough, elapsed (50 min) is not.
    state = None
    for i, v in enumerate(_calm(3000)):
        state = tt.update_state(state, v, 1_000_000.0 + i, CFG)
    eff = _eff(state, 1_000_000.0 + 3000)
    assert eff["_meta"]["reason"] == "cold_start"


def test_clock_skew_reads_stale_not_fresh():
    state, t = _feed(_calm(CFG.min_samples + 100))
    eff = _eff(state, t - 10_000)  # state is "from the future"
    assert eff["_meta"]["reason"] == "stale_state"


def test_reader_caches_state_within_ttl_and_failures_too(monkeypatch):
    calls = {"n": 0}

    class Counting:
        def hget(self, k, f):
            calls["n"] += 1
            raise ConnectionError("down")

    monkeypatch.setattr(tt, "_client", lambda url: Counting())
    for _ in range(5):
        tt.fetch_effective_thresholds("bus_synaptic_pressure", STATIC, "redis://x", CFG)
    assert calls["n"] == 1  # one timeout per TTL, not one per request
