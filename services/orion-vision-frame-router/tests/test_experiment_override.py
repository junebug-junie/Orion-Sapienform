"""Experiment override: orion:vision:experiment:<stream_id> forces the triggered tier (2026-10-29).

Written by orion-prediction's switchback runner with a TTL. The off arm is normal
operation: absent / "off" / no TTL / Redis down must all leave today's behaviour alone.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[3], Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from app.expectation import ExpectationCache, expect_key, experiment_key
from app.metrics import RouterMetrics, make_health_details
from app.state import RouterState
from test_walkway_policy_and_expectation import _FakeRedis, _HangingRedis, _decide, _policy

ON = {experiment_key("walkway"): (b"on", 60_000)}


def _cache(values=None, keys=None, **kw) -> ExpectationCache:
    cache = ExpectationCache(refresh_sec=5, **kw)
    cache.note_stream("walkway")
    asyncio.run(cache.refresh_once(_FakeRedis(keys or set(), values=values)))
    return cache


def test_on_forces_triggered_tier_attributed_to_experiment() -> None:
    cache = _cache(ON)
    assert cache.experiment_on("walkway") and cache.experiment_streams() == ["walkway"]
    d = _decide(_policy(cache), RouterState())
    assert d.dispatch_tier == "triggered" and d.triggered_by == "experiment"


def test_experiment_wins_attribution_over_labels_and_expectation() -> None:
    cache = _cache(ON, keys={expect_key("walkway")})
    state = RouterState()
    state.record_activity("walkway", ["person"], now=99.0)
    assert _decide(_policy(cache), state).triggered_by == "experiment"


def test_absent_off_or_garbage_leaves_today_behaviour() -> None:
    for values in ({}, {experiment_key("walkway"): (b"off", 60_000)},
                   {experiment_key("walkway"): (b"yes", 60_000)}):
        cache = _cache(values)
        assert not cache.experiment_on("walkway")
        assert _decide(_policy(cache), RouterState()).dispatch_tier == "baseline"
    # Off arm is normal operation: labels still raise the tier.
    state = RouterState()
    state.record_activity("walkway", ["dog"], now=99.0)
    d = _decide(_policy(_cache({experiment_key("walkway"): (b"off", 60_000)})), state)
    assert d.dispatch_tier == "triggered" and d.triggered_by == "labels"


def test_key_without_ttl_is_refused_so_it_can_never_stick() -> None:
    cache = _cache({experiment_key("walkway"): (b"on", -1)})
    assert not cache.experiment_on("walkway") and cache.experiment_no_ttl_refusals == 1
    assert _decide(_policy(cache), RouterState()).dispatch_tier == "baseline"


def test_expired_key_clears_on_next_refresh() -> None:
    cache = _cache(ON)
    assert cache.experiment_on("walkway")
    asyncio.run(cache.refresh_once(_FakeRedis(set())))   # TTL ran out: key gone
    assert not cache.experiment_on("walkway")


def test_redis_failure_or_hang_means_no_override() -> None:
    cache = _cache(ON)
    asyncio.run(cache.refresh_once(_FakeRedis(set(), fail=True, values=ON)))
    assert not cache.experiment_on("walkway") and cache.refresh_failures == 1
    cache = ExpectationCache(refresh_sec=0.5)
    cache.note_stream("walkway")
    asyncio.run(cache.refresh_once(_FakeRedis(set(), values=ON)))
    asyncio.run(cache.refresh_once(_HangingRedis(set())))
    assert not cache.experiment_on("walkway")


def test_flag_off_ignores_the_key_but_expectation_still_works() -> None:
    cache = _cache(ON, keys={expect_key("walkway")}, experiment_enabled=False)
    assert not cache.experiment_on("walkway") and cache.experiment_streams() == []
    assert _decide(_policy(cache), RouterState()).triggered_by == "expectation"
    cache = _cache(ON, keys={expect_key("walkway")}, expectation_enabled=False)
    assert not cache.is_open("walkway") and cache.open_streams() == []
    assert cache.experiment_on("walkway")


def test_dispatch_tier_counts_and_health_surface() -> None:
    m = RouterMetrics()
    m.record_dispatch("walkway", "triggered", "experiment")
    m.record_dispatch("walkway", "baseline", None)
    m.record_dispatch("walkway", "triggered", "experiment")
    m.record_dispatch()   # legacy call: total only
    assert m.frames_dispatched_total == 4
    assert m.dispatch_tier_counts == {"walkway": {"triggered:experiment": 2, "baseline:-": 1}}
    details = make_health_details(router_enabled=True, dry_run=False, policy_path="p", metrics=m,
                                  state=RouterState(), experiment_override_streams=["walkway"])
    assert details["experiment_override_streams"] == ["walkway"]
