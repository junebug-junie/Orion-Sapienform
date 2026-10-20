from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.event_pair import CLAIM_KIND, EventPairRefused, event_pair_effect, pair_stat
from app.effects import Effects, selector_key

DAY = 86400.0


def _poisson(rng, rate_per_day: float, span: float) -> np.ndarray:
    return np.sort(rng.uniform(0, span, rng.poisson(rate_per_day * span / DAY)))


def test_overlapping_windows_count_each_b_once_and_cover_their_union():
    # Windows [100, 160) and [130, 190) merge to [100, 190): 90 s, and the B at 150 counts once.
    st = pair_stat(np.array([100.0, 130.0]), np.array([10.0, 150.0, 189.0, 190.0]), 1000.0, 0.0, 60.0)
    assert st.covered_sec == pytest.approx(90.0)
    assert st.observed == 2                       # 150 and 189; 190 is the open end
    assert st.expected == pytest.approx(4 / 1000 * 90)


def test_lag_min_opens_the_window_later_and_windows_clip_at_the_range_end():
    st = pair_stat(np.array([900.0]), np.array([905.0, 950.0]), 1000.0, 30.0, 300.0)
    assert st.covered_sec == pytest.approx(70.0)  # [930, 1000) after clipping
    assert st.observed == 1                       # 905 is before lag_min


def test_b_following_a_by_30s_is_significant():
    rng = np.random.default_rng(1)
    span = 7 * DAY
    a = _poisson(rng, 6, span)
    b = np.sort(np.concatenate([a + 30.0, _poisson(rng, 20, span)]))
    res = event_pair_effect(a, b, span, 0.0, 60.0, n_placebo=999, seed=0)
    assert res.claim_kind == CLAIM_KIND == "observational_effect"
    assert res.rate_ratio > 20
    assert res.p_value <= 0.002
    assert res.observed_b_in_windows >= len(a[a + 30 < span])


def test_a_lag_window_that_misses_the_real_delay_finds_nothing():
    rng = np.random.default_rng(2)
    span = 7 * DAY
    a = _poisson(rng, 6, span)
    b = np.sort(np.concatenate([a + 30.0, _poisson(rng, 20, span)]))
    res = event_pair_effect(a, b, span, 120.0, 600.0, n_placebo=499, seed=0)
    assert res.p_value > 0.05


@pytest.mark.parametrize("placebo", ["same_time_of_day", "circular"])
def test_independent_poisson_streams_false_positive_rate_is_near_alpha(placebo):
    span = 7 * DAY
    hits = 0
    seeds = 300
    for seed in range(seeds):
        rng = np.random.default_rng(1000 + seed)
        a, b = _poisson(rng, 8, span), _poisson(rng, 30, span)
        hits += event_pair_effect(a, b, span, 0.0, 600.0, n_placebo=199, seed=seed,
                                  placebo=placebo).p_value <= 0.05
    rate = hits / seeds
    assert 0.015 <= rate <= 0.09, rate


def test_bursty_b_independent_of_a_stays_calibrated():
    """B in bursts (one visit = several rows). The circular shift keeps the bursts, so p stays honest."""
    span = 7 * DAY
    hits = 0
    seeds = 200
    for seed in range(seeds):
        rng = np.random.default_rng(5000 + seed)
        centres = _poisson(rng, 6, span)
        b = np.sort(np.concatenate([c + rng.uniform(0, 120, rng.integers(1, 8)) for c in centres]))
        a = _poisson(rng, 8, span)
        hits += event_pair_effect(a, b, span, 0.0, 300.0, n_placebo=199, seed=seed).p_value <= 0.05
    assert hits / seeds <= 0.09


def _daytime(rng, per_day: float, days: int) -> np.ndarray:
    """Events only between 08:00 and 20:00, otherwise independent."""
    n = rng.poisson(per_day * days)
    return np.sort(rng.integers(0, days, n) * DAY + rng.uniform(8 * 3600, 20 * 3600, n))


def test_shared_daily_rhythm_fools_the_free_shift_but_not_the_time_of_day_placebo():
    days, seeds = 14, 120
    hits = {"circular": 0, "same_time_of_day": 0}
    for seed in range(seeds):
        rng = np.random.default_rng(9000 + seed)
        a, b = _daytime(rng, 6, days), _daytime(rng, 20, days)
        for kind in hits:
            hits[kind] += event_pair_effect(a, b, days * DAY, 0.0, 600.0, n_placebo=199, seed=seed,
                                            placebo=kind).p_value <= 0.05
    assert hits["circular"] / seeds > 0.25            # measured 45/120: why the default keeps time of day
    assert hits["same_time_of_day"] / seeds <= 0.09, hits


def test_time_of_day_placebo_still_finds_a_real_30s_follow():
    rng = np.random.default_rng(11)
    a = _daytime(rng, 6, 14)
    b = np.sort(np.concatenate([a + 30.0, _daytime(rng, 20, 14)]))
    res = event_pair_effect(a, b, 14 * DAY, 0.0, 60.0, n_placebo=499, seed=0)
    assert res.p_value <= 0.002 and res.placebo == "same_time_of_day"


def test_time_of_day_placebo_refuses_lags_near_its_jitter():
    with pytest.raises(EventPairRefused, match="same_time_of_day"):
        event_pair_effect(np.arange(10) * 3600.0, np.arange(10) * 3600.0, 7 * DAY, 0.0, 3600.0,
                          placebo="same_time_of_day")


def test_same_seed_same_answer():
    rng = np.random.default_rng(3)
    a, b = _poisson(rng, 8, 3 * DAY), _poisson(rng, 30, 3 * DAY)
    r1 = event_pair_effect(a, b, 3 * DAY, 0.0, 300.0, n_placebo=199, seed=7)
    r2 = event_pair_effect(a, b, 3 * DAY, 0.0, 300.0, n_placebo=199, seed=7)
    assert r1.p_value == r2.p_value and r1.rate_ratio == r2.rate_ratio


@pytest.mark.parametrize("a,b,why", [
    (np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0]), "occurrence"),
    (np.arange(10) * 1000.0, np.array([]), "never occurs"),
])
def test_refuses_with_counts_instead_of_answering(a, b, why):
    with pytest.raises(EventPairRefused) as exc:
        event_pair_effect(a, b, DAY, 0.0, 60.0)
    assert why in exc.value.reason
    assert exc.value.detail["n_a"] == len(a) and exc.value.detail["n_b"] == len(b)


def test_refuses_a_range_too_short_for_placebos():
    with pytest.raises(EventPairRefused, match="lag_max"):
        event_pair_effect(np.arange(10.0), np.arange(10.0), 1000.0, 0.0, 600.0)


# --- orchestration: reading + storing ---------------------------------------------------------

NOW = datetime(2026, 10, 20, tzinfo=timezone.utc)


class _Engine:
    store = None

    @staticmethod
    def clock():
        return NOW


class _Store:
    def __init__(self, times: dict):
        self.times = times
        self.inserted: list[dict] = []
        self.reads: list[tuple] = []

    def event_times(self, source, event_type, stream_id, since, until, limit):
        self.reads.append((source, event_type, stream_id))
        return [t for t in self.times.get((event_type, stream_id), []) if since <= t < until]

    def insert_effect(self, row):
        row = {"estimate_id": "e1", **row}
        self.inserted.append(row)
        return row


def _fx(times):
    fx = Effects.__new__(Effects)
    fx.engine = _Engine()
    fx.store = _Store(times)
    return fx


def test_event_pair_is_stored_as_an_observational_estimate():
    rng = np.random.default_rng(4)
    start = NOW - timedelta(days=7)
    a = _poisson(rng, 6, 7 * DAY)
    b = np.concatenate([a + 30, _poisson(rng, 20, 7 * DAY)])
    ts = lambda xs: sorted(start + timedelta(seconds=float(x)) for x in xs if x < 7 * DAY)  # noqa: E731
    fx = _fx({("doorbell_ring", None): ts(a), ("person_presence", "walkway"): ts(b)})
    sa = {"source": "vision_events", "event_type": "doorbell_ring", "stream_id": None}
    sb = {"source": "vision_events", "event_type": "person_presence", "stream_id": "walkway"}
    out = fx.event_pair(sa, sb, start, NOW, 0.0, 60.0, n_placebo=199)
    row = fx.store.inserted[0]
    assert row["claim_kind"] == "observational_effect"
    assert row["action"] == "event_pair:vision_events:doorbell_ring"
    assert row["series_id"] == "vision_events:person_presence@walkway"
    assert row["effect"] == out["details"]["rate_ratio"] > 5 and row["p_value"] < 0.01
    assert row["ci_lo"] is None and row["details"]["estimator"] == "event_pair"
    assert ("vision_events", "person_presence", "walkway") in fx.store.reads


def test_refusal_stores_nothing():
    fx = _fx({("doorbell_ring", None): [], ("person_presence", "walkway"): [NOW - timedelta(hours=1)]})
    sa = {"source": "vision_events", "event_type": "doorbell_ring", "stream_id": None}
    sb = {"source": "vision_events", "event_type": "person_presence", "stream_id": "walkway"}
    with pytest.raises(EventPairRefused, match="occurrence"):
        fx.event_pair(sa, sb, NOW - timedelta(days=7), NOW, 0.0, 300.0)
    assert fx.store.inserted == []


def test_same_selector_needs_a_positive_lag_min():
    fx = _fx({})
    s = {"source": "vision_events", "event_type": "person_presence", "stream_id": "walkway"}
    with pytest.raises(EventPairRefused, match="lag_min"):
        fx.event_pair(s, s, NOW - timedelta(days=7), NOW, 0.0, 300.0)


def test_selector_key():
    assert selector_key({"source": "vision_events", "event_type": "x", "stream_id": None}) == "vision_events:x"


# --- endpoint ----------------------------------------------------------------------------------

def test_endpoint_returns_structured_refusal(monkeypatch):
    import asyncio

    import pydantic
    from fastapi import HTTPException

    import app.main as main

    fx = _fx({("person_presence", "walkway"): [NOW - timedelta(hours=1)]})
    monkeypatch.setattr(main, "engine", object())
    monkeypatch.setattr(main, "effects", fx)
    body = {"a": {"event_type": "doorbell_ring"}, "b": {"event_type": "person_presence", "stream_id": "walkway"},
            "start": (NOW - timedelta(days=7)).isoformat(), "end": NOW.isoformat(), "lag_max_sec": 300}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.post_event_pair(main.EventPairBody(**body)))
    assert exc.value.status_code == 422
    d = exc.value.detail
    assert d["error"] == "refused" and d["claim_kind"] == "observational_effect"
    assert d["n_a"] == 0 and d["n_b"] == 1 and fx.store.inserted == []
    with pytest.raises(pydantic.ValidationError):   # only vision_events is readable today
        main.EventPairBody(**{**body, "a": {"source": "other", "event_type": "x"}})
