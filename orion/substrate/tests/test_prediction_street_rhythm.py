"""orion/substrate/prediction_street_rhythm.py: the 30-min fold of walkway rhythm forecast surprise."""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from orion.substrate import prediction_street_rhythm as psr

NOW = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)


def _terms(p, happened):
    s = -math.log(p if happened else 1 - p)
    h = -p * math.log(p) - (1 - p) * math.log(1 - p)
    lg = math.log(p / (1 - p))
    return s - h, p * (1 - p) * lg * lg


def _ev(eid, p, happened, ago_min=1.0):
    ex, var = _terms(p, happened)
    return {"expectation_id": eid, "stream_id": "walkway", "scored_at": NOW - timedelta(minutes=ago_min),
            "excess": ex, "calm_var": var, "happened": happened, "p": p}


def test_ids_agree_with_the_attention_schema_and_the_seed_bar():
    from orion.schemas.attention_candidate import STREET_RHYTHM_NODE_ID
    from orion.substrate.curiosity_seed_bar import STREET_RHYTHM_NODE_ID as CSB_ID

    assert psr.STREET_RHYTHM_NODE_ID == STREET_RHYTHM_NODE_ID == CSB_ID == "node:prediction.street_rhythm"


def test_window_is_z_of_summed_excess_and_rests_at_exact_zero():
    st = psr.StreetRhythmState()
    for eid, p, hit in (("a", 0.7, True), ("b", 0.86, True)):
        assert st.add(_ev(eid, p, hit), now=NOW) == (psr.STREET_RHYTHM_NODE_ID, "accepted")
    r = st.reading(now=NOW)
    (ea, va), (eb, vb) = _terms(0.7, True), _terms(0.86, True)
    assert r.z == pytest.approx((ea + eb) / math.sqrt(va + vb))
    assert r.z < 0 and r.value == 0.0 and r.n == 2


def test_outcomes_leave_the_window_on_time_and_the_node_goes_absent():
    st = psr.StreetRhythmState()
    st.add(_ev("a", 0.9, False, ago_min=29), now=NOW)
    assert st.reading(now=NOW).value == 1.0
    assert st.reading(now=NOW + timedelta(minutes=2)) is None


def test_backlog_and_future_and_other_streams_are_refused():
    st = psr.StreetRhythmState()
    assert st.add(_ev("a", 0.9, False, ago_min=31), now=NOW)[1] == "older_than_window"
    assert st.add(_ev("b", 0.9, False, ago_min=-10), now=NOW)[1] == "future_scored_at"
    assert st.add({**_ev("c", 0.9, False), "stream_id": "doorbell"}, now=NOW) == (None, "unknown_stream")
    assert st.reading(now=NOW) is None


def test_calibrated_windows_rest_at_zero_about_half_the_time():
    rng = random.Random(5)
    zeros = 0
    n = 4000
    for k in range(n):
        st = psr.StreetRhythmState()
        for j in range(rng.randint(1, 4)):
            p = rng.choice([0.55, 0.62, 0.75, 0.86, 0.95])
            st.add(_ev(f"{k}-{j}", p, rng.random() < p), now=NOW)
        r = st.reading(now=NOW)
        zeros += r.value == 0.0
    assert 0.5 < zeros / n < 0.85


def test_replay_writes_once_per_outcome_like_live():
    samples = [psr.RhythmSample("a", NOW, *_terms(0.9, False), happened=False, p=0.9),
               psr.RhythmSample("b", NOW, *_terms(0.7, True), happened=True, p=0.7),
               psr.RhythmSample("c", NOW + timedelta(hours=1), *_terms(0.5, True), happened=True, p=0.5)]
    out = psr.replay_node_values(samples)
    assert [t for t, _ in out] == [NOW, NOW]       # the 50/50-only window carries no information: no write
    st = psr.StreetRhythmState()
    live = []
    for s in sorted(samples[:2], key=lambda s: s.expectation_id):
        st.add({"expectation_id": s.expectation_id, "stream_id": "walkway", "scored_at": s.scored_at,
                "excess": s.excess, "calm_var": s.calm_var, "happened": s.happened, "p": s.p}, now=NOW)
        live.append(st.reading(now=NOW).value)
    assert [v for _, v in out] == pytest.approx(live)


def test_a_republished_outcome_is_a_duplicate_not_fresh_news():
    st = psr.StreetRhythmState()
    assert st.add(_ev("a", 0.9, False), now=NOW)[1] == "accepted"
    assert st.add(_ev("a", 0.9, False), now=NOW + timedelta(minutes=3))[1] == "duplicate"
