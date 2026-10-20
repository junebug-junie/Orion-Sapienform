from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from app.causal import CausalError, design_switchback
from app.effects import Effects
from tests.helpers import T0
from tests.test_engine import Clock, _engine, _setup, _sync, fast_zoo  # noqa: F401  (fixture)


class EffectsStore:
    def __init__(self, rows):
        self.rows = rows

    def effects(self, action, series_id):
        return sorted(self.rows, key=lambda r: r["created_at"], reverse=True)


def _effects_with(rows):
    reg, spec, s, store, end = _setup()
    eng = _engine(reg, store, Clock(end), [])
    fx = Effects(eng)
    fx.store = EffectsStore(rows)
    return fx


def _obs(iv, effect, lo, hi, t):
    return {"estimate_id": f"e{t}", "claim_kind": "observational_effect", "intervention_id": iv, "effect": effect,
            "ci_lo": lo, "ci_hi": hi, "p_value": 0.1, "n": 20, "created_at": T0 + timedelta(minutes=t)}


def test_reestimating_one_intervention_does_not_narrow_the_pooled_interval():
    one = _effects_with([_obs("iv1", 1.0, 0.0, 2.0, 1)]).best_effect("x", "test.temp")
    many = _effects_with([_obs("iv1", 1.0, 0.0, 2.0, t) for t in range(5)]).best_effect("x", "test.temp")
    assert many["n"] == 1 and many["ci_hi"] - many["ci_lo"] == pytest.approx(one["ci_hi"] - one["ci_lo"])


def test_two_distinct_interventions_pool_and_experiment_outranks_them():
    fx = _effects_with([_obs("iv1", 1.0, 0.0, 2.0, 1), _obs("iv2", 2.0, 1.0, 3.0, 2)])
    r = fx.best_effect("x", "test.temp")
    assert r["claim_kind"] == "observational_effect" and r["n"] == 2 and r["effect"] == pytest.approx(1.5)
    fx.store.rows.append({**_obs(None, 0.7, 0.5, 0.9, 3), "claim_kind": "experimental_effect"})
    assert fx.best_effect("x", "test.temp")["claim_kind"] == "experimental_effect"


def _exp():
    return {"action": "gpu_burn", "design": design_switchback(T0, 20, 8, seed=3)}


def _log(exp, overrides=None):
    rows = []
    for b in exp["design"]["blocks"]:
        rows.append({"intervention_id": f"i{b['index']}", "action": exp["action"], "arm": b["arm"],
                     "started_at": T0 + timedelta(minutes=20 * b["index"]),
                     "ended_at": T0 + timedelta(minutes=20 * (b["index"] + 1))})
    for i, ch in (overrides or {}).items():
        rows[i].update(ch)
    return rows


def test_blocks_that_follow_the_design_are_accepted():
    exp = _exp()
    out = Effects._match_design(exp, _log(exp))
    assert [b["arm"] for b in out] == [b["arm"] for b in exp["design"]["blocks"]]


@pytest.mark.parametrize("bad", [
    lambda e: {0: {"arm": "off" if e["design"]["blocks"][0]["arm"] == "on" else "on"}},   # wrong arm
    lambda e: {1: {"started_at": T0 + timedelta(minutes=27), "ended_at": T0 + timedelta(minutes=47)}},  # off-schedule
    lambda e: {2: {"action": "something_else"}},                                          # other action
])
def test_blocks_that_do_not_follow_the_design_are_refused(bad):
    exp = _exp()
    with pytest.raises(CausalError, match="randomised design"):
        Effects._match_design(exp, _log(exp, bad(exp)))


def test_guard_says_unsafe_when_input_is_stale(monkeypatch):
    import tests.test_engine as te
    from tests.helpers import spec_doc as base_doc
    monkeypatch.setattr(te, "spec_doc", lambda **o: base_doc(thresholds=[
        {"name": "experiment_guard", "value": 40.0, "window": "30min", "max_probability": 0.05}], **o))
    reg, spec, s, store, end = te._setup()
    clock = Clock(end + timedelta(seconds=90))
    eng = _engine(reg, store, clock, [])
    asyncio.run(eng.tick(_sync))
    clock.t = end + timedelta(hours=3)   # sensor silent for 3 h
    g = Effects(eng).guard(spec)
    assert g["safe"] is False and "input stale" in g["reason"]


def test_intervention_estimate_refuses_a_history_window_that_crosses_a_break():
    reg, spec, s, store, end = _setup()
    eng = _engine(reg, store, Clock(end + timedelta(hours=2)), [])
    brk = end - timedelta(days=2)
    store.change_events = lambda: [{"scope": "*", "effect": "break", "started_at": brk, "ended_at": None}]
    store.intervention = lambda i: {"intervention_id": i, "action": "x", "experiment_id": None,
                                    "started_at": end - timedelta(hours=3), "ended_at": end - timedelta(hours=2)}
    with pytest.raises(CausalError, match="break"):
        Effects(eng).estimate_intervention("iv1", "test.temp")
