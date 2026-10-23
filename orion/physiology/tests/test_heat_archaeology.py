"""Heat archaeology: the 2026-10-14 circe step comes back, attributed, with its evidence."""
from __future__ import annotations

import random
from datetime import timedelta

import pytest

from orion.physiology import heat_archaeology as ha
from orion.physiology.tests import heat_fixture as fx


def _circe():
    readings, busy, truth = fx.build()
    [circe] = ha.analyze(readings, busy, window_start=fx.START, window_end=fx.END, nodes=["circe"])
    return circe, truth


def test_the_step_is_found_at_the_hour_it_happened():
    circe, truth = _circe()
    assert len(circe.steps) == 1
    step = circe.steps[0]
    # Hourly wall watts are noisy (harness alone swings ~0-285 W hour to hour), so the split
    # lands within a few hours; the load shift below is the crisp clock.
    assert abs((step.at - fx.STEP_AT).total_seconds()) <= 3 * 3600
    assert step.delta_w == pytest.approx(truth["step_w"], abs=10)
    assert step.before_hours + step.after_hours == 336
    # Composition: the crash loop is CPU watts, vision-council is GPU watts.
    assert step.cpu_delta_w == pytest.approx(70.0, abs=5)
    assert step.load_15m_before < 2.5 and step.load_15m_after > 13


def test_the_step_is_split_into_regime_and_workloads_with_truth_inside_tolerance():
    circe, truth = _circe()
    [attr] = circe.attributions
    regime = next(k for k in attr.contributions_w if k.startswith("CPU-load regime"))
    assert attr.contributions_w[regime] == pytest.approx(truth["crash_w"], abs=15)
    assert attr.contributions_w["vision-council"] == pytest.approx(truth["vision_w"], abs=10)
    assert abs(attr.unexplained_w) < 15
    assert circe.fit.r2 > 0.9 and circe.fit.n_hours == 336
    terms = {t.name: t for t in circe.fit.terms}
    assert terms["intercept"].watts_per_unit == pytest.approx(fx.FLOOR_W, abs=25)
    assert terms["harness"].watts_per_unit == pytest.approx(fx.HARNESS_W, abs=20)
    assert circe.gpu_fit is not None and circe.gpu_fit.r2 > 0.9


def test_durable_run_holds_are_reported_as_reservations_never_fitted():
    circe, _ = _circe()
    assert "run:curiosity.investigate" in circe.reservations
    assert all(not t.name.startswith("run:") for t in circe.fit.terms)


def test_summary_says_since_when_what_and_that_it_is_unverified():
    circe, _ = _circe()
    text = circe.summary
    assert "since around 2026-10-14" in text
    assert "shifts: 1.55 -> 13.98 at 2026-10-14 03:00 UTC" in text
    assert "CPU-load regime" in text and "no GPU lease explains" in text
    assert "container restarts" in text
    assert "vision-council" in text
    assert "UNVERIFIED" in text
    assert circe.verified is False


def test_every_step_and_fit_claim_carries_evidence():
    circe, _ = _circe()
    queries = {e.query for e in circe.evidence}
    assert queries == {"energy_hourly", "biometrics_hourly", "pool_busy_hours"}
    step_ev = next(e for e in circe.evidence if e.query == "energy_hourly")
    assert step_ev.values["hours"] == circe.steps[0].before_hours + circe.steps[0].after_hours
    assert step_ev.values["from"].startswith("2026-10-07")


def test_load_shift_is_reported_beside_the_watts():
    circe, _ = _circe()
    [shift] = [s for s in circe.regime_shifts if s.series == "load_15m"]
    assert shift.at == fx.STEP_AT and shift.delta > 10


def test_flat_node_has_no_step_and_no_invented_regime():
    readings, busy, _ = fx.build()
    [athena] = ha.analyze(readings, busy, window_start=fx.START, window_end=fx.END, nodes=["athena"])
    assert athena.steps == []
    assert athena.fit is None  # no workload columns, one-moded load: nothing to fit
    assert "stands out from the hour-to-hour wander" in athena.summary
    assert athena.fit_note and "nothing to attribute" in athena.fit_note
    assert athena.mean_wall_w == pytest.approx(400, abs=5)


def test_node_with_no_wall_reading_says_so_instead_of_zero():
    [r] = ha.analyze([], [], window_start=fx.START, window_end=fx.END, nodes=["hecate"])
    assert r.hours_with_wall == 0 and r.mean_wall_w is None
    assert "no wall-power reading" in r.summary


def test_missing_hours_are_skipped_not_read_as_zero():
    readings, busy, _ = fx.build()
    # Drop every 5th circe hour's wall reading: still one step at the same hour.
    thinned = [
        ha.HourReading(hour=r.hour, node=r.node, wall_w=None if i % 5 == 0 else r.wall_w,
                       gpu_w=r.gpu_w, cpu_w=r.cpu_w, load_15m=r.load_15m)
        for i, r in enumerate(readings)
    ]
    [circe] = ha.analyze(thinned, busy, window_start=fx.START, window_end=fx.END, nodes=["circe"])
    assert len(circe.steps) == 1
    assert abs((circe.steps[0].at - fx.STEP_AT).total_seconds()) <= 3 * 3600
    assert circe.steps[0].before_w > 780  # never pulled toward 0 by a missing hour


def test_results_sort_largest_step_first():
    readings, busy, _ = fx.build()
    out = ha.analyze(readings, busy, window_start=fx.START, window_end=fx.END)
    assert [r.node for r in out] == ["circe", "athena"]


def test_ols_recovers_exact_coefficients_and_drops_constant_and_collinear_columns():
    rng = random.Random(3)
    x1 = [rng.random() for _ in range(100)]
    x2 = [rng.random() for _ in range(100)]
    y = [5.0 + 2.0 * a - 3.0 * b for a, b in zip(x1, x2)]
    res = ha.ols({"a": x1, "b": x2, "flat": [1.0] * 100, "a_twice": [2 * v for v in x1]}, y)
    coef = dict(zip(res.names, res.coef))
    assert "flat" in res.dropped and ("a" in res.dropped or "a_twice" in res.dropped)
    # The surviving alias carries the whole slope in its own units.
    if "a" in coef:
        assert coef["a"] == pytest.approx(2.0)
    else:
        assert coef["a_twice"] == pytest.approx(1.0)
    assert coef["intercept"] == pytest.approx(5.0)
    assert coef["b"] == pytest.approx(-3.0)
    assert res.r2 == pytest.approx(1.0)


def test_find_steps_ignores_noise_and_small_shifts():
    rng = random.Random(5)
    flat = [800 + rng.gauss(0, 40) for _ in range(300)]
    assert ha.find_steps(flat) == []
    small = [800 + (40 if i >= 150 else 0) + rng.gauss(0, 5) for i in range(300)]
    assert ha.find_steps(small) == []  # under STEP_MIN_DELTA_W
    big = [800 + (300 if i >= 150 else 0) + rng.gauss(0, 40) for i in range(300)]
    assert ha.find_steps(big) == [150]


def test_otsu_rejects_one_moded_load_and_accepts_two_moded():
    rng = random.Random(9)
    normal = [rng.gauss(10, 2) for _ in range(500)]
    assert ha.otsu_threshold(normal)[1] < ha.REGIME_MIN_SEPARATION
    two = [rng.gauss(1.5, 0.4) for _ in range(250)] + [rng.gauss(14, 0.5) for _ in range(250)]
    thr, sep = ha.otsu_threshold(two)
    assert 3 < thr < 13 and sep > ha.REGIME_MIN_SEPARATION


def test_holder_groups():
    assert ha.holder_group("http:anthropic", None) == ("harness", False)
    assert ha.holder_group("http:openai", None) == ("ai-town", False)
    assert ha.holder_group("durable-runs:abc", "reverie.visual") == ("run:reverie.visual", True)
    assert ha.holder_group("durable-runs:abc", None) == ("run:?", True)
    assert ha.holder_group("vision-council", None) == ("vision-council", False)


def test_short_window_finds_no_step_rather_than_a_fake_one():
    readings, busy, _ = fx.build()
    end = fx.STEP_AT + timedelta(hours=12)
    short = [r for r in readings if fx.STEP_AT - timedelta(hours=12) <= r.hour < end]
    [circe] = ha.analyze(short, busy, window_start=fx.STEP_AT - timedelta(hours=12), window_end=end, nodes=["circe"])
    assert circe.steps == []


def _ar1(n, rho, sd, rng):
    x = [0.0]
    e = sd * (1 - rho * rho) ** 0.5
    for _ in range(n - 1):
        x.append(rho * x[-1] + rng.gauss(0, e))
    return x


def test_wandering_flat_series_rarely_reads_as_a_step():
    """Review 2026-10-23: with independent-hour SEs, AR(1) noise (rho 0.97) gave a 'step' in 62 of
    100 flat series. Live circe residuals run rho ~0.5-0.6."""
    rng = random.Random(11)
    live_like = sum(bool(ha.find_steps([900 + v for v in _ar1(336, 0.6, 120, rng)])) for _ in range(100))
    extreme = sum(bool(ha.find_steps([900 + v for v in _ar1(336, 0.97, 60, rng)])) for _ in range(100))
    assert live_like <= 5 and extreme <= 15


def test_unrelated_wandering_column_is_rarely_credible():
    rng = random.Random(12)
    hits = 0
    for _ in range(100):
        y = [900 + v for v in _ar1(336, 0.97, 60, rng)]
        x = [abs(v) for v in _ar1(336, 0.97, 1, rng)]
        res = ha.ols({"x": x}, y)
        hits += abs(res.coef[0]) >= ha.SUMMARY_MIN_T * res.se[0]
    assert hits <= 20  # was ~75 with independent-hour SEs


def test_a_load_shift_is_attributed_when_no_wall_step_clears_the_bar():
    readings, busy, truth = fx.build()
    # Make the wall step too small to clear STEP_MIN_DELTA_W by itself: drop the crash-loop watts
    # to 15 W, leaving the vision rise and the crisp load clock.
    lowered = [
        ha.HourReading(hour=r.hour, node=r.node, gpu_w=r.gpu_w, cpu_w=r.cpu_w, load_15m=r.load_15m,
                       wall_w=None if r.wall_w is None else (r.wall_w - (fx.CRASH_W - 15.0 if r.hour >= fx.STEP_AT else 0.0)))
        for r in readings
    ]
    [circe] = ha.analyze(lowered, busy, window_start=fx.START, window_end=fx.END, nodes=["circe"])
    assert circe.steps == []
    [attr] = [a for a in circe.attributions if a.anchor == "load_shift"]
    assert attr.step_at == fx.STEP_AT
    regime = next(v for k, v in attr.contributions_w.items() if k.startswith("CPU-load regime"))
    assert regime == pytest.approx(15.0, abs=12)
    assert "CPU load 1.5" in circe.summary and "across it wall power went" in circe.summary


def test_sentence_parts_add_up_to_the_delta():
    circe, _ = _circe()
    [attr] = circe.attributions
    assert sum(attr.contributions_w.values()) + attr.unexplained_w == pytest.approx(attr.delta_w, abs=1)
    # Every contribution is either named or folded into "other terms".
    text = ha._attribution_bits(attr, ha._credible(circe.fit))
    assert "unexplained" in text
