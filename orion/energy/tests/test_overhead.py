"""orion/energy/overhead.py: the per-host overhead factor and its fallback."""

from __future__ import annotations

import math

import pytest

from orion.energy import overhead as ov


def _hours(k: float, *, n: int = 72, rest: float = 541.0, gpu_idle: float = 190.0, cpu_idle: float | None = 100.0,
           noise: float = 0.0) -> list[dict]:
    out = []
    for i in range(n):
        g, c = gpu_idle + 10.0 + (i % 11) * 17.0, 100.0 + (i % 4) * 6.0
        x = (g - gpu_idle) + (c - 100.0)
        out.append({"ch": rest + k * x + (noise if i % 2 else -noise), "gpu": g, "cpu": c, "cpu_idle": cpu_idle})
    return out


def test_fit_recovers_the_slope_and_says_how():
    o = ov.fit_overhead("circe", _hours(2.35, noise=5.0), rest_w=541.0, gpu_idle_w=190.0, window_words="2026-10-16..10-30")
    assert o.fitted and o.factor == pytest.approx(2.35, abs=0.01) and o.r2 > 0.95 and o.hours == 72
    assert "fitted on circe 2026-10-16..10-30" in o.basis and "R^2" in o.basis and "541" in o.basis
    assert o.wall(100.0) == pytest.approx(100.0 * o.factor) and o.wall(-3.0) == 0.0 and o.wall(None) is None


def test_cpu_idle_missing_on_old_hours_uses_the_hosts_own_median():
    rows = _hours(1.27, rest=340.0, gpu_idle=76.0)
    for r in rows[:-10]:
        r["cpu_idle"] = None          # cpu_idle_watts began 2026-10-29: older hours lack it
    o = ov.fit_overhead("athena", rows, rest_w=340.0, gpu_idle_w=76.0)
    assert o.fitted and o.factor == pytest.approx(1.27, abs=0.01) and "CPU above 100 W idle" in o.basis


def test_the_fill_is_the_median_not_the_mean():
    # published idles 100, 100, 130 (median 100, mean 110): the old hours' true idle is 100, so only
    # the median fill recovers k exactly
    rows = _hours(2.0)
    for r in rows[:-3]:
        r["cpu_idle"] = None
    for r, ci in zip(rows[-3:], (100.0, 100.0, 130.0)):
        r["cpu_idle"] = ci
        r["ch"] = 541.0 + 2.0 * ((r["gpu"] - 190.0) + max(0.0, r["cpu"] - ci))
    o = ov.fit_overhead("circe", rows, rest_w=541.0, gpu_idle_w=190.0)
    assert o.factor == pytest.approx(2.0, abs=1e-3) and o.r2 == pytest.approx(1.0, abs=1e-6)


def test_drift_above_the_pinned_rest_lands_in_k_and_the_basis_says_how_much():
    rows = _hours(2.0)
    for r in rows:
        r["ch"] += 60.0               # 60 W of new always-on draw above the pinned rest
    o = ov.fit_overhead("circe", rows, rest_w=541.0, gpu_idle_w=190.0)
    assert o.fitted and o.factor > 2.0                     # charged to load, as the per-act meter does
    assert o.intercept_w == pytest.approx(60.0, abs=0.5)
    assert "with an intercept: 2.00 x load + 60 W (always-on draw above the pinned rest" in o.basis
    assert f"{o.hours} hourly means ({o.loaded_hours} with load)" in o.basis


@pytest.mark.parametrize("rows,kw,why", [
    (_hours(2.0, n=10), {}, "10 loaded hours"),
    (_hours(2.0, cpu_idle=None), {}, "no cpu_idle_watts"),
    (_hours(2.0), {"rest_w": None}, "no pinned rest"),
    (_hours(2.0), {"gpu_idle_w": None}, "no GPU rest"),
    (_hours(0.5), {}, "under 1"),                        # the wall cannot draw less than its parts
    ([{"ch": 541.0 + (900 if i % 2 else 0), "gpu": 190.0 + 30 + i % 3, "cpu": 100.0, "cpu_idle": 100.0}
      for i in range(60)], {}, "R^2"),                     # chassis does not follow load
])
def test_an_unusable_fit_falls_back_to_the_2992_value_and_says_why(rows, kw, why):
    args = {"rest_w": 541.0, "gpu_idle_w": 190.0, **kw}
    o = ov.fit_overhead("circe", rows, **args)
    assert not o.fitted and o.factor == ov.FALLBACK_FACTOR == 2.69 and o.r2 is None
    assert ov.FALLBACK_SOURCE in o.basis and why in o.basis


def test_non_numbers_are_dropped_not_fatal():
    rows = _hours(2.0) + [{"ch": "x", "gpu": None, "cpu": float("nan"), "cpu_idle": True}]
    o = ov.fit_overhead("circe", rows, rest_w=541.0, gpu_idle_w=190.0)
    assert o.fitted and math.isfinite(o.factor) and o.hours == 72
