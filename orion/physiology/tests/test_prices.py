"""Pure price math: rest-state return, anti-windup, ceiling, no_reading is never 0."""

from __future__ import annotations

import math

import pytest

from orion.physiology.prices import (
    EXOGENOUS_BIND,
    LIVE,
    NO_READING,
    PriceParams,
    bind_with_hysteresis,
    ceiling_for,
    ceiling_run_since,
    heat_budget,
    heat_contribution,
    hours_to_rest,
    is_stuck,
    month_end_forecast,
    size_ceiling,
    step_price,
    upper_bound_if_spans_zero,
    usable_sensitivity,
)

P = PriceParams(eta=0.5, ceiling=2.0, max_step=0.5, budget_floor=0.25)


def _run(prices0, seq, params=P):
    p = prices0
    out = []
    for use, budget, bind in seq:
        s = step_price(p, use=use, budget=budget, exogenous_bind=bind, params=params)
        out.append(s)
        p = s.price
    return out


def test_slack_constraint_stays_at_zero():
    steps = _run(0.0, [(0.5, 2.0, False)] * 24)
    assert all(s.price == 0.0 and s.status == LIVE for s in steps)


def test_price_climbs_only_while_over_budget_then_returns_to_rest():
    over = _run(0.0, [(3.0, 1.0, False)] * 3)
    assert [s.price for s in over] == [0.5, 1.0, 1.5]  # bounded by max_step, no jump
    after = _run(over[-1].price, [(0.0, 1.0, False)] * 6)  # constraint cleared, fully slack
    assert hours_to_rest([s.price for s in after]) == 2  # 1.5 -> 1.0 -> 0.5 -> 0.0
    assert after[-1].price == 0.0


def test_pure_rule_lingers_near_the_boundary_and_decay_floor_fixes_it():
    # use just under budget: excess = -0.05, pure dual ascent decays 0.025/tick
    pure = _run(1.0, [(0.95, 1.0, False)] * 6)
    assert pure[-1].price > 0.8
    floored = PriceParams(eta=0.5, ceiling=2.0, max_step=0.5, budget_floor=0.25, slack_decay_floor=2.0 / 6)
    fast = _run(1.0, [(0.95, 1.0, False)] * 6, floored)
    assert hours_to_rest([s.price for s in fast]) is not None
    assert hours_to_rest([s.price for s in fast]) <= 3


def test_anti_windup_freezes_at_ceiling_instead_of_integrating():
    steps = _run(0.0, [(5.0, 0.0, True)] * 48)
    assert all(s.status == EXOGENOUS_BIND for s in steps)
    # Rule A (Patch B default): ramps at max_step, then holds AT the ceiling, never past it.
    assert [s.price for s in steps[:5]] == [0.5, 1.0, 1.5, 2.0, 2.0]
    assert all(s.price == P.ceiling for s in steps[3:])
    assert all(abs(s.delta) <= P.max_step + 1e-12 for s in steps)
    # The old design rule (a one-tick jump to the ceiling) is still available for the replay.
    jump = _run(0.0, [(5.0, 0.0, True)] * 3, PriceParams(eta=0.5, ceiling=2.0, max_step=0.5, budget_floor=0.25,
                                                         bind_ramp=False))
    assert jump[0].price == P.ceiling
    # Without the flag, a zero budget would be a huge excess. The ceiling still bounds it.
    raw = _run(0.0, [(5.0, 0.0, False)] * 48)
    assert max(s.price for s in raw) == P.ceiling
    assert all(abs(s.delta) <= P.max_step + 1e-12 for s in raw)


def test_price_never_negative_and_never_above_ceiling():
    steps = _run(10.0, [(0.0, 1.0, False)] * 10)  # prev above ceiling is clamped first
    assert all(0.0 <= s.price <= P.ceiling for s in steps)


@pytest.mark.parametrize("use,budget", [(None, 1.0), (1.0, None), (float("nan"), 1.0), (None, None)])
def test_no_reading_holds_price_and_is_not_read_as_zero(use, budget):
    s = step_price(1.2, use=use, budget=budget, exogenous_bind=False, params=P)
    assert s.status == NO_READING
    assert s.price == 1.2  # held: 0 use would have pulled it down
    assert s.excess is None
    # Contrast: a real reading of 0 use is slack and does pull the price down.
    zero = step_price(1.2, use=0.0, budget=1.0, exogenous_bind=False, params=P)
    assert zero.status == LIVE and zero.price < 1.2


def test_heat_contribution_missing_host_is_unknown_not_zero():
    s = {"athena": 0.5, "circe": 0.1}
    assert heat_contribution(s, {"athena": 200.0, "circe": 400.0}) == pytest.approx(1.4)
    assert heat_contribution(s, {"athena": 200.0}) is None
    assert heat_contribution(s, {"athena": 200.0, "circe": None}) is None
    assert heat_contribution({"athena": None}, {"athena": 200.0}) is None
    # below-baseline watts are not negative heat
    assert heat_contribution({"athena": 0.5}, {"athena": -50.0}) == 0.0


def test_heat_budget_and_exogenous_bind():
    assert heat_budget(soft_line_c=31.0, forecast_cabinet_c=29.0, own_contribution_c=0.5) == (2.5, False)
    b, bind = heat_budget(soft_line_c=31.0, forecast_cabinet_c=33.0, own_contribution_c=0.5)
    assert b == 0.0 and bind is True
    b, bind = heat_budget(soft_line_c=31.0, forecast_cabinet_c=31.5, own_contribution_c=1.0)
    assert b == pytest.approx(0.5) and bind is False  # over the line, but Orion's share can fix it
    assert heat_budget(soft_line_c=31.0, forecast_cabinet_c=None, own_contribution_c=0.5) == (None, None)


def test_usable_sensitivity_is_the_point_floored_at_zero():
    """Rule C. The 10-27 short-run fit: athena 0.686 (0.234..1.060), circe 0.298 (0.145..0.449)."""
    assert usable_sensitivity(0.686, 0.234, 1.060) == 0.686
    # An interval spanning 0 no longer jumps to its upper bound (the 3-4x that pinned R3).
    assert usable_sensitivity(0.05, -0.1, 0.2) == 0.05
    assert usable_sensitivity(-0.3, -0.5, -0.1) == 0.0
    assert usable_sensitivity(0.9, 0.1, 0.5) == 0.5   # a point above its own interval is clipped
    assert usable_sensitivity(None, 0.0, 1.0) is None
    # The superseded rule, kept for the replay comparison.
    assert upper_bound_if_spans_zero(0.05, -0.1, 0.2) == 0.2


def test_bind_hysteresis_enters_at_the_line_and_leaves_only_under_the_margin():
    """Rule B. Generic in units: degC for heat (room without Orion vs soft line), USD for energy."""
    seq = [30.8, 31.0, 30.8, 30.6, 30.49, 30.8, 31.2]
    prev, out = False, []
    for x in seq:
        prev = bind_with_hysteresis(prev, floor_use=x, budget=31.0, rearm_margin=0.5)
        out.append(prev)
    assert out == [False, True, True, True, False, False, True]
    assert bind_with_hysteresis(True, floor_use=None, budget=31.0, rearm_margin=0.5) is None
    # Energy: the always-on floor already spends the $100 envelope -> bound until $95.
    assert bind_with_hysteresis(False, floor_use=100.0, budget=100.0, rearm_margin=5.0) is True
    assert bind_with_hysteresis(True, floor_use=96.0, budget=100.0, rearm_margin=5.0) is True
    assert bind_with_hysteresis(True, floor_use=94.9, budget=100.0, rearm_margin=5.0) is False


def test_per_tick_scaling_follows_the_same_path_per_hour():
    hourly = PriceParams(eta=0.25, ceiling=1.0, max_step=0.25, budget_floor=0.25, slack_decay_floor=1 / 6)
    tick = hourly.per_tick(120.0)
    assert tick.max_step == pytest.approx(0.25 / 30) and tick.ceiling == 1.0 and tick.budget_floor == 0.25
    # Bound for an hour: 30 ticks of 120 s reach the same price as one 1 h step.
    p = 0.0
    for _ in range(30):
        p = step_price(p, use=5.0, budget=0.0, exogenous_bind=True, params=tick).price
    assert p == pytest.approx(step_price(0.0, use=5.0, budget=0.0, exogenous_bind=True, params=hourly).price)
    with pytest.raises(ValueError):
        hourly.per_tick(0.0)


def test_ceiling_for_is_sized_only_from_real_wants_otherwise_provisional():
    """Rule D. No graded wants exist before Patch C: every live ceiling is provisional."""
    assert ceiling_for(1.0) == (1.0, "provisional")
    vals = [i / 100 for i in range(101)]
    assert ceiling_for(1.0, vals, [0.5, 1.0, 2.0]) == (pytest.approx(0.9), "sized")
    assert ceiling_for(1.0, vals, [0.0]) == (1.0, "provisional")
    with pytest.raises(ValueError):
        ceiling_for(0.0)


def test_stuck_check_counts_only_unflagged_runs_at_the_ceiling():
    """Rule E (R3): at the ceiling past one horizon without exogenous_bind is stuck."""
    since = None
    seen = []
    for t, (price, status) in enumerate([(0.5, LIVE), (1.0, LIVE), (1.0, LIVE), (1.0, LIVE),
                                         (1.0, EXOGENOUS_BIND), (1.0, LIVE), (0.9, LIVE)]):
        since = ceiling_run_since(since, price=price, ceiling=1.0, status=status, now=float(t))
        seen.append((since, is_stuck(since, now=float(t), horizon_sec=1.5)))
    assert seen == [(None, False), (1.0, False), (1.0, False), (1.0, True),
                    (None, False), (5.0, False), (None, False)]


def test_size_ceiling():
    vals = [i / 100 for i in range(101)]
    assert size_ceiling(vals, [0.5, 1.0, 2.0]) == pytest.approx(0.9 / 1.0)
    assert size_ceiling([], [1.0]) is None
    assert size_ceiling([0.5], [0.0]) is None


def test_month_end_forecast():
    assert month_end_forecast(to_date=100.0, trailing_rate_per_hour=0.5, hours_remaining=10) == 105.0
    assert month_end_forecast(to_date=100.0, trailing_rate_per_hour=None, hours_remaining=10) is None


def test_params_reject_decay_floor_above_max_step_and_nan_prev():
    with pytest.raises(ValueError):
        PriceParams(eta=0.5, ceiling=1.0, max_step=0.1, budget_floor=0.25, slack_decay_floor=0.5)
    with pytest.raises(ValueError):
        step_price(float("nan"), use=1.0, budget=1.0, exogenous_bind=False, params=P)


def test_empty_sensitivity_map_is_unknown():
    assert heat_contribution({}, {}) is None


def test_params_reject_nonpositive():
    with pytest.raises(ValueError):
        PriceParams(eta=0.0, ceiling=1.0, max_step=1.0, budget_floor=1.0)
    with pytest.raises(ValueError):
        PriceParams(eta=1.0, ceiling=1.0, max_step=1.0, budget_floor=1.0, slack_decay_floor=math.nan)


def test_bind_ramp_bounds_the_step_into_exogenous_bind():
    ramp = PriceParams(eta=0.5, ceiling=2.0, max_step=0.5, budget_floor=0.25, bind_ramp=True)
    steps = _run(0.0, [(5.0, 0.0, True)] * 6, ramp)
    assert [s.price for s in steps][:4] == [0.5, 1.0, 1.5, 2.0]
    assert all(s.status == EXOGENOUS_BIND and s.delta <= ramp.max_step + 1e-12 for s in steps)
    assert steps[-1].price == ramp.ceiling  # held at the ceiling, never past it


def test_bind_never_lowers_a_price():
    ramp = PriceParams(eta=0.5, ceiling=2.0, max_step=0.5, budget_floor=0.25, bind_ramp=True)
    s = step_price(1.8, use=None, budget=None, exogenous_bind=True, params=ramp)
    assert s.price == 2.0


def test_exogenous_bind_hysteresis():
    from orion.physiology.prices import exogenous_bind_with_hysteresis as h

    assert h(False, exogenous_c=30.9, soft_line_c=31.0, rearm_margin_c=0.5) is False
    assert h(False, exogenous_c=31.0, soft_line_c=31.0, rearm_margin_c=0.5) is True
    assert h(True, exogenous_c=30.6, soft_line_c=31.0, rearm_margin_c=0.5) is True  # still inside the band
    assert h(True, exogenous_c=30.4, soft_line_c=31.0, rearm_margin_c=0.5) is False
    assert h(True, exogenous_c=None, soft_line_c=31.0, rearm_margin_c=0.5) is None
