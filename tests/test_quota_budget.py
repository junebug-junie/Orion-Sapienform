"""Contested-scarcity quota reader.

Every expected value below is hand-computed and written as a literal. Nothing
here re-derives an answer by calling the code under test, which is the way a
fixture ends up asserting that a bug is faithfully reproduced.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from orion.autonomy.quota_budget import (
    WindowSpend,
    MIN_MEANINGFUL_ALLOWANCE_USD,
    LedgerTick,
    quota_state,
    sum_window,
    window_elapsed_fraction,
)

T0 = datetime(2026, 8, 26, 18, 14, tzinfo=timezone.utc)


def tick(minutes: int, tokens: int, cost: float | None, unpriced: int = 0) -> LedgerTick:
    return LedgerTick(
        observed_at=T0 + timedelta(minutes=minutes),
        total_tokens=tokens,
        cost_usd=cost,
        unpriced_session_count=unpriced,
    )


# --------------------------------------------------------------------------
# The regression that started this. Rows are per-tick DELTAS.
# --------------------------------------------------------------------------


def test_window_spend_is_the_sum_of_deltas_not_the_range():
    """Reading `dev_economics_ledger_log` with `max - min` understated real
    daily spend by an order of magnitude ($27-48/day reported against a real
    $51-583/day) because the rows are deltas, not cumulative totals.

    Costs 5.00, 10.00, 3.00 -> SUM is 18.00. The bug's answer is the range,
    10.00 - 3.00 = 7.00. Both are pinned so the wrong one can never pass.
    """
    spend = sum_window([tick(0, 100, 5.00), tick(15, 200, 10.00), tick(30, 50, 3.00)])

    assert spend.spend_usd == pytest.approx(18.00)
    assert spend.spend_usd != pytest.approx(7.00)  # max - min
    assert spend.tokens == 350
    assert spend.tick_count == 3
    assert spend.active_tick_count == 3


def test_a_cumulative_read_of_a_delta_series_is_not_what_this_returns():
    """A delta series can dip to zero; a cumulative counter never can. That
    `min(total_tokens) == 0` on every single day was the tell walked past the
    first time. Pinned so the semantics stay stated.
    """
    ticks = [tick(0, 500, 4.00), tick(15, 0, None), tick(30, 500, 4.00)]
    spend = sum_window(ticks)

    assert min(t.total_tokens for t in ticks) == 0
    assert spend.tokens == 1000
    assert spend.spend_usd == pytest.approx(8.00)


# --------------------------------------------------------------------------
# Absence reads as unknown, never as zero.
# --------------------------------------------------------------------------


def test_empty_window_is_unobserved_not_zero_spend():
    spend = sum_window([])

    assert spend.tick_count == 0
    assert spend.observed is False
    assert spend.spend_usd == 0.0  # arithmetic identity, NOT a claim about reality

    state = quota_state(allowance_usd=100.0, spend=spend, window_elapsed_fraction=0.5)
    assert state is not None
    assert state.spend_known is False
    # The dangerous substitution: a stopped producer rendering as a full tank.
    assert state.fraction_remaining is None
    assert state.pace is None
    assert state.projected_window_usd is None
    assert state.exhausted is False


def test_would_refuse_fails_closed_when_spend_is_unknown():
    """An unobserved window may be fully spent. Treating no data as $0 turns a
    producer outage into an unlimited budget.
    """
    state = quota_state(
        allowance_usd=100.0, spend=sum_window([]), window_elapsed_fraction=0.5
    )
    assert state is not None
    assert state.would_refuse(0.01) is True


def test_silent_ticks_are_observed_and_genuinely_zero():
    """18 consecutive all-zero ticks on 2026-08-26 18:14-22:31 UTC were real
    silence -- cross-checked against host transcript mtimes (zero
    ~/.claude/projects/*.jsonl files modified in that window). A silent tick
    still counts as having heard from the producer.
    """
    spend = sum_window([tick(15 * i, 0, None) for i in range(18)])

    assert spend.tick_count == 18
    assert spend.active_tick_count == 0
    assert spend.observed is True
    assert spend.spend_usd == 0.0
    assert spend.is_floor is False

    state = quota_state(allowance_usd=100.0, spend=spend, window_elapsed_fraction=0.5)
    assert state is not None
    assert state.spend_known is True
    assert state.fraction_remaining == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Undercounts are disclosed, not silently summed away.
# --------------------------------------------------------------------------


def test_unpriced_activity_makes_the_sum_a_floor():
    """Tokens with no priceable model. Live 2026-08-27: 0 of 1,273 ticks, but
    the case is reachable and must not pass as a complete total.
    """
    spend = sum_window([tick(0, 500, None), tick(15, 100, 2.00)])

    assert spend.spend_usd == pytest.approx(2.00)
    assert spend.unpriced_active_tick_count == 1
    assert spend.is_floor is True


def test_disclosed_unpriced_sessions_make_the_sum_a_floor():
    """A priced tick that could not price every session inside it. Live
    2026-08-27: exactly 1 of 1,273 ticks. Rare, real.
    """
    spend = sum_window([tick(0, 500, 4.00, unpriced=2)])

    assert spend.spend_usd == pytest.approx(4.00)
    assert spend.unpriced_session_count == 2
    assert spend.is_floor is True


def test_a_fully_priced_window_is_not_a_floor():
    spend = sum_window([tick(0, 500, 4.00), tick(15, 0, None)])
    assert spend.is_floor is False


# --------------------------------------------------------------------------
# Unconfigured is not the same as exhausted.
# --------------------------------------------------------------------------


def test_no_allowance_configured_returns_none():
    spend = sum_window([tick(0, 100, 5.00)])

    assert quota_state(allowance_usd=0.0, spend=spend, window_elapsed_fraction=0.5) is None
    assert quota_state(allowance_usd=0.005, spend=spend, window_elapsed_fraction=0.5) is None
    assert quota_state(allowance_usd=math.nan, spend=spend, window_elapsed_fraction=0.5) is None
    assert quota_state(allowance_usd=math.inf, spend=spend, window_elapsed_fraction=0.5) is None


def test_min_meaningful_allowance_is_admitted():
    spend = sum_window([tick(0, 100, 5.00)])
    state = quota_state(
        allowance_usd=MIN_MEANINGFUL_ALLOWANCE_USD, spend=spend, window_elapsed_fraction=0.5
    )
    assert state is not None


def test_unconfigured_and_exhausted_are_distinguishable():
    spend = sum_window([tick(0, 100, 20.00)])

    unconfigured = quota_state(allowance_usd=0.0, spend=spend, window_elapsed_fraction=0.5)
    exhausted = quota_state(allowance_usd=10.0, spend=spend, window_elapsed_fraction=0.5)

    assert unconfigured is None
    assert exhausted is not None
    assert exhausted.exhausted is True
    assert exhausted.remaining_usd == 0.0


# --------------------------------------------------------------------------
# Arithmetic, hand-computed.
# --------------------------------------------------------------------------


def test_state_arithmetic_hand_computed():
    """spend 18.00 of a 100.00 allowance, half the window elapsed.

    spent_fraction      = 18 / 100      = 0.18
    fraction_remaining  = 1 - 0.18      = 0.82
    remaining_usd       = 100 - 18      = 82.00
    pace                = 0.18 / 0.5    = 0.36
    projected_window    = 18 / 0.5      = 36.00
    """
    spend = sum_window([tick(0, 100, 5.00), tick(15, 200, 10.00), tick(30, 50, 3.00)])
    state = quota_state(allowance_usd=100.0, spend=spend, window_elapsed_fraction=0.5)
    assert state is not None

    assert state.spent_fraction == pytest.approx(0.18)
    assert state.fraction_remaining == pytest.approx(0.82)
    assert state.remaining_usd == pytest.approx(82.00)
    assert state.pace == pytest.approx(0.36)
    assert state.projected_window_usd == pytest.approx(36.00)
    assert state.exhausted is False


def test_would_refuse_boundary():
    """spend 18.00, allowance 100.00. 82.00 exactly fits; 82.01 does not."""
    spend = sum_window([tick(0, 100, 5.00), tick(15, 200, 10.00), tick(30, 50, 3.00)])
    state = quota_state(allowance_usd=100.0, spend=spend, window_elapsed_fraction=0.5)
    assert state is not None

    assert state.would_refuse(82.00) is False
    assert state.would_refuse(82.01) is True
    assert state.would_refuse(-5.0) is False  # negative cost clamps to 0


def test_pace_above_one_means_burning_faster_than_the_clock():
    """spend 60.00 of 100.00 with a quarter of the window gone.
    spent_fraction 0.6 / elapsed 0.25 = pace 2.4; projected 60/0.25 = 240.00.
    """
    spend = sum_window([tick(0, 100, 60.00)])
    state = quota_state(allowance_usd=100.0, spend=spend, window_elapsed_fraction=0.25)
    assert state is not None

    assert state.pace == pytest.approx(2.4)
    assert state.projected_window_usd == pytest.approx(240.00)
    assert state.exhausted is False  # not yet -- which is why pace is the number to watch


def test_pace_undefined_before_the_window_starts():
    spend = sum_window([tick(0, 100, 5.00)])
    state = quota_state(allowance_usd=100.0, spend=spend, window_elapsed_fraction=0.0)
    assert state is not None
    assert state.pace is None
    assert state.projected_window_usd is None


# --------------------------------------------------------------------------
# Advisory is not a flag.
# --------------------------------------------------------------------------


def test_mode_is_advisory_and_there_is_no_enforcing_option():
    """Hub holds the docker socket, so no software cap is enforceable wherever
    the logic lives. There is no honest value to flip this to.
    """
    spend = sum_window([tick(0, 100, 5.00)])
    state = quota_state(allowance_usd=100.0, spend=spend, window_elapsed_fraction=0.5)
    assert state is not None
    assert state.mode == "advisory"

    with pytest.raises((TypeError, AttributeError)):
        quota_state(  # type: ignore[call-arg]
            allowance_usd=100.0,
            spend=spend,
            window_elapsed_fraction=0.5,
            enforcing=True,
        )


# --------------------------------------------------------------------------
# Window clock.
# --------------------------------------------------------------------------


def test_window_elapsed_fraction_hand_computed():
    """2.5h into a 5h window = 0.5."""
    assert window_elapsed_fraction(T0 + timedelta(hours=2.5), T0, 5.0) == pytest.approx(0.5)


def test_window_elapsed_fraction_clamps_both_ends():
    assert window_elapsed_fraction(T0 + timedelta(hours=9), T0, 5.0) == 1.0
    assert window_elapsed_fraction(T0 - timedelta(hours=1), T0, 5.0) == 0.0
    assert window_elapsed_fraction(T0, T0, 0.0) == 0.0


# --------------------------------------------------------------------------
# Non-finite and negative costs must not defeat the fail-closed guarantee.
# Found by code review 2026-08-27; reachable from real data because
# `total_estimated_cost_usd` is `double precision` and Postgres accepts
# 'NaN'/'Infinity'.
# --------------------------------------------------------------------------


def test_nan_cost_is_unknown_not_a_summed_number():
    """Before the fix: NaN propagated into spend_usd, `max(0.0, nan)` returned
    0.0 so the display read "0.0% remaining", and would_refuse approved a
    billion-dollar ask. Silent inversion of the module's whole purpose.
    """
    spend = sum_window([tick(0, 100, float("nan")), tick(15, 100, 5.00)])

    assert spend.spend_usd == pytest.approx(5.00)
    assert math.isfinite(spend.spend_usd)
    assert spend.unpriced_active_tick_count == 1
    assert spend.is_floor is True

    state = quota_state(allowance_usd=100.0, spend=spend, window_elapsed_fraction=1.0)
    assert state is not None
    assert state.fraction_remaining == pytest.approx(0.95)
    assert state.would_refuse(1_000_000_000.0) is True


def test_infinite_cost_is_also_treated_as_unknown():
    spend = sum_window([tick(0, 100, float("inf"))])
    assert spend.spend_usd == 0.0
    assert spend.unpriced_active_tick_count == 1
    assert spend.is_floor is True


def test_negative_cost_is_rejected_not_netted_out():
    """A negative is corruption, not unknown. Netting it against real spend
    would let a bad row buy back budget.
    """
    with pytest.raises(ValueError, match="cost_usd"):
        sum_window([tick(0, 100, 500.00), tick(15, 100, -500.00)])


def test_quota_state_rejects_a_hand_built_nonfinite_window():
    """`sum_window` can no longer produce this, but `WindowSpend` is public."""
    bad = WindowSpend(
        tick_count=1,
        active_tick_count=1,
        spend_usd=float("nan"),
        tokens=100,
        unpriced_session_count=0,
        unpriced_active_tick_count=0,
    )
    with pytest.raises(ValueError, match="spend_usd"):
        quota_state(allowance_usd=100.0, spend=bad, window_elapsed_fraction=1.0)

    worse = WindowSpend(1, 1, -5.0, 100, 0, 0)
    with pytest.raises(ValueError, match="spend_usd"):
        quota_state(allowance_usd=100.0, spend=worse, window_elapsed_fraction=1.0)


def test_quota_state_re_exposes_is_floor():
    """A caller holding only a QuotaState would otherwise get a clean
    remaining_usd derived from a known-incomplete total, unmarked.
    """
    spend = sum_window([tick(0, 500, 80.00, unpriced=3)])
    state = quota_state(allowance_usd=150.0, spend=spend, window_elapsed_fraction=1.0)
    assert state is not None

    assert state.spend_is_floor is True
    assert state.remaining_usd == pytest.approx(70.00)  # a CEILING on remaining
    assert state.would_refuse(2.50) is False  # permissive, and now disclosed
