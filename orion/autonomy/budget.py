"""A dispatch budget denominated in something that can actually run out.

WHY THE OLD ONE IS NOT A BUDGET
-------------------------------
`_derive_daily_risk_cap` sets the daily ceiling to
`ewma + 3 * sd` of Orion's own past *demand* -- three standard deviations
above what it has already been wanting. That cannot bind, by construction. It
is a mirror held up to demand, not a constraint on it. Live drift over the
arc: 17 -> 29 -> 347 -> 554 -> 475 -> 3,475 -> 1,787.

It is also denominated in `risk_score`, which is five hand-typed constants in
a YAML file with 67% of dispatches carrying exactly 0.05. So the unit is
fabricated and the size is self-referential -- two independent reasons it was
never going to constrain anything.

WHAT REPLACES IT
----------------
Motor-seconds: the wall-clock an action actually occupies on the dispatch
path, measured at the send (`substrate_dispatch_results.latency_ms`, live
since 2026-08-21). Real, finite, and NOT under Orion's control -- the day is
24 hours long whatever Orion would prefer.

Measured on the live system the day this shipped: p50 5.0s per action, 1.7x
concurrency, ~40 motor-hours consumed per 24h day.

The allowance is set by an operator, not derived from usage. That is the
whole point: an exogenous number is the only kind that can say no. It also
sidesteps the exchange-rate problem -- denominating the budget in the scarce
resource itself makes allocation a knapsack rather than a conversion, so
nobody has to invent how many nats a GPU-second is worth.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not choose which actions to drop. That is the allocator (step 3), and
it needs this to exist first: you cannot rank by value-per-cost with no cost,
and you cannot enforce a ceiling that is defined as "slightly above whatever
you asked for".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

BudgetMode = Literal["advisory", "enforcing"]

# Below this, treat the allowance as "not configured" rather than as a real
# ceiling of zero. A budget of 0 would refuse every action forever, which is a
# configuration mistake far more likely than a genuine intent, and one that
# would look identical to a dead dispatcher.
MIN_MEANINGFUL_ALLOWANCE_SEC = 1.0


@dataclass(frozen=True)
class BudgetState:
    """Where the day stands. Pure arithmetic; the caller supplies the numbers."""

    allowance_sec: float
    spent_sec: float
    elapsed_fraction: float

    mode: BudgetMode

    @property
    def remaining_sec(self) -> float:
        return max(0.0, self.allowance_sec - self.spent_sec)

    @property
    def spent_fraction(self) -> float:
        if self.allowance_sec <= 0:
            return 0.0
        return self.spent_sec / self.allowance_sec

    @property
    def exhausted(self) -> bool:
        return self.spent_sec >= self.allowance_sec

    @property
    def pace(self) -> float:
        """Spend rate against clock rate. 1.0 means exactly on pace to finish
        the allowance as the day ends; 2.0 means burning it twice as fast as
        the day is passing.

        This is the number worth watching in advisory mode, not `exhausted`:
        by the time a budget is exhausted the interesting decision is hours
        past. Undefined before any of the day has elapsed, reported as 0.0.
        """
        if self.elapsed_fraction <= 0.0:
            return 0.0
        return self.spent_fraction / self.elapsed_fraction

    @property
    def projected_day_sec(self) -> float:
        """What the day ends at if the current pace holds."""
        if self.elapsed_fraction <= 0.0:
            return 0.0
        return self.spent_sec / self.elapsed_fraction

    def would_refuse(self, cost_sec: float) -> bool:
        """Would spending this much be refused, in ENFORCING mode?

        Reported in advisory mode too, and that is the point: an advisory
        budget whose only output is "still fine" is the switch-that-changes-
        nothing this repo bans. It has to say what it WOULD have stopped, or
        there is nothing to decide the flip on.
        """
        return (self.spent_sec + max(0.0, cost_sec)) > self.allowance_sec


def budget_state(
    *,
    allowance_sec: float,
    spent_sec: float,
    elapsed_fraction: float,
    enforcing: bool,
) -> BudgetState | None:
    """None when no meaningful allowance is configured.

    None means "no budget", which is different from "a budget with nothing
    left" and must not be collapsed into it -- one is unconfigured and the
    other is a real ceiling reached. Callers that treat them alike will
    either refuse everything or nothing, and both look like a broken
    dispatcher rather than a policy.
    """
    if not math.isfinite(allowance_sec) or allowance_sec < MIN_MEANINGFUL_ALLOWANCE_SEC:
        return None
    if not math.isfinite(spent_sec) or spent_sec < 0:
        raise ValueError(f"spent_sec must be finite and >= 0, got {spent_sec!r}")
    return BudgetState(
        allowance_sec=float(allowance_sec),
        spent_sec=float(spent_sec),
        elapsed_fraction=min(1.0, max(0.0, float(elapsed_fraction))),
        mode="enforcing" if enforcing else "advisory",
    )


def day_elapsed_fraction(now, day_start) -> float:
    """How far through the budget day we are, in [0, 1]."""
    total = 24 * 60 * 60
    elapsed = (now - day_start).total_seconds()
    return min(1.0, max(0.0, elapsed / total))
