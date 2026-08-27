"""A budget denominated in something somebody else also wants.

WHY MOTOR-SECONDS WERE NEVER GOING TO PRODUCE AN OPPORTUNITY COST
-----------------------------------------------------------------
`orion/autonomy/budget.py` fixed the two defects that made the old risk cap
fake: it is exogenous (an operator sets it, it is not derived from demand) and
it is denominated in a resource that really runs out (the day is 24 hours long
whatever Orion would prefer). Both correct, and both necessary.

They are not sufficient. Motor-seconds are scarce but **uncontested**. Nobody
else is drawing from that pool, so refusing an action returns seconds to a
place no one was waiting on, and "was this worth its cost" has no counterparty
to be worth it *against*. Live: the allowance has never bound, and at
`ENFORCE=false` it could not have.

Claude quota is the first resource in this system with a **second claimant**.
Orion and Juniper draw from the same window. Spending it is not a gauge moving;
it is Juniper not being able to code. That is what an opportunity cost is, and
no amount of allocator sophistication manufactures one where there is no rival.

WHAT THIS MODULE IS
-------------------
The read-only half, and deliberately only that. It reports where the window
stands and, crucially, **what it would have refused**. It changes no dispatch
behaviour and is wired into no allocator.

That ordering is the gate, not timidity. Per the design spec
(`docs/superpowers/specs/2026-08-27-claude-quota-contested-scarcity-design.md`):
run this against real spend for a week, and if it would never have refused
anything, do NOT wire it in -- the allowance is set too high or the action is
too cheap to matter, and shipping it would be the same ornamental scarcity the
2026-07-07 internal-economy spec was correctly refused for.

ADVISORY, AND NOT BY CHOICE
---------------------------
`mode` is hard-coded advisory. Hub holds `/var/run/docker.sock`, so a
Hub-resident agent is root-equivalent on the host and no software cap is
enforceable wherever the logic lives (resolved direction, Juniper 2026-08-14:
"advisory cap + reconciliation. Detect, do not pretend to prevent"). There is
no `enforcing` flag to flip here because there is nothing honest to flip it to.

THE DOLLAR DENOMINATOR HAS BEEN MEASURED AND REFUTED
----------------------------------------------------
Read this before configuring an allowance.

The design spec proposed discovering `allowance_usd` by "hitting the limit once
and recording cumulative spend at that moment". That procedure was run against
15 real rate-limit events inside the ledger's coverage. It does not converge:
the limit fired anywhere between **$85.39 and $289.76** of trailing 5h spend, a
3.4x spread, and the highest 5h window ever observed (**$420.07**) did not trip
it at all -- higher than 14 of the 15 events that did. No threshold separates
limited from not-limited on this axis. Full result:
`docs/superpowers/specs/2026-08-27-quota-window-calibration-finding.md`.

Likely cause: the subscription meters **per-session** windows, while
`dev_economics_ledger_log` is **machine-wide** across every concurrent session.
That is the right denominator for the contested-resource argument -- Orion and
Juniper really do draw from one pool -- and the wrong one for predicting a
per-session ceiling.

So `allowance_usd` is **not calibratable**, which is a stronger statement than
"not yet calibrated". Nothing here may render a fraction of it as though it
were read from, or fitted to, an authority. **Do not wire this into the
allocator on the dollar axis.**

WHAT SURVIVES, AND WHY THIS MODULE STILL EXISTS
-----------------------------------------------
Everything below is denominator-agnostic: summing deltas rather than ranges,
unknown-versus-zero, failing closed on unknown, disclosing undercounts, and
keeping "unconfigured" distinct from "exhausted". Only the units were wrong.

It is kept as (a) an honest spend *report*, and (b) the harness a better
denominator can be dropped into -- the leading candidate being to observe
rate-limit events directly rather than predict them, which needs no calibrated
denominator and cannot be tuned into non-binding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal

QuotaMode = Literal["advisory"]

# Below this, treat the allowance as unconfigured rather than as a real ceiling
# of zero -- same reasoning as budget.py's MIN_MEANINGFUL_ALLOWANCE_SEC. A $0
# ceiling would refuse everything forever, which is a config mistake far more
# likely than an intent, and indistinguishable from a dead reader.
MIN_MEANINGFUL_ALLOWANCE_USD = 0.01


@dataclass(frozen=True)
class LedgerTick:
    """One row of `dev_economics_ledger_log`.

    **This is a DELTA since the previous tick, not a cumulative total.** The
    producer says so plainly -- `services/orion-cocreation-signals/app/
    producers/dev_economics.py`: "the real *growth* in token/word/cost totals
    since the last check".

    The column names (`total_tokens`, `total_estimated_cost_usd`) describe what
    each number sums *within* a row, not whether rows accumulate *across* time,
    and reading them as cumulative understated real daily spend by an order of
    magnitude on the first attempt at this ($27-48/day reported against a real
    $51-583/day). Window spend is therefore SUM, never `max - min`.
    """

    observed_at: datetime
    total_tokens: int
    # None on a silent tick, and deliberately so: ledger_aggregate refuses to
    # emit a fabricated $0.00. None with total_tokens == 0 is real silence;
    # None with total_tokens > 0 is priced-out activity and makes any sum a
    # floor. Those are different facts and are counted separately below.
    cost_usd: float | None
    unpriced_session_count: int = 0


@dataclass(frozen=True)
class WindowSpend:
    """What a window of ticks actually shows. Pure arithmetic over real rows."""

    tick_count: int
    active_tick_count: int
    spend_usd: float
    tokens: int
    unpriced_session_count: int
    unpriced_active_tick_count: int

    @property
    def observed(self) -> bool:
        """Did we hear from the producer at all?

        `False` means the window is UNOBSERVED, which is emphatically not the
        same as "nothing was spent". A stopped producer and a genuinely quiet
        hour both present as no dollars; only the tick count separates them.
        Collapsing them is how an outage reads as unlimited budget.
        """
        return self.tick_count > 0

    @property
    def is_floor(self) -> bool:
        """True when `spend_usd` is a known undercount.

        Either a tick had real tokens but no priceable model, or a priced tick
        disclosed sessions it could not price. Live check 2026-08-27 across
        1,273 ticks: 0 of the former, 1 of the latter -- rare, but real, and it
        must not silently pass as a complete total.
        """
        return self.unpriced_session_count > 0 or self.unpriced_active_tick_count > 0


def sum_window(ticks: Iterable[LedgerTick]) -> WindowSpend:
    """Sum per-tick deltas. See `LedgerTick` for why this is SUM, not max-min."""
    tick_count = 0
    active = 0
    spend = 0.0
    tokens = 0
    unpriced_sessions = 0
    unpriced_active = 0

    for tick in ticks:
        tick_count += 1
        tokens += tick.total_tokens
        unpriced_sessions += tick.unpriced_session_count

        cost = tick.cost_usd
        if cost is not None and not math.isfinite(cost):
            # A NaN cost is "we do not know what this cost", which is exactly
            # what None already means -- so it is folded into unpriced rather
            # than summed. Reachable from real data: `total_estimated_cost_usd`
            # is `double precision` and Postgres accepts 'NaN'/'Infinity'.
            #
            # Summing it silently is not a cosmetic bug. NaN propagates through
            # spend_usd, and because `nan > 0.0` is False, `max(0.0, nan)`
            # returns 0.0 -- so the display reads "0.0% remaining" while
            # would_refuse() approves a billion-dollar ask. That is the exact
            # inversion this module exists to prevent, arriving silently.
            cost = None
        if cost is not None and cost < 0:
            raise ValueError(f"tick cost_usd must be >= 0, got {tick.cost_usd!r}")

        if tick.total_tokens > 0:
            active += 1
            if cost is None:
                unpriced_active += 1
        if cost is not None:
            spend += cost

    return WindowSpend(
        tick_count=tick_count,
        active_tick_count=active,
        spend_usd=spend,
        tokens=tokens,
        unpriced_session_count=unpriced_sessions,
        unpriced_active_tick_count=unpriced_active,
    )


@dataclass(frozen=True)
class QuotaState:
    """Where the window stands. Pure arithmetic; the caller supplies the rows."""

    allowance_usd: float
    spend: WindowSpend
    window_elapsed_fraction: float
    mode: QuotaMode = "advisory"

    @property
    def spend_known(self) -> bool:
        return self.spend.observed

    @property
    def spend_is_floor(self) -> bool:
        """True when every derived number here is a bound, not a value.

        Re-exposed from `WindowSpend` because a caller holding a `QuotaState`
        would otherwise get a clean `remaining_usd` / `would_refuse` answer
        derived from a known-incomplete total with nothing marking it. When
        this is True, `remaining_usd` is a CEILING on what remains and
        `would_refuse` is permissive -- both err toward letting spending
        through, which is the direction that needs disclosing.
        """
        return self.spend.is_floor

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.allowance_usd - self.spend.spend_usd)

    @property
    def spent_fraction(self) -> float:
        if self.allowance_usd <= 0:
            return 0.0
        return self.spend.spend_usd / self.allowance_usd

    @property
    def fraction_remaining(self) -> float | None:
        """Share of the CALIBRATED allowance still unspent, or None if unknown.

        None when the window is unobserved. A caller rendering this must not
        substitute 1.0 -- that is precisely the "producer down reads as full
        tank" failure, and it fails in the dangerous direction.
        """
        if not self.spend_known:
            return None
        return max(0.0, 1.0 - self.spent_fraction)

    @property
    def exhausted(self) -> bool:
        return self.spend_known and self.spend.spend_usd >= self.allowance_usd

    @property
    def pace(self) -> float | None:
        """Spend rate against clock rate; 2.0 is burning the window twice as
        fast as it is passing.

        As in `budget.py`, this is the number worth watching in advisory mode
        rather than `exhausted`: by the time a window is exhausted the
        interesting decision is hours past.
        """
        if not self.spend_known or self.window_elapsed_fraction <= 0.0:
            return None
        return self.spent_fraction / self.window_elapsed_fraction

    @property
    def projected_window_usd(self) -> float | None:
        """What the window ends at if the current pace holds."""
        if not self.spend_known or self.window_elapsed_fraction <= 0.0:
            return None
        return self.spend.spend_usd / self.window_elapsed_fraction

    def would_refuse(self, cost_usd: float) -> bool:
        """Would spending this be refused, if this budget could enforce?

        Reported even though nothing enforces, and that is the entire point of
        shipping the reader first. An advisory budget whose only output is
        "still fine" is the switch-that-changes-nothing this repo bans; it has
        to say what it WOULD have stopped or there is nothing to decide on.

        **Fails closed when spend is unknown.** An unobserved window may be
        fully spent, and the alternative -- treating no data as $0 -- turns a
        producer outage into an unlimited budget. Refusing on unknown is the
        safe direction, and in advisory mode it costs nothing but a log line.
        """
        if not self.spend_known:
            return True
        return (self.spend.spend_usd + max(0.0, cost_usd)) > self.allowance_usd


def quota_state(
    *,
    allowance_usd: float,
    spend: WindowSpend,
    window_elapsed_fraction: float,
) -> QuotaState | None:
    """None when no meaningful allowance is configured.

    None means "no budget", which is different from "a budget with nothing
    left" and must not be collapsed into it -- one is unconfigured and the
    other is a real ceiling reached. Same contract as `budget.budget_state`.
    """
    if not math.isfinite(allowance_usd) or allowance_usd < MIN_MEANINGFUL_ALLOWANCE_USD:
        return None
    # `sum_window` cannot produce either of these, but `WindowSpend` is public
    # and a hand-built one must not inherit that invariant silently. Same
    # contract as `budget.budget_state`'s check on `spent_sec`.
    if not math.isfinite(spend.spend_usd) or spend.spend_usd < 0:
        raise ValueError(f"spend_usd must be finite and >= 0, got {spend.spend_usd!r}")
    return QuotaState(
        allowance_usd=float(allowance_usd),
        spend=spend,
        window_elapsed_fraction=min(1.0, max(0.0, float(window_elapsed_fraction))),
    )


def window_elapsed_fraction(now: datetime, window_start: datetime, window_hours: float) -> float:
    """How far through the quota window we are, in [0, 1]."""
    if window_hours <= 0:
        return 0.0
    total = window_hours * 3600.0
    elapsed = (now - window_start).total_seconds()
    return min(1.0, max(0.0, elapsed / total))
