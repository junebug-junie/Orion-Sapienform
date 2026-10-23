"""Pure price math for the Budget organ (body-economy design, Organ 2).

One price per constraint. The price says what one unit of the resource is worth giving up
right now. It moves by dual ascent:

    price <- clip(price + step, 0, ceiling)
    step   = clip(eta * (use - budget) / max(budget, budget_floor), -max_step, +max_step)

Three rules from the design sit on top of the textbook rule:

1. ``no_reading`` is never 0. If the use or the budget is unknown, the price is held where it
   was and the step is marked ``no_reading``. Unknown is not "nothing used".
2. Anti-windup. If the constraint cannot be met even with Orion's discretionary load at zero
   (``exogenous_bind``), the price does not integrate. It is set to the bounded ceiling and
   held there, so a hot room can never turn into an ever-rising veto.
3. Bounded step. ``max_step`` caps how far one tick can move the price.

``bind_ramp`` is a second optional addition (default False = the design's rule exactly). The
design freezes a bound price AT the ceiling, which is a one-tick jump from 0 to the ceiling and
breaks the bounded-step rule. With ``bind_ramp`` the price walks toward the ceiling at
``max_step`` per tick and holds there; it still never integrates past the ceiling.

``slack_decay_floor`` is an optional addition (default 0 = the design's rule exactly). When the
constraint is slack, the price falls by at least this much per tick. Pure dual ascent falls by
``eta * |excess|``, which is tiny when use sits just under the budget, so a price can linger
long after the constraint clears. The replay reports both settings.

No I/O, no clock, no settings. The caller supplies every number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

LIVE = "live"
EXOGENOUS_BIND = "exogenous_bind"
NO_READING = "no_reading"


@dataclass(frozen=True)
class PriceParams:
    eta: float
    ceiling: float
    max_step: float
    budget_floor: float
    slack_decay_floor: float = 0.0
    bind_ramp: bool = False

    def __post_init__(self) -> None:
        for name in ("eta", "ceiling", "max_step", "budget_floor"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a finite positive number, got {v!r}")
        if not (math.isfinite(self.slack_decay_floor) and self.slack_decay_floor >= 0):
            raise ValueError("slack_decay_floor must be finite and >= 0")


@dataclass(frozen=True)
class PriceStep:
    price: float
    status: str
    excess: Optional[float]
    delta: float


def _known(x: Optional[float]) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def step_price(prev_price: float, *, use: Optional[float], budget: Optional[float],
               exogenous_bind: bool, params: PriceParams) -> PriceStep:
    """One dual-ascent tick with the no_reading, anti-windup and bounded-step rules."""
    prev = min(max(float(prev_price), 0.0), params.ceiling)
    if exogenous_bind:
        target = min(params.ceiling, prev + params.max_step) if params.bind_ramp else params.ceiling
        target = max(target, prev)  # never falls while bound
        return PriceStep(price=target, status=EXOGENOUS_BIND, excess=None, delta=target - prev)
    if not (_known(use) and _known(budget)):
        return PriceStep(price=prev, status=NO_READING, excess=None, delta=0.0)
    budget_v = max(float(budget), 0.0)
    excess = (float(use) - budget_v) / max(budget_v, params.budget_floor)
    delta = max(-params.max_step, min(params.max_step, params.eta * excess))
    if excess <= 0.0:
        delta = min(delta, -params.slack_decay_floor)
    price = min(max(prev + delta, 0.0), params.ceiling)
    return PriceStep(price=price, status=LIVE, excess=excess, delta=price - prev)


def usable_sensitivity(point: Optional[float], lo: Optional[float], hi: Optional[float]) -> Optional[float]:
    """The design's rule for an uncertain sensitivity: if the interval includes 0, use its
    upper bound. Heat added by load is never negative, so the result is floored at 0.
    Unknown stays unknown (None)."""
    if not (_known(point) and _known(lo) and _known(hi)):
        return None
    value = hi if lo <= 0.0 <= hi else point
    return max(0.0, float(value))


def heat_contribution(sensitivity_c_per_100w: Mapping[str, Optional[float]],
                      incremental_watts: Mapping[str, Optional[float]]) -> Optional[float]:
    """Orion's own heat share in degC: sum over hosts of sensitivity x incremental watts / 100.

    Any host with a sensitivity but no watts reading (or vice versa) makes the whole level
    unknown (None). A missing host is not a host drawing 0 W."""
    total = 0.0
    for host, s in sensitivity_c_per_100w.items():
        w = incremental_watts.get(host)
        if not (_known(s) and _known(w)):
            return None
        total += float(s) * max(0.0, float(w)) / 100.0
    return total


def heat_budget(*, soft_line_c: float, forecast_cabinet_c: Optional[float],
                own_contribution_c: Optional[float]) -> tuple[Optional[float], Optional[bool]]:
    """Heat budget = soft line - the forecast cabinet temperature WITHOUT Orion's load, floored at 0.

    Returns (budget, exogenous_bind). exogenous_bind is True when the room would sit at or over
    the line even with Orion's discretionary load at zero. (None, None) when unknown."""
    if not (_known(forecast_cabinet_c) and _known(own_contribution_c)):
        return None, None
    exogenous_c = float(forecast_cabinet_c) - float(own_contribution_c)
    return max(0.0, soft_line_c - exogenous_c), exogenous_c >= soft_line_c


def exogenous_bind_with_hysteresis(prev_bind: bool, *, exogenous_c: Optional[float], soft_line_c: float,
                                   rearm_margin_c: float) -> Optional[bool]:
    """Enter exogenous_bind at the soft line; leave it only once the room (without Orion's share)
    is ``rearm_margin_c`` under the line. Without this, a forecast wandering across the line flips
    the price between the ceiling and the integrator every tick (seen in the 10-07..22 replay).
    None when the exogenous level is unknown."""
    if not _known(exogenous_c):
        return None
    line = soft_line_c - rearm_margin_c if prev_bind else soft_line_c
    return float(exogenous_c) >= line


def size_ceiling(want_values: Sequence[float], costs: Sequence[float], *, top_frac: float = 0.1) -> Optional[float]:
    """Data-sized ceiling: the highest price at which a want at the top ``top_frac`` of value
    still clears at the median cost (price x cost <= value). None if either input is empty
    or the median cost is not positive."""
    vals = sorted(float(v) for v in want_values if _known(v))
    cs = sorted(float(c) for c in costs if _known(c))
    if not vals or not cs:
        return None
    v_q = _quantile(vals, 1.0 - top_frac)
    c_med = _quantile(cs, 0.5)
    if c_med <= 0.0:
        return None
    return v_q / c_med


def month_end_forecast(*, to_date: float, trailing_rate_per_hour: Optional[float],
                       hours_remaining: float) -> Optional[float]:
    """Month-end total = what is already spent + the trailing rate for the hours left."""
    if not (_known(to_date) and _known(trailing_rate_per_hour)):
        return None
    return float(to_date) + float(trailing_rate_per_hour) * max(0.0, hours_remaining)


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def hours_to_rest(prices: Iterable[float], *, eps: float = 1e-9) -> Optional[int]:
    """Number of ticks until the price first reaches 0 (<= eps). None if it never does."""
    for i, p in enumerate(prices):
        if p <= eps:
            return i
    return None
