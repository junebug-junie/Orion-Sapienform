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

Patch B rule changes, from the 10-07..22 replay (docs/superpowers/specs/2026-10-23-body-economy-
heat-and-replay-findings.md, "What this means for Patch B"):

A. **Bounded ramp into bind** (``bind_ramp``, now the default). The design froze a bound price AT
   the ceiling: a one-tick jump from 0 to the ceiling, which breaks the bounded-step rule. The
   price now walks toward the ceiling at ``max_step`` per tick and holds there. It still never
   integrates past the ceiling. ``bind_ramp=False`` keeps the old jump for the replay comparison.
B. **Hysteresis on the bind flag** (``bind_with_hysteresis``). Enter at the line; leave only once
   the floor is ``rearm_margin`` under it. Without it a forecast wandering across the line flips
   the price between the ceiling and the integrator every tick.
C. **The sensitivity rule** (``usable_sensitivity``). The point estimate, floored at 0 and never
   above the interval's upper bound. The old rule (use the upper bound when the interval spans 0;
   kept as ``upper_bound_if_spans_zero`` for the replay) made Orion's share 3-4x the point and held
   the heat price at its ceiling, unflagged, for 18-20 h.
D. **Data-sized ceilings** (``ceiling_for``). ``size_ceiling`` when graded want values and costs
   exist; otherwise the provisional ceiling, labelled ``provisional`` wherever it is shown.
E. **The stuck-price check** (``ceiling_run_since`` / ``is_stuck``). A price at its ceiling for more
   than one horizon without ``exogenous_bind`` is flagged, never left silently pinned.

``slack_decay_floor``: when the constraint is slack the price falls by at least this much per tick.
Pure dual ascent falls by ``eta * |excess|``, which is tiny when use sits just under the budget, so
a price can linger long after the constraint clears. Set it to ``ceiling / horizon`` to return to 0
within one horizon (R3).

``PriceParams.per_tick`` scales the per-hour rates to the real tick length, so a 120 s live tick and
the replay's 1 h tick follow the same path per hour.

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
    bind_ramp: bool = True

    def __post_init__(self) -> None:
        for name in ("eta", "ceiling", "max_step", "budget_floor"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a finite positive number, got {v!r}")
        if not (math.isfinite(self.slack_decay_floor) and self.slack_decay_floor >= 0):
            raise ValueError("slack_decay_floor must be finite and >= 0")
        if self.slack_decay_floor > self.max_step:
            raise ValueError("slack_decay_floor must not exceed max_step (bounded-step rule)")

    def per_tick(self, tick_sec: float, *, rate_sec: float = 3600.0) -> "PriceParams":
        """These params read as per-``rate_sec`` rates, scaled to one tick of ``tick_sec``.
        ``eta``, ``max_step`` and ``slack_decay_floor`` scale; ``ceiling`` and ``budget_floor``
        are levels and do not."""
        if not (math.isfinite(tick_sec) and tick_sec > 0):
            raise ValueError(f"tick_sec must be a finite positive number, got {tick_sec!r}")
        k = tick_sec / rate_sec
        return PriceParams(eta=self.eta * k, ceiling=self.ceiling, max_step=self.max_step * k,
                           budget_floor=self.budget_floor, slack_decay_floor=self.slack_decay_floor * k,
                           bind_ramp=self.bind_ramp)


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
    if not _known(prev_price):
        raise ValueError(f"prev_price must be a finite number, got {prev_price!r}")
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
    """Patch B rule (C): the point estimate, floored at 0 (heat added by load is never negative)
    and never above the interval's upper bound. Unknown stays unknown (None).

    Why not the upper bound: with the corrected fit the upper bounds were 3-4x the points, which
    made Orion's own share look large and held the heat price at its ceiling, unflagged, for
    18-20 h in the 10-07..22 replay (R3 FAIL). The point passes. Use the short-run (2 h step
    response) fit, whose interval clears 0 for both GPU hosts."""
    if not (_known(point) and _known(lo) and _known(hi)):
        return None
    return max(0.0, min(float(point), float(hi)))


def upper_bound_if_spans_zero(point: Optional[float], lo: Optional[float], hi: Optional[float]) -> Optional[float]:
    """SUPERSEDED by ``usable_sensitivity`` (Patch B). The design's first rule: if the interval
    includes 0, use its upper bound. Kept only so the replay can show why it was dropped."""
    if not (_known(point) and _known(lo) and _known(hi)):
        return None
    value = hi if lo <= 0.0 <= hi else point
    return max(0.0, float(value))


def bind_with_hysteresis(prev_bind: bool, *, floor_use: Optional[float], budget: Optional[float],
                         rearm_margin: float) -> Optional[bool]:
    """Patch B rule (B), any unit. ``floor_use`` is the use Orion cannot avoid (the room without
    Orion's share; the month's spend at the always-on floor). Enter bind when it reaches the
    budget; while bound, leave only once it is ``rearm_margin`` under. None when unknown."""
    if not (_known(floor_use) and _known(budget)):
        return None
    line = float(budget) - (max(0.0, rearm_margin) if prev_bind else 0.0)
    return float(floor_use) >= line


def heat_contribution(sensitivity_c_per_100w: Mapping[str, Optional[float]],
                      incremental_watts: Mapping[str, Optional[float]]) -> Optional[float]:
    """Orion's own heat share in degC: sum over hosts of sensitivity x incremental watts / 100.

    Any host with a sensitivity but no watts reading (or vice versa) makes the whole level
    unknown (None). A missing host is not a host drawing 0 W. An empty sensitivity map is
    also unknown, not a known zero."""
    if not sensitivity_c_per_100w:
        return None
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
    return bind_with_hysteresis(prev_bind, floor_use=exogenous_c, budget=soft_line_c, rearm_margin=rearm_margin_c)


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


def ceiling_for(provisional: float, want_values: Sequence[float] = (), costs: Sequence[float] = (),
                *, top_frac: float = 0.1) -> tuple[float, str]:
    """Patch B rule (D): (ceiling, source). ``sized`` when ``size_ceiling`` has graded want values
    and per-act costs to work from; otherwise the ``provisional`` ceiling, which every display
    must label as such. Until Appetite (Patch C) produces graded wants, every ceiling is
    provisional: there is nothing yet to size it against."""
    sized = size_ceiling(want_values, costs, top_frac=top_frac)
    if sized is not None and math.isfinite(sized) and sized > 0:
        return float(sized), "sized"
    if not (math.isfinite(provisional) and provisional > 0):
        raise ValueError(f"provisional ceiling must be a finite positive number, got {provisional!r}")
    return float(provisional), "provisional"


def ceiling_run_since(prev_since: Optional[float], *, price: float, ceiling: float, status: str,
                      now: float, rel_tol: float = 1e-6) -> Optional[float]:
    """Patch B rule (E), part 1: when the current run AT the ceiling WITHOUT exogenous_bind began
    (any clock in seconds), or None when the price is not in such a run. A flagged
    exogenous_bind hold is the anti-windup clamp doing its job; it is not counted."""
    at_ceiling = price >= ceiling * (1.0 - rel_tol)
    if not at_ceiling or status == EXOGENOUS_BIND:
        return None
    return prev_since if prev_since is not None else now


def is_stuck(run_since: Optional[float], *, now: float, horizon_sec: float) -> bool:
    """Patch B rule (E), part 2: at the ceiling, unflagged, for more than one horizon (R3)."""
    return run_since is not None and (now - run_since) > horizon_sec


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
