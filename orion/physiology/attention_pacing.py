"""The price of one more message to Juniper today: a pace-based threshold (one attention budget).

Juniper (2026-11-10): reach-out should follow how much Orion wants to say something, not a
schedule, and land around 3 to 6 messages a day. Every door that messages her now counts on one
line (``attention:juniper``), and the price of the next message grows with the day's pace.

Theory anchor: the online knapsack threshold of Zhou, Chakrabarty and Lukose (2008, "Budget
constrained bidding in keyword auctions and online knapsack problems"). Items arrive one at a time
with value per unit of capacity in [L, U]; accept an item when its value is at least

    psi(z) = (L / e) * (U * e / L) ** z,     z = the fraction of the budget already used.

psi(0) = L / e (the first message is cheap: anything worth sending goes), psi(1) = U (the last
unit costs the most any message is worth). It is (ln(U/L) + 1)-competitive against an adversarial
arrival order, and no online rule does better -- for a fixed capacity and items small against it. Here
the capacity is ~5 whole messages and the budget accrues, so this rule is ANCHORED on that threshold
(its shape and end points), not a claim to inherit the guarantee. The budget is not the whole day's cap at once:
it accrues over Juniper's waking day (budget pacing, as in Balseiro and Gur 2019: spend tracks a
target rate). So ``z = sent_today / allowance(now)``, with one message allowed at the start of the
day and the rest accruing evenly until the quiet hours begin, up to ``pace_budget`` (the cap less one:
see ``PACE_HEADROOM``). Ahead of pace, z passes 1 and the
price sits at U: only a message worth the most any message can be worth goes, and it still needs
net > 0.

L and U, from live data (want_ledger, family hub.outreach, 2026-10-27 .. 11-10):

* U = 0.65: the family's weight (0.648 on 11-10). value = weight x rank and rank <= 1, so no
  outreach want is worth more than its weight.
* L = 0.30: the lowest outreach want value in 14 days was 0.306 (1,797 bids).

Typical value ~0.38: it goes while z < ~0.70 (``psi`` = 0.38). Paced to 5, that is about the 4th
message by the end of the waking day; a want worth 0.60 (a long silence, an unusual event) goes
until z ~0.95.

The hard cap (``JUNIPER_DAILY_CAP``) is the one owner of the number: the Hub's daily_cap gate and
the body's attention:juniper line both import it. The env key it replaced
(``HUB_ENDOGENOUS_OUTREACH_DAILY_CAP``) is refused anywhere it reappears
(scripts/check_env_key_single_source.py RETIRED_KEYS).

Pure.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

# Messages to Juniper per local day, every door counted (Juniper, 2026-11-10: "3 to 6"). The one owner.
JUNIPER_DAILY_CAP = 6
# Sends that do not count on the line (``endogenous_outreach_decisions.result_json.source``): the strain
# report exists to speak when strain holds everything else. Forced (operator) sends are excluded by their
# own ``forced`` column. Collapse-mirror replies and the Claude room relay answer Juniper's own input and
# never write an outreach decision row. Read by Hub's daily cap and the body's attention:juniper line.
CAP_EXEMPT_SOURCES = ("strain_report",)
# Sends whose decision row says Juniper asked for them (``result_json.door_policy``): a curiosity run she
# queued in chat (door policy, 2026-11-10). A send she triggered is not Orion spending her attention.
CAP_EXEMPT_DOOR_POLICIES = ("asked_by_juniper",)
# The price paces the day to one message under the cap. The last slot stays free for messages that do not
# wait on the price: an office visit at the end of its 5-minute window, a curiosity Door-A send (its run
# already bid). Replay (evals/run_outreach_attention_budget_replay.py, 14 days to 11-10): pacing to the full
# cap of 6 hit the cap on 13 of 14 days and refused 2 of 4 office visits; pacing to 5 hit it on 2 days and
# refused none, at 4.9 messages a day.
PACE_HEADROOM = 1


def pace_budget(cap: int = JUNIPER_DAILY_CAP) -> int:
    """The messages the price paces the day to: the cap less the headroom (at least 1). Pure."""
    return max(1, int(cap) - PACE_HEADROOM)


# Want-value bounds of one message (see module docstring for the data behind them).
VALUE_LOW = 0.30
VALUE_HIGH = 0.65


def _window(now: datetime, *, tz: str, start_hour: int, end_hour: int) -> tuple[datetime, datetime]:
    zone = ZoneInfo(tz)
    local = now.astimezone(zone)
    day = datetime(local.year, local.month, local.day, tzinfo=zone)
    if start_hour < 0 or end_hour < 0 or start_hour >= end_hour:
        return day, day + timedelta(days=1)
    return day + timedelta(hours=start_hour), day + timedelta(hours=end_hour)


def waking_span_hours(now: datetime, *, tz: str, start_hour: int, end_hour: int) -> float:
    """Length of today's waking day in hours (DST-correct). Pure."""
    start, end = _window(now, tz=tz, start_hour=start_hour, end_hour=end_hour)
    return (end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds() / 3600.0


def waking_end(now: datetime, *, tz: str, start_hour: int, end_hour: int) -> datetime:
    """When today's waking day ends (UTC). Pure."""
    return _window(now, tz=tz, start_hour=start_hour, end_hour=end_hour)[1].astimezone(timezone.utc)


def waking_fraction(now: datetime, *, tz: str, start_hour: int, end_hour: int) -> float:
    """Share of Juniper's waking day [start_hour, end_hour) local that has passed, 0..1. Pure.

    Before the window: 0. After it: 1. Quiet hours off or wrapping midnight (start >= end, or either
    < 0): the whole local day."""
    start, end = _window(now, tz=tz, start_hour=start_hour, end_hour=end_hour)
    # Through UTC: same-tzinfo subtraction is wall-clock and is 1 h off on DST days.
    span = (end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds()
    done = (now.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds()
    return min(1.0, max(0.0, done / span)) if span > 0 else 1.0


def allowance(fraction: float, budget: int) -> float:
    """Messages the day has budgeted by now: 1 at the start, ``budget`` at the end. Pure."""
    budget = max(1, int(budget))
    return 1.0 + (budget - 1) * min(1.0, max(0.0, float(fraction)))


def pace(sent: float, fraction: float, budget: int) -> float:
    """z: messages already sent today over what the day has budgeted by now (> 1 = ahead). Pure."""
    return max(0.0, float(sent)) / allowance(fraction, budget)


def message_price(z: float, *, low: float = VALUE_LOW, high: float = VALUE_HIGH) -> float:
    """psi(z): the want-value one more message must beat. z clamped to [0, 1]. Pure."""
    z = min(1.0, max(0.0, float(z)))
    return (low / math.e) * ((high * math.e / low) ** z)


def pace_for_value(value: float, *, low: float = VALUE_LOW, high: float = VALUE_HIGH) -> Optional[float]:
    """The largest z at which a message worth ``value`` still beats the price (psi^-1). None: never
    (worth no more than the cheapest price). Above ``high``: 1 (any pace up to the allowance). Pure."""
    if not math.isfinite(value) or value <= low / math.e:
        return None
    if value >= high:
        return 1.0
    return math.log(value * math.e / low) / math.log(high * math.e / low)


# Relief aims just past equality: at price == value the net is 0, which still defers (review finding).
RELIEF_VALUE_MARGIN = 1e-3


def relief_sec(sent: float, value: float, *, allowance_now: float, rate_per_hour: float,
               budget: Optional[int] = None) -> Optional[float]:
    """Seconds until the day's allowance grows enough that a message worth ``value`` beats the price.

    ``allowance_now`` and ``rate_per_hour`` as Budget posts them on the reading; ``budget`` the
    day's pace budget (the allowance never passes it). None: not by this accrual (the value never
    beats the price, the allowance it needs is past the day's budget, or the allowance is not
    growing: outside the waking day). 0: it already does. Pure."""
    z_star = pace_for_value(value - RELIEF_VALUE_MARGIN)
    if z_star is None or z_star <= 0:
        return None
    need = max(0.0, float(sent)) / z_star
    if need <= allowance_now:
        return 0.0
    if budget is not None and need > float(budget):
        return None
    if not (rate_per_hour > 0 and math.isfinite(rate_per_hour)):
        return None
    return (need - allowance_now) / rate_per_hour * 3600.0
