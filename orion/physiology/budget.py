"""Organ 2 Budget and Organ 5's ``strain`` dial, SHADOW (body economy Patch B2).

Every physiology tick, after the Body readings (``orion.physiology.body``), one price per
constraint by the dual-ascent rule in ``orion.physiology.prices`` (with the Patch B rules: bounded
ramp into bind, bind hysteresis, the point sensitivity, provisional/sized ceilings, the stuck check).

What each price charges (``use`` against ``budget``, in the reading's own unit):

* ``heat``: ORION'S OWN heat share in degC, sum over hosts of the short-run sensitivity
  (``config/physiology/heat_sensitivity.yaml``) x watts above the host's baseline / 100, against
  the room's headroom: soft line - (cabinet, sensor offset removed, - Orion's share). The forecast
  is persistence (the newest cabinet reading). ``exogenous_bind`` (with 0.5 C hysteresis) when the
  room is over the line even with Orion's share at 0: "the room is too hot and my load is not the
  cause". A host whose sensitivity is missing or older than 30 days makes the heat price
  no_reading, never 0. A host asleep (no watts, or hecate's chassis under 150 W) adds nothing.
* ``energy``: the month-end Orion cost forecast (to-date + trailing hourly rate x hours left)
  against the monthly envelope. ``exogenous_bind`` (5% hysteresis) when the always-on floor alone
  (to-date + the p10 hourly cost x hours left) reaches the envelope.
* ``lane:<role>``: lease-seconds asked of the seat in the last hour (``detail.lease_s``: occupied
  slot time plus gap borrowers) against slots x window. A seat with 0 live slots that still holds a
  lease is ``exogenous_bind``. The waiting backlog is hunger, NOT an input here.
* ``attention:juniper``: today's cap-counted outreach, extrapolated to the end of the local day,
  against the daily cap.
* ``claude``: ``limited`` is ``exogenous_bind`` (the limit resets on its own clock); ``clear`` is a
  use of 0; ``unknown`` is no_reading.

SHADOW: prices are published on ``PhysiologyStateV1`` and read by nothing that decides.

Pure: the caller passes ``now``, the readings, the raw inputs and the carry from the last tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional
from zoneinfo import ZoneInfo

from orion.physiology.body import BodyInputs, priced_hours
from orion.physiology.heat_config import (
    CABINET_SERIES,
    DEFAULT_MAX_AGE_DAYS,
    HeatSensitivityConfig,
    host_sensitivity,
    offset_at,
)
from orion.physiology.prices import (
    EXOGENOUS_BIND,
    NO_READING,
    PriceParams,
    bind_with_hysteresis,
    ceiling_for,
    ceiling_run_since,
    is_stuck,
    month_end_forecast,
    step_price,
)
from orion.schemas.physiology import (
    ATTENTION_JUNIPER,
    CLAUDE,
    ENERGY,
    HEAT,
    LANE_PREFIX,
    PhysiologyConstraintReadingV1,
    PhysiologyPriceV1,
    PhysiologyStateV1,
    PhysiologyStrainV1,
)


@dataclass(frozen=True)
class PriceSpec:
    """Per-HOUR rates (scaled to the real tick by ``PriceParams.per_tick``)."""
    eta: float
    max_step: float
    horizon_sec: float
    ceiling: float = 1.0          # provisional (prices.ceiling_for); want-value per unit
    budget_floor: float = 1.0     # in the reading's unit; guards the excess against a ~0 budget

    def params(self, *, budget_floor: Optional[float] = None) -> PriceParams:
        # slack_decay_floor = ceiling / horizon: back to 0 within one horizon after it clears (R3).
        return PriceParams(eta=self.eta, ceiling=self.ceiling, max_step=self.max_step,
                           budget_floor=budget_floor if budget_floor is not None else self.budget_floor,
                           slack_decay_floor=self.ceiling / (self.horizon_sec / 3600.0), bind_ramp=True)


# The replayed heat/energy rule (findings doc: eta = max_step = 0.25/h, 6 h horizon). A lane's
# horizon is its 1 h window, Juniper's a day; their steps are sized so the decay floor
# (ceiling / horizon) never exceeds max_step (the bounded-step rule).
DEFAULT_SPECS: Mapping[str, PriceSpec] = {
    HEAT: PriceSpec(eta=0.25, max_step=0.25, horizon_sec=6 * 3600.0, budget_floor=0.25),
    ENERGY: PriceSpec(eta=0.25, max_step=0.25, horizon_sec=6 * 3600.0),
    "lane": PriceSpec(eta=1.0, max_step=1.0, horizon_sec=3600.0, budget_floor=60.0),
    ATTENTION_JUNIPER: PriceSpec(eta=0.25, max_step=0.25, horizon_sec=24 * 3600.0, budget_floor=1.0),
    CLAUDE: PriceSpec(eta=0.25, max_step=0.25, horizon_sec=6 * 3600.0),
}


@dataclass(frozen=True)
class BudgetConfig:
    tz: str = "America/Denver"
    heat_soft_line_c: float = 31.0
    heat_rearm_c: float = 0.5
    heat_config: Optional[HeatSensitivityConfig] = None
    heat_config_error: Optional[str] = "not_loaded"
    heat_max_age_days: float = DEFAULT_MAX_AGE_DAYS
    # hecate is on demand and not in the short-run fit: under this chassis draw it is asleep.
    hecate_awake_watts: float = 150.0
    energy_rearm_frac: float = 0.05
    energy_min_hours: int = 24
    energy_trailing_hours: int = 168
    juniper_min_elapsed_h: float = 1.0
    nominal_tick_sec: float = 120.0
    max_tick_sec: float = 600.0     # after a restart or a gap, never step more than 10 min at once
    specs: Mapping[str, PriceSpec] = field(default_factory=lambda: dict(DEFAULT_SPECS))

    def spec(self, constraint: str) -> PriceSpec:
        return self.specs["lane" if constraint.startswith(LANE_PREFIX) else constraint]


@dataclass(frozen=True)
class _Ask:
    """What one constraint asks of the price rule this tick."""
    use: Optional[float]
    budget: Optional[float]
    exogenous_bind: Optional[bool]     # None: unknown (the use/budget is then no_reading too)
    why: str
    no_reading: Optional[str] = None
    budget_floor: Optional[float] = None


def _f(v: Optional[float], nd: int = 2) -> str:
    return "?" if v is None else f"{v:.{nd}f}"


def _parse_dt(v: Any) -> Optional[datetime]:
    if not isinstance(v, str):
        return None
    try:
        d = datetime.fromisoformat(v)
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


# --- one ask per constraint ---------------------------------------------------------------------

def heat_ask(reading: PhysiologyConstraintReadingV1, body: BodyInputs, now: datetime, cfg: BudgetConfig,
             prev_bind: bool) -> _Ask:
    if reading.level is None:
        return _Ask(None, None, None, f"no cabinet reading ({reading.no_reading})", no_reading=reading.no_reading)
    if cfg.heat_config is None:
        reason = f"heat_sensitivity:{cfg.heat_config_error or 'not_loaded'}"
        return _Ask(None, None, None, f"no heat sensitivity config ({reason})", no_reading=reason)
    hw = body.host_watts
    if hw is None or not hw.read_ok:
        return _Ask(None, None, None, "host watts unread", no_reading="read_failed:orion_biometrics_cluster")
    share = 0.0
    parts: list[str] = []
    for host in sorted(set(cfg.heat_config.hosts) | set(hw.watts)):
        watts = hw.watts.get(host)
        entry, sens, why_not = host_sensitivity(cfg.heat_config, host, now, max_age_days=cfg.heat_max_age_days)
        if entry is None and why_not and why_not.startswith("not_fitted"):
            # Not in the fit: it adds nothing only while it is asleep.
            if watts is None or (host == "hecate" and watts < cfg.hecate_awake_watts):
                parts.append(f"{host} asleep")
                continue
            return _Ask(None, None, None, f"{host} is drawing {watts:.0f} W and has no fitted sensitivity",
                        no_reading=f"heat_sensitivity:{why_not}")
        if sens is None:
            return _Ask(None, None, None, f"{host}: {why_not}", no_reading=f"heat_sensitivity:{why_not}")
        if watts is None or not math.isfinite(watts):
            return _Ask(None, None, None, f"{host}: no watts this hour", no_reading=f"no_watts:{host}")
        over = max(0.0, float(watts) - entry.baseline_watts)
        c = sens * over / 100.0
        share += c
        parts.append(f"{host} +{over:.0f} W x {sens:.2f} = {c:.2f} C")
    raw = float(reading.level)
    off, rec = offset_at(cfg.heat_config, CABINET_SERIES, now)
    cabinet = raw - off
    room = cabinet - share
    bind = bind_with_hysteresis(prev_bind, floor_use=room, budget=cfg.heat_soft_line_c, rearm_margin=cfg.heat_rearm_c)
    budget = max(0.0, cfg.heat_soft_line_c - room)
    sensor = f" (sensor offset -{off:.1f} C, {rec.change_event})" if rec is not None else ""
    why = (f"cabinet {raw:.1f} C{sensor} -> {cabinet:.1f} C; Orion's share {share:.2f} C "
           f"[{'; '.join(parts) or 'no host'}]; room without Orion {room:.1f} C vs line {cfg.heat_soft_line_c:.1f} C")
    if bind:
        why = "room over the line without Orion's load (not my load); " + why
    return _Ask(share, budget, bind, why)


def _local_month_bounds(now: datetime, tz: ZoneInfo) -> tuple[datetime, datetime]:
    local = now.astimezone(tz)
    start = datetime(local.year, local.month, 1, tzinfo=tz)
    end = datetime(local.year + (local.month == 12), local.month % 12 + 1, 1, tzinfo=tz)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def energy_ask(reading: PhysiologyConstraintReadingV1, body: BodyInputs, now: datetime, cfg: BudgetConfig,
               prev_bind: bool) -> _Ask:
    env = reading.budget
    if reading.level is None:
        return _Ask(None, env, None, f"no energy reading ({reading.no_reading})", no_reading=reading.no_reading)
    if env is None:
        return _Ask(None, None, None, "no monthly envelope set", no_reading="no_envelope")
    month_start, month_end = _local_month_bounds(now, ZoneInfo(cfg.tz))
    hours = [h for h in body.energy.hours if h.hour_start >= month_start]
    usd = [h.kwh * rate for h, rate, how in priced_hours(hours, body.energy.prices) if rate is not None]
    if len(usd) < cfg.energy_min_hours:
        return _Ask(None, env, None, f"{len(usd)} priced hours this month (< {cfg.energy_min_hours})",
                    no_reading=f"forecast_needs_{cfg.energy_min_hours}h")
    trail = usd[-cfg.energy_trailing_hours:]
    rate = sum(trail) / len(trail)
    floor_rate = sorted(trail)[int(0.1 * (len(trail) - 1))]
    left = max(0.0, (month_end - now).total_seconds() / 3600.0)
    to_date = float(reading.level)
    use = month_end_forecast(to_date=to_date, trailing_rate_per_hour=rate, hours_remaining=left)
    floor = to_date + floor_rate * left
    bind = bind_with_hysteresis(prev_bind, floor_use=floor, budget=env, rearm_margin=cfg.energy_rearm_frac * env)
    why = (f"month to date ${to_date:.2f}; forecast ${_f(use)} by month end at ${rate:.3f}/h; "
           f"the always-on floor alone (${floor_rate:.3f}/h) reaches ${floor:.2f} vs envelope ${env:.0f}")
    if bind:
        why = "the envelope is spent by what stays switched on (not by what I do); " + why
    return _Ask(use, env, bind, why, budget_floor=0.05 * env)


def lane_ask(reading: PhysiologyConstraintReadingV1, now: datetime, cfg: BudgetConfig, prev_bind: bool) -> _Ask:
    if reading.level is None:
        return _Ask(None, reading.budget, None, f"no lane reading ({reading.no_reading})", no_reading=reading.no_reading)
    if reading.budget is None:
        return _Ask(None, None, None, "seat capacity unread", no_reading="capacity_unread")
    d = reading.detail
    lease = d.get("lease_s")
    use = float(lease) if isinstance(lease, (int, float)) else float(reading.level)
    window_min = float(d.get("window_sec") or 3600.0) / 60.0
    hunger = d.get("deferrable_queued")
    tail = f"; {hunger} deferrable waiting (hunger, not priced)" if isinstance(hunger, int) and hunger else ""
    if reading.budget <= 0.0:
        bind = use > 0.0
        return _Ask(use, 0.0, bind, f"seat has no live slot{'; a lease still holds it' if bind else ''}{tail}")
    lent = d.get("lent_overlap_s")
    lent_s = f", {float(lent):.0f} s of it lent into idle holds" if isinstance(lent, (int, float)) and lent else ""
    why = f"{use:.0f} lease-s asked of {int(d.get('slots') or 0)} slot(s) in {window_min:.0f} min{lent_s}{tail}"
    return _Ask(use, float(reading.budget), False, why)


def juniper_ask(reading: PhysiologyConstraintReadingV1, now: datetime, cfg: BudgetConfig) -> _Ask:
    if reading.level is None:
        return _Ask(None, reading.budget, None, f"no outreach reading ({reading.no_reading})", no_reading=reading.no_reading)
    tz = ZoneInfo(cfg.tz)
    local = now.astimezone(tz)
    elapsed_h = (local - datetime(local.year, local.month, local.day, tzinfo=tz)).total_seconds() / 3600.0
    sent = float(reading.level)
    pace = sent * 24.0 / elapsed_h if elapsed_h >= cfg.juniper_min_elapsed_h else sent
    return _Ask(pace, reading.budget, False,
                f"{sent:.0f} of {_f(reading.budget, 0)} cap-counted messages by {local:%H:%M} local; day-end pace {pace:.1f}")


def claude_ask(reading: PhysiologyConstraintReadingV1) -> _Ask:
    st = reading.state
    if reading.level is None or st not in ("clear", "limited"):
        return _Ask(None, 0.0, None, f"claude limit state {st or 'unknown'} ({reading.no_reading or 'not both kinds read fresh'})",
                    no_reading=reading.no_reading or "state_unknown")
    if st == "limited":
        resets = [f"{k} resets {v}" for k, v in reading.detail.items() if k.endswith("_resets_at")]
        return _Ask(1.0, 0.0, True, "Claude limit reached (resets on its own clock)" + (f"; {', '.join(resets)}" if resets else ""))
    return _Ask(0.0, 0.0, False, "Claude limits clear")


# --- the tick -----------------------------------------------------------------------------------

def _ask_for(reading: PhysiologyConstraintReadingV1, body: BodyInputs, now: datetime, cfg: BudgetConfig,
             prev_bind: bool) -> Optional[_Ask]:
    c = reading.constraint
    if c == HEAT:
        return heat_ask(reading, body, now, cfg, prev_bind)
    if c == ENERGY:
        return energy_ask(reading, body, now, cfg, prev_bind)
    if c.startswith(LANE_PREFIX):
        return lane_ask(reading, now, cfg, prev_bind)
    if c == ATTENTION_JUNIPER:
        return juniper_ask(reading, now, cfg)
    if c == CLAUDE:
        return claude_ask(reading)
    return None   # e.g. the bare "lane" no_reading placeholder: nothing to price


def price_constraint(reading: PhysiologyConstraintReadingV1, ask: _Ask, prev: Mapping[str, Any], now: datetime,
                     tick_sec: float, cfg: BudgetConfig) -> tuple[PhysiologyPriceV1, dict]:
    spec = cfg.spec(reading.constraint)
    ceiling, source = ceiling_for(spec.ceiling)
    params = spec.params(budget_floor=ask.budget_floor).per_tick(tick_sec)
    prev_price = float(prev.get("price") or 0.0)
    if not (math.isfinite(prev_price) and prev_price >= 0.0):
        prev_price = 0.0
    if ask.no_reading is not None or ask.exogenous_bind is None:
        step = step_price(prev_price, use=None, budget=None, exogenous_bind=False, params=params)
    else:
        step = step_price(prev_price, use=ask.use, budget=ask.budget, exogenous_bind=bool(ask.exogenous_bind),
                          params=params)
    now_s = now.timestamp()
    since = ceiling_run_since(prev.get("ceiling_since"), price=step.price, ceiling=ceiling, status=step.status, now=now_s)
    if step.status == NO_READING:
        state = "no_reading"
        since = prev.get("ceiling_since")   # unknown does not end, or start, a run
    elif step.status == EXOGENOUS_BIND:
        state = "exogenous_bind"
    elif is_stuck(since, now=now_s, horizon_sec=spec.horizon_sec):
        state = "stuck_at_ceiling"
    elif step.excess is not None and step.excess > 0:
        state = "binding"
    else:
        state = "slack"
    why = ask.why if ask.no_reading is None else f"{ask.why}; price held at its last value"
    price = PhysiologyPriceV1(
        value=round(step.price, 6), ceiling=ceiling, ceiling_source=source, bind_state=state,
        use=None if ask.use is None else round(float(ask.use), 4),
        budget=None if ask.budget is None else round(float(ask.budget), 4),
        excess=None if step.excess is None else round(step.excess, 4),
        horizon_sec=spec.horizon_sec, why=why[:600])
    bind = bool(ask.exogenous_bind) if ask.exogenous_bind is not None else bool(prev.get("bind"))
    carry = {"price": step.price, "bind": bind, "ceiling_since": since, "at": now.isoformat()}
    return price, carry


def strain_dial(constraints: list[PhysiologyConstraintReadingV1]) -> PhysiologyStrainV1:
    """Organ 5: the largest live price / its ceiling, and which constraint. exogenous_bind and
    no_reading constraints are listed apart and do not move the dial. A pure function of the
    stored prices, so it can be recomputed from any physiology_state row."""
    exo, stuck, none_ = [], [], []
    best: Optional[tuple[float, str, PhysiologyPriceV1]] = None
    for c in constraints:
        p = c.price
        if p is None:
            continue
        if p.bind_state == "exogenous_bind":
            exo.append(c.constraint)
            continue
        if p.bind_state == "no_reading":
            none_.append(c.constraint)
            continue
        if p.bind_state == "stuck_at_ceiling":
            stuck.append(c.constraint)
        ratio = min(1.0, max(0.0, p.value / p.ceiling))
        if best is None or ratio > best[0]:
            best = (ratio, c.constraint, p)
    if best is None:
        return PhysiologyStrainV1(value=None, constraint=None, exogenous_bind=exo, stuck=stuck, no_reading=none_,
                                  why="no constraint priced live")
    ratio, name, p = best
    why = (f"strain {ratio:.2f}: {name} at {p.value:.3f} of a {p.ceiling_source} ceiling {p.ceiling:g} ({p.bind_state})"
           if ratio > 0 else "strain 0: every live constraint is slack")
    if exo:
        why += f"; out of my hands: {', '.join(exo)}"
    return PhysiologyStrainV1(value=round(ratio, 6), constraint=name if ratio > 0 else None, exogenous_bind=exo,
                              stuck=stuck, no_reading=none_, why=why)


def price_state(state: PhysiologyStateV1, body: BodyInputs, carry: Optional[Mapping[str, Any]], now: datetime,
                cfg: BudgetConfig) -> tuple[PhysiologyStateV1, dict]:
    """One Budget tick. Returns the state with a price on every priceable reading and the strain
    dial, plus the carry for the next tick ({constraint: {price, bind, ceiling_since, at}})."""
    carry = dict(carry or {})
    out_carry: dict = {}
    readings: list[PhysiologyConstraintReadingV1] = []
    warnings = list(state.warnings)
    for r in state.constraints:
        prev = carry.get(r.constraint) or {}
        ask = _ask_for(r, body, now, cfg, bool(prev.get("bind")))
        if ask is None:
            readings.append(r)
            continue
        prev_at = _parse_dt(prev.get("at"))
        tick = (now - prev_at).total_seconds() if prev_at is not None else cfg.nominal_tick_sec
        tick = min(max(tick, 1.0), cfg.max_tick_sec)
        price, out_carry[r.constraint] = price_constraint(r, ask, prev, now, tick, cfg)
        if price.bind_state == "stuck_at_ceiling":
            warnings.append(f"price_stuck_at_ceiling:{r.constraint}")
        readings.append(r.model_copy(update={"price": price}))
    priced = state.model_copy(update={"constraints": readings, "warnings": warnings})
    return priced.model_copy(update={"strain": strain_dial(readings)}), out_carry


__all__ = [
    "BudgetConfig",
    "DEFAULT_SPECS",
    "PriceSpec",
    "energy_ask",
    "heat_ask",
    "juniper_ask",
    "lane_ask",
    "claude_ask",
    "price_constraint",
    "price_state",
    "strain_dial",
]
