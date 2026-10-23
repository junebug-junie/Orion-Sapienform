"""Organ 1, Body: one reading per constraint, in its own unit, with an explicit no_reading.

Pure and I/O-free: no clock, no database, no bus. orion-durable-runs reads the sources
(``services/orion-durable-runs/app/physiology_store.py``) and passes plain values here.

Constraints and their sources (design doc, Organ 1 table; A3 scope, no forecast, no price):

* ``heat``: athena ``cabinet_temp_c`` as the regulate step already read it
  (``ArousalInputsV1.cabinet_*``, ``orion.autonomy.cabinet_heat``'s own verdict). Line: the soft
  line, ``ORION_PHYSIOLOGY_HEAT_SOFT_LINE_C`` (31 C). The 34 C reflex is untouched.
* ``energy``: Orion's month-to-date dollars. kWh per hour per source from
  ``orion_energy_hourly`` (athena, circe, hecate, cooling; sql-writer's rollup), each hour priced
  at that hour's next-kWh rate from ``energy_cost_accrued`` (newest ``computed_at`` per interval,
  or costs triple). RMP prices arrive about a day late: an hour with no price uses the newest
  earlier rate (``carried_kwh`` says how much). Line: ``ORION_PHYSIOLOGY_ENERGY_MONTHLY_ENVELOPE_USD``
  (unset: no line yet, Juniper sets it).
* ``lane:<role>``: GPU slot-seconds held in the last hour from ``gpu_pool_events`` (each lease's
  held interval, clipped to the window; an open grant counts up to now). Line: the role's
  discovered slots x window, from the pool's own state RPC.
* ``attention:juniper``: Orion's outreach messages sent today (local date), every source.
  Line: the outreach daily cap (``ORION_PHYSIOLOGY_JUNIPER_DAILY_CAP``, Hub's cap 4).
* ``claude``: ``ClaudeLimitObservationV1`` (``orion:substrate:claude_limit``) state and limit
  events. Not dollars: the dollar denominator was measured and refuted (orion/schemas/claude_limit.py).

Every reader of a source failing, a source going stale, or a source never observed yields
``level=None`` with a ``no_reading`` reason. Never 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Optional, Sequence

from orion.schemas.physiology import (
    ATTENTION_JUNIPER,
    CLAUDE,
    ENERGY,
    HEAT,
    LANE_PREFIX,
    PhysiologyConstraintReadingV1,
    PhysiologyStateV1,
)

# gpu_pool_events kinds that close a lease and carry held_ms.
LEASE_CLOSE_EVENTS = frozenset({"released", "expired", "aborted"})
LEASE_OPEN_EVENTS = frozenset({"granted"})


@dataclass(frozen=True)
class BodyConfig:
    heat_soft_line_c: float = 31.0
    energy_monthly_envelope_usd: Optional[float] = None
    # The newest orion_energy_hourly hour older than this: the rollup stopped -> stale.
    energy_max_lag_sec: float = 3 * 3600.0
    juniper_daily_cap: int = 4
    # No outreach decision at all (any outcome; live ~6/min) for this long: the decision log or
    # the outreach loop is off, so a count of 0 would be a lie -> stale.
    juniper_decision_max_age_sec: float = 3600.0
    lane_window_sec: float = 3600.0
    # A claude observation received longer ago than this is stale (the producer's slowest
    # window, weekly, emits every 3600 s).
    claude_max_age_sec: float = 3 * 3600.0


# --- inputs (filled by the host's reads) --------------------------------------------------------


@dataclass(frozen=True)
class HeatInput:
    read_ok: bool
    temp_c: Optional[float] = None
    thermal_state: Optional[str] = None   # cabinet_heat verdict; "unknown" = newest reading past grace
    observed_at: Optional[datetime] = None


@dataclass(frozen=True)
class EnergyHour:
    hour_start: datetime
    kwh: float                       # summed over every Orion source that hour


@dataclass(frozen=True)
class EnergyPrice:
    hour_start: datetime
    marginal_usd_per_kwh: float


@dataclass(frozen=True)
class EnergyInput:
    read_ok: bool
    month_start: Optional[datetime] = None
    hours: Sequence[EnergyHour] = ()
    # Rates from before month_start are welcome: they seed the carry-forward for early hours.
    prices: Sequence[EnergyPrice] = ()
    failed_source: Optional[str] = None


@dataclass(frozen=True)
class LeaseEvent:
    lease_id: Optional[str]
    role: Optional[str]
    event: str
    at: datetime
    held_ms: Optional[float] = None


@dataclass(frozen=True)
class LaneInput:
    events_read_ok: bool
    events: Sequence[LeaseEvent] = ()
    # role -> discovered slots. None: the pool state could not be read.
    capacity: Optional[Mapping[str, int]] = None


@dataclass(frozen=True)
class JuniperInput:
    read_ok: bool
    sent_today: Optional[int] = None
    # Sent today that the Hub's daily cap counts (curiosity/reading-finding outreach is exempt).
    cap_counted_today: Optional[int] = None
    newest_decision_at: Optional[datetime] = None


@dataclass(frozen=True)
class ClaudeObservation:
    kind: str                         # session_limit | weekly_limit
    state: str                        # clear | limited | unknown
    observed: bool
    event_count: int
    received_at: datetime
    resets_at: Optional[datetime] = None


@dataclass(frozen=True)
class BodyInputs:
    heat: HeatInput
    energy: EnergyInput
    lanes: LaneInput
    juniper: JuniperInput
    # Latest observation per kind. Empty: nothing received since the host started.
    claude: Sequence[ClaudeObservation] = field(default_factory=tuple)


# --- one function per constraint ----------------------------------------------------------------


def _no(constraint: str, unit: str, reason: str, **kw) -> PhysiologyConstraintReadingV1:
    return PhysiologyConstraintReadingV1(constraint=constraint, unit=unit, level=None, no_reading=reason, **kw)


def heat_reading(inp: HeatInput, cfg: BodyConfig) -> PhysiologyConstraintReadingV1:
    line = dict(budget=cfg.heat_soft_line_c, budget_kind="soft_line")
    if not inp.read_ok:
        return _no(HEAT, "c", "read_failed:cabinet", **line)
    if inp.temp_c is None:
        return _no(HEAT, "c", "no_points:cabinet", **line)
    if inp.thermal_state in (None, "unknown"):
        # cabinet_heat's own grace rule: the newest reading is past grace. Its temp is old news.
        return _no(HEAT, "c", "stale:cabinet", detail={"last_temp_c": round(float(inp.temp_c), 3)}, **line)
    temp = float(inp.temp_c)
    return PhysiologyConstraintReadingV1(
        constraint=HEAT, unit="c", level=round(temp, 3), source_at=inp.observed_at,
        detail={"thermal_state": inp.thermal_state, "over_line_c": round(temp - cfg.heat_soft_line_c, 3)},
        **line)


def energy_reading(inp: EnergyInput, now: datetime, cfg: BodyConfig) -> PhysiologyConstraintReadingV1:
    env = cfg.energy_monthly_envelope_usd
    line = dict(budget=env, budget_kind="envelope" if env is not None else None)
    if not inp.read_ok:
        return _no(ENERGY, "usd", f"read_failed:{inp.failed_source or 'orion_energy_hourly'}", **line)
    month_start = inp.month_start
    hours = [h for h in inp.hours if month_start is None or h.hour_start >= month_start]
    if not hours:
        return _no(ENERGY, "usd", "no_rows:orion_energy_hourly", **line)
    newest = max(h.hour_start for h in hours)
    # An hour row is written as the hour fills; its start is at most an hour old when fresh.
    lag = (now - newest).total_seconds()
    if lag > cfg.energy_max_lag_sec:
        return _no(ENERGY, "usd", "stale:orion_energy_hourly", source_at=newest,
                   detail={"lag_sec": round(lag, 1)}, **line)
    prices = sorted(inp.prices, key=lambda p: p.hour_start)
    if not prices:
        return _no(ENERGY, "usd", "no_rate:energy_cost_accrued", source_at=newest, **line)
    by_hour = {p.hour_start: p.marginal_usd_per_kwh for p in prices}
    usd = kwh_total = priced_kwh = carried_kwh = 0.0
    unpriced_kwh = 0.0
    idx = 0
    last_rate: Optional[float] = None
    for h in sorted(hours, key=lambda x: x.hour_start):
        kwh_total += h.kwh
        while idx < len(prices) and prices[idx].hour_start <= h.hour_start:
            last_rate = prices[idx].marginal_usd_per_kwh
            idx += 1
        rate = by_hour.get(h.hour_start)
        if rate is not None:
            priced_kwh += h.kwh
        elif last_rate is not None:
            rate = last_rate
            carried_kwh += h.kwh
        else:
            unpriced_kwh += h.kwh
            continue
        usd += h.kwh * rate
    if unpriced_kwh > 0 and priced_kwh + carried_kwh == 0:
        return _no(ENERGY, "usd", "no_rate:energy_cost_accrued", source_at=newest, **line)
    return PhysiologyConstraintReadingV1(
        constraint=ENERGY, unit="usd", level=round(usd, 4), source_at=newest,
        detail={"kwh_month_to_date": round(kwh_total, 4), "priced_kwh": round(priced_kwh, 4),
                "carried_rate_kwh": round(carried_kwh, 4), "unpriced_kwh": round(unpriced_kwh, 4),
                "latest_rate_usd_per_kwh": prices[-1].marginal_usd_per_kwh,
                "month_start": month_start.isoformat() if month_start else None},
        **line)


def _lease_intervals(events: Iterable[LeaseEvent], now: datetime) -> tuple[dict[str, list[tuple[datetime, datetime]]], int]:
    """Per role, one (start, end) interval per lease. A lease's closing events (released, and an
    expired before it) each carry held_ms; their union is the lease. A grant with no close yet
    is open: it holds up to ``now``."""
    by_lease: dict[str, dict] = {}
    for e in events:
        if not e.lease_id:
            continue
        rec = by_lease.setdefault(e.lease_id, {"role": None, "start": None, "end": None, "granted_at": None,
                                               "closed": False})
        if e.role:
            rec["role"] = e.role
        if e.event in LEASE_CLOSE_EVENTS:
            rec["closed"] = True
            if e.held_ms is not None and e.held_ms >= 0:
                start = e.at - timedelta(milliseconds=float(e.held_ms))
                rec["start"] = start if rec["start"] is None else min(rec["start"], start)
                rec["end"] = e.at if rec["end"] is None else max(rec["end"], e.at)
        elif e.event in LEASE_OPEN_EVENTS:
            rec["granted_at"] = e.at if rec["granted_at"] is None else min(rec["granted_at"], e.at)
    out: dict[str, list[tuple[datetime, datetime]]] = {}
    open_count = 0
    for rec in by_lease.values():
        role = rec["role"]
        if not role:
            continue
        if rec["closed"]:
            if rec["start"] is None:
                continue
            out.setdefault(role, []).append((rec["start"], rec["end"]))
        elif rec["granted_at"] is not None:
            open_count += 1
            out.setdefault(role, []).append((rec["granted_at"], now))
    return out, open_count


def lane_readings(inp: LaneInput, now: datetime, cfg: BodyConfig) -> list[PhysiologyConstraintReadingV1]:
    window = cfg.lane_window_sec
    capacity = dict(inp.capacity) if inp.capacity is not None else None
    if not inp.events_read_ok:
        roles = sorted(capacity) if capacity else []
        if not roles:
            return [_no(LANE_PREFIX.rstrip(":"), "slot_s", "read_failed:gpu_pool_events")]
        return [_no(f"{LANE_PREFIX}{r}", "slot_s", "read_failed:gpu_pool_events",
                    budget=float(capacity[r]) * window, budget_kind="capacity") for r in roles]
    intervals, open_count = _lease_intervals(inp.events, now)
    w_start = now - timedelta(seconds=window)
    roles = sorted(set(intervals) | set(capacity or {}))
    out: list[PhysiologyConstraintReadingV1] = []
    for role in roles:
        held = 0.0
        leases = 0
        for start, end in intervals.get(role, []):
            s, e = max(start, w_start), min(end, now)
            if e > s:
                held += (e - s).total_seconds()
                leases += 1
        slots = (capacity or {}).get(role)
        budget = float(slots) * window if slots else None
        detail: dict = {"leases": leases, "window_sec": window}
        if slots is not None:
            detail["slots"] = int(slots)
        if capacity is None:
            detail["capacity"] = "pool_state_unread"
        out.append(PhysiologyConstraintReadingV1(
            constraint=f"{LANE_PREFIX}{role}", unit="slot_s", level=round(held, 3),
            budget=budget, budget_kind="capacity" if budget is not None else None, detail=detail))
    if open_count and out:
        out[0].detail["open_leases_all_lanes"] = open_count
    return out


def juniper_reading(inp: JuniperInput, now: datetime, cfg: BodyConfig) -> PhysiologyConstraintReadingV1:
    line = dict(budget=float(cfg.juniper_daily_cap), budget_kind="daily_cap")
    if not inp.read_ok or inp.sent_today is None:
        return _no(ATTENTION_JUNIPER, "messages", "read_failed:endogenous_outreach_decisions", **line)
    if inp.newest_decision_at is None:
        return _no(ATTENTION_JUNIPER, "messages", "no_rows:endogenous_outreach_decisions", **line)
    age = (now - inp.newest_decision_at).total_seconds()
    if age > cfg.juniper_decision_max_age_sec:
        return _no(ATTENTION_JUNIPER, "messages", "stale:endogenous_outreach_decisions",
                   source_at=inp.newest_decision_at, detail={"age_sec": round(age, 1)}, **line)
    return PhysiologyConstraintReadingV1(
        constraint=ATTENTION_JUNIPER, unit="messages", level=float(inp.sent_today),
        source_at=inp.newest_decision_at,
        detail={"cap_counted_today": inp.cap_counted_today}, **line)


_STATE_RANK = {"limited": 2, "unknown": 1, "clear": 0}


def claude_reading(observations: Sequence[ClaudeObservation], now: datetime, cfg: BodyConfig) -> PhysiologyConstraintReadingV1:
    line = dict(budget=0.0, budget_kind="not_limited")
    if not observations:
        return _no(CLAUDE, "limit_events", "no_observation", state="unknown", **line)
    fresh = [o for o in observations if 0 <= (now - o.received_at).total_seconds() <= cfg.claude_max_age_sec]
    newest = max(o.received_at for o in observations)
    if not fresh:
        return _no(CLAUDE, "limit_events", "stale:claude_limit", state="unknown", source_at=newest, **line)
    seen = [o for o in fresh if o.observed]
    if not seen:
        return _no(CLAUDE, "limit_events", "unobserved:claude_limit", state="unknown", source_at=newest, **line)
    # Any kind limited -> limited. A kind unread beside a clear one -> unknown, never clear.
    state = max((o.state if o.observed else "unknown" for o in fresh), key=lambda s: _STATE_RANK.get(s, 1))
    detail: dict = {}
    for o in fresh:
        detail[f"{o.kind}_state"] = o.state if o.observed else "unknown"
        detail[f"{o.kind}_events"] = o.event_count
        if o.resets_at is not None:
            detail[f"{o.kind}_resets_at"] = o.resets_at.isoformat()
    return PhysiologyConstraintReadingV1(
        constraint=CLAUDE, unit="limit_events", level=float(sum(o.event_count for o in seen)),
        state=state, source_at=newest, detail=detail, **line)


def build_physiology_state(now: datetime, inputs: BodyInputs, cfg: BodyConfig = BodyConfig(),
                           warnings: Sequence[str] = ()) -> PhysiologyStateV1:
    constraints = [heat_reading(inputs.heat, cfg), energy_reading(inputs.energy, now, cfg)]
    constraints += lane_readings(inputs.lanes, now, cfg)
    constraints += [juniper_reading(inputs.juniper, now, cfg), claude_reading(inputs.claude, now, cfg)]
    return PhysiologyStateV1(tick_id=f"physiology:{now.isoformat()}", generated_at=now,
                             constraints=constraints, warnings=list(warnings))


__all__ = [
    "BodyConfig",
    "BodyInputs",
    "ClaudeObservation",
    "EnergyHour",
    "EnergyInput",
    "EnergyPrice",
    "HeatInput",
    "JuniperInput",
    "LaneInput",
    "LeaseEvent",
    "build_physiology_state",
    "claude_reading",
    "energy_reading",
    "heat_reading",
    "juniper_reading",
    "lane_readings",
]
