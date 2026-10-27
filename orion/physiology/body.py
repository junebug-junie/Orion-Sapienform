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
* ``lane:<role>``: GPU slot-seconds held in the last hour. Closed leases from ``gpu_pool_events``
  (each close's ``held_ms`` interval, merged per lease, clipped to the window); open leases from
  the pool's own projection ``gpu_pool_leases`` (status granted/recalling), held up to now. A
  child call (``hold_lease_id`` set) runs inside its durable-run hold's slot and is not counted
  again. Each held span is charged to the role ITS OWN event names: a lease the pool re-placed
  on another seat (aborted on agent-gpu2, re-granted on agent) is not charged to its last seat
  for time it held elsewhere (Patch B1: that put 2,390 s of agent-gpu2 time on a 1-slot
  lane:agent, 2026-10-23 10:14). The level is slot-seconds OCCUPIED: at most ``slots`` leases
  count at any instant. More can be live at once by design (gpu_pool scheduler H3: a
  higher-priority one-off call runs in an idle durable-run hold's gap), so lease-seconds above
  the slots are reported apart as ``lent_overlap_s``, not as held. Line: the role's discovered
  slots x window, from the pool's own state RPC. The events feed going quiet reads stale, never 0.
  Beside each lane, in ``detail``: Orion's own deferrable self-work waiting for it (queued
  background durable-run holds, ``gpu_pool_state_history.deferrable_depth``) and how long that
  queue has been non-empty. That is hunger, the seed of Appetite; it is not strain and not a
  price input.
* ``attention:juniper``: Orion's outreach messages today (local date) that the Hub's daily cap
  counts (curiosity and reading-finding outreach are exempt from the cap; their sends are in
  ``detail``). Line: that cap (``HUB_ENDOGENOUS_OUTREACH_DAILY_CAP``, 4).
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

# gpu_pool_events kinds that end a lease's hold. released/expired/aborted/retried carry held_ms;
# cancelled/dead_lettered may not (then the lease is closed with no interval).
LEASE_CLOSE_EVENTS = frozenset({"released", "expired", "aborted", "retried", "cancelled", "dead_lettered"})
LEASE_OPEN_EVENTS = frozenset({"granted"})
# Claude announces two independent limits; both must be read fresh before "clear" is true.
CLAUDE_KINDS = ("session_limit", "weekly_limit")


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
    # Newest gpu_pool_events row (any kind; live ~1,500 grants/h) older than this: the events feed
    # (pool -> bus -> sql-writer) stopped -> every lane is stale, not 0.
    lane_events_max_age_sec: float = 900.0
    # A claude observation received longer ago than this is stale (the producer's slowest
    # window, weekly, emits every 3600 s).
    claude_max_age_sec: float = 3 * 3600.0
    # A pool host's newest gpu_pool_state_history snapshot older than this is not read (the pool
    # publishes every ~5 s; arousal's S2 uses the same 30 s).
    backlog_max_age_sec: float = 30.0


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
    event: str                       # a close event, or "granted" for a lease still open
    at: datetime                     # the close time, or the grant time of an open lease
    held_ms: Optional[float] = None
    # Set on a child call: it runs inside that durable-run hold's slot (gpu_pool_leases).
    hold_lease_id: Optional[str] = None


@dataclass(frozen=True)
class LaneInput:
    events_read_ok: bool
    events: Sequence[LeaseEvent] = ()
    # role -> discovered slots. None: the pool state could not be read.
    capacity: Optional[Mapping[str, int]] = None
    # Newest gpu_pool_events row of any kind (freshness of the feed). None: table empty.
    newest_event_at: Optional[datetime] = None


@dataclass(frozen=True)
class BacklogSnapshot:
    host: str
    at: datetime
    # work_class -> queued deferrable self-work (orion.gpu_pool.shed.deferrable_queued).
    # None: this snapshot did not record it (a pool or sql-writer older than #2873) -> unknown.
    deferrable: Optional[Mapping[str, int]]


@dataclass(frozen=True)
class BacklogInput:
    read_ok: bool
    # The newest snapshot per pool host.
    snapshots: Sequence[BacklogSnapshot] = ()
    # (host, work_class) -> the newest snapshot inside the lookback where that class's deferrable
    # queue was empty (or unrecorded). None: non-empty for the whole lookback.
    last_empty_at: Mapping[tuple[str, str], Optional[datetime]] = field(default_factory=dict)
    lookback_sec: float = 6 * 3600.0


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
    # Waiting deferrable self-work (hunger). None: the host did not read it.
    backlog: Optional[BacklogInput] = None


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


def _merge(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    out: list[tuple[datetime, datetime]] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _lease_intervals(events: Iterable[LeaseEvent], now: datetime
                     ) -> tuple[dict[str, list[tuple[datetime, datetime]]], dict]:
    """Per role, the held intervals of every non-child lease.

    Each close carries its own held_ms interval ON ITS OWN ROLE (the seat that event names): a
    lease the pool re-placed (aborted on one seat, re-granted on another) holds on each seat only
    for its time there. Overlapping spans of one lease on one role merge (expired then released is
    one hold); disjoint ones stay apart (a retried lease's wait is not held). A lease still open
    (granted/recalling in the pool's projection) holds from its CURRENT grant up to ``now`` on its
    current role, as well as any closed spans it already had (re-granted after an abort)."""
    by_lease: dict[str, dict] = {}
    for e in events:
        if not e.lease_id:
            continue
        rec = by_lease.setdefault(e.lease_id, {"role": None, "spans": [], "open": None, "child": False})
        if e.hold_lease_id:
            rec["child"] = True
        if e.event in LEASE_CLOSE_EVENTS:
            if e.held_ms is not None and e.held_ms >= 0:
                rec["spans"].append((e.role, e.at - timedelta(milliseconds=float(e.held_ms)), e.at))
            if e.role and rec["role"] is None:
                rec["role"] = e.role
        elif e.event in LEASE_OPEN_EVENTS:
            rec["open"] = (e.role, e.at) if rec["open"] is None or e.at > rec["open"][1] else rec["open"]
    out: dict[str, list[tuple[datetime, datetime]]] = {}
    counts = {"open": 0, "child_skipped": 0, "cross_role": 0}
    for rec in by_lease.values():
        if rec["child"]:
            if rec["spans"] or rec["open"]:
                counts["child_skipped"] += 1
            continue
        fallback = rec["role"] or (rec["open"][0] if rec["open"] else None)
        per_role: dict[str, list[tuple[datetime, datetime]]] = {}
        for role, start, end in rec["spans"]:
            role = role or fallback
            if role:
                per_role.setdefault(role, []).append((start, end))
        if rec["open"] is not None and (rec["open"][0] or fallback):
            counts["open"] += 1
            per_role.setdefault(rec["open"][0] or fallback, []).append((rec["open"][1], now))
        if len(per_role) > 1:
            counts["cross_role"] += 1
        for role, spans in per_role.items():
            out.setdefault(role, []).extend(_merge(spans))
    return out, counts


def occupied_seconds(spans: Iterable[tuple[datetime, datetime]], w_start: datetime, w_end: datetime,
                     slots: Optional[int]) -> tuple[float, float, int]:
    """(slot-seconds occupied, lease-seconds, spans in window) for one role's lease spans.

    Lease-seconds sum every span. Occupied caps the number of leases counted at any instant at
    ``slots``: one slot cannot be held twice, and the pool lends an idle hold's slot to a one-off
    call by design (scheduler H3), so the excess is lending, not extra slot time. ``slots`` None
    (pool state unread) or 0 (a seat with no live slot still holding a lease): uncapped, so a held
    lease is never hidden."""
    marks: list[tuple[datetime, int]] = []
    n_spans = 0
    for start, end in spans:
        s0, e0 = max(start, w_start), min(end, w_end)
        if e0 > s0:
            marks += [(s0, 1), (e0, -1)]
            n_spans += 1
    marks.sort(key=lambda m: (m[0], m[1]))   # an end before a start at the same instant
    cap = slots if slots is not None and slots >= 1 else None
    occupied = lease = 0.0
    live = 0
    prev: Optional[datetime] = None
    for at, d in marks:
        if prev is not None and live > 0:
            dt = (at - prev).total_seconds()
            lease += dt * live
            occupied += dt * (live if cap is None else min(live, cap))
        live += d
        prev = at
    return occupied, lease, n_spans


def deferrable_backlog(inp: Optional[BacklogInput], now: datetime, cfg: BodyConfig
                       ) -> tuple[Optional[dict[str, tuple[int, float, bool]]], Optional[str]]:
    """Orion's own deferrable self-work waiting for a GPU seat, per work class:
    ``{work_class: (queued, waiting_sec, waiting_capped)}``, or ``(None, reason)`` when unknown.

    Count: the newest fresh snapshot of every pool host, summed. Waiting: how long that class's
    queue has been non-empty on the host where it has waited longest (``waiting_capped``: the
    whole lookback, so at least that long). Hunger, not strain: nothing prices it."""
    if inp is None:
        return None, "not_read"
    if not inp.read_ok:
        return None, "read_failed:gpu_pool_state_history"
    fresh = [s for s in inp.snapshots if 0 <= (now - s.at).total_seconds() <= cfg.backlog_max_age_sec]
    if not fresh:
        return None, ("stale:gpu_pool_state_history" if inp.snapshots else "no_rows:gpu_pool_state_history")
    if any(s.deferrable is None for s in fresh):
        return None, "unrecorded:deferrable_depth"
    out: dict[str, tuple[int, float, bool]] = {}
    for snap in fresh:
        for wc, n in (snap.deferrable or {}).items():
            n = int(n or 0)
            if n <= 0:
                continue
            since = inp.last_empty_at.get((snap.host, wc))
            capped = since is None
            waited = inp.lookback_sec if capped else max(0.0, (now - since).total_seconds())
            q, w, c = out.get(wc, (0, 0.0, False))
            out[wc] = (q + n, max(w, waited), c or capped)
    return out, None


def lane_readings(inp: LaneInput, now: datetime, cfg: BodyConfig,
                  backlog: Optional[BacklogInput] = None
                  ) -> tuple[list[PhysiologyConstraintReadingV1], list[str]]:
    """One reading per lane, plus warnings for the whole state."""
    window = cfg.lane_window_sec
    capacity = dict(inp.capacity) if inp.capacity is not None else None
    roles_known = sorted(capacity) if capacity else []
    hunger, hunger_missing = deferrable_backlog(backlog, now, cfg)

    def _hunger(role: str) -> dict:
        # Keyed by work class; a class's queue sits beside the lane of the same name (agent ->
        # lane:agent, diffusion -> lane:diffusion). A class with no such lane is a warning.
        if hunger is None:
            return {"deferrable_no_reading": hunger_missing}
        q, w, c = hunger.get(role, (0, 0.0, False))
        d: dict = {"deferrable_queued": q, "deferrable_waiting_sec": round(w, 1)}
        if c:
            d["deferrable_waiting_capped"] = True
        return d

    def _cap(role: str) -> dict:
        if capacity is None or role not in capacity:
            return {}
        return {"budget": float(capacity[role]) * window, "budget_kind": "capacity"}

    def all_no(reason: str, detail: Optional[dict] = None, **kw) -> list[PhysiologyConstraintReadingV1]:
        if not roles_known:
            return [_no(LANE_PREFIX.rstrip(":"), "slot_s", reason, detail=dict(detail or {}), **kw)]
        return [_no(f"{LANE_PREFIX}{r}", "slot_s", reason, detail={**(detail or {}), **_hunger(r)},
                    **_cap(r), **kw) for r in roles_known]

    if not inp.events_read_ok:
        return all_no("read_failed:gpu_pool_events"), []
    newest = inp.newest_event_at
    if newest is None:
        return all_no("no_rows:gpu_pool_events"), []
    age = (now - newest).total_seconds()
    if age > cfg.lane_events_max_age_sec:
        return all_no("stale:gpu_pool_events", source_at=newest, detail={"age_sec": round(age, 1)}), []
    intervals, counts = _lease_intervals(inp.events, now)
    w_start = now - timedelta(seconds=window)
    roles = sorted(set(intervals) | set(capacity or {}))
    if not roles:
        # Fresh events, no lease in the window, and no pool state to name the lanes.
        return [_no(LANE_PREFIX.rstrip(":"), "slot_s", "read_failed:gpu_pool_state", source_at=newest)], []
    out: list[PhysiologyConstraintReadingV1] = []
    for role in roles:
        slots = int(capacity[role]) if capacity is not None and role in capacity else None
        # Leases are NOT merged across each other: concurrent leases hold separate slots, up to
        # the role's slots (occupied_seconds).
        held, lease_s, spans = occupied_seconds(intervals.get(role, []), w_start, now, slots)
        detail: dict = {"held_spans": spans, "window_sec": window, "lease_s": round(lease_s, 3)}
        if slots is not None:
            detail["slots"] = slots
            if slots >= 1:
                detail["lent_overlap_s"] = round(lease_s - held, 3)
            elif held > 0:
                detail["held_without_live_slot"] = True
        if capacity is None:
            detail["capacity"] = "pool_state_unread"
        detail.update(_hunger(role))
        out.append(PhysiologyConstraintReadingV1(
            constraint=f"{LANE_PREFIX}{role}", unit="slot_s", level=round(held, 3), source_at=newest,
            detail=detail, **_cap(role)))
    warnings = []
    if counts["open"]:
        warnings.append(f"lane_open_leases:{counts['open']}")
    if counts["child_skipped"]:
        warnings.append(f"lane_child_leases_in_hold_slots:{counts['child_skipped']}")
    if counts["cross_role"]:
        warnings.append(f"lane_leases_held_on_two_seats:{counts['cross_role']}")
    if hunger:
        for wc in sorted(set(hunger) - set(roles)):
            warnings.append(f"lane_deferrable_unplaced:{wc}:{hunger[wc][0]}")
    return out, warnings


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
    if inp.cap_counted_today is None:
        return _no(ATTENTION_JUNIPER, "messages", "read_failed:endogenous_outreach_decisions", **line)
    # Level and line count the same thing: the sends the Hub's cap counts. Every send that reached
    # Juniper (cap-exempt curiosity / reading-finding outreach included) is in detail.
    return PhysiologyConstraintReadingV1(
        constraint=ATTENTION_JUNIPER, unit="messages", level=float(inp.cap_counted_today),
        source_at=inp.newest_decision_at,
        detail={"sent_today_all_sources": inp.sent_today}, **line)


_STATE_RANK = {"limited": 2, "unknown": 1, "clear": 0}


def claude_reading(observations: Sequence[ClaudeObservation], now: datetime, cfg: BodyConfig) -> PhysiologyConstraintReadingV1:
    """Both limits (session, weekly) must be read fresh before the state can be ``clear``: a kind
    not yet received (the weekly one emits hourly, so for up to an hour after every restart) or
    gone stale reads ``unread`` and makes the state ``unknown``."""
    line = dict(budget=0.0, budget_kind="not_limited")
    if not observations:
        return _no(CLAUDE, "limit_events", "no_observation", state="unknown", **line)
    newest = max(o.received_at for o in observations)
    fresh = {o.kind: o for o in observations
             if 0 <= (now - o.received_at).total_seconds() <= cfg.claude_max_age_sec}
    if not fresh:
        return _no(CLAUDE, "limit_events", "stale:claude_limit", state="unknown", source_at=newest, **line)
    seen = [o for o in fresh.values() if o.observed]
    if not seen:
        return _no(CLAUDE, "limit_events", "unobserved:claude_limit", state="unknown", source_at=newest, **line)
    detail: dict = {}
    states: list[str] = []
    for kind in sorted(set(CLAUDE_KINDS) | set(fresh)):
        o = fresh.get(kind)
        if o is None:
            detail[f"{kind}_state"] = "unread"
            states.append("unknown")
            continue
        st = o.state if o.observed else "unknown"
        detail[f"{kind}_state"] = st
        detail[f"{kind}_events"] = o.event_count
        if o.resets_at is not None:
            detail[f"{kind}_resets_at"] = o.resets_at.isoformat()
        states.append(st)
    # Any kind limited -> limited. Any kind unread beside clear ones -> unknown, never clear.
    state = max(states, key=lambda st: _STATE_RANK.get(st, 1))
    return PhysiologyConstraintReadingV1(
        constraint=CLAUDE, unit="limit_events", level=float(sum(o.event_count for o in seen)),
        state=state, source_at=newest, detail=detail, **line)


def build_physiology_state(now: datetime, inputs: BodyInputs, cfg: BodyConfig = BodyConfig(),
                           warnings: Sequence[str] = ()) -> PhysiologyStateV1:
    warnings = list(warnings)
    energy = energy_reading(inputs.energy, now, cfg)
    if energy.level is not None and (energy.detail.get("unpriced_kwh") or 0) > 0:
        warnings.append("energy_partial:unpriced_kwh")
    lanes, lane_warnings = lane_readings(inputs.lanes, now, cfg, inputs.backlog)
    warnings += lane_warnings
    constraints = [heat_reading(inputs.heat, cfg), energy, *lanes,
                   juniper_reading(inputs.juniper, now, cfg), claude_reading(inputs.claude, now, cfg)]
    return PhysiologyStateV1(tick_id=f"physiology:{now.isoformat()}", generated_at=now,
                             constraints=constraints, warnings=warnings)


__all__ = [
    "BacklogInput",
    "BacklogSnapshot",
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
    "deferrable_backlog",
    "energy_reading",
    "heat_reading",
    "juniper_reading",
    "lane_readings",
    "occupied_seconds",
]
