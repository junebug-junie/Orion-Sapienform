#!/usr/bin/env python3
"""Live smoke for the Budget organ (body economy Patch B2), read-only.

Takes the last ``--hours`` of ``physiology_state`` rows (the Body readings the live node already
stored), re-derives what the live node would also have had at each tick, and runs
``orion.physiology.budget.price_state`` over them in order, carrying the state tick to tick exactly
as the node does. Prints the price path per constraint and checks the Patch B sanity rules:

- the heat price is ~0 whenever the raw cabinet reading is under the soft line, except
  the decay tail within one horizon after the price was last above 0.05 (as R1 excuses it);
- no price sits at its ceiling unflagged for longer than its horizon (consecutive, by timestamp);
- energy's bind state is REPORTED (expected ``exogenous_bind`` while the envelope is $100), not gated:
  it describes today's world, not a rule.

What is re-derived per tick (the stored rows predate these inputs):
- host watts: the 1 h mean of ``orion_biometrics_cluster`` ending at the tick;
- energy hours and rates: ``orion_energy_hourly`` / ``energy_cost_accrued`` up to the tick;
- lanes: re-read through the Patch B1 reducer from ``gpu_pool_events`` (a lease whose close came
  after the tick was open at the tick), because rows written before B1 carry the old over-count.
  The pool's capacity is taken from the stored reading's budget.

Usage:
  ORION_ANALYSIS_DSN=postgresql://... .venv/bin/python scripts/analysis/smoke_body_economy_prices.py [--hours 24]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from orion.physiology.body import (  # noqa: E402
    BodyInputs,
    EnergyHour,
    EnergyInput,
    EnergyPrice,
    HeatInput,
    HostWattsInput,
    JuniperInput,
    LaneInput,
    LeaseEvent,
    lane_readings,
    BodyConfig,
)
from orion.physiology.budget import BudgetConfig, price_state  # noqa: E402
from orion.physiology.heat_config import load_heat_sensitivity  # noqa: E402
from orion.schemas.physiology import LANE_PREFIX, PhysiologyStateV1  # noqa: E402

ROWS_SQL = "SELECT tick_id, generated_at, constraints, warnings FROM physiology_state WHERE generated_at >= %(since)s ORDER BY generated_at"
WATTS_SQL = """
SELECT date_trunc('minute', observed_at) AS m,
       avg((measurements_by_node->'athena'->>'cpu_watts_total')::float + (measurements_by_node->'athena'->>'gpu_watts_total')::float) AS athena,
       avg((measurements_by_node->'circe'->>'cpu_watts_total')::float + (measurements_by_node->'circe'->>'gpu_watts_total')::float) AS circe,
       avg((measurements_by_node->'hecate'->>'chassis_watts')::float) AS hecate
FROM orion_biometrics_cluster WHERE observed_at >= %(since)s GROUP BY 1 ORDER BY 1
"""
ENERGY_SQL = "SELECT hour_start, SUM(kwh) AS kwh FROM orion_energy_hourly WHERE hour_start >= %(since)s GROUP BY 1 ORDER BY 1"
RATES_SQL = """
SELECT date_trunc('hour', interval_start) AS hour_start,
       SUM(marginal_usd_per_kwh * energy_kwh) / NULLIF(SUM(energy_kwh), 0) AS rate
FROM (SELECT DISTINCT ON (usage_point_id, interval_start) interval_start, energy_kwh, marginal_usd_per_kwh
      FROM energy_cost_accrued WHERE interval_start >= %(since)s
      ORDER BY usage_point_id, interval_start, computed_at DESC) x GROUP BY 1 ORDER BY 1
"""
LEASES_SQL = """
SELECT e.lease_id, e.role, e.event, e.generated_at, e.held_ms, l.hold_lease_id
FROM gpu_pool_events e LEFT JOIN gpu_pool_leases l ON l.lease_id = e.lease_id
WHERE e.generated_at >= %(since)s AND e.event = ANY(%(events)s)
"""
OPEN_SQL = "SELECT lease_id, role, granted_at, hold_lease_id FROM gpu_pool_leases WHERE status IN ('granted','recalling') AND granted_at IS NOT NULL"


def _utc(v: datetime) -> datetime:
    return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)


def lease_events_at(closes: list[dict], opens: list[dict], tick: datetime, window: timedelta) -> list[LeaseEvent]:
    """The lease events the live reducer would have seen at ``tick``: closes inside the window,
    plus every lease open at the tick (a close after the tick whose span began before it, or a
    lease still open now that was granted before it)."""
    out = []
    for c in closes:
        at, held = c["at"], c["held_ms"]
        if tick - window <= at <= tick:
            out.append(LeaseEvent(c["lease_id"], c["role"], c["event"], at, held, c["hold_lease_id"]))
        elif at > tick and held is not None and at - timedelta(milliseconds=held) < tick:
            out.append(LeaseEvent(c["lease_id"], c["role"], "granted", at - timedelta(milliseconds=held),
                                  None, c["hold_lease_id"]))
    for o in opens:
        if o["granted_at"] < tick:
            out.append(LeaseEvent(o["lease_id"], o["role"], "granted", o["granted_at"], None, o["hold_lease_id"]))
    return out


def main(argv: list[str] | None = None) -> int:
    import psycopg
    from psycopg.rows import dict_row

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--dsn", default=os.environ.get("ORION_ANALYSIS_DSN"))
    ap.add_argument("--heat-config", default=str(_HERE.parents[1] / "config" / "physiology" / "heat_sensitivity.yaml"))
    ap.add_argument("--tz", default="America/Denver")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args(argv)
    if not a.dsn:
        raise SystemExit("--dsn or ORION_ANALYSIS_DSN required")
    heat_cfg, err = load_heat_sensitivity(a.heat_config)
    cfg = BudgetConfig(tz=a.tz, heat_config=heat_cfg, heat_config_error=err)
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=a.hours)
    window = timedelta(hours=1)
    with psycopg.connect(a.dsn, row_factory=dict_row, options="-c default_transaction_read_only=on") as conn:
        rows = conn.execute(ROWS_SQL, {"since": since}).fetchall()
        minutes = conn.execute(WATTS_SQL, {"since": since - window}).fetchall()
        month = datetime(now.year, now.month, 1, tzinfo=timezone.utc) - timedelta(days=2)
        ehours = conn.execute(ENERGY_SQL, {"since": month}).fetchall()
        rates = conn.execute(RATES_SQL, {"since": month - timedelta(days=14)}).fetchall()
        closes = [{**r, "at": _utc(r["generated_at"])} for r in conn.execute(LEASES_SQL, {
            "since": since - window, "events": ["released", "expired", "aborted", "retried", "cancelled", "dead_lettered"]}).fetchall()]
        opens = [{**r, "granted_at": _utc(r["granted_at"])} for r in conn.execute(OPEN_SQL).fetchall()]
    mins = [(_utc(m["m"]), m) for m in minutes]
    ehours = [EnergyHour(_utc(r["hour_start"]), float(r["kwh"])) for r in ehours if r["kwh"] is not None]
    rates = [EnergyPrice(_utc(r["hour_start"]), float(r["rate"])) for r in rates if r["rate"] is not None]

    carry, path = None, []
    for r in rows:
        tick = _utc(r["generated_at"])
        st = PhysiologyStateV1.model_validate({"tick_id": r["tick_id"], "generated_at": tick,
                                               "constraints": r["constraints"], "warnings": r["warnings"]})
        cap = {c.constraint[len(LANE_PREFIX):]: int(round((c.budget or 0) / 3600.0))
               for c in st.constraints if c.constraint.startswith(LANE_PREFIX) and c.budget is not None}
        lanes, _ = lane_readings(LaneInput(True, lease_events_at(closes, opens, tick, window), cap or None,
                                           newest_event_at=tick), tick, BodyConfig())
        st = st.model_copy(update={"constraints": [c for c in st.constraints if not c.constraint.startswith("lane")] + lanes})
        win = [m for t, m in mins if tick - window <= t <= tick]
        watts = {h: (sum(m[h] for m in win if m[h] is not None) / max(1, sum(1 for m in win if m[h] is not None))
                     if any(m[h] is not None for m in win) else None) for h in ("athena", "circe", "hecate")}
        # Complete hours only (no look-ahead into the hour that was still filling at the tick).
        body = BodyInputs(heat=HeatInput(True), energy=EnergyInput(True, hours=[h for h in ehours if h.hour_start + timedelta(hours=1) <= tick],
                                                                       prices=[p for p in rates if p.hour_start <= tick]),
                          lanes=LaneInput(True), juniper=JuniperInput(True), host_watts=HostWattsInput(bool(win), watts))
        priced, carry = price_state(st, body, carry, tick, cfg)
        path.append(priced)

    by = defaultdict(list)
    for s in path:
        for c in s.constraints:
            if c.price is not None:
                by[c.constraint].append((s.generated_at, c, c.price))
    print(f"{len(path)} ticks {path[0].generated_at:%m-%d %H:%M} .. {path[-1].generated_at:%m-%d %H:%M} UTC" if path else "no rows")
    summary = {}
    for name in sorted(by):
        pts = by[name]
        states = Counter(p.bind_state for _, _, p in pts)
        vals = [p.value for _, _, p in pts]
        last = pts[-1][2]
        summary[name] = {"ticks": len(pts), "states": dict(states), "min": min(vals), "max": max(vals), "last": last.value,
                         "last_why": last.why}
        print(f"- {name}: states {dict(states)}; price min {min(vals):.3f} max {max(vals):.3f} last {last.value:.3f} ({last.bind_state})")
        print(f"    last why: {last.why}")
    strain = [s.strain for s in path if s.strain is not None]
    sv = [x.value for x in strain if x.value is not None]
    drivers = Counter((x.constraint or "none(rest)") for x in strain if x.value is not None)
    at_rest = sum(1 for v in sv if v == 0.0)
    print(f"- strain: min {min(sv) if sv else None} max {max(sv) if sv else None}; at rest (0) {at_rest}/{len(sv)} ticks; "
          f"driven by {dict(drivers)}; exogenous listed apart: {dict(Counter(n for x in strain for n in x.exogenous_bind))}")
    print(f"    last: {strain[-1].why if strain else None}")

    checks = {}
    tail = timedelta(seconds=cfg.spec("heat").horizon_sec)
    last_hot = None
    bad = under = 0
    for t, c, p in by.get("heat", []):
        if p.value > 0.05 and (c.level is None or c.level >= cfg.heat_soft_line_c):
            last_hot = t
        if c.level is not None and c.level < cfg.heat_soft_line_c:
            under += 1
            if p.value > 0.05 and not (last_hot is not None and t - last_hot <= tail):
                bad += 1
    checks["heat_~0_when_cabinet_under_line"] = {"ticks_under_line": under, "ticks_priced_over_0.05_outside_decay_tail": bad,
                                                 "ok": bad == 0}
    e_states = Counter(p.bind_state for _, _, p in by.get("energy", []))
    checks["energy_bind_state (reported, not gated)"] = {"states": dict(e_states), "ok": True}
    longest: dict = {}
    for n, pts in by.items():
        start = None
        for t, _, p in pts:
            if p.value >= p.ceiling * (1 - 1e-6) and p.bind_state not in ("exogenous_bind", "stuck_at_ceiling"):
                start = start or t
                longest[n] = max(longest.get(n, 0.0), (t - start).total_seconds())
            else:
                start = None
    checks["no_price_at_ceiling_unflagged_past_its_horizon"] = {
        "longest_unflagged_ceiling_run_sec": {n: round(v) for n, v in longest.items()},
        "stuck_flagged_ticks": {n: sum(1 for _, _, p in pts if p.bind_state == "stuck_at_ceiling") for n, pts in by.items()},
        "ok": all(v <= cfg.spec(n).horizon_sec for n, v in longest.items())}
    print("checks:", json.dumps(checks, indent=1, default=str))
    if a.out_json:
        Path(a.out_json).write_text(json.dumps({"summary": summary, "checks": checks,
                                                "strain_last": strain[-1].model_dump(mode="json") if strain else None},
                                               indent=1, default=str))
    return 0 if all(c["ok"] for c in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
