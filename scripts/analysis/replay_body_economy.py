#!/usr/bin/env python3
"""Deterministic replay of the body-economy price rule over live hourly data (read-only).

Design: docs/superpowers/specs/2026-10-23-organism-body-economy-design.md, Organ 2 and acceptance
checks R1-R7. The price math is ``orion/physiology/prices.py``; this script only feeds it hourly
aggregates and grades the result.

Constraints replayed:

- ``heat``: level = Orion's own heat share (fitted sensitivity x incremental watts per host, from
  ``fit_heat_sensitivity.py``'s JSON). Budget = soft line - forecast cabinet temperature without
  Orion's share, floored at 0. ``exogenous_bind`` when that floor is hit.
- ``energy``: level = forecast 30-day Orion cost; budget = a monthly envelope. Juniper has not set
  the envelope (design missing question), so a few hypothetical envelopes are replayed to
  exercise the dynamics. They are labelled hypothetical everywhere.

Evaluated: R1, R3, R4. R2, R5, R6, R7 need wants, a want ledger, lane bids or per-act downstream
use rates. None exist yet, so they are reported NOT YET EVALUABLE with the reason. R6
(reservations) is deferred to Patch C: it needs a bid stream.

**The Patch B gate** (``patch_b`` in the output): the rules the live Budget organ runs
(``orion/physiology/budget.py``): bounded ramp into bind, 0.5 C bind hysteresis, the point
sensitivity of the short-run fit and the baselines from ``config/physiology/heat_sensitivity.yaml``,
the cabinet reading corrected for #2780's sensor offset, slack decay of ceiling / horizon, and for
energy the $100 envelope with 5% bind hysteresis. R1 is graded with one horizon of decay tail
excused (findings doc, Patch B rule 3). The older variants stay beside it for comparison.

Usage:
  .venv/bin/python scripts/analysis/replay_body_economy.py --fit-json heat_fit.json \
      --out-json replay.json --out-md replay.md [--cache extract.json] \
      [--heat-config config/physiology/heat_sensitivity.yaml]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))
import _body_economy_data as data_mod  # noqa: E402
from orion.physiology.heat_config import CABINET_SERIES, load_heat_sensitivity, offset_at  # noqa: E402
from orion.physiology.prices import (  # noqa: E402
    EXOGENOUS_BIND,
    NO_READING,
    PriceParams,
    bind_with_hysteresis,
    exogenous_bind_with_hysteresis,
    heat_budget,
    heat_contribution,
    month_end_forecast,
    step_price,
    upper_bound_if_spans_zero,
)

TARIFF_USD_PER_KWH = 0.1213  # design Fact 1, measured pre-tax
HORIZON_H = 6  # the heat forecast horizon in the design's constraint table
ENERGY_SOURCES = ("athena", "circe", "hecate", "cooling")
# Regressor per host must match the fit: compute watts for athena/circe, chassis for hecate.
HOST_REGRESSOR = {"athena": "cw_athena", "circe": "cw_circe", "hecate": "w_hecate"}


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


# ---------------------------------------------------------------- hourly aggregates

def hourly_from_extract(extract: dict) -> dict:
    """{hour: {"cabinet_c", "<host>" watts, "kwh": {source: kwh}}}. Missing stays None."""
    acc: dict[datetime, dict[str, list[float]]] = {}
    for r in extract["bins15"]:
        h = _ts(r["t"]).replace(minute=0)
        row = acc.setdefault(h, {})
        if r.get("cabinet_c") is not None:
            row.setdefault("cabinet_c", []).append(r["cabinet_c"])
        for host, key in HOST_REGRESSOR.items():
            if r.get(key) is not None:
                row.setdefault(host, []).append(r[key])
    out: dict[datetime, dict] = {}
    for h, row in acc.items():
        out[h] = {k: (sum(v) / len(v) if len(v) >= 2 else None) for k, v in row.items()}
        out[h].setdefault("cabinet_c", None)
        for host in HOST_REGRESSOR:
            out[h].setdefault(host, None)
        out[h]["kwh"] = {}
    for r in extract["energy_hourly"]:
        h = _ts(r["t"])
        if r["source"] in ENERGY_SOURCES and (r.get("covered_sec") or 0) > 1800:
            out.setdefault(h, {"cabinet_c": None, **{x: None for x in HOST_REGRESSOR}, "kwh": {}})
            # scale a partly covered hour up to a full hour
            out[h]["kwh"][r["source"]] = r["kwh"] * 3600.0 / r["covered_sec"]
    return dict(sorted(out.items()))


def _p10(vals: list[float]) -> Optional[float]:
    v = sorted(x for x in vals if x is not None)
    if not v:
        return None
    return v[int(0.1 * (len(v) - 1))]


# ---------------------------------------------------------------- heat replay

@dataclass(frozen=True)
class HourResult:
    hour: str
    cabinet_c: Optional[float]
    forecast_c: Optional[float]
    own_c: Optional[float]
    budget_c: Optional[float]
    status: str
    price: float
    excess: Optional[float]


def run_heat_replay(hours: dict, *, sensitivity: dict[str, float], baselines: dict[str, float],
                    params: PriceParams, soft_line_c: float, start: datetime, end: datetime,
                    forecast_mode: str, bind_rearm_margin_c: float = 0.0) -> list[HourResult]:
    """forecast_mode:
    - ``oracle``: forecast cabinet = mean of the realized hourly means over [t, t+6h). This grades
      the price rule alone, with a perfect forecaster.
    - ``persistence``: forecast cabinet = last hour's realized mean. A forecaster any live system
      could run today.
    Orion's share uses the same convention (oracle: this hour's watts; persistence: last hour's)."""
    keys = sorted(hours)
    idx = {h: i for i, h in enumerate(keys)}
    price = 0.0
    prev_bind = False
    out = []
    t = start
    while t < end:
        i = idx.get(t)
        fc = own = None
        if i is not None:
            if forecast_mode == "oracle":
                window = [hours[keys[j]]["cabinet_c"] for j in range(i, min(i + HORIZON_H, len(keys)))
                          if keys[j] < t + timedelta(hours=HORIZON_H)]
                known = [x for x in window if x is not None]
                fc = sum(known) / len(known) if len(known) >= HORIZON_H // 2 else None
                src = hours[t]
            else:
                prev = hours.get(t - timedelta(hours=1))
                fc = prev["cabinet_c"] if prev else None
                src = prev or {}
            inc = {h: (src.get(h) - baselines[h]) if src.get(h) is not None else None for h in sensitivity}
            own = heat_contribution(sensitivity, inc)
        budget, bind = heat_budget(soft_line_c=soft_line_c, forecast_cabinet_c=fc, own_contribution_c=own)
        if bind_rearm_margin_c > 0 and bind is not None:
            bind = exogenous_bind_with_hysteresis(prev_bind, exogenous_c=fc - own, soft_line_c=soft_line_c,
                                                  rearm_margin_c=bind_rearm_margin_c)
        if bind is not None:
            prev_bind = bind
        s = step_price(price, use=own, budget=budget, exogenous_bind=bool(bind), params=params)
        price = s.price
        cab = hours.get(t, {}).get("cabinet_c")
        out.append(HourResult(t.isoformat(), cab, fc, own, budget, s.status, s.price, s.excess))
        t += timedelta(hours=1)
    return out


def eval_r1(res: list[HourResult], *, soft_line_c: float, ceiling: float, near_zero_frac: float = 0.05,
            min_coverage: float = 0.8, min_exo_recall: float = 0.9) -> dict:
    """R1: price ~0 when the forecast is under the soft line; positive above it; every hour that
    is unmeetable with zero Orion load flagged exogenous_bind.

    "Unmeetable" is judged on the REALIZED cabinet temperature (cabinet - Orion's share >= line),
    not the forecast the flag was set from, so the check measures how well the forecast-driven
    flag tracks reality instead of passing by construction. "Never climbing past the ceiling" is
    guaranteed by step_price's clip and is unit-tested there, not graded here.
    Fails when fewer than ``min_coverage`` of hours have a reading: no data is not a pass."""
    known = [r for r in res if r.forecast_c is not None and r.status != NO_READING]
    under = [r for r in known if r.forecast_c < soft_line_c]
    over = [r for r in known if r.forecast_c >= soft_line_c]
    eps = near_zero_frac * ceiling
    under_zero = sum(1 for r in under if r.price <= eps)
    over_pos = sum(1 for r in over if r.price > 0)
    # Under-line hours still carrying price, split by cause: the decay tail within one horizon
    # after a binding hour (the integrator's lag) versus anything else.
    binding_idx = [i for i, r in enumerate(res) if r.status == EXOGENOUS_BIND or (r.excess is not None and r.excess > 0)]
    pos = {r.hour: i for i, r in enumerate(res)}
    tail = other = 0
    for r in under:
        if r.price > eps:
            i = pos[r.hour]
            if any(0 < i - j <= HORIZON_H for j in binding_idx):
                tail += 1
            else:
                other += 1
    exo = [r for r in known if r.own_c is not None and r.cabinet_c is not None
           and (r.cabinet_c - r.own_c) >= soft_line_c]
    exo_flagged = sum(1 for r in exo if r.status == EXOGENOUS_BIND)
    flagged_total = sum(1 for r in known if r.status == EXOGENOUS_BIND)
    coverage = len(known) / len(res) if res else 0.0
    frac = lambda a, b: (a / b) if b else None  # noqa: E731
    out = {
        "hours_known": len(known), "hours_no_reading": len(res) - len(known),
        "hours_under_line": len(under), "under_line_price_near_zero": under_zero,
        "under_line_near_zero_frac": frac(under_zero, len(under)),
        "under_line_mean_price": (sum(r.price for r in under) / len(under)) if under else None,
        "hours_over_line": len(over), "over_line_price_positive": over_pos,
        "over_line_positive_frac": frac(over_pos, len(over)),
        "hours_exogenous_realized": len(exo), "exogenous_flagged": exo_flagged,
        "exogenous_recall": frac(exo_flagged, len(exo)), "hours_flagged_total": flagged_total,
        "coverage": coverage,
        "near_zero_threshold": eps,
        "under_line_nonzero_decay_tail": tail, "under_line_nonzero_other": other,
    }
    base = bool(
        coverage >= min_coverage and under and over
        and out["over_line_positive_frac"] >= 0.95
        and (not exo or out["exogenous_recall"] >= min_exo_recall)
    )
    out["pass"] = base and out["under_line_near_zero_frac"] >= 0.95
    # Same check with the integrator's decay tail (<= one horizon after a binding hour) excused.
    out["pass_excluding_decay_tail"] = base and other == 0
    return out


def eval_r3(prices: list[float], statuses: list[str], binding: list[bool], *, ceiling: float,
            horizon: int = HORIZON_H, amplitude: float, eps: float = 1e-9,
            min_leg_frac: float = 0.05) -> dict:
    """R3: back to 0 within one horizon after the constraint clears; peak-to-peak oscillation under
    ``amplitude``; no run at the ceiling longer than one horizon unless flagged exogenous_bind.

    ``binding[i]`` is True when the constraint binds in hour i (over budget or exogenous_bind).
    A move smaller than ``min_leg_frac`` x ceiling is not counted as a reversal leg, so a 0.01
    wiggle on a falling price is not called an oscillation."""
    n = len(prices)
    returns, censored, stuck = [], 0, 0
    for i in range(1, n):
        if binding[i - 1] and not binding[i]:
            k, got = 0, None
            while i + k < n and not binding[i + k]:
                if prices[i + k] <= eps:
                    got = k
                    break
                k += 1
            if got is None:
                if k > horizon:
                    stuck += 1  # clear for longer than a horizon and the price never reached 0
                else:
                    censored += 1  # re-bound (or ran out of data) within a horizon
            else:
                returns.append(got)
    # oscillation: peak-to-peak in any horizon window with >= 2 direction reversals
    min_leg = min_leg_frac * ceiling
    worst_osc = 0.0
    for i in range(0, max(n - horizon, 0) + 1):
        w = prices[i:i + horizon + 1]
        d = [b - a for a, b in zip(w, w[1:]) if abs(b - a) >= max(min_leg, eps)]
        rev = sum(1 for a, b in zip(d, d[1:]) if (a > 0) != (b > 0))
        if rev >= 2:
            worst_osc = max(worst_osc, max(w) - min(w))
    # ceiling runs without the exogenous flag. Split by cause: an hour at the ceiling while the
    # constraint is NOT binding is windup (the price failed to fall); while it IS binding every
    # hour it is saturation of an open-loop replay (nothing lowers Orion's load as the price rises).
    longest, run = 0, 0
    windup_h = saturated_h = 0
    for p, st, b in zip(prices, statuses, binding):
        if p >= ceiling - eps and st != EXOGENOUS_BIND:
            run += 1
            longest = max(longest, run)
            if b:
                saturated_h += 1
            else:
                windup_h += 1
        else:
            run = 0
    out = {
        "clear_events": len(returns) + censored + stuck, "returned_to_zero": len(returns),
        "censored_rebound_before_zero": censored, "stuck_positive_past_horizon": stuck,
        "max_hours_to_zero": max(returns) if returns else None,
        "returns_within_horizon": sum(1 for r in returns if r <= horizon),
        "max_peak_to_peak_oscillating_window": worst_osc, "declared_amplitude": amplitude, "min_leg": min_leg,
        "longest_unflagged_ceiling_run_h": longest,
        "unflagged_ceiling_hours_binding": saturated_h, "unflagged_ceiling_hours_not_binding": windup_h,
    }
    out["pass"] = bool(
        stuck == 0 and all(r <= horizon for r in returns) and worst_osc < amplitude and longest <= horizon
    )
    return out


# ---------------------------------------------------------------- energy

def fill_energy(hours: dict, start: datetime, end: datetime) -> tuple[dict[datetime, float], dict]:
    """Orion kWh per hour over [start, end). A missing source-hour is filled with that source's
    median hourly kWh in the window (never 0). Returns (kwh_by_hour, fill report)."""
    by_src: dict[str, list[float]] = {s: [] for s in ENERGY_SOURCES}
    t = start
    while t < end:
        for s in ENERGY_SOURCES:
            v = hours.get(t, {}).get("kwh", {}).get(s)
            if v is not None:
                by_src[s].append(v)
        t += timedelta(hours=1)
    med = {s: (sorted(v)[len(v) // 2] if v else None) for s, v in by_src.items()}
    total_hours = int((end - start).total_seconds() // 3600)
    out: dict[datetime, float] = {}
    filled = {s: 0 for s in ENERGY_SOURCES}
    filled_kwh = 0.0
    t = start
    while t < end:
        k = 0.0
        for s in ENERGY_SOURCES:
            v = hours.get(t, {}).get("kwh", {}).get(s)
            if v is None:
                if med[s] is None:
                    raise ValueError(f"no {s} energy readings at all in the window; cannot fill")
                v = med[s]
                filled[s] += 1
                filled_kwh += v
            k += v
        out[t] = k
        t += timedelta(hours=1)
    total = sum(out.values())
    return out, {"filled_source_hours": filled, "filled_kwh": filled_kwh, "total_kwh": total,
                 "filled_share": filled_kwh / total if total else None, "hours": total_hours,
                 "median_fill_kwh": med}


def eval_r4(kwh: dict[datetime, float], *, cycle_start: datetime, cycle_end: datetime, replay_start: datetime,
            replay_end: datetime, trailing_h: int = 168, tariff: float = TARIFF_USD_PER_KWH, tol: float = 0.10) -> dict:
    """R4: month-end forecast (to-date + trailing-7-day rate x hours left) within +-10% of the
    realized 30-day Orion cost. Also reports a harder rate-only forecast (30 days x trailing rate)."""
    keys = sorted(kwh)
    realized = sum(kwh[h] for h in keys if cycle_start <= h < cycle_end) * tariff
    errs, rate_errs, rows = [], [], []
    t = replay_start
    while t < replay_end:
        hist = [kwh[h] for h in keys if cycle_start <= h < t]
        trail = [kwh[h] for h in keys if t - timedelta(hours=trailing_h) <= h < t]
        if len(trail) >= trailing_h * 0.9:
            rate = sum(trail) / len(trail) * tariff
            remaining = (cycle_end - t).total_seconds() / 3600
            fc = month_end_forecast(to_date=sum(hist) * tariff, trailing_rate_per_hour=rate, hours_remaining=remaining)
            rate_only = rate * (cycle_end - cycle_start).total_seconds() / 3600
            errs.append(fc / realized - 1)
            rate_errs.append(rate_only / realized - 1)
            if t.hour == 0:
                rows.append({"at": t.isoformat(), "forecast_usd": fc, "rate_only_usd": rate_only})
        t += timedelta(hours=1)
    worst = max(errs, key=abs) if errs else None
    worst_rate = max(rate_errs, key=abs) if rate_errs else None
    return {
        "cycle": [cycle_start.isoformat(), cycle_end.isoformat()], "realized_usd": realized,
        "tariff_usd_per_kwh": tariff, "forecasts": len(errs),
        "worst_error": worst, "mean_abs_error": (sum(abs(e) for e in errs) / len(errs)) if errs else None,
        "rate_only_worst_error": worst_rate,
        "rate_only_mean_abs_error": (sum(abs(e) for e in rate_errs) / len(rate_errs)) if rate_errs else None,
        "daily_forecasts": rows, "tolerance": tol,
        "pass": bool(errs) and abs(worst) <= tol,
        "rate_only_pass": bool(rate_errs) and abs(worst_rate) <= tol,
    }


def run_energy_replay(kwh: dict[datetime, float], *, envelope_usd: float, fixed_kwh_per_h: float, params: PriceParams,
                      cycle_start: datetime, cycle_end: datetime, start: datetime, end: datetime,
                      trailing_h: int = 168, tariff: float = TARIFF_USD_PER_KWH,
                      rearm_frac: float = 0.0) -> tuple[list[float], list[str], list[bool]]:
    keys = sorted(kwh)
    price, prices, statuses, binding = 0.0, [], [], []
    prev_bind = False
    t = start
    while t < end:
        to_date = sum(kwh[h] for h in keys if cycle_start <= h < t) * tariff
        trail = [kwh[h] for h in keys if t - timedelta(hours=trailing_h) <= h < t]
        remaining = (cycle_end - t).total_seconds() / 3600
        use = month_end_forecast(to_date=to_date, trailing_rate_per_hour=(sum(trail) / len(trail) * tariff) if trail else None,
                                 hours_remaining=remaining)
        floor_forecast = to_date + fixed_kwh_per_h * tariff * remaining
        if rearm_frac > 0:
            bind = bool(bind_with_hysteresis(prev_bind, floor_use=floor_forecast, budget=envelope_usd,
                                             rearm_margin=rearm_frac * envelope_usd))
        else:
            bind = floor_forecast >= envelope_usd
        prev_bind = bind
        s = step_price(price, use=use, budget=envelope_usd, exogenous_bind=bind, params=params)
        price = s.price
        prices.append(price)
        statuses.append(s.status)
        binding.append(bind or (use is not None and use > envelope_usd))
        t += timedelta(hours=1)
    return prices, statuses, binding


# ---------------------------------------------------------------- driver

NOT_YET = {
    "R2": "Needs replayed non-GPU wants (reading, outreach text, dream bookkeeping) with values and costs. "
          "WantV1 and want_ledger do not exist yet (design patch C), and grammar_acts.cost_json is null.",
    "R5": "Needs a posted lane price and background bid-before-lease (patch E). No lane price exists, so "
          "there is nothing to replay against the 10-21..22 refusal baseline.",
    "R6": "DEFERRED to Patch C: needs the reservation rule and simultaneous bids. No bid stream exists to replay.",
    "R7": "Needs per-act downstream-use rates per family (patch A measurement spine / C). Not recorded yet.",
}


def corrected_hours(hours: dict, heat_cfg) -> dict:
    """The cabinet reading with the recorded sensor measurement offset removed (the live heat
    price does the same, orion.physiology.heat_config.offset_at)."""
    out = {}
    for t, row in hours.items():
        off, _ = offset_at(heat_cfg, CABINET_SERIES, t)
        c = row.get("cabinet_c")
        out[t] = {**row, "cabinet_c": (c - off) if c is not None else None}
    return out


def run_patch_b(hours: dict, heat_cfg, *, soft_line_c: float, start: datetime, end: datetime, kwh: dict,
                cycle_start: datetime, cycle_end: datetime, envelope_usd: float = 100.0) -> dict:
    """The Patch B gate: the live Budget organ's rules (see the module docstring)."""
    from orion.physiology.budget import DEFAULT_SPECS
    from orion.schemas.physiology import ENERGY, HEAT

    sens = {h: e.usable for h, e in heat_cfg.hosts.items()}
    base = {h: e.baseline_watts for h, e in heat_cfg.hosts.items()}
    hspec = DEFAULT_SPECS[HEAT]
    params = hspec.params()
    hc = corrected_hours(hours, heat_cfg)
    heat = {}
    for mode in ("oracle", "persistence"):
        res = run_heat_replay(hc, sensitivity=sens, baselines=base, params=params, soft_line_c=soft_line_c,
                              start=start, end=end, forecast_mode=mode, bind_rearm_margin_c=0.5)
        binding = [r.status == EXOGENOUS_BIND or (r.excess is not None and r.excess > 0) for r in res]
        r1 = eval_r1(res, soft_line_c=soft_line_c, ceiling=params.ceiling)
        r3 = eval_r3([r.price for r in res], [r.status for r in res], binding, ceiling=params.ceiling,
                     amplitude=0.5 * params.ceiling)
        heat[mode] = {"R1": r1, "R3": r3, "R1_gate": r1["pass_excluding_decay_tail"], "R3_gate": r3["pass"],
                      "live_forecaster": mode == "persistence",
                      "status_counts": {st: sum(1 for r in res if r.status == st) for st in {r.status for r in res}},
                      "max_price": max((r.price for r in res), default=None),
                      "hours": [asdict(r) for r in res]}
    espec = DEFAULT_SPECS[ENERGY]
    eparams = espec.params(budget_floor=0.05 * envelope_usd)
    fixed = _p10([kwh[h] for h in kwh])
    pr, st, bd = run_energy_replay(kwh, envelope_usd=envelope_usd, fixed_kwh_per_h=fixed, params=eparams,
                                   cycle_start=cycle_start, cycle_end=cycle_end, start=start, end=end, rearm_frac=0.05)
    energy = {"envelope_usd": envelope_usd, "R3": eval_r3(pr, st, bd, ceiling=eparams.ceiling, amplitude=0.5),
              "hours": len(pr), "hours_binding": sum(bd),
              "hours_exogenous_bind": sum(1 for x in st if x == EXOGENOUS_BIND),
              "max_price": max(pr) if pr else None, "first_hours": pr[:6]}
    return {"sensitivity_c_per_100w": sens, "baseline_watts": base,
            "sensor_offsets": [{"since": o.since.isoformat(), "offset_c": o.offset_c, "change_event": o.change_event}
                               for o in heat_cfg.offsets],
            "heat_params_per_hour": asdict(params), "heat": heat, "energy": energy}


def run(extract: dict, fit: dict, *, soft_line_c: float, replay_start: datetime, replay_end: datetime,
        cycle_start: datetime, cycle_end: datetime, envelopes: list[float], heat_cfg=None) -> dict:
    hours = hourly_from_extract(extract)
    sens_raw = fit["replay_sensitivity"]
    # The design's first rule (upper bound when the interval spans 0), for the comparison variants.
    sens = {h: upper_bound_if_spans_zero(v["point"], v["lo"], v["hi"]) for h, v in sens_raw.items()}
    bad = [h for h, v in sens.items() if v is None]
    if bad or not sens:
        raise SystemExit(f"fit gave no usable sensitivity for {bad or 'any host'}; refusing to replay "
                         "(every hour would be no_reading)")
    window_hours = [h for h in hours if replay_start <= h < replay_end]
    baselines = {h: _p10([hours[t][h] for t in window_hours]) for h in sens}
    ceiling = 1.0
    heat_variants = {
        "pure_rule": PriceParams(eta=0.25, ceiling=ceiling, max_step=0.25, budget_floor=0.25, bind_ramp=False),
        "decay_floor": PriceParams(eta=0.25, ceiling=ceiling, max_step=0.25, budget_floor=0.25,
                                   slack_decay_floor=ceiling / HORIZON_H, bind_ramp=False),
        "decay_floor+bind_ramp": PriceParams(eta=0.25, ceiling=ceiling, max_step=0.25, budget_floor=0.25,
                                             slack_decay_floor=ceiling / HORIZON_H, bind_ramp=True),
    }
    heat = {}
    sens_point = {h: max(0.0, v["point"]) for h, v in sens_raw.items()}
    variants = [(k, heat_variants[k], 0.0, sens) for k in ("pure_rule", "decay_floor")] + [
        ("decay_floor+bind_hysteresis_0.5C", heat_variants["decay_floor"], 0.5, sens),
        ("decay_floor+bind_hysteresis_0.5C+point_sensitivity", heat_variants["decay_floor"], 0.5, sens_point),
        ("decay_floor+bind_hysteresis_0.5C+bind_ramp", heat_variants["decay_floor+bind_ramp"], 0.5, sens),
        ("decay_floor+bind_hysteresis_0.5C+bind_ramp+point_sensitivity", heat_variants["decay_floor+bind_ramp"], 0.5, sens_point),
    ]
    for mode in ("oracle", "persistence"):
        for vname, params, margin, sens_v in variants:
            res = run_heat_replay(hours, sensitivity=sens_v, baselines=baselines, params=params, soft_line_c=soft_line_c,
                                  start=replay_start, end=replay_end, forecast_mode=mode, bind_rearm_margin_c=margin)
            binding = [r.status == EXOGENOUS_BIND or (r.excess is not None and r.excess > 0) for r in res]
            heat[f"{mode}/{vname}"] = {
                "params": {**asdict(params), "bind_rearm_margin_c": margin}, "sensitivity": sens_v,
                "R1": eval_r1(res, soft_line_c=soft_line_c, ceiling=params.ceiling),
                "R3": eval_r3([r.price for r in res], [r.status for r in res], binding,
                              ceiling=params.ceiling, amplitude=0.5 * params.ceiling),
                "status_counts": {s: sum(1 for r in res if r.status == s) for s in {r.status for r in res}},
                "hours": [asdict(r) for r in res],
            }
    kwh, fill = fill_energy(hours, cycle_start, cycle_end)
    r4 = eval_r4(kwh, cycle_start=cycle_start, cycle_end=cycle_end, replay_start=replay_start, replay_end=replay_end)
    # Cross-check without any imputation: athena + circe are metered for the whole cycle.
    metered = {}
    t = cycle_start
    while t < cycle_end:
        k = hours.get(t, {}).get("kwh", {})
        if k.get("athena") is not None and k.get("circe") is not None:
            metered[t] = k["athena"] + k["circe"]
        t += timedelta(hours=1)
    r4_metered = eval_r4(metered, cycle_start=cycle_start, cycle_end=cycle_end, replay_start=replay_start,
                         replay_end=replay_end)
    r4_metered["hours_present"] = len(metered)
    r4_metered.pop("daily_forecasts", None)
    fixed = _p10([kwh[h] for h in kwh])
    energy = {}
    for env in envelopes:
        for vname, params in (("pure_rule", PriceParams(eta=0.25, ceiling=1.0, max_step=0.25, budget_floor=0.05 * env,
                                                        bind_ramp=False)),
                              ("decay_floor", PriceParams(eta=0.25, ceiling=1.0, max_step=0.25, budget_floor=0.05 * env,
                                                          slack_decay_floor=1.0 / HORIZON_H, bind_ramp=False))):
            pr, st, bd = run_energy_replay(kwh, envelope_usd=env, fixed_kwh_per_h=fixed, params=params,
                                           cycle_start=cycle_start, cycle_end=cycle_end, start=replay_start, end=replay_end)
            energy[f"envelope_{env:.0f}/{vname}"] = {
                "envelope_usd_hypothetical": env,
                "R3": eval_r3(pr, st, bd, ceiling=1.0, amplitude=0.5),
                "hours_binding": sum(bd), "hours_exogenous_bind": sum(1 for s in st if s == EXOGENOUS_BIND),
                "max_price": max(pr) if pr else None, "final_price": pr[-1] if pr else None,
            }
    patch_b = None
    if heat_cfg is not None:
        patch_b = run_patch_b(hours, heat_cfg, soft_line_c=soft_line_c, start=replay_start, end=replay_end, kwh=kwh,
                              cycle_start=cycle_start, cycle_end=cycle_end)
        patch_b["R4"] = {"pass": r4["pass"], "worst_error": r4["worst_error"], "realized_usd": r4["realized_usd"]}
    return {
        "patch_b": patch_b,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "replay_window": [replay_start.isoformat(), replay_end.isoformat()],
        "soft_line_c": soft_line_c, "horizon_h": HORIZON_H,
        "sensitivity_used_c_per_100w": sens, "sensitivity_fit": sens_raw,
        "baseline_watts_p10": baselines,
        "ceiling_note": ("Provisional ceiling = 1.0 want-value per degC. NOT data-sized: size_ceiling() needs "
                         "graded want values and per-act costs, and neither exists (no WantV1; cost_json null)."),
        "heat": heat, "energy_fill": fill, "R4": r4, "R4_athena_circe_unfilled": r4_metered,
        "energy_price_dynamics": energy,
        "baseline_note": "p10 baselines are taken over the whole replay window (mild look-ahead in persistence mode; the forecast itself uses only hour t-1).",
        "not_yet_evaluable": NOT_YET,
    }


def render_md(r: dict) -> str:
    L = ["# Body-economy replay", "", f"Generated {r['generated_at']}; replay {r['replay_window'][0]} .. {r['replay_window'][1]}; "
         f"soft line {r['soft_line_c']} C; horizon {r['horizon_h']} h.", "",
         "Sensitivity used (degC per 100 W held; upper bound where the interval spans 0): "
         + ", ".join(f"{h} {v:.3f}" for h, v in r["sensitivity_used_c_per_100w"].items() if v is not None), "",
         r["ceiling_note"], ""]
    pb = r.get("patch_b")
    if pb:
        f2 = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
        L += ["## Patch B gate (the live Budget organ's rules)", "",
              "Sensitivity (point, short-run fit, degC/100 W): " + ", ".join(f"{h} {v:.3f}" for h, v in pb["sensitivity_c_per_100w"].items())
              + "; baselines (W): " + ", ".join(f"{h} {v:.0f}" for h, v in pb["baseline_watts"].items())
              + "; sensor offsets: " + (", ".join(f"-{o['offset_c']} C from {o['since'][:16]}" for o in pb["sensor_offsets"]) or "none"), "",
              "The live forecaster is persistence (the newest cabinet reading); oracle grades the rule with a perfect 6 h mean.", "",
              "| forecast | R1 (decay tail excused) | under-line ~0 | over-line >0 | unmeetable flagged | R3 | max h to 0 | stuck | worst osc | unflagged ceiling run h | unflagged ceiling h: binding / not binding (windup) | max price |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for mode, v in pb["heat"].items():
            a, b = v["R1"], v["R3"]
            L.append(f"| {mode}{' (live)' if v.get('live_forecaster') else ''} | {'PASS' if v['R1_gate'] else 'FAIL'} | {f2(a['under_line_near_zero_frac'])} ({a['hours_under_line']} h; tail {a['under_line_nonzero_decay_tail']}, other {a['under_line_nonzero_other']}) | "
                     f"{f2(a['over_line_positive_frac'])} ({a['hours_over_line']} h) | {a['exogenous_flagged']}/{a['hours_exogenous_realized']} | "
                     f"{'PASS' if v['R3_gate'] else 'FAIL'} | {b['max_hours_to_zero']} | {b['stuck_positive_past_horizon']} | "
                     f"{b['max_peak_to_peak_oscillating_window']:.2f} | {b['longest_unflagged_ceiling_run_h']} | "
                     f"{b['unflagged_ceiling_hours_binding']} / {b['unflagged_ceiling_hours_not_binding']} | {f2(v['max_price'])} |")
        e = pb["energy"]
        L += ["", f"- Energy at the ${e['envelope_usd']:.0f} envelope: exogenous_bind {e['hours_exogenous_bind']} of {e['hours']} h, "
              f"max price {f2(e['max_price'])}, first hours {', '.join(f'{x:.2f}' for x in e['first_hours'])} (bounded ramp, no jump); "
              f"R3 {'PASS' if e['R3']['pass'] else 'FAIL'} (unflagged ceiling run {e['R3']['longest_unflagged_ceiling_run_h']} h).",
              f"- R4 (month-end energy forecast): {'PASS' if pb['R4']['pass'] else 'FAIL'}, worst error {pb['R4']['worst_error']:+.1%} "
              f"vs realized ${pb['R4']['realized_usd']:.2f}.",
              "- R6 (reservations): DEFERRED to Patch C (needs bids).", ""]
    L += ["## Heat price (R1, R3), comparison variants (pre-Patch B rules)", "",
         "| forecast / rule | R1 | under-line ~0 | under-line >0: decay tail / other | over-line >0 | realized-unmeetable hours flagged | R3 | max h to 0 | stuck / censored | worst osc | unflagged ceiling run h |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in r["heat"].items():
        a, b = v["R1"], v["R3"]
        f = lambda x: "n/a" if x is None else f"{x:.2f}"  # noqa: E731
        L.append(f"| {k} | {'PASS' if a['pass'] else 'FAIL'} ({'PASS' if a['pass_excluding_decay_tail'] else 'FAIL'} excl. tail) | {f(a['under_line_near_zero_frac'])} ({a['hours_under_line']} h) | "
                 f"{a['under_line_nonzero_decay_tail']} / {a['under_line_nonzero_other']} | "
                 f"{f(a['over_line_positive_frac'])} ({a['hours_over_line']} h) | {a['exogenous_flagged']}/{a['hours_exogenous_realized']} | "
                 f"{'PASS' if b['pass'] else 'FAIL'} | {b['max_hours_to_zero']} | {b['stuck_positive_past_horizon']} / {b['censored_rebound_before_zero']} | "
                 f"{b['max_peak_to_peak_oscillating_window']:.2f} | {b['longest_unflagged_ceiling_run_h']} |")
    r4 = r["R4"]
    L += ["", "## Month-end energy forecast (R4)", "",
          f"- Realized Orion cost {r4['cycle'][0][:10]}..{r4['cycle'][1][:10]}: ${r4['realized_usd']:.2f} at ${r4['tariff_usd_per_kwh']}/kWh "
          f"(gap-filled share of kWh: {r['energy_fill']['filled_share']:.1%}).",
          f"- To-date + trailing-7d forecast, {r4['forecasts']} hourly forecasts: worst error {r4['worst_error']:+.1%}, "
          f"mean |error| {r4['mean_abs_error']:.1%} -> {'PASS' if r4['pass'] else 'FAIL'}",
          f"- Harder rate-only forecast (30 d x trailing-7d rate): worst {r4['rate_only_worst_error']:+.1%}, "
          f"mean |error| {r4['rate_only_mean_abs_error']:.1%} -> {'PASS' if r4['rate_only_pass'] else 'FAIL'}",
          f"- No-imputation cross-check (athena + circe only, {r['R4_athena_circe_unfilled']['hours_present']} h metered): "
          f"worst {r['R4_athena_circe_unfilled']['worst_error']:+.1%}, rate-only worst {r['R4_athena_circe_unfilled']['rate_only_worst_error']:+.1%} "
          f"-> {'PASS' if r['R4_athena_circe_unfilled']['pass'] else 'FAIL'}",
          "", "## Energy price dynamics (hypothetical envelopes, R3 only)", ""]
    for k, v in r["energy_price_dynamics"].items():
        b = v["R3"]
        L.append(f"- {k}: binding {v['hours_binding']} h, exogenous {v['hours_exogenous_bind']} h, max price {v['max_price']:.2f}, "
                 f"R3 {'PASS' if b['pass'] else 'FAIL'} (max h to 0 {b['max_hours_to_zero']}, censored {b['censored_rebound_before_zero']}, "
                 f"osc {b['max_peak_to_peak_oscillating_window']:.2f}, unflagged ceiling run {b['longest_unflagged_ceiling_run_h']} h)")
    L += ["", "## Not yet evaluable", ""] + [f"- **{k}**: {v}" for k, v in r["not_yet_evaluable"].items()]
    return "\n".join(L) + "\n"


def _dt(s: str) -> datetime:
    return _ts(s if "T" in s else s + "T00:00:00Z")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit-json", required=True, help="output of fit_heat_sensitivity.py")
    ap.add_argument("--extract-start", default="2026-09-23T00:00:00Z")
    ap.add_argument("--extract-end", default="2026-10-23T00:00:00Z")
    ap.add_argument("--replay-start", default="2026-10-07")
    ap.add_argument("--replay-end", default="2026-10-23", help="exclusive; 10-23 00:00 = through 10-22")
    ap.add_argument("--cycle-start", default="2026-09-23", help="30-day realized window start")
    ap.add_argument("--soft-line-c", type=float, default=31.0, help="design: Juniper-set, proposed 31 C")
    ap.add_argument("--envelopes", default="100,180,197,230", help="monthly envelopes, USD ($100 = Juniper's, 10-23)")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--heat-config", default=str(_HERE.parents[1] / "config" / "physiology" / "heat_sensitivity.yaml"),
                    help="the live heat sensitivities (Patch B gate); '' to skip the gate")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args(argv)
    extract = data_mod.load_or_fetch(a.extract_start, a.extract_end, a.cache)
    fit = json.loads(Path(a.fit_json).read_text())
    cs = _dt(a.cycle_start)
    heat_cfg = None
    if a.heat_config:
        heat_cfg, err = load_heat_sensitivity(a.heat_config)
        if err:
            raise SystemExit(f"--heat-config {a.heat_config}: {err}")
    r = run(extract, fit, soft_line_c=a.soft_line_c, replay_start=_dt(a.replay_start), replay_end=_dt(a.replay_end),
            cycle_start=cs, cycle_end=cs + timedelta(days=30), envelopes=[float(x) for x in a.envelopes.split(",")],
            heat_cfg=heat_cfg)
    Path(a.out_json).write_text(json.dumps(r, indent=1, default=str))
    Path(a.out_md).write_text(render_md(r))
    print(render_md(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
