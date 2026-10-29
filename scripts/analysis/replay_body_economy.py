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

Evaluated: R1, R3, R4. With ``--wants`` / ``--wants-only`` (Patch C1): R2, R5, R6, R7 on synthetic
wants built from real act history (grammar_acts, act_downstream_use) and decided by the live Choice
code (``orion/physiology/choice.py``). R5 is reported NOT EVALUABLE with its reason (no posted lane
price exists for its baseline). ``render_md``'s "Not yet evaluable" list below is the pre-C1 note.

**The Patch B gate** (``patch_b`` in the output): the rules the live Budget organ runs
(``orion/physiology/budget.py``): bounded ramp into bind, 0.5 C bind hysteresis, the point
sensitivity of the short-run fit and the baselines from ``config/physiology/heat_sensitivity.yaml``,
the raw cabinet reading (no sensor offset: the sensor never moved), slack decay of ceiling / horizon, and for
energy the $100 envelope with 5% bind hysteresis. R1 is graded with one horizon of decay tail
excused (findings doc, Patch B rule 3). The older variants stay beside it for comparison.

Usage:
  .venv/bin/python scripts/analysis/replay_body_economy.py --fit-json heat_fit.json \
      --out-json replay.json --out-md replay.md [--cache extract.json] \
      [--heat-config config/physiology/heat_sensitivity.yaml] [--wants]
  .venv/bin/python scripts/analysis/replay_body_economy.py --wants-only \
      --out-json wants.json --out-md wants.md [--acts-cache acts.json] [--cache extract.json]
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
from orion.physiology.heat_config import load_heat_sensitivity  # noqa: E402
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

# Pre-Patch C1 notes, kept for the price-only report. R2, R5, R6 and R7 now run with --wants.
NOT_YET = {
    "R2": "Needs replayed non-GPU wants (reading, outreach text, dream bookkeeping) with values and costs. "
          "WantV1 and want_ledger do not exist yet (design patch C), and grammar_acts.cost_json is null.",
    "R5": "Needs a posted lane price and background bid-before-lease (patch E). No lane price exists, so "
          "there is nothing to replay against the 10-21..22 refusal baseline.",
    "R6": "DEFERRED to Patch C: needs the reservation rule and simultaneous bids. No bid stream exists to replay.",
    "R7": "Needs per-act downstream-use rates per family (patch A measurement spine / C). Not recorded yet.",
}


def run_patch_b(hours: dict, heat_cfg, *, soft_line_c: float, start: datetime, end: datetime, kwh: dict,
                cycle_start: datetime, cycle_end: datetime, envelope_usd: float = 100.0) -> dict:
    """The Patch B gate: the live Budget organ's rules (see the module docstring)."""
    from orion.physiology.budget import DEFAULT_SPECS
    from orion.schemas.physiology import ENERGY, HEAT

    sens = {h: e.usable for h, e in heat_cfg.hosts.items()}
    base = {h: e.baseline_watts for h, e in heat_cfg.hosts.items()}
    hspec = DEFAULT_SPECS[HEAT]
    params = hspec.params()
    heat = {}
    for mode in ("oracle", "persistence"):
        res = run_heat_replay(hours, sensitivity=sens, baselines=base, params=params, soft_line_c=soft_line_c,
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
              + "; cabinet reading raw (no sensor offset)", "",
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


# ---------------------------------------------------------------- Patch C1: wants (R2, R5, R6, R7)
#
# Synthetic wants built from REAL act history: every act a bidding family really opened is replayed
# as a want at its opened_at, with the family's real cost profile, hunger from its real previous
# close, rank against its real 30-day history and weights from real downstream use. Decided by the
# live Choice code (orion/physiology/choice.py). Nothing is tuned here: the rules are the live ones.

# Pool seats and their slots (config/gpu_pool.yaml + discovery, 10-27). No physiology_state exists
# before 10-23, so lanes are replayed at 0 use and price 0 ("not replayed"), stated in the output.
POOL_SLOTS = {"agent": 1, "agent-gpu2": 2, "agent-deep": 1, "metacog": 4, "fast": 4, "chat": 1, "diffusion": 1}
REFLEX_ON_C, REFLEX_OFF_C = 34.0, 33.0     # the spinal heat reflex (orion/autonomy/thermal_gate.py)


def acts_from_extract(ex: dict):
    from orion.physiology.appetite import ActRecord
    from orion.physiology.weights import UseRecord

    def dt(v):
        return _ts(v) if v else None

    acts = [ActRecord(act_id=r["act_id"], family=r["actor"], opened_at=dt(r.get("opened_at")),
                      closed_at=dt(r.get("closed_at")), outcome_value=r.get("outcome_value"),
                      outcome_source=r.get("outcome_source"), signal_id=r.get("signal_id"),
                      cost=r.get("cost_json") if isinstance(r.get("cost_json"), dict) else {}) for r in ex["acts"]]
    uses = [UseRecord(act_id=r["act_id"], family=r["actor"], started_at=dt(r.get("started_at")),
                      link_status=r["link_status"], used=r.get("used"), window_closed=bool(r.get("window_closed")))
            for r in ex["uses"]]
    return acts, uses


def _ts_any(v: str) -> datetime:
    d = datetime.fromisoformat(v.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def reflex_armed(hours: dict, start: datetime, end: datetime) -> dict:
    """Per hour: the 34 C reflex armed (enter at 34, re-arm below 33) on the RAW hourly cabinet mean
    (the reflex reads the raw sensor). An hourly mean understates minute peaks: a lower bound."""
    armed, out, t = False, {}, start
    while t < end:
        c = hours.get(t, {}).get("cabinet_c")
        if c is not None:
            armed = c >= REFLEX_ON_C or (armed and c >= REFLEX_OFF_C)
        out[t] = armed
        t += timedelta(hours=1)
    return out


def synthetic_state(t: datetime, *, heat_price: float, heat_status: str, cabinet_c, juniper_sent: float):
    from orion.schemas.physiology import PhysiologyConstraintReadingV1 as R, PhysiologyStateV1

    bind = {EXOGENOUS_BIND: "exogenous_bind", NO_READING: "no_reading"}.get(heat_status, "binding" if heat_price > 0 else "slack")
    heat = (R(constraint="heat", unit="c", level=cabinet_c, budget=31.0, budget_kind="soft_line",
              price={"value": heat_price, "ceiling": 1.0, "bind_state": bind, "horizon_sec": 21600.0, "why": "replayed"})
            if cabinet_c is not None else
            R(constraint="heat", unit="c", no_reading="replay:no_cabinet", budget=31.0, budget_kind="soft_line"))
    lanes = [R(constraint=f"lane:{role}", unit="slot_s", level=0.0, budget=slots * 3600.0, budget_kind="capacity",
               detail={"slots": slots, "window_sec": 3600.0, "replay": "not_replayed"},
               price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 3600.0, "why": "not replayed"})
             for role, slots in POOL_SLOTS.items()]
    jun = R(constraint="attention:juniper", unit="messages", level=juniper_sent, budget=4.0, budget_kind="daily_cap",
            price={"value": 0.0, "ceiling": 1.0, "bind_state": "slack", "horizon_sec": 86400.0, "why": "replayed level"})
    return PhysiologyStateV1(tick_id=f"replay:{t.isoformat()}", generated_at=t, constraints=[heat, jun, *lanes])


def _replay_decisions(acts, uses, families, *, start, end, heat_by_hour, cabinet, armed, lane_price=None):
    """Decide every real act of a bidding family in [start, end) as a want, in time order, with one
    reservation book. ``lane_price`` overrides every lane price (the R7 stress run)."""
    from zoneinfo import ZoneInfo

    from orion.physiology.choice import ReservationBook, build_context, decide
    from orion.physiology.weights import weight_series
    from orion.schemas.physiology_want import WantV1

    tz = ZoneInfo("America/Denver")
    series = weight_series(families, acts, uses, start=(start - timedelta(days=1)).date(), end=end.date())
    book = ReservationBook()
    # The weekly floor counts THIS replay's own funded wants (closed loop), exactly as live counts
    # want_ledger: an empty ledger at the start, then every funded decision (Patch C3a).
    book.seed_funded([], start)
    rows = []
    wants = sorted((a for a in acts if a.opened_at is not None and start <= a.opened_at < end
                    and families.get(a.family) is not None and families.get(a.family).bids), key=lambda a: a.opened_at)
    for a in wants:
        t = a.opened_at
        hour = t.replace(minute=0, second=0, microsecond=0)
        cfg = families.get(a.family)
        gpu = bool(cfg.lanes)
        local_day = t.astimezone(tz).date()
        sent = sum(1 for b in acts if b.family == "hub.outreach" and b.opened_at is not None and b.opened_at < t
                   and b.opened_at.astimezone(tz).date() == local_day)
        hp, hs = heat_by_hour.get(hour, (0.0, NO_READING))
        hot = bool(armed.get(hour)) or hs == EXOGENOUS_BIND or hp > 0
        before = [b for b in acts if (b.closed_at or b.opened_at) is not None and (b.closed_at or b.opened_at) < t]
        st = synthetic_state(t, heat_price=hp, heat_status=hs, cabinet_c=cabinet.get(hour), juniper_sent=float(sent))
        if lane_price is not None:
            st = st.model_copy(update={"constraints": [
                c.model_copy(update={"price": c.price.model_copy(update={"value": lane_price, "bind_state": "binding"})})
                if c.constraint.startswith("lane:") else c for c in st.constraints]})
        if gpu and armed.get(hour):
            rows.append({"t": t.isoformat(), "family": a.family, "gpu": gpu, "hot": hot, "outcome": "deferred",
                         "reason": "spinal:heat_reflex", "net": None, "floor_share": None})
            continue
        ctx = build_context(families, before, [u for u in uses if u.started_at and u.started_at < t], st, t,
                            weights=series.get(t.date(), {}))
        share = book.funded_share(a.family, cfg.tier, t)
        # Not "replay": that producer name marks hand-run probes, which the floor never counts.
        w = WantV1(want_id=f"replay:{a.act_id}", family=a.family, producer="replay.act_history", verb="replay", tier=cfg.tier,
                   opened_at=t)
        d = decide(w, st, ctx, book, t)
        if d.outcome == "funded":
            book.release(w.want_id)   # the real act ran and closed; its use is history now
        rows.append({"t": t.isoformat(), "family": a.family, "tier": cfg.tier, "gpu": gpu, "hot": hot,
                     "outcome": d.outcome, "reason": d.reason, "net": d.net, "floor_share": share,
                     "intensity": d.intensity, "binding": d.binding_constraint})
    return rows, series


def _rate(rows):
    return (sum(1 for r in rows if r["outcome"] == "funded") / len(rows)) if rows else None


def eval_r2(rows, *, heat_start, heat_end) -> dict:
    """R2: over the heat window, NON-GPU wants are funded at a reduced, non-zero rate; Juniper's turns
    100%. Graded literally: non-zero AND strictly below the same wants' rate outside the heat."""
    def in_win(r):
        return heat_start <= _ts_any(r["t"]) < heat_end
    hot_ng = [r for r in rows if not r["gpu"] and in_win(r) and r["hot"]]
    cool_ng = [r for r in rows if not r["gpu"] and not r["hot"]]
    hot_g = [r for r in rows if r["gpu"] and in_win(r) and r["hot"]]
    rh, rc = _rate(hot_ng), _rate(cool_ng)
    nonzero = rh is not None and rh > 0
    reduced = rh is not None and rc is not None and rh < rc
    by_family = {}
    for r in hot_ng:
        f = by_family.setdefault(r["family"], {"n": 0, "funded": 0})
        f["n"] += 1
        f["funded"] += r["outcome"] == "funded"
    return {"pass": bool(nonzero and reduced), "nonzero": nonzero, "reduced": reduced,
            "non_gpu_hot_wants": len(hot_ng), "non_gpu_hot_funded_rate": rh,
            "non_gpu_cool_wants": len(cool_ng), "non_gpu_cool_funded_rate": rc,
            "gpu_hot_wants": len(hot_g), "gpu_hot_funded_rate": _rate(hot_g),
            "gpu_hot_held_by_reflex": sum(1 for r in hot_g if r["reason"] == "spinal:heat_reflex"),
            "non_gpu_hot_by_family": by_family,
            "non_gpu_hot_reasons": {k: sum(1 for r in hot_ng if r["reason"] == k) for k in sorted({r["reason"] for r in hot_ng})},
            "non_gpu_cool_reasons": {k: sum(1 for r in cool_ng if r["reason"] == k) for k in sorted({r["reason"] for r in cool_ng})},
            "juniper_turns_funded": "100% by construction: Tier 0 (hub.chat_reply) never bids and is never deferred",
            "structural": "In C1 heat is unpriced per act, so heat cannot reduce a non-GPU want: this FAIL is structural, not a tuning miss.",
            "note": ("Non-GPU here = families whose families.yaml lanes are empty (their acts record no pool lease). "
                     "Heat is UNPRICED per act in C1 (wh_by_host is null on every act), so the heat price cannot "
                     "touch a non-GPU want; only Juniper's cap and the weights can.")}


def eval_r5(ex: dict) -> dict:
    lp = [r["max_lane_price"] for r in ex.get("lane_prices", []) if r.get("max_lane_price") is not None]
    ref = ex.get("background_refusals", [])
    base = [r for r in ref if r["day"] in ("2026-10-21", "2026-10-22")]
    return {"pass": None, "evaluable": False,
            "reason": ("No posted lane price exists for the 10-21..22 baseline (physiology_state starts 10-23) and no "
                       "background caller bids before it leases (Patch E). Nothing to replay against the baseline."),
            "baseline_10_21_22_background_refused": sum(r["refused"] for r in base),
            "baseline_10_21_22_background_shed": sum(r.get("shed", 0) for r in base),
            "baseline_10_21_22_background_granted": sum(r["granted"] for r in base),
            "physiology_ticks_with_lane_price": len(lp),
            "ticks_any_lane_price_positive": sum(1 for x in lp if x > 0),
            "max_lane_price_seen": max(lp) if lp else None}


def eval_r6(acts, families, *, now: datetime, burst: int = 20) -> dict:
    """R6: a burst of ``burst`` simultaneous bids at a ZERO price cannot reserve more than 1.2x a
    lane. Per placeable family with real slot history, on an empty lane (worst case)."""
    from orion.physiology.choice import ReservationBook, build_context, decide
    from orion.schemas.physiology_want import WantV1

    st = synthetic_state(now, heat_price=0.0, heat_status="live", cabinet_c=28.0, juniper_sent=0.0)
    ctx = build_context(families, acts, [], st, now, weights={})
    out, worst = {}, 0.0
    for f, c in sorted(families.families.items()):
        app = ctx.appetite.get(f)
        if not c.bids or c.placement != "one_of" or app is None or app.total_slot_s_q is None:
            continue
        book = ReservationBook()
        ds = [decide(WantV1(want_id=f"r6:{f}:{i}", family=f, producer="replay", verb="burst", tier=c.tier), st, ctx, book, now)
              for i in range(burst)]
        per_lane = {}
        for ln in c.lanes:
            budget = POOL_SLOTS.get(ln, 1) * 3600.0
            per_lane[ln] = round(book.total(f"lane:{ln}", now) / budget, 4)
        worst = max([worst, *per_lane.values()])
        out[f] = {"slot_s_p90": round(app.total_slot_s_q, 1), "funded": sum(d.outcome == "funded" for d in ds),
                  "deferred_capacity": sum(d.reason == "capacity_reserved" for d in ds), "reserved_over_budget": per_lane}
    # Mixed burst: every placeable family interleaved, lanes already half used, diffusion unloaded.
    half = st.model_copy(update={"constraints": [
        (c.model_copy(update={"level": 0.0, "budget": 0.0, "detail": {**c.detail, "slots": 0}})
         if c.constraint == "lane:diffusion" else
         c.model_copy(update={"level": 0.5 * c.budget}) if c.constraint.startswith("lane:") else c)
        for c in st.constraints]})
    ctx2 = build_context(families, acts, [], half, now, weights={})
    book = ReservationBook()
    fams = [f for f, c in sorted(families.families.items()) if c.bids]
    for i in range(burst):
        for f in fams:
            decide(WantV1(want_id=f"r6m:{f}:{i}", family=f, producer="replay", verb="burst", tier=families.get(f).tier),
                   half, ctx2, book, now)
    mixed = {}
    for c in half.constraints:
        if c.constraint.startswith("lane:"):
            budget = c.budget if c.budget else 3600.0
            mixed[c.constraint] = round(((c.level or 0.0) + book.total(c.constraint, now)) / budget, 4)
    worst_mixed = max(mixed.values()) if mixed else 0.0
    return {"pass": worst <= 1.2 and worst_mixed <= 1.2, "worst_reserved_over_budget": worst,
            "mixed_burst_worst_use_plus_reserved_over_budget": worst_mixed, "mixed_burst": mixed,
            "by_family": out,
            "note": ("A design property: reservations are only added after the forecast + reservations + cost <= line "
                     "check, so this confirms the code enforces the rule; it cannot find a pricing failure.")}


def weekly_floor_report(rows, families, *, floor: float = 0.05, tier: int = 2) -> dict:
    """The literal weekly clause, per family of ``tier``, over a decided replay: at every want time,
    the family's share of the tier's funded wants over the trailing 7 days. A family passes when that
    share never sat under ``floor`` while the family had an unfunded, HUNGRY bid (intensity > 0)
    deferred on PRICE in that same week -- i.e. demand-limited families (everything they asked for
    was funded) and capacity-only deferrals do not count against the quota, a price deferral does."""
    fund = [(_ts_any(r["t"]), r["family"]) for r in rows if r["outcome"] == "funded" and r.get("tier") == tier]
    out = {}
    for f, c in sorted(families.families.items()):
        if not c.bids or c.tier != tier:
            continue
        mine = [r for r in rows if r["family"] == f]
        worst, misses = None, []
        for r in mine:
            t = _ts_any(r["t"])
            week = [g for (ft, g) in fund if t - timedelta(days=7) <= ft <= t]
            share = (sum(1 for g in week if g == f) / len(week)) if week else None
            if share is not None:
                worst = share if worst is None else min(worst, share)
            if (r["outcome"] == "deferred" and r["reason"] == "price" and (r.get("intensity") or 0) > 0
                    and (share is None or share < floor)):
                misses.append(r["t"])
        end = mine[-1] if mine else None
        last_t = _ts_any(rows[-1]["t"]) if rows else None
        week = [g for (ft, g) in fund if last_t and last_t - timedelta(days=7) <= ft <= last_t]
        out[f] = {"bids": len(mine), "funded": sum(1 for r in mine if r["outcome"] == "funded"),
                  "floor_funded": sum(1 for r in mine if r["reason"] == "exploration_floor"),
                  "deferred_price_hungry": sum(1 for r in mine if r["outcome"] == "deferred" and r["reason"] == "price"
                                               and (r.get("intensity") or 0) > 0),
                  "deferred_price_sated": sum(1 for r in mine if r["outcome"] == "deferred" and r["reason"] == "price"
                                              and not (r.get("intensity") or 0) > 0),
                  "deferred_capacity": sum(1 for r in mine if r["reason"] == "capacity_reserved"),
                  "min_trailing_week_share": round(worst, 4) if worst is not None else None,
                  "final_week_share": round(sum(1 for g in week if g == f) / len(week), 4) if week else None,
                  "quota_misses": len(misses), "first_miss": misses[0] if misses else None,
                  "last_bid": end["t"] if end else None}
    return out


def eval_r7(rows_stress, series, acts, uses, families) -> dict:
    """R7: drift <= 0.1/day; every Tier 2 family keeps >= 5% of Tier 2's funded wants per rolling
    7 days (Patch C3a's weekly quota, graded literally by ``weekly_floor_report``: no hungry price
    deferral while the family sat under 5%); a zero-use family ends below its prior. Tier 1 has no
    floor (its reserved lane share is its guarantee): its shares are reported, not graded."""
    from orion.physiology.weights import MAX_DRIFT_PER_DAY, max_daily_move, weight_series

    drift = max_daily_move(series)
    last = series[max(series)]
    floor_fired = sum(1 for r in rows_stress if r["reason"] == "exploration_floor")
    weekly = weekly_floor_report(rows_stress, families, tier=2)
    weekly_t1 = weekly_floor_report(rows_stress, families, tier=1)
    violations = [f for f, x in weekly.items() if x["quota_misses"]]
    # Literal whole-window shares in the stress run.
    shares = {}
    for f, c in families.families.items():
        if not c.bids:
            continue
        mine = [r for r in rows_stress if r["family"] == f and r["outcome"] == "funded"]
        tier = [r for r in rows_stress if families.get(r["family"]).tier == c.tier and r["outcome"] == "funded"]
        shares[f] = round(len(mine) / len(tier), 4) if tier else None
    live_zero = {f: {"n": w.evidence.n, "use_rate": w.evidence.use_rate, "basis": w.basis, "final": round(w.weight, 4)}
                 for f, w in last.items() if w.evidence.use_rate == 0.0}
    # Synthetic zero-use family on real timing: thought.painting's real linked rows, every one unused.
    from orion.physiology.weights import UseRecord

    synth_uses = [u for u in uses if u.family != "thought.painting"] + [
        UseRecord(u.act_id, u.family, u.started_at, "linked", False, True) for u in uses if u.family == "thought.painting"]
    s2 = weight_series(families, acts, synth_uses, start=min(series), end=max(series))
    w2 = s2[max(s2)]["thought.painting"]
    prior = families.get("thought.painting").prior_weight
    synth = {"family": "thought.painting (all real linked rows set unused)", "n": w2.evidence.n,
             "final_weight": round(w2.weight, 4), "prior": prior, "below_prior": w2.weight < prior}
    live_ok = all(v["final"] < 1.0 for v in live_zero.values() if v["basis"] == "evidence")
    return {"pass": bool(drift <= MAX_DRIFT_PER_DAY + 1e-9 and not violations and synth["below_prior"] and live_ok),
            "max_daily_move": round(drift, 4), "floor_violations": len(violations), "floor_violation_families": violations,
            "floor_fired": floor_fired, "weekly_quota_tier2": weekly, "weekly_shares_tier1_ungraded": weekly_t1,
            "stress_lane_price": 1.0, "funded_share_in_tier_stress": shares,
            "zero_use_live": live_zero,
            "by_construction_note": ("The weekly-quota clause confirms the code, not the economics: the grader recomputes "
                                     "the same trailing-7-day Tier 2 share the floor branch reads, from the same replay "
                                     "rows, and the floor branch runs before the price branch, so a hungry price deferral "
                                     "under 5% can only appear if that bookkeeping is broken. The drift cap is applied "
                                     "inside weight_series (also by construction). The synthetic zero-use clause is the "
                                     "part that could fail. Sated and capacity deferrals are listed, never counted."),
            "zero_use_live_note": ("A zero-use family under 10 evidenced acts stays at its prior by the ungraded rule; "
                                   "the 'ends below prior' clause can only bite at n >= 10."),
            "zero_use_synthetic": synth,
            "final_weights": {f: {"weight": round(w.weight, 4), "target": round(w.target, 4), "basis": w.basis,
                                  "use": w.evidence.use_rate, "use_n": w.evidence.use_n,
                                  "engagement": w.evidence.engagement_rate, "engagement_n": w.evidence.engagement_n}
                              for f, w in sorted(last.items())}}


def run_wants(extract: dict, acts_ex: dict, heat_cfg, *, soft_line_c: float, acts_start: datetime, acts_end: datetime,
              heat_start: datetime, heat_end: datetime) -> dict:
    from orion.physiology.families import load_families

    families, err = load_families(_HERE.parents[1] / "config" / "physiology" / "families.yaml")
    if err:
        raise SystemExit(f"families.yaml: {err}")
    acts, uses = acts_from_extract(acts_ex)
    hours = hourly_from_extract(extract)
    heat_by_hour = {}
    pb = None
    if heat_cfg is not None:
        # The Patch B gate's heat rule with the live forecaster (persistence), heat only.
        from orion.physiology.budget import DEFAULT_SPECS
        from orion.schemas.physiology import HEAT

        pb = run_heat_replay(corrected_hours(hours, heat_cfg),
                             sensitivity={h: e.usable for h, e in heat_cfg.hosts.items()},
                             baselines={h: e.baseline_watts for h, e in heat_cfg.hosts.items()},
                             params=DEFAULT_SPECS[HEAT].params(), soft_line_c=soft_line_c, start=acts_start,
                             end=acts_end, forecast_mode="persistence", bind_rearm_margin_c=0.5)
        for r in pb:
            heat_by_hour[_ts_any(r.hour)] = (r.price, r.status)
    cabinet = {t: v.get("cabinet_c") for t, v in hours.items()}
    armed = reflex_armed(hours, acts_start, acts_end)
    rows, series = _replay_decisions(acts, uses, families, start=acts_start, end=acts_end, heat_by_hour=heat_by_hour,
                                     cabinet=cabinet, armed=armed)
    rows_stress, _ = _replay_decisions(acts, uses, families, start=acts_start, end=acts_end, heat_by_hour=heat_by_hour,
                                       cabinet=cabinet, armed=armed, lane_price=1.0)
    return {
        "window": [acts_start.isoformat(), acts_end.isoformat()], "heat_window": [heat_start.isoformat(), heat_end.isoformat()],
        "acts": len(acts), "uses": len(uses), "wants_replayed": len(rows),
        "reflex_armed_hours": sum(1 for v in armed.values() if v),
        "heat_price_source": "Patch B persistence replay (the live forecaster)" if pb else "none (no heat config)",
        "lanes_note": "No physiology_state before 10-23: lanes replayed at 0 use and price 0 (capacity lines still apply).",
        "R2": eval_r2(rows, heat_start=heat_start, heat_end=heat_end),
        "R5": eval_r5(acts_ex),
        "R6": eval_r6(acts, families, now=acts_end),
        "R7": eval_r7(rows_stress, series, acts, uses, families),
        "reasons": {k: sum(1 for r in rows if r["reason"] == k) for k in sorted({r["reason"] for r in rows})},
    }


def render_wants_md(w: dict) -> str:
    f = lambda x: "n/a" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))  # noqa: E731
    v = lambda p: "PASS" if p is True else ("FAIL" if p is False else "NOT EVALUABLE")  # noqa: E731
    r2, r5, r6, r7 = w["R2"], w["R5"], w["R6"], w["R7"]
    L = ["## Patch C1 wants replay (R2, R5, R6, R7)", "",
         f"Synthetic wants from real act history {w['window'][0][:10]}..{w['window'][1][:10]}: {w['wants_replayed']} wants "
         f"({w['acts']} acts, {w['uses']} downstream-use rows); reflex armed {w['reflex_armed_hours']} h; heat price: "
         f"{w['heat_price_source']}. {w['lanes_note']}", "",
         f"- **R2 {v(r2['pass'])}**: non-GPU wants in the heat {w['heat_window'][0][:10]}..{w['heat_window'][1][:10]}: "
         f"{r2['non_gpu_hot_wants']} wants, funded {f(r2['non_gpu_hot_funded_rate'])} (non-zero: {r2['nonzero']}); outside "
         f"the heat {r2['non_gpu_cool_wants']} wants, funded {f(r2['non_gpu_cool_funded_rate'])} (reduced: {r2['reduced']}). "
         f"Reasons in the heat {r2['non_gpu_hot_reasons']}, outside {r2['non_gpu_cool_reasons']}. "
         f"GPU wants in the heat: {r2['gpu_hot_wants']}, funded {f(r2['gpu_hot_funded_rate'])}, held by the reflex "
         f"{r2['gpu_hot_held_by_reflex']}. Juniper's turns: {r2['juniper_turns_funded']}. {r2['note']} {r2['structural']}",
         f"- **R5 {v(r5['pass'])}**: {r5['reason']} Baseline 10-21..22: {r5['baseline_10_21_22_background_refused']} background "
         f"requests ended unavailable ({r5['baseline_10_21_22_background_shed']} shed) vs {r5['baseline_10_21_22_background_granted']} "
         f"granted. Since 10-23, {r5['ticks_any_lane_price_positive']} of {r5['physiology_ticks_with_lane_price']} ticks had any lane "
         f"price > 0 (max {f(r5['max_lane_price_seen'])}).",
         f"- **R6 {v(r6['pass'])}**: 20 simultaneous bids at price 0 on an empty lane: worst reserved / lane budget "
         f"{r6['worst_reserved_over_budget']:.2f} (limit 1.2); mixed burst (all families x20, lanes half used, "
         f"diffusion unloaded): worst (use + reserved) / budget {r6['mixed_burst_worst_use_plus_reserved_over_budget']:.2f}. "
         f"{r6['note']} "
         + "; ".join(f"{k}: {d['funded']} funded, {d['deferred_capacity']} deferred on capacity, {d['reserved_over_budget']}"
                     for k, d in r6["by_family"].items()),
         f"- **R7 {v(r7['pass'])}**: max weight move {r7['max_daily_move']:.3f}/day (cap 0.1); weekly exploration quota with every "
         f"lane price at 1.0: {r7['floor_fired']} wants funded by the floor, "
         f"{r7['floor_violations']} Tier 2 families with a hungry price deferral while under 5% of the trailing week "
         f"({', '.join(r7['floor_violation_families']) or 'none'}); synthetic zero-use family {r7['zero_use_synthetic']['family']} (n {r7['zero_use_synthetic']['n']}) ends at "
         f"{r7['zero_use_synthetic']['final_weight']} vs prior {r7['zero_use_synthetic']['prior']} (below: "
         f"{r7['zero_use_synthetic']['below_prior']}); live zero-use families: {r7['zero_use_live'] or 'none'}. "
         f"{r7['zero_use_live_note']} {r7['by_construction_note']}", "",
         "| family | weight | target | basis | use rate (n) | Juniper reply rate (n) | funded share in tier (stress) |",
         "|---|---|---|---|---|---|---|"]
    for fam, x in r7["final_weights"].items():
        L.append(f"| {fam} | {x['weight']:.3f} | {x['target']:.3f} | {x['basis']} | {f(x['use'])} ({x['use_n']}) | "
                 f"{f(x['engagement'])} ({x['engagement_n']}) | {f(r7['funded_share_in_tier_stress'].get(fam))} |")
    L += ["", "Weekly quota, Tier 2, stress run (trailing-7-day share of Tier 2's funded wants):", "",
          "| family | bids | funded | by floor | price-deferred hungry | price-deferred sated | capacity | min week share | final week share | quota misses |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for fam, x in r7["weekly_quota_tier2"].items():
        L.append(f"| {fam} | {x['bids']} | {x['funded']} | {x['floor_funded']} | {x['deferred_price_hungry']} | "
                 f"{x['deferred_price_sated']} | {x['deferred_capacity']} | {f(x['min_trailing_week_share'])} | "
                 f"{f(x['final_week_share'])} | {x['quota_misses']} |")
    L += ["", "Tier 1 (no floor by design; its reserved lane share is its guarantee), stress run: "
          + "; ".join(f"{fam} final week share {f(x['final_week_share'])} ({x['funded']}/{x['bids']} funded)"
                      for fam, x in r7["weekly_shares_tier1_ungraded"].items())]
    L += ["", "Replay decision reasons: " + ", ".join(f"{k} {n}" for k, n in w["reasons"].items()), ""]
    return "\n".join(L) + "\n"


def _dt(s: str) -> datetime:
    return _ts(s if "T" in s else s + "T00:00:00Z")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit-json", default=None, help="output of fit_heat_sensitivity.py (required unless --wants-only)")
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
    ap.add_argument("--wants", action="store_true", help="also run the Patch C1 wants replay (R2, R5, R6, R7)")
    ap.add_argument("--wants-only", action="store_true", help="only the wants replay; --fit-json not needed")
    ap.add_argument("--acts-start", default="2026-10-16T00:00:00Z", help="grammar_acts begin 10-16")
    ap.add_argument("--acts-end", default=None, help="default: now (UTC hour)")
    ap.add_argument("--heat-start", default="2026-10-18", help="R2's heat window (design: 10-18..22)")
    ap.add_argument("--heat-end", default="2026-10-23")
    ap.add_argument("--acts-cache", default=None)
    a = ap.parse_args(argv)
    if a.wants or a.wants_only:
        acts_end = a.acts_end or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
        acts_ex = data_mod.load_or_fetch_acts(a.acts_start, acts_end, a.acts_cache)
        heat_cfg = None
        if a.heat_config:
            heat_cfg, err = load_heat_sensitivity(a.heat_config)
            if err:
                raise SystemExit(f"--heat-config {a.heat_config}: {err}")
        wx = data_mod.load_or_fetch(a.acts_start, acts_end, a.cache if a.wants_only else None)
        wants = run_wants(wx, acts_ex, heat_cfg, soft_line_c=a.soft_line_c, acts_start=_ts_any(a.acts_start),
                          acts_end=_ts_any(acts_end), heat_start=_dt(a.heat_start), heat_end=_dt(a.heat_end))
        if a.wants_only:
            Path(a.out_json).write_text(json.dumps({"wants": wants}, indent=1, default=str))
            md = "# Body-economy replay (wants only)\n\n" + render_wants_md(wants)
            Path(a.out_md).write_text(md)
            print(md)
            return 0
    if not a.fit_json:
        raise SystemExit("--fit-json is required unless --wants-only")
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
    md = render_md(r)
    if a.wants:
        r["wants"] = wants
        md += "\n" + render_wants_md(wants)
    Path(a.out_json).write_text(json.dumps(r, indent=1, default=str))
    Path(a.out_md).write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
