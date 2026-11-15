"""Air-path replay eval (read-only, live DB): does the closet top tell Orion anything the cabinet does not?

Re-runs the checks of docs/superpowers/specs/2026-11-12-orion-air-path-design.md (#3234) over the last
N days and prints one line each:

  LEAD_LAG   closet-top vs cabinet: cross-correlation of 5-min and 15-min changes; the peak lag
             (positive = the closet top moves FIRST). The design found lag 0 (no lead).
  FORECAST   one-hour cabinet change, linear model fitted on the first 70% and tested on the last 30%:
             MAE with and without closet heat (closet top - office), paired one-sided test
             (app.backtest.paired_win_p, Newey-West lag 12 = the 1 h horizon's overlap at 5-min origins).
  SLOPES     circe and hecate heat per 100 W at the closet top, through the real
             GET /effects/heat-per-watt path (app.effects.Effects.heat_per_watt; window capped at the
             fit's MAX_DAYS). hecate reads "refused: <reason>" until it runs enough hours.
  RESIDUAL   daily mean residual of the hourly closet-top gap fit (the stuck-air watch: a day the closet
             top runs hotter than the machines' watts explain). Days above +0.5 C are marked.
  GATE       the design's gate for a second patch that feeds the cooling act: closet heat lowers the
             one-hour cabinet MAE with p < 0.05, OR the residual is high (> +0.5 C mean over the 6 h
             before) ahead of a cabinet 32 C crossing on >= 3 separate days.

The office-sound line from the design is dropped: Juniper (2026-11-12) said office noise is not a cost.
Nothing is written: every store method that could write raises.

    cd services/orion-prediction && PYTHONPATH=../..:. POSTGRES_URI=postgresql://...@localhost:55432/conjourney \\
        python evals/run_air_path_eval.py --days 30

Exit 0 when every line could be computed (the gate verdict is a finding, not a failure); 1 when the
data needed for a line is missing (e.g. the closet board went dark).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SERVICE = Path(__file__).resolve().parents[1]
REPO = SERVICE.parents[1]
sys.path[:0] = [str(SERVICE), str(REPO)]

CLOSET, CABINET, ROOM = "closet.top_temp_c", "athena.cabinet_temp_c", "office.temp_c"
WATTS = ("athena.chassis_watts", "circe.chassis_watts", "hecate.chassis_watts", "athena.cabinet_cooling_watts")
HOT_C = 32.0
STUCK_C = 0.5


class ReadOnly:
    def __init__(self, st):
        self.st = st

    def __getattr__(self, k):
        if k.startswith(("insert", "write", "set_", "upsert", "prune", "update", "delete")):
            raise RuntimeError(f"read-only eval: {k} blocked")
        return getattr(self.st, k)


def engine():
    from app.engine import Engine
    from app.series import load_registry
    from app.store import PostgresStore

    reg = load_registry(REPO / "config" / "prediction" / "series.v1.yaml")
    return Engine(reg, ReadOnly(PostgresStore(os.environ["POSTGRES_URI"])), backtest_windows=40,
                  backtest_step_min=180, select_interval_sec=21600, settle_sec=60, publish_enabled=False)


def five_min(eng, start: datetime, end: datetime) -> pd.DataFrame:
    """Real 5-min buckets (no fill), each series' own exclusions masked."""
    from app.frame import covariate_series, mask_windows, to_grain

    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="5min", inclusive="left")
    cols = {}
    for sid in (CLOSET, CABINET, ROOM):
        spec = eng.effective(eng.registry.get(sid), eng.clock())
        cols[sid] = mask_windows(to_grain(eng.store.points(spec, start, end), 5, spec.agg, 0, start=start, end=end),
                                 spec.exclude).reindex(idx)
    for sid in WATTS:
        cols[sid] = covariate_series(eng.store, eng.registry.get(sid), start, end, idx)
    return pd.DataFrame(cols, index=idx)


def lead_lag(f: pd.DataFrame) -> str:
    out = []
    for step in (1, 3):                       # 5-min and 15-min changes
        g = f[[CLOSET, CABINET]].resample(f"{5 * step}min", origin="epoch").mean() if step > 1 else f[[CLOSET, CABINET]]
        d = g.diff()
        rs = {k: d[CLOSET].corr(d[CABINET].shift(-k)) for k in range(-3, 4)}   # k > 0: closet first
        peak = max(rs, key=lambda k: rs[k] if np.isfinite(rs[k]) else -9)
        out.append(f"{5 * step}-min changes peak at lag {5 * step * peak:+d} min (r {rs[peak]:.2f}; lag 0 r "
                   f"{rs[0]:.2f}; -{5 * step}/+{5 * step} min r {rs[-1]:.2f}/{rs[1]:.2f}; n {int(d.dropna().shape[0])})")
    return "LEAD_LAG closet top vs cabinet (+ = closet top first): " + "; ".join(out)


def forecast(f: pd.DataFrame) -> tuple[str, float | None]:
    from app.backtest import paired_win_p

    h = 12                                    # 1 h ahead in 5-min steps
    x = pd.DataFrame({
        "cab": f[CABINET], "room": f[ROOM], "room_trend": f[ROOM] - f[ROOM].shift(3),
        "athena": f["athena.chassis_watts"], "circe": f["circe.chassis_watts"],
        "ac": f["athena.cabinet_cooling_watts"], "heat": f[CLOSET] - f[ROOM]})
    x["y"] = f[CABINET].shift(-h) - f[CABINET]
    x = x.dropna()
    if len(x) < 200:
        raise SystemExit(f"FORECAST: only {len(x)} complete 5-min rows")
    cut = int(0.7 * len(x))
    train, test = x.iloc[:cut], x.iloc[cut:]
    base_cols = ["cab", "room", "room_trend", "athena", "circe", "ac"]

    def err(cols):
        A = np.column_stack([np.ones(len(train)), train[cols].to_numpy()])
        beta = np.linalg.lstsq(A, train["y"].to_numpy(), rcond=None)[0]
        B = np.column_stack([np.ones(len(test)), test[cols].to_numpy()])
        return test["y"].to_numpy() - B @ beta

    e0, e1 = err(base_cols), err(base_cols + ["heat"])
    p, n = paired_win_p(e1, e0, lag=h)
    ptxt = "n/a" if p is None else f"{p:.3f}"
    return (f"FORECAST one-hour cabinet change, test {test.index[0]:%m-%d %H:%M}..{test.index[-1]:%m-%d %H:%M} UTC "
            f"({n} points): MAE without closet heat {np.mean(np.abs(e0)):.3f} C, with {np.mean(np.abs(e1)):.3f} C; "
            f"paired one-sided p (with < without) {ptxt}"), p


def slopes(eng, days: float) -> str:
    from app.effects import Effects
    from app.heat_balance import MAX_DAYS, HeatBalanceRefused

    d = min(days, MAX_DAYS)
    try:
        fit = Effects(eng).heat_per_watt(CLOSET, d, None, False)
    except HeatBalanceRefused as exc:
        return f"SLOPES closet top ({d:g} d): whole fit refused: {exc.reason}"
    parts = []
    for w in ("circe.chassis_watts", "hecate.chassis_watts"):
        s = fit["slopes"].get(w)
        if s is not None:
            parts.append(f"{w.split('.')[0]} {s['per_100w']:+.2f} C per 100 W (95% {s['ci_lo_per_100w']:+.2f}.."
                         f"{s['ci_hi_per_100w']:+.2f}, p {s['p_value']:.3g}, mean {s['mean']:.0f} W)")
        else:
            parts.append(f"{w.split('.')[0]} refused: {fit['refused'].get(w)}")
    return (f"SLOPES closet top minus office ({d:g} d window, {fit['n_hours']} h, R^2 {fit['r2']:.2f}): "
            + "; ".join(parts))


def residual(eng, start: datetime, end: datetime, f5: pd.DataFrame) -> tuple[str, int]:
    """Hourly gap fit on the identifiable watts/controls at lags 0..2 h, plain OLS; daily mean residual."""
    from app.heat_balance import HEAT_BALANCE, _lagged, identifiable, load_frame

    class Fx:
        pass

    fx = Fx()
    fx.engine, fx.store = eng, eng.store
    frame, cfg = load_frame(fx, CLOSET, start, end)
    seen = frame["gap"].notna()             # judge each machine only on the hours the closet top has data
    names = [w for w in (*cfg["watts"], *cfg["controls"]) if identifiable(frame[w][seen]) is None]
    X, _ = _lagged(frame, names)
    data = pd.concat([frame["gap"].rename("y"), X], axis=1).dropna()
    A = np.column_stack([np.ones(len(data)), data.drop(columns="y").to_numpy()])
    beta = np.linalg.lstsq(A, data["y"].to_numpy(), rcond=None)[0]
    res = pd.Series(data["y"].to_numpy() - A @ beta, index=data.index)
    daily = res.groupby(res.index.floor("1D")).agg(["mean", "count"])
    days = ", ".join(f"{d:%m-%d} {m:+.2f}{'*' if m > STUCK_C else ''}" for d, (m, c) in daily.iterrows() if c >= 12)
    # Cabinet 32 C crossings: the first 5-min mean >= 32 C on a day; was the residual high in the 6 h before?
    cab = f5[CABINET][f5[CLOSET].notna()].dropna()   # only days the closet top was read
    hot = cab[cab >= HOT_C]
    first = hot.groupby(hot.index.floor("1D")).apply(lambda s: s.index[0]) if not hot.empty else pd.Series(dtype=object)
    led = 0
    for t in first:
        before = res[(res.index >= t - pd.Timedelta(hours=6)) & (res.index < t)]
        led += bool(len(before) and before.mean() > STUCK_C)
    return (f"RESIDUAL daily mean of the closet-top gap fit (C; * = above +{STUCK_C} C, the stuck-air watch; "
            f"fit on {', '.join(n.split('.')[0] if 'chassis' in n else n for n in names)}, {len(data)} h): {days}. "
            f"Cabinet {HOT_C:g} C crossing days: {len(first)}, residual high in the 6 h before: {led}"), led


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=30.0)
    args = ap.parse_args()
    eng = engine()
    end = (eng.latest_complete_bucket(eng.registry.get(CLOSET), eng.clock()) + pd.Timedelta(minutes=5)).floor("60min")
    end = end.to_pydatetime()
    start = end - timedelta(days=args.days)
    f5 = five_min(eng, start, end)
    have = f5[CLOSET].dropna()
    if have.empty:
        print(f"NO DATA: no closet-top readings in {start:%Y-%m-%d}..{end:%Y-%m-%d}")
        return 1
    print(f"WINDOW {start:%Y-%m-%d %H:%M}..{end:%Y-%m-%d %H:%M} UTC asked ({args.days:g} d); closet-top readings "
          f"from {have.index[0]:%Y-%m-%d %H:%M} ({len(have)} 5-min buckets)")
    print(lead_lag(f5))
    line, p = forecast(f5)
    print(line)
    print(slopes(eng, args.days))
    line, led = residual(eng, start, end, f5)
    print(line)
    passed = (p is not None and p < 0.05) or led >= 3
    print(f"GATE second patch (closet heat into the cabinet forecast / cooling act): {'PASS' if passed else 'NOT MET'} "
          f"(paired p {'n/a' if p is None else f'{p:.3f}'} vs < 0.05; residual-led 32 C days {led} vs >= 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
