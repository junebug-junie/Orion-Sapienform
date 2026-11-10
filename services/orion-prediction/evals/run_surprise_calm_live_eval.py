"""Surprise calm check on live scored forecasts, read-only (nothing is stored).

Spec acceptance (docs/superpowers/specs/2026-11-08-prediction-error-suite-rebuild-design.md):
"over a quiet 24 h, excess surprise mean within the null band around 0". Prints, per target series
and horizon: mean surprisal, mean excess vs the calm point (0.95), its 95 % null band from
independent origins, share of p <= 0.05 and <= 0.2 (calibrated: 0.05 / 0.20), and live 80 %
coverage. Negative excess = ranges too wide; positive = too narrow or the world changed.

    cd services/orion-prediction && POSTGRES_URI=... PYTHONPATH=../..:. python evals/run_surprise_calm_live_eval.py --hours 168

Baseline 2026-11-10 06:30 UTC (champion rows). The calm check FAILS today:
- 7 days: cabinet 15 min excess -0.115 +-0.056, office 15 min -0.116 +-0.061 (coverage in band).
- quiet 24 h: cabinet 15 min -0.281 +-0.101, 60 min -0.372 +-0.145; office 15 min -0.355 +-0.085.
The ranges are calibrated on average (80 % coverage ~0.84) but not conditionally: one width for calm
and busy hours, so a calm day reads far below 0 and the misses cluster in busy periods. Fix belongs
in the forecaster (volatility-adaptive conformal ranges), not in this transform. No consumer may
read prediction.surprise.v1 until this check passes.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SERVICE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SERVICE), str(SERVICE.parents[1])]

from app.series import load_registry  # noqa: E402
from app.store import PostgresStore  # noqa: E402
from app.surprise import CALM_SURPRISAL, summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()
    store = PostgresStore(os.environ["POSTGRES_URI"])
    since = datetime.now(timezone.utc) - timedelta(hours=args.hours)
    print(f"calm surprisal = {CALM_SURPRISAL:.3f}; window {args.hours:g} h")
    for spec in load_registry(SERVICE.parents[1] / "config" / "prediction" / "series.v1.yaml").targets():
        rows = [r for r in store.scored_with_quantiles(spec.series_id, since) if r["role"] == "champion"]
        cov = {}
        for r in store.live_scores(spec.series_id, since):
            if r["role"] == "champion":
                cov.setdefault(r["horizon_min"], []).append(bool(r["in_80"]))
        for hm, h in summarize(rows, spec.grain_min).items():
            c = cov.get(int(hm)) or []
            band = "n/a" if h["null_band_95"] is None else f"+-{h['null_band_95']:.3f}"
            print(f"{spec.series_id:24s} {hm:>5s} min  n={h['n']:5d} (ind {h['n_independent']:4d})  "
                  f"excess {h['mean_excess']:+.3f} {band:>8s} calm={h['calm']}  "
                  f"p<=.05 {h['share_p_le_0_05']:.3f}  p<=.2 {h['share_p_le_0_2']:.3f}  "
                  f"cov80 {sum(c) / len(c) if c else float('nan'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
