#!/usr/bin/env python3
"""What has each autonomous action actually been worth?

Reports a BASELINE-MATCHED CONTRAST, not a raw post-action delta.

The distinction is the whole point of this script. An action is dispatched
because a pressure is high, and a high pressure falls on its own. So "what
the signal did after the action ran" is dominated by mean reversion. Live
over 3 days, `prune_dangling_images` reads as a 5.8x effect on the raw
number (-0.148 vs -0.026) and INVERTS in 6 of 8 baseline deciles. The raw
column is therefore not printed as a value; it is printed next to the
contrast so the gap between them stays visible.

Columns:

  n            LIFETIME treated observations in bins that have control
               coverage. Deliberately labelled: the contrast is computed
               from cumulative posterior cells and is NOT affected by
               --days, which filters only the ledger columns (`raw`,
               `nats/act`, `alone%`, `upheld%`). Mixing a windowed and a
               lifetime quantity in one row without saying so was review
               finding 5.
  contrast     sum_b w_b * (treated_mean[b] - control_mean[b]), the effect
               over the conditions this action actually runs in. Can be
               zero. Can be positive when the action claims `decrease` --
               an action that cannot lose is not competing.
  +/-          one standard deviation of that contrast
  raw          the unconditional mean delta over --days, i.e. the phase-1
               number. Shown only to expose the size of the confound. Note
               the window mismatch above before reading the gap as pure
               confound.
  alone%       share of treated observations from a tick where this was the
               ONLY dispatched action (frame_dispatch_count == 1). The field
               delta is frame-wide, so a low number means the whole row is
               shared attribution.
  cover        share of the treated arm's volume that HAS a usable control
               bin. A low number means the contrast describes a minority of
               the action's real behaviour. `thin`/`frozen` name the bins
               that were refused and why -- too few untreated observations,
               versus an instrument that stopped moving there.
  arm          which control arm produced it. `no_action` is
               quasi-experimental (ticks where nothing ran are systematically
               calmer ticks; binning absorbs most of that, not all).
               `randomized_holdback` is experimental. Never merged.
  nats/act     mean Bayesian surprise per run -- the information one run
               buys. Converges toward 0 for an action whose effect is
               already known, which is the point: a tic earns nothing
               without anyone writing a rule against tics.
  sole%        share of observations where no other dispatched candidate in
               the same tick claimed the same signal.

An action/signal pair with no control coverage prints `NO CONTROL` and no
number, because a number there is what would be believed.

Read-only. Safe to run any time.

    python3 scripts/analysis/report_action_value.py --days 7
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from orion.autonomy.contrast import ControlCell, contrast  # noqa: E402
from orion.autonomy.prediction import EffectPosterior  # noqa: E402

LEDGER_QUERY = """
SELECT dispatch_kind,
       target_id,
       signal_id,
       direction,
       count(*)                                    AS n,
       avg(surprise_nats)                          AS mean_nats,
       sum(surprise_nats)                          AS total_nats,
       avg(observed_delta)                         AS raw_delta,
       count(*) FILTER (WHERE frame_dispatch_count = 1) AS alone_n,
       count(*) FILTER (WHERE co_predictors = 0)   AS sole_n,
       count(*) FILTER (WHERE claim_upheld)        AS upheld_n,
       count(*) FILTER (WHERE claim_upheld IS NOT NULL) AS decidable_n
  FROM substrate_action_outcomes
 WHERE observed_at > now() - make_interval(days => :days)
   AND arm = 'dispatched'
 GROUP BY 1, 2, 3, 4
 ORDER BY total_nats DESC
"""

TREATED_CELLS_QUERY = """
SELECT dispatch_kind, target_id, signal_id, baseline_bin,
       posterior_mean, posterior_variance, posterior_n
  FROM substrate_action_effect_posterior
"""

# move_rate is NOT optional here. `ControlCell.is_frozen` reads ONLY move_rate
# (orion/autonomy/contrast.py), and the dataclass defaults it to 1.0 = presumed
# alive. Omitting it from this SELECT -- as this query did until 2026-08-21 --
# silently made every cell read healthy, so `contrast()` accepted pinned control
# cells as coverage, `est.frozen_bins` was always empty, and the
# "FROZEN CONTROL CELLS -- refused as coverage" block below could never print.
#
# Live at the time of the fix, three cells satisfied the freeze condition
# (n >= 200, move_rate < 0.25) and none were reported: continuity_pressure
# 0.099, introspection_pressure 0.109, and reliability_pressure 0.120 --
# the last of which IS a PredictableSignal, so an action claiming it while the
# signal sat in bin 0 would have had its raw delta handed back wearing the
# contrast's name and confidence. That is precisely what contrast.py exists to
# refuse.
CONTROL_CELLS_QUERY = """
SELECT signal_id, arm, baseline_bin,
       posterior_mean, posterior_variance, posterior_n, moved_n, move_rate
  FROM substrate_signal_control_cells
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--dsn",
        default=os.environ.get(
            "ORION_PG_DSN", "postgresql://postgres:postgres@localhost:55432/conjourney"
        ),
    )
    args = parser.parse_args()

    from sqlalchemy import create_engine, text

    engine = create_engine(args.dsn)
    with engine.connect() as conn:
        rows = conn.execute(text(LEDGER_QUERY), {"days": args.days}).mappings().all()
        treated = {
            (r["dispatch_kind"], r["target_id"], r["signal_id"], int(r["baseline_bin"])):
            EffectPosterior(
                mean=float(r["posterior_mean"]),
                variance=float(r["posterior_variance"]),
                n=int(r["posterior_n"]),
            )
            for r in conn.execute(text(TREATED_CELLS_QUERY)).mappings().all()
        }
        control = {
            (r["signal_id"], r["arm"], int(r["baseline_bin"])): ControlCell(
                posterior=EffectPosterior(
                    mean=float(r["posterior_mean"]),
                    variance=float(r["posterior_variance"]),
                    n=int(r["posterior_n"]),
                ),
                moved_n=int(r["moved_n"]),
                move_rate=float(r["move_rate"]),
            )
            for r in conn.execute(text(CONTROL_CELLS_QUERY)).mappings().all()
        }

    if not rows:
        print(f"No scored actions in the last {args.days} days.")
        return 0

    header = (
        f"{'kind':<10} {'target':<24} {'signal':<20} {'dir':<9} "
        f"{'n(life)':>8} {'contrast':>10} {'+/-':>8} {'raw(win)':>9} {'cover':>6} "
        f"{'arm':<20} {'nats/act':>9} {'sole%':>6} {'alone%':>7} {'upheld%':>8}"
    )
    print(header)
    print("-" * len(header))

    frozen = sorted(
        (k[0], k[2], cell.move_rate)
        for k, cell in control.items()
        if cell.is_frozen
    )
    if frozen:
        print("FROZEN CONTROL CELLS -- refused as coverage:")
        for signal, b, rate in frozen:
            print(f"  {signal} bin {b}: movement rate {rate:.4f}")
        print()

    no_control: list[str] = []
    for r in rows:
        est = contrast(
            treated, control, r["dispatch_kind"], r["target_id"], r["signal_id"]
        )
        sole_pct = 100.0 * r["sole_n"] / r["n"] if r["n"] else 0.0
        alone_pct = 100.0 * r["alone_n"] / r["n"] if r["n"] else 0.0
        upheld_pct = (
            100.0 * r["upheld_n"] / r["decidable_n"] if r["decidable_n"] else float("nan")
        )
        label = (
            f"{r['dispatch_kind']:<10} {r['target_id']:<24.24} "
            f"{r['signal_id']:<20.20} {r['direction']:<9}"
        )
        if est is None:
            no_control.append(
                f"{r['dispatch_kind']}/{r['target_id']}/{r['signal_id']}"
            )
            print(
                f"{label} {r['n']:>7}w {'NO CONTROL':>10} {'':>8} "
                f"{r['raw_delta']:>9.4f} {'0%':>6} {'-':<20} "
                f"{r['mean_nats']:>9.5f} {sole_pct:>5.0f}% {alone_pct:>6.0f}% "
                f"{upheld_pct:>7.0f}%"
            )
            continue
        print(
            f"{label} {est.treated_n:>8} {est.value:>10.4f} {est.sd:>8.4f} "
            f"{r['raw_delta']:>9.4f} {1.0 - est.uncovered_weight:>5.0%} "
            f"{est.control_arm + '/' + est.evidence_class[:5]:<20} "
            f"{r['mean_nats']:>9.5f} {sole_pct:>5.0f}% {alone_pct:>6.0f}% "
            f"{upheld_pct:>7.0f}%"
        )
        if est.thin_bins or est.frozen_bins:
            detail = []
            if est.thin_bins:
                detail.append(f"thin={list(est.thin_bins)}")
            if est.frozen_bins:
                detail.append(f"FROZEN={list(est.frozen_bins)}")
            print(f"{'':<64}   refused bins: {' '.join(detail)}")

    print()
    if no_control:
        print(
            f"{len(no_control)} action/signal pair(s) have NO control coverage and "
            "are reported without a value:"
        )
        for name in no_control:
            print(f"  - {name}")
        print()
    print(
        "contrast is quasi-experimental unless the arm says randomized_holdback. "
        "Ticks where nothing ran are calmer ticks; the baseline bin absorbs most "
        "of that, not all of it."
    )
    print(
        "The +/- is a MODEL interval: posterior variance comes from a fixed "
        "observation variance (0.04) divided by n, so it is 0.2/sqrt(n) by "
        "construction. Successive field ticks are autocorrelated, so the "
        "effective sample size is well below n and the real interval is wider "
        "than this in both directions."
    )
    print(
        "`n(life)` is cumulative; `raw(win)` and the trailing percentages honour "
        "--days. They are not the same window."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
