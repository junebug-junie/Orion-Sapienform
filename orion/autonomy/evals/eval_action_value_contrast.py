#!/usr/bin/env python3
"""Does the contrast actually remove the confound? Replayed on live data.

This is acceptance check 3 of docs/superpowers/specs/2026-08-21-action-value-
control-arm-design.md, and the only one that matters -- the rest is plumbing.

The claim under test: phase 1's value for the docker-prune family was
`mean(after - before)` over the ticks it ran, roughly -0.15, and that number
is regression to the mean rather than an effect. Replaying the same real
frames through the baseline-matched contrast must land materially closer to
zero. If it does not, either the confound is real and the estimator failed
to remove it, or the estimator is broken -- both are reasons to stop, not to
ship a number.

Deliberately NOT a pass-by-construction eval. It fails if the contrast comes
back close to the raw delta, and it also fails if the replay finds nothing to
measure, because "no data" reported as a pass is how a dead eval survives.

Reads the live database. Writes nothing. Safe to run any time.

    python3 orion/autonomy/evals/eval_action_value_contrast.py --days 3
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from orion.autonomy.contrast import (  # noqa: E402
    MOVEMENT_EPSILON,
    ControlCell,
    baseline_bin,
    contrast,
)
from orion.autonomy.prediction import EffectPosterior, update_posterior  # noqa: E402

SIGNAL = "resource_pressure"

# The three prune templates are ~21% of everything Orion dispatches.
TARGETS = ("host:docker_images", "host:docker_containers", "host:docker_build_cache")

REPLAY_QUERY = """
SELECT (f.feedback_frame_json->'pressure_before'->>'resource_pressure')::float AS before_v,
       (f.feedback_frame_json->'pressure_after'->>'resource_pressure')::float  AS after_v,
       COALESCE(
         jsonb_array_length(d.dispatch_frame_json->'dispatched_candidates'), 0
       ) AS n_disp,
       COALESCE((
         SELECT jsonb_agg(DISTINCT c->>'target_id')
           FROM jsonb_array_elements(
                  COALESCE(d.dispatch_frame_json->'dispatched_candidates', '[]'::jsonb)
                ) c
       ), '[]'::jsonb) AS dispatched_targets
  FROM substrate_feedback_frames f
  JOIN substrate_execution_dispatch_frames d
    ON d.frame_id = f.source_execution_dispatch_frame_id
 WHERE f.created_at > now() - make_interval(days => :days)
   AND f.feedback_frame_json->'pressure_before'->>'resource_pressure' IS NOT NULL
   AND f.feedback_frame_json->'pressure_after'->>'resource_pressure' IS NOT NULL
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument(
        "--dsn",
        default=os.environ.get(
            "ORION_PG_DSN", "postgresql://postgres:postgres@localhost:55432/conjourney"
        ),
    )
    parser.add_argument(
        "--max-contrast-share",
        type=float,
        default=0.5,
        help=(
            "For a target whose RAW delta is materially large, fail if "
            "|contrast| still exceeds this share of |raw|. Most of a large raw "
            "number is supposed to be mean reversion."
        ),
    )
    parser.add_argument(
        "--material-raw",
        type=float,
        default=0.05,
        help=(
            "Only targets whose |raw delta| reaches this are gated on the share "
            "above. Gating a target whose raw number is already ~0 is a "
            "ratio-of-two-small-numbers test: it fails on noise and says "
            "nothing about whether the confound was removed. First version of "
            "this eval did exactly that and failed two targets whose raw "
            "deltas were +0.048 and -0.021."
        ),
    )
    args = parser.parse_args()

    from sqlalchemy import create_engine, text

    engine = create_engine(args.dsn)
    with engine.connect().execution_options(stream_results=True) as conn:
        rows = conn.execute(text(REPLAY_QUERY), {"days": args.days})

        treated: dict[tuple[str, str, str, int], EffectPosterior] = {}
        treated_deltas: dict[str, list[float]] = {t: [] for t in TARGETS}
        control: dict[tuple[str, str, int], ControlCell] = {}
        idle_ticks = 0
        total = 0

        for row in rows:
            total += 1
            before, after = float(row.before_v), float(row.after_v)
            delta = after - before
            b = baseline_bin(before)

            if row.n_disp == 0:
                idle_ticks += 1
                key = (SIGNAL, "no_action", b)
                cell = control.get(key) or ControlCell(EffectPosterior.cold(), 0)
                control[key] = ControlCell(
                    posterior=update_posterior(cell.posterior, delta),
                    moved_n=cell.moved_n + (1 if abs(delta) >= MOVEMENT_EPSILON else 0),
                )
                continue

            hit = set(row.dispatched_targets or []) & set(TARGETS)
            for target in hit:
                tkey = ("maintain", target, SIGNAL, b)
                prior = treated.get(tkey) or EffectPosterior.cold()
                treated[tkey] = update_posterior(prior, delta)
                treated_deltas[target].append(delta)

    print(f"replayed {total} feedback frames over {args.days} days")
    print(f"  idle (untreated) ticks: {idle_ticks}")
    frozen_bins = sorted(k[2] for k, c in control.items() if c.is_frozen)
    if frozen_bins:
        print(f"  FROZEN control bins refused as coverage: {frozen_bins}")
    print()

    header = f"{'target':<28} {'n':>6} {'raw':>10} {'contrast':>10} {'+/-':>9} {'cover':>7} {'verdict':<12}"
    print(header)
    print("-" * len(header))

    failures: list[str] = []
    measured = 0
    gated_ok = 0
    for target in TARGETS:
        deltas = treated_deltas[target]
        if not deltas:
            print(f"{target:<28} {0:>6} {'-':>10} {'-':>10} {'-':>9} {'-':>7} {'NO DATA':<12}")
            continue
        raw = sum(deltas) / len(deltas)
        est = contrast(treated, control, "maintain", target, SIGNAL)
        if est is None:
            print(
                f"{target:<28} {len(deltas):>6} {raw:>10.4f} {'-':>10} {'-':>9} "
                f"{'0%':>7} {'NO CONTROL':<12}"
            )
            continue
        measured += 1
        share = abs(est.value) / abs(raw) if abs(raw) > 1e-9 else 0.0
        gated = abs(raw) >= args.material_raw
        ok = (not gated) or share <= args.max_contrast_share
        if not ok:
            failures.append(
                f"{target}: contrast {est.value:+.4f} is {share:.0%} of raw {raw:+.4f}"
            )
        verdict = "OK" if gated and ok else ("FAIL" if not ok else "raw~0")
        print(
            f"{target:<28} {est.treated_n:>6} {raw:>10.4f} {est.value:>10.4f} "
            f"{est.sd:>9.4f} {1.0 - est.uncovered_weight:>6.0%} "
            f"{verdict:<12}"
        )
        if gated and ok:
            gated_ok += 1

    print()
    if measured == 0:
        print(
            "FAIL: nothing was measurable. An eval that reports a pass on an "
            "empty replay is how a dead eval survives -- this is a failure, not "
            "a clean run."
        )
        return 1
    if failures:
        print("FAIL: the contrast did not remove the confound:")
        for f in failures:
            print(f"  - {f}")
        return 1
    if gated_ok == 0:
        print(
            "FAIL: no target had a raw delta large enough to gate "
            f"(|raw| >= {args.material_raw}). Nothing here demonstrates the "
            "confound was removed, so this is not a pass. Widen --days or "
            "lower --material-raw deliberately, do not accept this as green."
        )
        return 1
    print(
        f"PASS: {gated_ok} of {measured} measured target(s) had a materially "
        f"large raw delta, and every one of them shrank to within "
        f"{args.max_contrast_share:.0%} of it. Targets marked `raw~0` had no "
        "confound to remove and are reported, not gated."
    )
    print(
        "Reminder 1: this arm is `no_action`, which is QUASI-experimental. Ticks "
        "where nothing ran are calmer ticks; the baseline bin absorbs most of "
        "that and not all of it."
    )
    print(
        "Reminder 2: the +/- comes from a FIXED observation variance "
        "(orion.autonomy.prediction.DEFAULT_OBSERVATION_VARIANCE = 0.04, fitted "
        "to 68,715 real pressure deltas), not from the empirical spread inside "
        "each bin. It is a model interval, not a measured standard error, and "
        "it is wrong wherever a bin's real spread departs from that constant."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
