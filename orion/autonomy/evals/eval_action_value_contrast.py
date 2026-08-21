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

KNOWN LIMITATION, stated because this artifact stands behind the headline
number. The replay reads `feedback_frame_json->'pressure_before'`, which
`orion/feedback/builder.py` populates via `extract_field_pressure_snapshot` --
the helper `orion.feedback.outcome_resolution._present_pressures` was
deliberately written NOT to use, because it returns 0.0 for a channel the
field did not produce and applies clamp01. The `IS NOT NULL` filter below
cannot tell a fabricated 0.0 from a measured one. So this replays a slightly
different quantity than the shipped code will write. For `resource_pressure`
the gap should be nil (it maps from a real channel that is always present),
but "should be" is not "is". Once phase 2 has run for a day, re-run this
against `substrate_action_outcomes.baseline/observed_delta`, which is the
production read path, and compare.

Reads the live database. Writes nothing. Safe to run any time.

    python3 orion/autonomy/evals/eval_action_value_contrast.py --days 3
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from orion.autonomy.contrast import (  # noqa: E402
    ControlCell,
    baseline_bin,
    contrast,
)
from orion.feedback.extractors import PRESSURE_DELTA_EPSILON  # noqa: E402
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
 ORDER BY f.created_at
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
                cell = control.get(key) or ControlCell(EffectPosterior.cold())
                control[key] = cell.observe(
                    update_posterior(cell.posterior, delta),
                    moved=abs(delta) >= PRESSURE_DELTA_EPSILON,
                )
                continue

            hit = set(row.dispatched_targets or []) & set(TARGETS)
            for target in hit:
                tkey = ("maintain", target, SIGNAL, b)
                prior = treated.get(tkey) or EffectPosterior.cold()
                treated[tkey] = update_posterior(prior, delta)
                treated_deltas[target].append(delta)

    # ORDER BY f.created_at above is load-bearing, not tidiness. ControlCell's
    # move_rate is an EWMA -- a WINDOWED statistic -- and folding an unordered
    # result set into it computes the rate over an arbitrary permutation,
    # which is not a rate of anything. The first version of this replay had no
    # ORDER BY and reported 100% coverage on a bin whose instrument had been
    # pinned for twelve hours.

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

    # Positive control (review finding 6). Every gate above is a
    # "did the number SHRINK" test, which an estimator hardwired to return
    # 0.0 would pass with the maximum possible margin -- and finding 1 showed
    # a degenerate control cell pushes the answer toward zero, which those
    # gates would read as extra evidence of success. This is the check that
    # separates "the confound was removed" from "the estimator is dead":
    # shift every treated cell by a known constant and require it back.
    INJECTED = 0.25
    shifted = {
        key: EffectPosterior(post.mean + INJECTED, post.variance, post.n)
        for key, post in treated.items()
    }
    probe_target = next(
        (t for t in TARGETS if treated_deltas[t] and contrast(treated, control, "maintain", t, SIGNAL)),
        None,
    )
    if probe_target is None:
        print("FAIL: no target with control coverage to run the positive control on.")
        return 1
    base = contrast(treated, control, "maintain", probe_target, SIGNAL)
    probe = contrast(shifted, control, "maintain", probe_target, SIGNAL)
    recovered = probe.value - base.value
    print(
        f"positive control on {probe_target}: injected {INJECTED:+.4f}, "
        f"recovered {recovered:+.4f}"
    )
    if abs(recovered - INJECTED) > 1e-9:
        print(
            "FAIL: the estimator did not recover a known injected shift. It is "
            "not measuring anything, and every 'shrank to zero' verdict above "
            "is meaningless."
        )
        return 1

    # Instrument-sensitivity band (review finding 1). `is_frozen` is a LIVE
    # guard: it refuses a cell while the instrument is pinned. It does not
    # retroactively remove contamination a cell absorbed before recovering,
    # and the 2026-08-21 pin is inside this very window. So report how much
    # the answer depends on the degenerate bins instead of asserting a single
    # number -- if the conclusion is not robust across the band, there is no
    # conclusion.
    print()
    print("instrument-sensitivity band (bins whose control arm is mostly frozen):")
    suspect = {
        k for k, cell in control.items()
        if cell.posterior.n >= 200 and (cell.moved_n / cell.posterior.n) < 0.25
    }
    if not suspect:
        print("  none -- every control bin moved on at least 25% of its ticks.")
    else:
        for k in sorted(suspect):
            cell = control[k]
            print(
                f"  {k[0]} bin {k[2]}: {cell.moved_n}/{cell.posterior.n} moved "
                f"({cell.moved_n / cell.posterior.n:.1%}), "
                f"control mean {cell.posterior.mean:+.4f}"
            )
        cleaned = {k: v for k, v in control.items() if k not in suspect}
        for target in TARGETS:
            if not treated_deltas[target]:
                continue
            full = contrast(treated, control, "maintain", target, SIGNAL)
            trimmed = contrast(treated, cleaned, "maintain", target, SIGNAL)
            print(
                f"  {target:<28} as-computed {full.value:+.4f}"
                if full
                else f"  {target:<28} as-computed NO CONTROL",
                end="",
            )
            print(
                f"   suspect-bins-dropped {trimmed.value:+.4f}"
                if trimmed
                else "   suspect-bins-dropped NO CONTROL"
            )

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
