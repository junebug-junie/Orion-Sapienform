# PR: baseline-matched control arm for action value

Branch: `feat/action-value-control-arm`
Base: `main` @ `f9d30ade5`
Supersedes the value definition shipped in PR #1798 (phase 1, same day).

## Summary

- Replaces the action-outcome ledger's value with a **baseline-matched
  contrast** against an untreated control arm. Phase 1's value was the
  unconditional field delta `after - before`, which measures mean reversion,
  because actions fire *because* a pressure is high.
- **The design spec's proposed control arm does not work, and this patch uses
  a different one.** Checked against live frames before building it. The
  amendment is recorded in the spec rather than a rewrite of it.
- Adds a **frozen-instrument guard** the design did not anticipate: a control
  cell that has never seen its signal move is not a calm baseline, and a
  Normal-Normal posterior with fixed observation variance structurally cannot
  detect the difference.
- Found, while checking that guard against live data, that
  `resource_pressure` -- the signal all three docker-prune templates claim --
  had been a **constant equal to a hand-typed YAML edge weight** for most of
  the day, reporting a blind camera. Documented, not fixed here.
- Still measurement-only. Nothing in this patch changes what Orion dispatches.

## Outcome moved

Replayed over the live 3-day corpus (86,938 feedback frames, 80,438 of them
untreated), the ledger's headline number does not merely shrink -- it changes
sign:

```
target                      n        raw   contrast       +/-   cover  verdict
host:docker_images       3385    -0.1350    +0.0371    0.0037    90%   OK
host:docker_containers    741    +0.0477    +0.0646    0.0074    78%   raw~0
host:docker_build_cache  1261    -0.0205    +0.0468    0.0057    53%   raw~0
```

Phase 1 would have converged on *"pruning dangling images reduces
resource_pressure by 0.14"*, with shrinking variance and a Bayesian surprise
term decaying to ~0 precisely because the wrong estimate was stable. The
matched contrast says pruning slightly **raises** it. The entire -0.1350 was
regression to the mean.

`cover` is below 100% because the frozen-cell guard is refusing the baseline
bins whose control arm was pinned during this window. That is the guard
working. An earlier run of this eval reported 100% coverage and +0.0172; it
folded its EWMA over an **unordered** result set, which is not a rate of
anything, and therefore failed to notice a bin whose instrument had been dead
for twelve hours. Fixed with an `ORDER BY created_at` before these numbers
were taken. The independent per-bin SQL reconstruction below, and the
reviewer's own arithmetic, both put the bin-8-excluded answer at +0.038.

### The same result, per bin, computed independently in SQL

`host:docker_images`, 3 days. Recomputed straight from the frames with a
`GROUP BY floor(before*10)`, i.e. without going through `contrast()` at all:

```
bin  treated_n  treated_mean   control_n  control_mean      diff
 0          11       +0.2936        3396       +0.0212   +0.2724
 1         285       +0.1624       21480       +0.0276   +0.1348
 2         416       +0.1020       12333       +0.0352   +0.0668
 3         705       +0.0123       13915       -0.0387   +0.0510
 4         536       -0.0795        5785       -0.1078   +0.0283
 5          93       -0.1906         494       -0.2057   +0.0151
 6         154       -0.3160         497       -0.3069   -0.0091
 7        1135       -0.3759        3015       -0.3820   +0.0061
 8         363       -0.1890       19695       -0.0117   -0.1773
```

Volume-weighted over all nine bins: **+0.0170**, and the estimator agreed
(+0.0172) back when it was still accepting bin 8. Excluding bin 8, which the
fixed guard now refuses: **+0.0381**, against the shipped estimator's
+0.0371. The raw treated mean recomputes to -0.1362 against the estimator's
-0.1350 on a slightly later window. The estimator is doing the arithmetic it
claims to.

Bin 7 is the whole story: 1,135 of 3,698 prunes happen when resource_pressure
is between 0.7 and 0.8, and it then falls by -0.376. On untreated ticks
starting in the same band it falls by **-0.382**. The prune is not what moved
it.

**Bin 8 is a poisoned cell, and finding it is what fixed the guard.** Its
control mean is -0.0117 across 19,695 ticks -- far calmer than bins 6 and 7
either side of it, which is not physical. That is the frozen block: 19,233 of
those 19,695 ticks contributed exactly 0.0 (only 462, or 2.3%, moved at all),
dragging the bin's control mean toward zero. Bin 8 carries ~10% of the
treated weight and contributed -0.0174 of the original result -- one
degenerate bin moving the answer by more than the entire estimate, and in the
opposite direction.

The shipped guard refuses it (`move_rate` 0.023 against a 0.25 threshold), so
the reported contrast is +0.0371 with 90% coverage rather than +0.0170 with a
fabricated 100%. `eval_action_value_contrast.py` now prints an
instrument-sensitivity band on every run, so this is a standing check rather
than a one-time hand analysis.

**What the guard still does not do:** it refuses a cell *while* the
instrument is pinned; it does not retroactively remove contamination a cell
absorbed before recovering. Recorded as a MEDIUM risk below.

That is the whole point of the patch: the confounded estimator does not fail
loudly like a zero-filled metric does. It converges on a plausible,
low-variance, confident number and looks maximally trustworthy exactly when
it is wrong.

## Current architecture (before this patch)

- `orion/feedback/outcome_resolution.py` scored only `dispatched_candidates`,
  computed `observed_delta = after - before`, and folded it into one
  Normal-Normal posterior per `(dispatch_kind, target_id, signal_id)`.
- `substrate_action_effect_posterior` held that pooled belief; the dispatch
  builder read it back as `ExpectedEffectV1.predicted_delta`.
- No comparison group existed anywhere in the path. `posterior_mean`
  estimated *"what the signal did in windows where this action ran"*, and was
  named as though it estimated *"what this action did to the signal"*.

## Architecture touched

- **New:** `orion/autonomy/contrast.py` -- `baseline_bin`, `ControlCell`,
  `contrast()`, `pooled_treated_mean()`, the arm vocabulary and its
  precedence order.
- **Changed:** the feedback resolver now scores two arms and emits untreated
  observations; the dispatch builder attaches `expected_effect` to blocked
  candidates; both runtimes' cell loaders are keyed by baseline bin.
- **New table:** `substrate_signal_control_cells`.
- **Contract note:** `substrate_action_effect_posterior.posterior_mean` keeps
  its meaning (*what this action expects to observe*). The **contrast** is a
  separate derived quantity read off the cells, deliberately NOT stamped onto
  `ExpectedEffectV1` -- see "why the contrast is not predicted_delta" below.

## The control arm the spec proposed, and why it is not the one used

The spec proposed contrasting a dispatched candidate against a
**capacity-blocked** one (a candidate approved by policy that lost the
`max_dispatch_candidates:5` race). Live frame shape, 2h sample:

```
n_dispatched  n_blocked  frames
0             0          667
5             5           34
3             7            5
```

Blocked candidates only ever exist in ticks where five *other* candidates did
go out, because the cap only binds when there is a queue. Two consequences,
both fatal:

1. The field delta is measured **frame-wide**, so a dispatched and a blocked
   candidate in the same tick read the same `before` and the same `after`. A
   within-tick contrast between them is identically zero by construction.
   (Pinned by `test_within_tick_the_two_arms_are_identical_...`.)
2. Across ticks, every capacity-blocked observation is contaminated by the
   five siblings that did run. There is no clean capacity-blocked control
   frame at all.

**The arm used instead is `no_action`:** one untreated observation per
(tick, signal), drawn from the ~94% of ticks in which nothing was dispatched.
The condition is deliberately *"nothing ran"*, not *"nothing claiming this
signal ran"* -- 5 of 16 templates declare no signal at all and are 72% of
dispatch volume, and an undeclared action still acts.

`capacity_blocked` rows are still written (a lost candidate's reading is real
evidence, and it is the direct measurement of the contamination above) but
are **not admissible as a control arm**; `contrast()` will not accept them as
one, and they do not advance any action's own posterior.

## Why the contrast is not `predicted_delta`

`prediction_error` on a ledger row is `observed_delta - predicted_delta`.
`observed_delta` is a raw field delta. The contrast is counterfactual-
adjusted. Storing the contrast as `predicted_delta` would make the residual a
units mismatch wearing a residual's name. So `predicted_delta` stays
"what this action expects to *observe* when it runs" (the volume-weighted
pooled treated mean), and the contrast -- "what this action *does*" -- is read
off the cells by the report and, in phase 3, by the budget.

## Metric quality gate (CLAUDE.md 0A), for the contrast

1. **Provenance.** `contrast()` reads `substrate_action_effect_posterior`
   (written by `_write_action_outcomes`, only for `arm == 'dispatched'`) and
   `substrate_signal_control_cells` (written by `_write_control_cells`, from
   `resolve_action_outcomes`' idle-tick loop). Both trace to
   `orion.field.pressure.field_pressures()` on real `FieldStateV1` rows.
2. **Independence.** The contrast is not a transform of anything already in
   the model -- it is the *difference* between two populations, one of which
   (untreated ticks) had no representation in the pipeline at all before this
   patch.
3. **Theory anchor.** Baseline-matched difference-in-means over a
   quasi-experimental control arm; the confound removed is named
   (selection-on-the-signal / regression to the mean), not gestured at.
4. **Live-data sanity.** Run on 86,938 real frames, printed above. The
   estimate is not degenerate, and the `docker_images` case reproduces the
   predicted sign inversion. **Failure found in this step:** see the
   instrument section below.
5. **Existing mechanism.** Searched. Nothing in `orion/` computed a
   counterfactual or a matched comparison for any action. The nearest thing,
   `action_outcomes.surprise`, is a global bus reading stamped identically
   onto every candidate in a tick.
6. **Reversibility.** Cheap. Two additive columns, one new table, one changed
   index. The phase-1 pooled posterior rows are copied to a backup table
   rather than reinterpreted.

## Live instrument failure found during step 4

`resource_pressure` -- 26.5% of declared-claim dispatch volume, the signal all
three prune templates claim -- sat at **exactly 0.85, stddev exactly 0.0,
across ~12,000 consecutive frames** on 2026-08-21:

```
day          frames   pinned at 0.85    distinct values
2026-08-19    26868       12  (0.0%)          2596
2026-08-20    38668       32  (0.1%)          2063
2026-08-21    20754    20510 (98.8%)            38
```

Traced end to end:

1. `node:substrate.vision.prediction_error` saturated at exactly **1.0** for
   12+ hours. That reading is *correct* -- it is
   `vision_channel_staleness_pressure` reporting that no vision artifact has
   arrived at all.
2. The `node:substrate.vision -> capability:vision` edge in
   `config/field/orion_field_topology.v1.yaml` maps it to the `pressure`
   channel with **weight 0.85**. 1.0 x 0.85 = 0.85.
3. `capability:vision` wins the `max()` merge into the `resource_pressure`
   dimension.

So for most of a day, Orion's *resource* pressure was a constant equal to a
hand-typed YAML edge weight, and what it was really reporting was that a
camera was off. **No staleness or freshness check could catch this**: the
value was rewritten every single tick. It was fresh, present, and constant.
It cleared on a container rebuild, which makes it recurring, not a one-off.

This is why `ControlCell.moved_n` exists. Without it, the bin-8 control cell
would have accumulated 12,000 observations of delta exactly 0.0, with
variance shrinking as 1/n, and every prune's contrast would have been its raw
delta reported with enormous confidence.

**Cross-reference, checked after the fact: PR #1800 (opened 19:11 UTC today)
independently root-caused and fixed the camera half of this.**
`orion-vision-host` was bricked from 2026-08-20 22:00 UTC by a self-defeating
VRAM config (`free 4191 - reserve 3500 = 691 < hard_floor 1400`, unsatisfiable
once the models warm up), refusing every task for ~21 hours while the
container reported healthy. That timeline matches this pin exactly, and the
restart that cleared the pin at ~19:00 UTC was that fix landing.

That closes the *input*. It does not close the defect found here, and #1800
does not mention it: **a blind camera should not have been able to move
Orion's resource pressure at all.** The staleness reading was correct the
whole time; the wiring took a correct perception alarm, multiplied it by a
hand-typed edge weight, and let it win a `max()` merge into the dimension
that decides whether to prune Docker images. Fixing the camera makes the
symptom go away and leaves the mechanism in place for the next perception
outage -- and #1800 itself reports that nothing alerted for 21 hours, so
there will be a next one.

Three follow-ups, none done here:

- **The merge is wrong, not just the input** (still open after PR #1800).
  A blind camera should not be able to raise a resource dimension. `max()` over a capability set that
  mixes perception with disk/CPU/memory collapses "my eye is off" into "I am
  out of resources", and the prune templates fire on it.
- **Saturation is invisible.** A channel pegged at its ceiling and a channel
  genuinely at its ceiling are indistinguishable downstream. `moved_n`
  catches it in the ledger; nothing catches it in the field.
- **The camera itself** was reporting maximum staleness for 12 hours.

## Files changed

- `orion/autonomy/contrast.py`: new. The estimator, the arm vocabulary, the
  frozen-cell guard.
- `orion/autonomy/evals/eval_action_value_contrast.py`: new. Replays the live
  corpus and fails if the contrast does not remove the confound, or if
  nothing was measurable.
- `orion/feedback/outcome_resolution.py`: scores both arms; emits untreated
  observations on idle ticks; counts real movement.
- `orion/execution_dispatch/builder.py`: `expected_effect` on blocked
  candidates; `build_expected_effect` pools over binned cells.
- `orion/schemas/action_prediction.py`: `arm`, `baseline_bin`,
  `frame_dispatch_count`.
- `services/orion-feedback-runtime/app/store.py`: control-cell load/write;
  treated upsert restricted to the dispatched arm.
- `services/orion-feedback-runtime/app/worker.py`: passes control priors and
  cells through; logs the untreated count.
- `services/orion-execution-dispatch-runtime/app/{store,worker}.py`: binned
  cell loader.
- `services/orion-sql-db/manual_migration_action_value_control_arm.sql`: new.
- `scripts/analysis/report_action_value.py`: reports the contrast, arm
  coverage, uncovered share, frozen cells, and `NO CONTROL`.
- `docs/superpowers/specs/2026-08-21-action-value-control-arm-design.md`:
  amended in place with what live data changed.
- `tests/test_action_value_contrast.py`: new, 25 tests.
- `tests/test_action_outcome_resolution.py`,
  `tests/test_expected_effect_declaration.py`: cell keys now carry the bin.

## Schema / bus / API changes

- **Added:** `substrate_action_outcomes.arm`, `.baseline_bin`,
  `.frame_dispatch_count`.
- **Added:** table `substrate_signal_control_cells`.
- **Added:** `substrate_action_effect_posterior.baseline_bin`, promoted into
  the primary key.
- **Changed:** the ledger's unique index moves from `(dispatch_id,
  signal_id)` to `(dispatch_id, signal_id, dispatch_frame_id)`, so the dedup
  unit matches the real observation unit -- each tick is a separate field
  window. A reprocessed feedback pass over the same dispatch frame still
  inserts nothing.

  **Scope stated honestly.** The first draft of this claimed the old key was
  live-losing rows, reasoning that `dispatch_id` is a pure function of
  `(proposal_id, policy_id)` and starvation re-blocks the same action for
  many consecutive ticks. Measured before shipping the claim: over 24h,
  21,895 distinct dispatch_ids across every blocked and dispatched candidate,
  and **zero** appear in more than one frame -- proposal_ids are regenerated
  every tick and starvation is keyed on `(kind, target)`, not on the
  proposal. This is a defensive correctness fix, not a fix for an observed
  loss.
- **Added (schema field, no migration):** `ExpectedEffectV1` now appears on
  blocked candidates. The field already existed and is optional.
- **Bus:** none. No new channels, no envelope changes.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: not applicable, nothing changed.
- local `.env` synced: not applicable, nothing changed.
- Skipped keys requiring operator action: none.

`ORION_DISPATCH_HOLDBACK_FRACTION` (step 3 of the design, the randomized arm)
is deliberately **not** in this patch. A randomized arm feeding an unproven
estimator is worse than no arm at all; it ships after the contrast is proven,
which is what the eval above now does.

## Tests run

```text
cd /mnt/scripts/Orion-Sapienform-action-value-control-arm && \
  PYTHONPATH=.:services/orion-feedback-runtime:services/orion-execution-dispatch-runtime \
  .venv/bin/python -m pytest \
    tests/test_action_outcome_resolution.py tests/test_action_prediction.py \
    tests/test_action_value_contrast.py tests/test_dispatch_starvation.py \
    tests/test_execution_dispatch_builder.py tests/test_execution_dispatch_frame_schemas.py \
    tests/test_expected_effect_declaration.py tests/test_feedback_builder.py \
    tests/test_feedback_extractors.py tests/test_feedback_runtime_store.py \
    tests/test_feedback_scoring.py tests/test_execution_dispatch_runtime_store.py -q

205 passed in 2.75s
```

35 of those are new (`tests/test_action_value_contrast.py`). Every arithmetic
fixture in the contrast tests is hand-computed in a comment next to its
assertion; the review recomputed all of them independently and they hold.

**Not smoke-tested:** `scripts/analysis/report_action_value.py` cannot run
against the live database until the migration is applied -- its query selects
`arm` and `frame_dispatch_count`, which do not exist yet. Verified to parse
and import; its output shape is unverified. Run it as step 4 of the deploy.

Deterministic gates:

```text
git diff --check                                        -> clean
scripts/check_metric_dead_wiring.py                     -> clean
scripts/check_service_env_compose_parity.py orion-feedback-runtime
    -> OK, all 15 .env_example keys exposed
scripts/check_service_env_compose_parity.py orion-execution-dispatch-runtime
    -> 2 of 25 keys missing from compose (EXECUTION_DISPATCH_STALENESS_MIN_SEC,
       EXECUTION_DISPATCH_STALENESS_MAX_SEC). PRE-EXISTING: this patch adds no
       env keys and touches no .env_example.
```

## Evals run

```text
cd /mnt/scripts/Orion-Sapienform-action-value-control-arm && \
  PYTHONPATH=. .venv/bin/python orion/autonomy/evals/eval_action_value_contrast.py --days 3

replayed 87642 feedback frames over 3 days
  idle (untreated) ticks: 81018
  FROZEN control bins refused as coverage: [8]

target                            n        raw   contrast       +/-   cover verdict
host:docker_images             3385    -0.1350     0.0371    0.0037    90% OK
host:docker_containers          741     0.0477     0.0646    0.0074    78% raw~0
host:docker_build_cache        1261    -0.0205     0.0468    0.0057    53% raw~0

positive control on host:docker_images: injected +0.2500, recovered +0.2500

instrument-sensitivity band (bins whose control arm is mostly frozen):
  resource_pressure bin 8: 462/19695 moved (2.3%), control mean -0.0117
  host:docker_images       as-computed +0.0371   suspect-bins-dropped +0.0371
  host:docker_containers   as-computed +0.0646   suspect-bins-dropped +0.0646
  host:docker_build_cache  as-computed +0.0468   suspect-bins-dropped +0.0468

PASS: 1 of 3 measured target(s) had a materially large raw delta, and every
one of them shrank to within 50% of it.
```

Three gates, not one. The shrink test is the headline; the **positive
control** (inject a known +0.25, require it back) is what separates "the
confound was removed" from "the estimator is dead", since a hardwired 0.0
would pass every shrink test with the maximum possible margin; and the
**sensitivity band** shows what the answer depends on. The band reads
identical on both sides here because the frozen guard has already refused the
suspect bin -- that is the guard and the band agreeing, not a no-op.

This is acceptance check 3 of the design spec, and the only one that matters.
The first version of this eval gated **every** target on
`|contrast| <= 0.5 * |raw|` and failed two of them -- a ratio-of-two-small-
numbers test that fails on noise and says nothing about the confound. Fixed
to gate only targets whose raw delta is materially large, and to fail
explicitly when no target qualifies, so an empty replay cannot report green.

## Docker/build/smoke checks

Not run. This patch is not deployed -- see "Restart required". The migration
below has **not** been applied to the live database, on purpose: it drops an
index the currently-running phase-1 code names in an `ON CONFLICT` clause, so
applying it early would make every ledger write fail (gracefully, inside the
existing savepoint, but pointlessly) until the new code is up.

## Review findings fixed

Adversarial review (subagent, CLAUDE.md section 12) returned 21 findings, 3
of them blocking. It also independently replayed the same 3-day corpus and
reproduced +0.01697 against the then-reported +0.0172, so its numbers are
computed against the real thing rather than a model of it.

**Finding 1 (HIGH) -- `is_frozen` was a cold-start-only guard.**
  - Fix: `moved_n == 0` on a monotone LIFETIME counter can only ever catch a
    channel that was *born* dead; once a cell has seen one movement it can
    never be frozen again. The scenario the guard exists for -- a healthy
    channel that freezes later, i.e. exactly what happened on 2026-08-21 --
    was structurally undetectable. It passed the live replay only because the
    replay built its cells from scratch inside the pinned window.
    `is_frozen` now reads `move_rate`, an EWMA (alpha 1/1000, ~50 min at the
    live untreated rate) against 0.25, a threshold read off the live per-bin
    movement fractions (healthy 0.73-0.92, pinned 0.024).
  - Evidence: `test_a_cell_that_was_healthy_and_then_froze_is_caught` folds
    2,000 real pinned observations into a cell with `moved_n = 40,000` and
    asserts it is caught; `test_the_freeze_is_detected_inside_a_bounded_
    number_of_ticks` pins the horizon at exactly 1,386 observations
    (`ln(0.25)/ln(1-alpha)`, hand-derived then confirmed by the loop);
    `test_a_recovering_channel_stops_being_frozen` proves it does not latch.
    Live: the guard now refuses bin 8 and the headline moves from +0.0170 to
    +0.0371, matching the reviewer's independent +0.0381.

**Finding 1b (HIGH, found while fixing 1) -- the eval fed its EWMA an
unordered result set.**
  - Fix: `ORDER BY f.created_at` in the replay query. A windowed statistic
    computed over an arbitrary permutation is not a rate of anything, and it
    is why the first run reported a fabricated 100% coverage.
  - Evidence: same eval, 90%/78%/53% coverage after the fix, with bin 8
    named in the `FROZEN control bins refused as coverage` line.

**Finding 2 (HIGH) -- control cells had no idempotency guard, and the
docstring asserted one that did not exist.**
  - Fix: the monotone `posterior_n <` comparison stops the belief moving
    backwards and does nothing against double-counting (a replayed tick reads
    n=N+k, recomputes n=N+2k, lands again). Added
    `last_dispatch_frame_id` to the cell and
    `AND ... IS DISTINCT FROM EXCLUDED.last_dispatch_frame_id` to the upsert.
    Docstring rewritten to say what each guard actually does.
  - Evidence: not triggerable today (nothing prunes
    `substrate_feedback_frames`, so `reconcile_feedback_pending` finds no
    aged frame to re-queue) -- but that reconciler exists precisely to replay,
    and this repo has ~8.3 GB of unbounded substrate tables being worked
    through. A replay would have corrupted ONE arm of the contrast and not
    the other.

**Finding 3 (HIGH) -- the migration broke running phase-1 code two ways.**
  - Fix: `baseline_bin` now carries `DEFAULT 0`. Without it, `ADD COLUMN`
    (nullable) + `SET NOT NULL` made every phase-1 INSERT fail a not-null
    violation, swallowed by the savepoint into a silent ledger stall with a
    healthy-looking pipeline. The backfill is now unconditional so the
    existing rows still get exact bins.
  - Evidence: the index drop remains an unavoidable brief gap (phase-1's
    `ON CONFLICT (dispatch_id, signal_id)` cannot survive it), which is why
    the deploy order below says to apply the migration immediately before the
    deploy rather than ahead of time.

**Finding 4 (MEDIUM) -- `frame_dispatch_count` was write-only with a
self-contradictory rationale.**
  - Fix: the comment claimed it let an analysis filter contaminated CONTROL
    observations -- impossible, since control observations are emitted only
    when the count is 0 and get no ledger row at all. Rewritten to say what
    it is (treated-arm contamination) and wired into
    `report_action_value.py` as the `alone%` column
    (`frame_dispatch_count == 1` = sole actor).

**Finding 5 (MEDIUM) -- the report printed windowed and lifetime quantities
in the same row.**
  - Fix: columns renamed `n(life)` and `raw(win)`, the docstring states that
    `--days` does not touch the contrast, and a closing line repeats it.

**Finding 6 (MEDIUM) -- the eval could not fail for an estimator hardwired to
return 0.0.**
  - Fix: added a positive control. Every treated cell is shifted by a known
    +0.25 and the estimator must return the shift.
  - Evidence: `positive control on host:docker_images: injected +0.2500,
    recovered +0.2500`.

**Finding 7 (MEDIUM) -- the eval replays the zero-filled pressure snapshot,
not the production read path.** Disclosed in the eval docstring alongside the
existing limitation blocks, with the concrete follow-up (re-run against
`substrate_action_outcomes.baseline/observed_delta` once phase 2 has run a
day). Not silently fixed, because it cannot be until there are rows.

**Finding 8 (MEDIUM) -- `min_control_n = 1`.**
  - Fix: `MIN_CONTROL_CELL_N = 30`, justified from live bin populations
    (3 to 21,483). Bins refused for thin coverage are now reported as
    `thin_bins`, separately from `frozen_bins` and from "no control at all" --
    three different facts that were collapsing into one.

**Finding 10/11/13 (MEDIUM) -- specced, not silently fixed.** The
`prediction_error`/`surprise_nats` inconsistency is now documented on the
schema field itself with the reason the obvious fix is deferred (bin-matching
`predicted_delta` needs a field read on the dispatch tick path, where the
existing daily-risk-cap read is already 49.8% of this database's buffer
traffic). The model-interval and autocorrelation caveats are printed by both
the eval and the report. Within-bin baseline imbalance (reviewer's estimate:
~±0.002 on a +0.037 estimate) is recorded as a risk.

**Finding 12 (MEDIUM) -- one savepoint covered both arms.**
  - Fix: two adjacent savepoints, so a constraint violation on one scored
    action no longer discards that tick's untreated observations.

**Findings 14-21 (LOW).** Dead `if False` branch removed from a test; the
frozen-cell report line now prints bins and rates instead of bare signal
names; unused `mean_abs_err` dropped from the query; CHECK constraints added
for both `arm` vocabularies, both bin ranges and `moved_n <= posterior_n`;
`claim_upheld`'s docstring corrected (it described a comparison the
`no_action` arm structurally cannot support); `ControlObservation` trimmed
from seven write-only fields to four and given a real consumer
(`summarize_control_observations`, logged per tick as
`signal@bin:moved/total`, so an operator can tell four healthy channels from
one pinned one); the duplicate `1e-6` unified into
`orion.feedback.extractors.PRESSURE_DELTA_EPSILON`; tests added for
`pooled_treated_mean`'s variance and for multi-bin `build_expected_effect`,
which was the actual behaviour change in the builder and had no coverage.

**Not changed, on purpose:**
- **Finding 9 (unlocked read-modify-write on control cells).** Real. Two
  concurrent feedback runtimes would interleave and undercount rather than
  inflate. One runs today. Doing the fold in SQL is the right fix and is a
  separate patch; recorded as a risk rather than half-done.
- **`pooled_treated_mean` double-counts the prior across cells.** The
  reviewer confirmed the effect and worked the magnitude: with K bins the
  extra precision is `0.16 * (K-1)` observation-equivalents, so for the live
  cell shape the pooled sd is 0.008940 against a correct 0.008943. Immaterial
  above n of order 10. Not worth a fix; worth the comment it now has.

**Explicitly confirmed correct by the review, recorded so it is not
re-litigated:** migration idempotency and the first-run `DO $$` guard; every
hand-computed test fixture (all recomputed independently); `baseline_bin`
float boundaries at all eleven decile edges; `contrast()`'s weighted
difference and variance propagation; the refusal to pool `randomized_holdback`
with `no_action`; and `_present_pressures` genuinely not zero-filling.

## Restart required

**Only two services need rebuilding.** Phase 1's deploy order was
consumers-before-producers across five services; that constraint was
re-derived for this patch rather than inherited, and it does not apply here.
Verified two ways:

```text
$ grep -rl "orion.autonomy.contrast"          services/ -> feedback-runtime, execution-dispatch-runtime
$ grep -rl "orion.feedback.outcome_resolution" services/ -> feedback-runtime
$ grep -rl "orion.execution_dispatch.builder"  services/ -> execution-dispatch-runtime
$ grep -rl "orion.schemas.action_prediction"   services/ -> feedback-runtime
```

and the only new *frame* content is `expected_effect` on blocked candidates,
a field that already exists and is already optional in the deployed schema:

```text
$ docker exec orion-athena-feedback-runtime python -c \
    "from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1 as C; \
     print('expected_effect' in C.model_fields)"
True
```

So there is no silent-discard hazard in this patch. Hub, policy-runtime and
proposal-runtime do not need to be touched.

```bash
# 1. Migration FIRST, immediately before the feedback runtime goes up.
#    It drops the (dispatch_id, signal_id) unique index that the CURRENTLY
#    RUNNING phase-1 code names in its ON CONFLICT clause, so between the
#    migration and the deploy every ledger write fails -- gracefully, inside
#    the existing savepoint, but pointlessly. Keep the gap short.
docker exec -i <postgres-container> psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_action_value_control_arm.sql

# 2. Feedback runtime (writes both arms), then dispatch runtime (reads the
#    binned cells). Either order works -- they share no frame contract that
#    changed -- but feedback first means the cells exist before anything
#    tries to read them.
./scripts/safe_docker_build.sh orion-feedback-runtime up -d --build
./scripts/safe_docker_build.sh orion-execution-dispatch-runtime up -d --build

# 3. Verify the arm is actually being recorded.
python3 scripts/check_sql_migrations_applied.py \
  --file manual_migration_action_value_control_arm.sql
psql -h localhost -p 55432 -U postgres -d conjourney -c \
  "SELECT arm, count(*) FROM substrate_action_outcomes GROUP BY 1;"
psql -h localhost -p 55432 -U postgres -d conjourney -c \
  "SELECT signal_id, arm, baseline_bin, posterior_n, moved_n
     FROM substrate_signal_control_cells ORDER BY posterior_n DESC LIMIT 20;"
```

The last query is the one that matters. If `moved_n` is 0 on a cell with a
large `posterior_n`, the instrument is frozen and the contrast is correctly
refusing to use it -- that is the guard working, not a bug.

The migration is guarded against a destructive re-run: every other statement
in it is idempotent, including the primary-key swap (Postgres re-auto-names
the constraint, so a second run drops and re-adds it happily), which would
have made a bare `DELETE FROM substrate_action_effect_posterior` look
harmless while silently wiping every accumulated posterior. The backup-and-
delete block is wrapped in a `DO $$ ... $$` guarded on the absence of the
`baseline_bin` column, i.e. on genuinely being the first run.

## Risks / concerns

- **Severity: HIGH.** `resource_pressure` is structurally able to be pinned
  by an unrelated blind camera (documented above). The contrast degrades
  honestly when that happens -- one bin, no variance, `moved_n == 0`, refused
  as coverage -- but the *actions* keep firing on it. The field-merge fix is
  a separate patch and should not wait long.
  **Mitigation:** `moved_n` guard; the failure is now visible in
  `report_action_value.py` rather than silent.

- **Severity: MEDIUM.** `ControlCell.is_frozen` refuses a cell *while* its
  instrument is pinned; it does not retroactively remove contamination the
  cell absorbed before recovering. Bin 8 is currently refused, so it does not
  reach the number -- but if that instrument recovers and the cell's rate
  climbs back above 0.25, its posterior mean (-0.0117, diluted by 19,233
  fabricated zeros out of 19,695) becomes coverage again.
  **Mitigation:** `eval_action_value_contrast.py` prints an
  instrument-sensitivity band on every run, so the dependence is a standing
  output rather than a one-time hand analysis. A real fix means decaying the
  posterior itself, not just the movement rate -- a separate design question.

- **Severity: MEDIUM.** Control cells are advanced by an unlocked
  read-modify-write: `load_control_posteriors()` reads on its own connection
  at tick start, the write happens later with no row lock. Two concurrent
  feedback runtimes would interleave and the belief would advance by fewer
  than the true number of observations.
  **Mitigation:** one instance runs today, and the failure direction is
  undercounting rather than inflation. Doing the fold atomically in SQL is
  the right fix and is a separate patch.

- **Severity: MEDIUM.** The `no_action` arm is **quasi-experimental**, not
  causal. Ticks where nothing ran are systematically calmer ticks. Binning on
  the baseline absorbs most of that selection and provably not all of it.
  Every artifact says so; nothing may describe this number as causal.
  **Mitigation:** step 3's randomized holdback is the fix and is specced.

- **Severity: MEDIUM.** The reported `+/-` derives from a *fixed* observation
  variance (`DEFAULT_OBSERVATION_VARIANCE = 0.04`, fitted to 68,715 real
  pressure deltas), not from each bin's empirical spread. It is a model
  interval, not a measured standard error, and it is wrong wherever a bin's
  real spread departs from that constant.
  **Mitigation:** stated in the eval output and here. A per-cell running
  variance is a small, separate patch.

- **Severity: LOW.** Control cells have no per-observation ledger, so their
  only anti-double-count guard is the monotone `posterior_n` comparison in
  the upsert. Two concurrently running feedback runtimes would interleave
  rather than duplicate, but the belief would advance by fewer than the true
  number of observations.
  **Mitigation:** one feedback runtime runs today; the guard is
  fail-toward-undercounting, not toward inflation.

- **Severity: MEDIUM.** Ledger growth roughly triples. Phase 1 wrote only
  dispatched rows (~5,400/day); `capacity_blocked` adds ~6,000/day (1,536
  capacity blocks measured in a 6h window), and dispatch_ids never repeat
  across ticks, so none of it collapses. `substrate_action_outcomes` has an
  `observed_at DESC` index from day one but **no retention policy**, and this
  repo already has ~8.3 GB of unbounded substrate tables.
  **Mitigation:** none in this patch. A retention window belongs with the
  other substrate retention work, not bolted on here.

- **Severity: LOW.** `capacity_blocked` rows are written and are not used as
  a control arm. That is a deliberate record of the contamination, not a
  write-only field -- `frame_dispatch_count` is what a later analysis needs
  to quantify it -- but neither has a consumer inside this patch.

## Note for concurrent agents

Merging this turns `scripts/check_sql_migrations_applied.py` red repo-wide
until an operator hand-applies the migration -- it currently reports
`1 migration file(s) are NOT fully applied` with 8 named missing objects.
That is the gate working correctly, but if it runs inside anyone's
`agent-check`, every concurrent branch goes red for reasons unrelated to
their work. Apply the migration promptly after merge, or expect the noise.

## PR link

<filled in on push>
