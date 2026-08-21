# Action value needs a control arm

**Date:** 2026-08-21
**Status:** IMPLEMENTED, with one central amendment -- see "Amendment" below
**Supersedes the value definition in:** PR "action-outcome ledger" (phase 1)
**Blocks:** any decision budget that reads `substrate_action_outcomes.posterior_mean`

## Amendment (2026-08-21, during implementation)

**The control arm proposed below does not work, and the implementation uses a
different one.** Kept visible rather than rewritten, because the reasoning
that produced the wrong arm is the same reasoning a later patch will reach
for.

The proposal was to contrast a dispatched candidate against a
capacity-blocked one. Checked against live frames before building it:

```
n_dispatched  n_blocked  frames (2h live sample)
0             0          667
5             5           34
3             7            5
```

Blocked candidates only ever exist in ticks where five other candidates DID
go out, because the cap only binds when there is a queue. Two consequences,
both fatal:

1. The field delta is measured frame-wide, so a dispatched and a
   capacity-blocked candidate in the same tick read the **same** before and
   the **same** after. A within-tick contrast between them is identically
   zero by construction.
2. Across ticks, every capacity-blocked observation is contaminated by the
   five siblings that did run. There is no clean capacity-blocked control
   frame at all.

**The arm actually used is `no_action`:** one untreated observation per
(tick, signal) drawn from the ~94% of ticks in which NOTHING was dispatched.
Live over 3 days that is 80,414 untreated ticks against 86,910 total. The
condition is "nothing ran", not "nothing claiming this signal ran", because
5 of 16 templates declare no signal at all and are 72% of dispatch volume --
an undeclared action still acts.

`capacity_blocked` rows are still written to the ledger (a lost candidate's
reading is real evidence) and are **not admissible as a control arm**;
`contrast()` will not accept them as one.

**Result, replayed on the live 3-day corpus** (acceptance check 3):

```
target                      n        raw   contrast       +/-   cover
host:docker_images       3674    -0.1369    +0.0172    0.0036   100%
host:docker_containers    933    +0.0477    +0.0530    0.0066   100%
host:docker_build_cache  2349    -0.0207    +0.0268    0.0042   100%
```

The headline number does not merely shrink, it **changes sign**: phase 1
would have reported "pruning dangling images reduces resource_pressure by
0.14"; the matched contrast says it very slightly raises it. The whole
-0.1369 was mean reversion.

**Second amendment -- a guard the design did not anticipate.** A control cell
that has never seen its signal move is not a calm baseline, it is a frozen
instrument, and contrasting against it hands the treated arm's entire raw
delta back wearing the contrast's name and confidence. This is not
hypothetical: see "Live instrument failure found during implementation"
at the end of this document. `ControlCell.moved_n` counts observations that
left the dead band, and `contrast()` refuses a cell with `moved_n == 0` and
`n >= 200` as coverage. A Normal-Normal posterior with a fixed observation
variance cannot detect this on its own -- its variance shrinks as 1/n whether
the data varies or is one constant repeated, so a frozen channel produces the
most confident-looking cell in the table.

**Binning also changed:** fixed-width deciles of [0,1], not trailing-window
quantiles. Quantile edges make bin identity mean something different at
different times, so pooling records across time would silently mix
conditions -- the same class of defect as the confound itself.

---

## Arsonist summary

Phase 1 built the right plumbing around the wrong number.

An action's recorded value is currently the unconditional field delta across
its window: `after - before`. That is not what the action did. It is what
happened, and what happens is dominated by the fact that actions fire when a
pressure is high and high pressures fall on their own.

Measured live over 3 days, `prune_dangling_images`:

```
pruned?    n        mean resource_pressure delta
no      66036       -0.0257
yes      3426       -0.1481     <- looks like a 5.8x effect
```

Condition on where the pressure started, and it inverts. In 6 of 8
comparable baseline deciles the prune arm falls *less* than the no-prune arm:

```
baseline_decile  pruned    n      mean_delta
2                no     21515      +0.0284
2                yes      277      +0.1647    <- prune arm went UP more
3                no     12441      +0.0370
3                yes      395      +0.0928    <- UP more
4                no     14412      -0.0377
4                yes      688      +0.0114    <- UP while control went down
5                no      6054      -0.1056
5                yes      536      -0.0795    <- down LESS
6                no       499      -0.2014
6                yes       93      -0.1906    <- down LESS
7                no       516      -0.3182
7                yes      154      -0.3160    <- identical
8                no      3177      -0.3808
8                yes     1128      -0.3750    <- identical
```

The entire raw gap is regression to the mean.

Left as-is, the ledger converges on a confident `posterior_mean` near -0.15
for each prune, `surprise_nats` decays to ~0 because the estimate is stable,
and `report_action_value.py` prints a low-uncertainty row that reads as
*"confirmed: docker prune decreases resource_pressure by 0.15."* It is the
same failure class as the zero-fill phase 1 does guard against, except it
arrives as a plausible non-zero number, which is strictly harder to catch.

`resource_pressure` carries 26.5% of all declared-claim dispatch volume and
the three prune templates are ~21% of all dispatch. This is the ledger's main
population, not an edge case.

## Current architecture

- **Producer:** `orion/execution_dispatch/builder.py::build_expected_effect`
  stamps `ExpectedEffectV1` onto each dispatched candidate. `predicted_delta`
  comes from the stored posterior; the template author declares only
  `(signal, direction)`.
- **Scorer:** `orion/feedback/outcome_resolution.py::resolve_action_outcomes`
  reads `field_pressures()` before and after, computes
  `observed_delta = after - before`, and updates a Normal-Normal posterior
  (`orion/autonomy/prediction.py`).
- **Store:** `substrate_action_outcomes` (append-only ledger) and
  `substrate_action_effect_posterior` (one row per
  `(dispatch_kind, target_id, signal_id)`), written in the same transaction
  as the feedback frame.
- **Consumers:** the dispatch builder reads the posterior back into the next
  prediction; `scripts/analysis/report_action_value.py` reads the ledger.
  Nothing changes what gets dispatched. Phase 1 is measurement-only.

The gap: no arm to compare against, so `posterior_mean` estimates
*"what the signal did in windows where this action ran"*, not *"what this
action did to the signal"*.

## Missing questions, answered before designing

**Is there already a control arm available?** Yes, and it is large. The
dispatch policy admits at most 5 candidates per tick; the rest are recorded
on the same frame in `blocked_candidates` with
`blocked_by = ["max_dispatch_candidates:5"]`. Live: 58,285 such blocks in
7 days, versus 38,138 real dispatches. These candidates were built, scored,
approved by policy, and then lost a capacity race. They are the same actions,
in the same field conditions, in the same tick — minus the execution.

**Is losing that race random?** No, and this must not be overclaimed. The
admit order is effective priority (real priority plus starvation aging), so
the blocked arm is systematically slightly lower-priority. Priority correlates
with the motivating pressure, which is the confounder. Matching on baseline
absorbs most of that; it does not absorb all of it. This is a natural
experiment, not a randomized one, and every artifact it produces must say so.

**Can we get a clean arm?** Yes, cheaply: deliberately withhold a small
random fraction of candidates that *would* have been dispatched. That is a
real randomized holdback and it is the only thing that licenses a causal
claim. It costs a small, bounded amount of capability.

**Does the existing `co_predictors` field address this?** No. It counts other
declared claimants on the same signal in the same frame. Different problem.

## Proposed design

### 1. Two arms, both recorded

Add to `ActionOutcomeRecordV1` and `substrate_action_outcomes`:

```
arm  TEXT NOT NULL   -- 'dispatched' | 'capacity_blocked' | 'randomized_holdback'
```

`resolve_action_outcomes` iterates `dispatch_frame.blocked_candidates` as
well as `dispatched_candidates`, scoring each identically. A blocked
candidate carries the same `expected_effect`, so this requires the dispatch
builder to attach `expected_effect` on the blocked path too (today only the
dispatched path gets it — `orion/execution_dispatch/builder.py`'s
`make_blocked` does not call `build_expected_effect`).

Only `blocked_by == ["max_dispatch_candidates:5"]` becomes a control record.
`requires_operator_review` and `deferred` blocks are excluded: those are
blocked for reasons correlated with the action's own content, which is a
different and much worse confounder than losing a capacity race.

### 2. Value is a matched contrast, not a delta

Bin by baseline. Deciles of the signal's own `before` value, computed over a
trailing window and stored on the record as `baseline_bin` so a reader never
has to re-derive the binning and get a different answer than the writer did.

An action's effect on a signal is then:

```
effect(action, signal) = sum over bins b of  w_b * ( mean_delta[dispatched, b]
                                                   - mean_delta[control, b] )
```

with `w_b` the dispatched arm's share of volume in bin `b`, so the reported
number answers "what did this action do, over the conditions it actually
runs in" rather than over a uniform prior on conditions.

The posterior in `substrate_action_effect_posterior` moves from a belief
about the raw delta to a belief about **this contrast**. That single change
is what makes the number a denominator worth dividing by. It also lets a
value be zero or negative, which nothing in the current pipeline can express
— and an action that cannot lose is not competing.

Bins with an empty control arm contribute nothing and are reported as
uncovered coverage, never silently dropped to the raw delta.

### 3. Randomized holdback (the upgrade, gated off by default)

`ORION_DISPATCH_HOLDBACK_FRACTION` (default `0.0`). When set, that fraction
of candidates that passed every gate and won a slot are withheld and recorded
as `arm='randomized_holdback'`. At 0.05 that is ~270 withheld actions/day
against ~5,400 dispatched — enough to detect an effect the size of the raw
gap within days, and small enough that it is not a capability regression.

This is the only arm that licenses the word "causal". The report must label
contrasts computed from `capacity_blocked` as **quasi-experimental** and
contrasts computed from `randomized_holdback` as **experimental**, and never
merge the two into one number.

### 4. Reporting refuses the unsafe claim

`scripts/analysis/report_action_value.py` stops printing a single
`posterior_mean` column. It prints the contrast, the arm counts per bin, the
uncovered-bin share, and the arm label. A row with no control coverage prints
`NO CONTROL` rather than a number, because a number there is what would be
believed.

## Schema / bus / API changes

- **Added:** `substrate_action_outcomes.arm TEXT NOT NULL DEFAULT 'dispatched'`,
  `substrate_action_outcomes.baseline_bin SMALLINT`.
- **Changed meaning:** `substrate_action_effect_posterior.posterior_mean`
  becomes a belief about the dispatched-minus-control contrast, not the raw
  delta. This is a semantic change to an existing column with an existing
  reader (`build_expected_effect`). Because phase 1 has not been deployed,
  the safe migration is to truncate both tables at cutover rather than
  reinterpret rows written under the old meaning. If phase 1 HAS been
  deployed by then, the rows must be dropped, not reinterpreted — a
  silently-redefined column is exactly the class of defect this whole arc
  exists to stop.
- **Added:** `ExecutionDispatchCandidateV1.expected_effect` populated on
  blocked candidates (no schema change; the field already exists and is
  optional).
- **Bus:** none. No new channels, no envelope changes.

## Env/config changes

- `ORION_DISPATCH_HOLDBACK_FRACTION` (default `0.0`, off). Requires
  `services/orion-execution-dispatch-runtime/.env_example` plus a local
  `.env` sync via `python scripts/sync_local_env_from_example.py`.

## Files likely to touch

- `orion/schemas/action_prediction.py` — `arm`, `baseline_bin`
- `orion/feedback/outcome_resolution.py` — score both arms
- `orion/execution_dispatch/builder.py` — `expected_effect` on blocked
  candidates; holdback selection
- `services/orion-execution-dispatch-runtime/app/settings.py` + `.env_example`
- `services/orion-feedback-runtime/app/store.py` — persist the new columns
- `services/orion-sql-db/manual_migration_action_value_control_arm.sql`
- `scripts/analysis/report_action_value.py` — contrast reporting
- `orion/autonomy/evals/` — an eval that reproduces the decile table above
  from live data and fails if the raw and matched estimates diverge beyond a
  threshold without the report saying so

## Non-goals

- Changing what gets dispatched, or how the daily risk cap is derived. Still
  measurement-only. The budget rewrite reads this number; it is not part of
  this design.
- Widening the action vocabulary. 5 of 16 templates declare no claim at all
  and account for **72%** of live dispatch volume over 24h (corrects the
  "~62%" in the phase-1 commit message, which was computed over a 7-day
  dispatch mix rather than 24h volume). Giving those templates claims is
  worth doing and is a separate patch.
- Fixing `reliability_pressure`, which is decayed to ~1e-190 in 91% of
  frames. It carries 16 of 17,983 dispatches/day, so it is not urgent — but
  it must not be declared on a high-volume template before it is fixed. See
  "Known bad instruments" below.
- Any causal claim from the `capacity_blocked` arm alone.

## Acceptance checks

1. A dispatched and a capacity-blocked candidate in the same frame, same
   signal, produce two ledger rows with different `arm` and identical
   `baseline`/`observed_after`.
2. A `requires_operator_review` block produces **no** control record.
3. Replaying the live 3-day prune data through the contrast produces a value
   near zero, not near -0.15 — i.e. the design reproduces the decile
   inversion above rather than the raw gap. This is the acceptance check that
   actually matters; the others are plumbing.
4. A signal/action pair with zero control coverage reports `NO CONTROL`, and
   no posterior row is written for it.
5. `ORION_DISPATCH_HOLDBACK_FRACTION=0.0` (the default) changes dispatch
   behaviour not at all — byte-identical frames against a recorded fixture.
6. With holdback enabled, withheld candidates are recorded and are NOT
   counted as dispatched in `sum_risk_dispatched_today`.

## Known bad instruments this design must not launder

Recorded here so a later patch does not rediscover them as new findings.

- **`reliability_pressure`** — 91.1% of 50,680 real frames sit below 1e-12,
  live values around 3.7e-190. A geometric decay artifact, present-not-absent,
  so `_present_pressures`' absence guard structurally cannot fire on it. The
  phase-1 metric gate measured a pooled sd of 0.29 across four channels and
  missed it; the per-channel breakdown is the check that catches this class.
- **`deviation_pressure`** — injected unconditionally at
  `orion/field/pressure.py:503` from a field that defaults to `0.0`. If the
  tension subsystem stops publishing, it reads present-and-0.0 forever and
  perfectly confirms any `no_change` claim. Healthy today (3,812 distinct
  values over 12h, 65.5% at genuine rest), but unprotected, and nothing would
  announce the transition.
- **`resource_pressure`** — genuinely healthy over 36h (2,768 distinct
  values, delta sd 0.226), but was frozen at exactly 0.85 with sd 0.0 for
  3,281 consecutive frames spanning 2026-08-21 04:00-06:00 UTC, which is the
  window the phase-1 live checks ran in. Confirm it is unfrozen before
  trusting any phase-1 output.
- **`action_outcomes.surprise`** (the pre-existing column, 133,058 rows) —
  `latest_bus_synaptic_prediction_error()`, a global bus reading stamped
  identically onto every candidate in a tick. Once this design's contrast
  exists, that column should be replaced by it and the bus-metric write
  retired outright, per CLAUDE.md's "kill means kill" rule. Not partially
  excluded from one consumer — retired.

## Recommended next patch

Steps 1 and 2 only (`arm`, `baseline_bin`, contrast reporting), with
acceptance check 3 as the gate. Holdback (step 3) ships after the contrast is
proven to reproduce the decile inversion on real data, because a randomized
arm that feeds a broken estimator is worse than no arm at all.


---

## Live instrument failure found during implementation (2026-08-21)

Recorded here because it poisons this design's main signal and is a separate
patch, not because it was fixed.

`resource_pressure` -- 26.5% of declared-claim dispatch volume, the signal all
three docker-prune templates claim -- sat at **exactly 0.85, stddev exactly
0.0, across ~12,000 consecutive frames** on 2026-08-21 (03:00 to ~18:00 UTC),
against 2,600 distinct values/day on every preceding day:

```
day          frames   pinned at 0.85    distinct values
2026-08-19    26868       12  (0.0%)          2596
2026-08-20    38668       32  (0.1%)          2063
2026-08-21    20754    20510 (98.8%)            38
```

Mechanism, traced end to end:

1. `node:substrate.vision.prediction_error` saturated at exactly **1.0** for
   12+ consecutive hours. That reading is correct -- it is
   `vision_channel_staleness_pressure` (orion/substrate/prediction_error.py)
   reporting that no vision artifact has arrived at all. The eye is blind.
2. The `node:substrate.vision -> capability:vision` edge in
   `config/field/orion_field_topology.v1.yaml` maps `prediction_error` to the
   `pressure` channel with **weight 0.85**. 1.0 x 0.85 = 0.85.
3. `capability:vision`'s pressure wins the `max()` merge into the
   `resource_pressure` dimension (`orion/field/pressure.py`,
   `CHANNEL_DIMENSION_MAP["pressure"] = "resource_pressure"`).

So for most of a day, Orion's *resource* pressure -- the thing that drives
~21% of everything it does, all of it docker pruning -- was a constant equal
to a hand-typed YAML edge weight, and what it was actually reporting was that
a camera was off. **No staleness or freshness check could catch this**: the
value was rewritten every single tick. It was fresh, present, and constant.

The freeze cleared when the runtimes were rebuilt at ~19:00 UTC, which makes
it a recurring state, not a one-off.

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

Three separate follow-ups, none of them done here:

- **The merge is wrong**, not just the input. A blind camera should not be
  able to raise a resource dimension. `max()` over a capability set that
  mixes perception with disk/CPU/memory collapses "my eye is off" into "I am
  out of resources", and the prune templates then fire on it.
- **Saturation needs to be visible.** A channel pegged at its ceiling and a
  channel genuinely at its ceiling are indistinguishable downstream. The
  `moved_n` guard above catches it in the ledger; nothing catches it in the
  field.
- **The camera.** `node:substrate.vision` reporting maximum staleness for 12
  hours is itself a real, unhandled outage.
