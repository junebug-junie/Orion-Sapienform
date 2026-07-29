# Precision-weighted proposal scoring — design spec

Status: **`dimension_confidence()` half IMPLEMENTED 2026-07-28** (PR #1442). The
`proposal_priority()`/weight-calibration half's **measurement pass ran 2026-07-29 and returned a
real, decisive STOP** -- see "2026-07-29 calibration-measurement update" below. `proposal_risk()`
stays untouched (Non-goals, unchanged).

## 2026-07-29 calibration-measurement update

Followed Missing Questions 3/4 exactly, "measure before minting": built
`scripts/analysis/measure_proposal_feedback_correlation.py`, a read-only probe over real
`substrate_feedback_frames` (175,688 rows, 2026-07-23T23:38 -> 2026-07-29T00:56, 121.3h real
span), joined against `substrate_proposal_frames`/`substrate_policy_decision_frames`/
`substrate_execution_dispatch_frames` to confirm chain completeness first (**100% -- every
`source_proposal_frame_id`/`source_policy_frame_id`/`source_execution_dispatch_frame_id`
reference resolves to a real row**, not assumed from the schema).

**Missing Question 3 (per-template vs per-dimension) resolved empirically, not by default lean**:
per-candidate `FeedbackFrameV1.observations[].score` is a *fixed constant per `outcome_kind`* by
construction (`orion/feedback/builder.py::_score_for_outcome_kind`, backed by
`FeedbackScoringV1`'s 8 hand-set constants) -- checked against all 175,688 real rows, not assumed:
every `(template, source_kind, outcome_kind)` bucket's real score stddev is float noise (~1e-13 to
~1e-17). The only source_kind carrying a real, non-deterministic outcome is `cortex_result`'s
`completed`/`failed` split -- the one signal reflecting an actually-dispatched action's actual
result rather than a deterministic function of policy-gate/dispatch-mode config. That signal is
attributable per-template via the same real ID contract `orion/proposals/builder.py::
stable_proposal_id()` already uses (`proposal:{template_key}:...` embedded verbatim in every
downstream dispatch_id/result_id -- confirmed live on real rows before writing the script, not
assumed), with no separate join needed. Per-dimension attribution was checked too (via
`config/proposals/proposal_policy.v1.yaml`'s real per-template `dimensions` weights, not guessed)
and found redundant with per-template here: templates map close to 1:1 onto a single dominant
dimension, so a per-dimension analysis on this data cannot be distinguished from a per-template one
-- moot given Step 2 didn't proceed either way (see below).

**Missing Question 4 (online vs. offline) confirmed via this same measurement pass**: this offline,
one-shot batch script *is* the "periodic offline refit, not live online update" mechanism the doc's
own strong prior called for -- no online/incremental update was built or is proposed.

**Real numbers, decisive STOP verdict**:

| template | dominant dimension | n_completed | n_failed | n_total | completion_rate | usable? |
| --- | --- | --- | --- | --- | --- | --- |
| `inspect_node_resource_pressure` | `resource_pressure` | 15,879 | 15 | 15,894 | 99.91% | yes |
| `inspect_transport_status` | `contract_pressure` | 60 | 2 | 62 | 96.77% | yes |
| `inspect_execution_pressure` | `execution_pressure` | 0 | 8 | 8 | 0.00% | NO -- below n=30 floor |
| (9 remaining templates) | various | 0 | 0 | 0 | n/a | NO -- zero real observations ever |

Only 2 of 12 configured templates (`inspect_node_resource_pressure`, `inspect_transport_status`)
have >= 30 real `cortex_result` observations to estimate a completion rate from at all. The other
10 templates have either a statistically meaningless sample (`inspect_execution_pressure`, n=8) or
**zero real completion observations ever** -- not because they perform badly, but because they
never get dispatched far enough to produce a real outcome under the current dispatch-mode/policy-
gate configuration (a separate architectural fact, out of scope to change here per this doc's own
Non-goals on `proposal_risk()`/dispatch mode). Fitting priority weights from this data would only
ever be informed by 17% of templates and would silently conflate "never given the chance to run"
with "ran and failed" for the rest unless explicitly excluded -- exactly the "found a real,
non-degenerate signal without checking whether it predicts anything real" gap this doc's own
Acceptance Check 3 warned against.

**Step 2 (the offline calibration mechanism / persisted weight artifact) was correctly NOT built.**
Per this task's explicit scope: "a measurement-only result... is an acceptable, correct outcome."
No new schema, table, or artifact was persisted; `proposal_priority()`'s `0.4/0.2/0.1` coefficients
and `config/proposals/proposal_policy.v1.yaml`'s `dimension_weights`/`base_priority`/`base_risk`
remain exactly as they were -- untouched, unfit, still hand-typed. This is not a temporary
placeholder pending more history; it's a real finding that the current dispatch-mode gate
structure, not sample count over time, is the actual blocker -- re-running this same script after
more real-time elapses will not change the verdict unless the underlying dispatch-mode/policy-gate
configuration changes first to let more templates actually complete real dispatches.

Full report: `scripts/analysis/measure_proposal_feedback_correlation.py`'s own output
(`/tmp/measure-proposal-feedback-correlation/report.md` at measurement time, plus
`template_outcome_counts.csv` and `progress.log` in the same directory).

## 2026-07-28 implementation update

Followed the "Recommended next patch" section exactly: measured first
(`scripts/analysis/measure_proposal_dimension_variance.py`, real `substrate_field_state` replay
via the real `orion.field.pressure.field_pressures()`, no reimplementation), then built the
EWMA-based confidence replacement only after the real data supported it.

**Missing Question 1 (real per-dimension variance) answered with real numbers**, 16h window
(28,271 real ticks, `2026-07-28T07:44 -> 23:44 UTC`):

| dimension | mean | stddev | variance | min | max | real events (upward, non-decay) |
| --- | --- | --- | --- | --- | --- | --- |
| `execution_pressure` | 0.238 | 0.031 | 9.63e-4 | 0.0 | 0.392 | 720 (avg gap 79s) |
| `resource_pressure` | 0.683 | 0.022 | 4.88e-4 | 0.68 | 0.9 | 22 (avg gap 2637s) |
| `reliability_pressure` | 0.024 | 0.104 | 1.09e-2 | ~0 | 0.9 | 35 (avg gap 1472s) |
| `reasoning_pressure` | 0.043 | 0.0015 | 2.11e-6 | 0.0041 | 0.045 | 13 (avg gap 4138s) |

All four are real, non-degenerate, and genuinely refreshed -- none are the `node:substrate.route`
"zero real events ever, decaying unopposed" disease (that check was run explicitly, not assumed:
the measurement script's first pass used a flat 1-hour recency cutoff and mislabeled
`reasoning_pressure` as an abandoned producer; widening the window and checking real event *counts*
rather than recency-since-last-event corrected this -- `reasoning_pressure`'s ~69min average event
gap is real, sparse cadence, not abandonment).

**Missing Question 2 (durable history) confirmed live**: `substrate_field_state`, 128k+ real rows,
72.5h span, exactly the table `services/orion-proposal-runtime/app/store.py::
ProposalRuntimeStore.load_latest_field()` already reads in production.

**Missing Question 4 (online vs. offline) decided implicitly by scope**: this patch is the
confidence-only half: EWMA state updates online, once per digestion tick, same shape as the
already-shipped `recent_perturbation_ewma*` precedent (PR #1433) -- not the separate weight-
calibration question Missing Question 4 was really asking about, which stays deferred (see
Non-goals).

**Domain-specific EWMA floors, hand-verified not assumed** (per the `execution_prediction_error`
lesson that a borrowed floor silently dominates real z-scores): each dimension's `min_variance` is
~1/10th its own real measured population variance, verified by replaying the real historical
series through `compute_ewma_update` at that floor (no permanent near-zero-variance blowup during
long held/flat stretches -- a real, live risk here since this baseline updates every digestion
tick including unchanged-value ticks, unlike `execution_prediction_error`'s per-event cadence; and
no permanent saturation either -- real p99 |z| in the 1.3-4.3 range across all four dimensions).
Cold-start guard (`DIMENSION_PRECISION_EWMA_MIN_SAMPLES = 8`) hand-verified via a synthetic
seed-at-zero-then-jump replay, scale-invariant across all four dimensions given a shared alpha.

**What shipped**: `orion/proposals/scoring.py::dimension_confidence()` replaced with a real
inverse-z-score confidence (1 minus a saturating surprise score, same shape as
`execution_prediction_error`, inverted); new `dimension_precision_ewma`/`_var`/`_n`/`_zscore` dict
fields on `FieldStateV1` (dict-keyed by dimension_id, additive/safe-default, same
backward-compat precedent as PR #1433); producer-side update in a new
`services/orion-field-digester/app/digestion/precision.py::update_dimension_precision_baseline()`,
wired into `run_digestion_tick()` after decay/diffusion/suppression settle each tick;
`proposal_confidence()` and `orion/proposals/builder.py::_build_candidate()` updated to thread
`field: FieldStateV1` through. Full report:
`scripts/analysis/measure_proposal_dimension_variance.py`'s own output
(`/tmp/measure-proposal-dimension-variance/report.md` at measurement time).

**What did NOT ship (explicitly out of scope, unchanged)**: `proposal_priority()`'s fixed
`0.4/0.2/0.1` coefficients, `proposal_risk()`'s flat bumps, `config/proposals/
proposal_policy.v1.yaml`'s `dimension_weights`/`base_priority`/`base_risk`, and the
`FeedbackFrameV1` calibration-loop closure (Missing Questions 3/5 and the weight-calibration half
of Missing Question 4 remain genuinely open, not answered by this patch).

---

Status (original, 2026-07-28 pre-implementation): **design mode, not implemented.** Touches
`orion/proposals/scoring.py`, part of the
proposal→policy→execution-dispatch→consolidation→feedback pipeline that ultimately governs what
Orion is permitted to autonomously do — a cognition-loop/autonomy-adjacent surface gated by
CLAUDE.md §0A's proposal-mode requirement. This document proposes; it does not build.

## Arsonist summary

Tonight's arbitration-layer investigation (2026-07-28, same session as the metacog Draft/Enrich
cut, the `recent_perturbations`/Layer 5 attention fix PR #1433, and the `execution_prediction_error`
fix PR #1434) found the proposal/policy pipeline's own scoring math is the same disease, never
fixed: `orion/proposals/scoring.py::proposal_priority()` is
`clamp01(base_priority + 0.4*match_score + 0.2*urgency + 0.1*confidence)` — fixed, hand-typed
coefficients, never calibrated against real outcomes. `dimension_confidence()` is a binary
data-presence flag (`1.0 if dimension_id in field_pressures else 0.0`) wearing a name that implies
epistemic certainty it doesn't compute. `proposal_risk()` is a stack of flat additive bumps
(+0.10/+0.05/+0.10). `config/proposals/proposal_policy.v1.yaml`'s `dimension_weights` and every
template's `base_priority`/`base_risk` are hand-typed once and never adjusted — confirmed by grep
this session: zero write-sites for any of these values outside static config, despite
`orion/feedback/builder.py::build_feedback_frame()` genuinely recording real outcomes
(`FeedbackFrameV1`) that nothing ever reads back in.

This is not a new problem needing new theory — it's the same disease already fixed three times
tonight (`bus_synaptic_prediction_error` 2026-07-26, `recent_perturbations`/Layer 5 attention PR
#1433, `execution_prediction_error` PR #1434), all via the same real, already-shared mechanism:
`orion/bus/ewma.py::compute_ewma_update`, comparing each tick against a live rolling baseline
instead of a frozen constant. And the theoretical anchor for *why* this specific fix (precision
from variance, not a flat weight) is the right one isn't invented for this doc — it's already named
twice elsewhere in this codebase tonight: the heartbeat design doc
(`docs/superpowers/specs/2026-07-28-precision-weighted-attention-organ-and-heartbeat-discrimination-design.md`)
cites Active Inference / Free Energy Principle precision-weighting as "the theoretically correct
anchor for attention as a computation," and the Sentience Striving Program charter
(`orion/sentience_striving_program/README.md`, §9a item 3) already names "Free-energy/active-
inference reframing — `capability_policy` as literal expected-free-energy action selection" as a
blue-sky direction. Nobody has built either for proposal scoring specifically.

This spec is the thin, buildable step toward that theory using parts already proven live tonight —
not the full expected-free-energy reframing, which is a separate, larger ambition already named
and explicitly out of scope here (see Non-goals).

## Current architecture

- `orion/proposals/scoring.py` — five pure functions, no state, no history, operating on a single
  tick's `field_pressures: dict[str, float]`:
  - `dimension_confidence()` (lines 26-35): `1.0 if dimension_id in field_pressures else 0.0`.
    Presence flag, not confidence. Its own docstring already discloses this: "No principled
    per-dimension confidence formula survives the SelfStateV1 burn... an honest simplification: a
    binary presence flag, not a fabricated continuous score."
  - `template_match_score()` (lines 38-53): per-dimension `dimension_score() * template_weight *
    policy_weight`, takes the max.
  - `proposal_urgency()` (lines 56-70): max real-time pressure score across the template's declared
    dimensions (or all four `PRESSURE_DIMENSIONS` if none declared). Single-tick, no history.
  - `proposal_priority()` (lines 87-96): `clamp01(base_priority + 0.4*match_score + 0.2*urgency +
    0.1*confidence)`. No calibration record found anywhere for these three coefficients.
  - `proposal_risk()` (lines 99-116): `base_risk` plus flat bumps for proposal kind, policy gate,
    and `reliability_pressure >= 0.5`; capped at 0.15 for read-only observe/inspect/summarize kinds.
- `config/proposals/proposal_policy.v1.yaml` — `dimension_weights` (execution_pressure: 0.30,
  resource_pressure: 0.25, reasoning_pressure: 0.15, reliability_pressure: 0.35, plus three more)
  and every `proposal_templates` entry's `base_priority`/`base_risk` — hand-typed, static.
- `orion/feedback/builder.py::build_feedback_frame()` — genuinely builds real `FeedbackFrameV1`
  rows from real dispatch/policy/cortex-result observations and real field-pressure deltas
  (confirmed honest this session, not fabricated: `_score_for_outcome_kind()`,
  `_aggregate_outcome_status()`, real `reliability_pressure` before/after deltas). Grepped this
  session for every write-site of `base_priority`/`base_risk`/`dimension_weights` across the whole
  repo: only ever read (`orion/proposals/builder.py`), never written back to from anything that
  consumes `FeedbackFrameV1`. The loop observes, never learns — confirmed, not assumed.
- `orion/bus/ewma.py::compute_ewma_update` — already-shared EWMA mean/variance tracker. Used by
  `bus_synaptic_prediction_error` (2026-07-26) and, as of tonight, `execution_prediction_error`
  (PR #1434) and both consumers of `recent_perturbations` (PR #1433). Not yet used anywhere in
  `orion/proposals/`.
- **Live-verified, hard-won lesson directly applicable here** (PR #1434, memory
  `feedback_borrowed_calibrated_constants_dont_transfer_across_domains`): `compute_ewma_update`'s
  shared `_MIN_VARIANCE` default (`1e-6`) does not transfer across domains without re-verification —
  it was calibrated for `orion-bus-mirror`'s real-time-gap domain and was five orders of magnitude
  too large for `execution_prediction_error`'s real variance scale. The four `PRESSURE_DIMENSIONS`
  here (execution/resource/reasoning/reliability pressure) almost certainly each have their own real,
  different natural scale — the same re-verification discipline applies before picking any floor.

## Missing questions

1. **What real historical range/variance does each of the four `PRESSURE_DIMENSIONS` actually
   have?** Needed before picking any EWMA `min_variance` floor per dimension — per the lesson above,
   not assumed transferable from `execution_prediction_error`'s or any other domain's calibrated
   constant.
2. **Does `field_pressures`' source (`FieldStateV1`/`substrate_field_state`, via
   `orion.field.pressure.field_pressures()`) have durable, queryable historical rows** to compute a
   real EWMA baseline against, the same way `recent_perturbations` and `execution_prediction_error`
   did? Not yet checked for this specific pipeline.
3. **What's the right unit for "outcome" to calibrate against?** `FeedbackFrameV1.outcome_score` is
   one scalar per feedback frame, but a tick can carry multiple competing `ProposalCandidateV1`s.
   Does calibration need to be per-template (do proposals built from this template tend to succeed)
   or per-dimension (does high `execution_pressure` reliably predict a good/bad outcome)? Real design
   fork, not a detail — affects the schema of whatever gets persisted.
4. **Online incremental update, or periodic offline refit?** An online live-updating weight has the
   same replay/inspectability tension the heartbeat doc already worked through for stochastic
   trajectories (a claim "the weights adjusted because of outcome X" needs to be reconstructable
   after the fact). Default assumption for this spec is offline refit (safer, matches "measure
   before minting"), but this needs an explicit decision, not a default reached by omission.
5. **Does `proposal_risk()`'s flat-bump structure need the same treatment, or is risk deliberately
   conservative by design** (a safety gate, not an optimization target) and should stay hand-set on
   purpose? Worth deciding explicitly — applying one uniform treatment to both priority and risk
   without asking this first would be assuming the answer.
6. **Is this spec a genuine step toward Active Inference's expected-free-energy formalism, or a
   superficially-similar patch that borrows the word "precision" without the rest of the theory?**
   Worth being honest in this doc, not oversold: this proposes real precision-weighting (gating a
   term's influence by its own estimated reliability) — a real, partial piece of the theory, not the
   full epistemic-value + pragmatic-value expected-free-energy decomposition the charter's bigger
   §9a.3 ambition names. Say so plainly wherever this gets referenced later.

## Proposed schema / API changes

- `orion/proposals/scoring.py::dimension_confidence()` replaced with a genuine precision estimate:
  given a per-dimension EWMA mean/variance, something like a normalized inverse-variance or
  z-score-derived reliability measure — not a binary flag. Exact form pending Missing Question 1's
  real variance data (do not guess a formula before seeing real numbers, same discipline as every
  other fix tonight).
- New persisted EWMA state for each of the 4 `PRESSURE_DIMENSIONS` — likely additive fields on
  `FieldStateV1` itself, same precedent as PR #1433's `recent_perturbation_ewma`/`_var`/`_n`
  (additive, safe defaults, `extra="forbid"`-compatible), or a new lightweight companion schema —
  exact home pending Missing Question 2.
- `proposal_priority()`'s fixed `0.4/0.2/0.1` coefficients become either a periodically-refit set
  computed offline from real `FeedbackFrameV1` history, or a live incremental update — pending
  Missing Question 4. Whichever is chosen, the resulting weights must be inspectable (logged, same
  spirit as the heartbeat doc's seed-logging requirement for its own stochastic mechanism), not a
  black box.
- Recommend exposing the new precision term on `ProposalCandidateV1` (or its `motivating_dimensions`
  map) so "why did this candidate win" stays real and visible rather than hidden inside an opaque
  weighted sum — matches this codebase's own "no empty-shell cognition" standard.

## Files likely to touch

- `orion/proposals/scoring.py` — the core rewrite.
- `orion/bus/ewma.py` — reused; may need a per-dimension `min_variance` override parameter, same
  pattern PR #1434 already added.
- `orion/schemas/field_state.py` (or a new schema) — wherever the new EWMA state persists, pending
  Missing Question 2.
- `config/proposals/proposal_policy.v1.yaml` — `dimension_weights`/`base_priority`/`base_risk`'s
  role changes if calibration supersedes static config; needs careful handling so the file's
  existing declarative-policy role isn't silently broken for anything still reading it directly.
- `orion/feedback/builder.py` or a new consumer — the write-back path from `FeedbackFrameV1` to
  calibration, if periodic offline refit is chosen (Missing Question 4).
- `scripts/analysis/measure_proposal_scoring_calibration.py` (new) — offline measurement/calibration
  script, matching tonight's convention (`measure_attention_salience_normalization.py`,
  `measure_ast_hot_reducer.py`, `measure_emergent_clustering_probe.py`).
- Tests for all of the above, plus a determinism/reproducibility test for whatever calibration
  mechanism ships (same spirit as the heartbeat doc's Missing Question 10).

## Non-goals

- **2026-07-29 addendum**: not building the offline calibration/persisted-weight mechanism this
  patch's own "Proposed schema / API changes" section anticipated -- the 2026-07-29
  calibration-measurement update above found real data does not support it yet (only 2/12
  templates have a statistically usable real completion-rate sample; the other 10 have zero or
  single-digit real observations, gated by dispatch-mode/policy-gate config rather than by their
  own quality). Revisit only after that gate itself changes, not merely after more wall-clock time
  passes.
- Not implementing full expected-free-energy action selection (the charter's larger §9a.3 ambition:
  epistemic value + pragmatic value, precision-weighted, as a formal EFE minimization). This spec is
  the thin precision-weighting step, not the full reframing — see Missing Question 6.
- Not changing `proposal_risk()`'s flat-bump structure unless Missing Question 5 concludes it should
  — risk is plausibly conservative by design on purpose, not folded into this patch by default.
- Not building a live/online weight-update mechanism unless Missing Question 4 concludes that's
  actually needed over periodic offline refit.
- Not touching `orion-heartbeat`'s tensor-network/ensemble-dissipation machinery — established this
  session as the wrong tool for a plain-scalar problem like this one; heartbeat's fix stays
  heartbeat's own.
- Not changing `ProposalCandidateV1`/`ProposalFrameV1`'s existing consumers' expectations without an
  explicit backward-compatibility check, same discipline as PR #1433's additive-fields-with-safe-
  defaults approach.

## Acceptance checks

1. A measurement script shows real historical variance for each of the four pressure dimensions is
   non-degenerate (same metric-quality-gate discipline as every other fix tonight — flat/degenerate
   fails this check) before any EWMA floor is picked.
2. A before/after replay against real historical `ProposalFrameV1` data shows the new precision term
   is not just a relabeled binary flag — it genuinely varies (non-zero stddev) across real ticks.
3. If weight calibration ships: a real correlation check between calibrated scoring and actual
   recorded outcomes, ideally checked in-sample and on a held-out window — avoiding the recurring gap
   from tonight's other findings ("found a real, non-degenerate signal" without ever checking whether
   it predicts anything real).
4. Ships disabled or shadow-measured first, same precedent as every live-scoring change tonight
   (`bus_synaptic`, PR #1433, PR #1434) — before flipping the live proposal-scoring path that actually
   gates real candidate selection.

## Recommended next patch

Same sequencing discipline as tonight's other specs: measure before minting. Concretely — real
historical-variance data for the four pressure dimensions (Missing Question 1) and confirmation that
`field_pressures`' source has durable, queryable history (Missing Question 2), before any EWMA code
gets written. Once that's real, build the precision-weighted `confidence` replacement alone first
(the smaller, more clearly justified half — an honest confidence signal is valuable on its own even
before any weight-calibration work happens) and treat closing the `FeedbackFrameV1` calibration loop
as its own separate follow-up patch, per Missing Question 4 — don't build both speculatively at once.
