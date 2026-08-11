# Deleting `base_priority`: making the proposal arena's rate track state

Date: 2026-08-11
Status: **design mode, awaiting sign-off.** Nothing implemented. Cognition-loop-adjacent per
`CLAUDE.md` §0A ("Proposal mode before invasive cognition changes").

Targets Work Outcome **O1** from `orion/sentience_striving_program/README.md:110-113`:

> **O1 — Capability actually varies with state.** Orion's autonomous-action budget demonstrably
> rises and falls with real internal pressure, not a flat per-cycle allowance, with a demonstrated,
> verified ceiling.

## Arsonist summary

The arena dispatches exactly 5 actions per tick, every tick, forever. It has never once varied with
state. Three independent mechanisms were suspected; only one turned out to matter, and the fix is a
deletion of a key that already has a working replacement sitting unused two lines above it in the
same file.

`config/proposals/proposal_policy.v1.yaml` carries both a per-template `base_priority` (0.20–0.42)
and a global `thresholds.min_priority: 0.10`. `proposal_priority()` is additive:

```python
# orion/proposals/scoring.py:281
return clamp01(base_priority + confidence * max(match_score, urgency))
```

Every candidate therefore *starts* at 3–4x the threshold that is supposed to gate it. `min_priority`
has never bound on a single tick in the arena's recorded history. The computed half of the
expression — the entire precision-weighted formula, its Feldman & Friston anchor, the EWMA
baselines — moves the score by at most ~0.09, which is smaller than the spacing between adjacent
hand-typed `base_priority` values. **The scoring is decorative. A hand-typed float decides the
ranking and a hand-typed float defeats the threshold.**

Delete `base_priority`. `min_priority: 0.10` starts working. Measured against 60,237 real ticks,
Orion would propose on ~6% of them instead of 100%, and would propose *more* when pressure is
genuinely elevated. That is O1, in the charter's own words, achieved by removing a key.

## Current architecture

```text
substrate_field_state
  → orion-proposal-runtime      (Layer 7)  scoring.py::proposal_priority()
  → orion-policy-runtime        (Layer 8)  evaluator.py
  → orion-execution-dispatch-runtime (L9)  max_dispatches_per_tick: 5
  → orion-cortex-exec                      substrate.inspect / summarize / observe
  → substrate_dispatch_results
  → orion-feedback-runtime      (Layer 10) FeedbackFrameV1
  → ∅
```

- Config: `config/proposals/proposal_policy.v1.yaml` (12 templates, hand-authored)
- Loader: `orion/proposals/policy.py` (`extra="forbid"`)
- Scoring: `orion/proposals/scoring.py`
- Threshold consumer: `orion/proposals/builder.py`

### What is actually measured, live

Live priority scores, `substrate_proposal_frames`, 30-minute window, 707 ticks per template:

| template | priority | confidence | `base_priority` | `dimensions` |
| --- | --- | --- | --- | --- |
| `inspect_bus_channel_catalog` | 0.7585 | 0.8633 | 0.38 | `{}` |
| `summarize_transport_contract_drift` | 0.7392 | 0.8633 | 0.36 | `{}` |
| `inspect_attended_target` | 0.7196 | 0.8633 | 0.34 | `{}` |
| `inspect_field_topology_catalog` | 0.7098 | 0.8633 | 0.33 | `{}` |
| `watch_transport_backpressure` | 0.6998 | 0.8633 | 0.32 | `{}` |
| `summarize_loaded_state` | 0.6598 | 0.7281 | 0.35 | 1 dim |
| `inspect_node_resource_pressure` | 0.6498 | 0.7281 | 0.34 | 1 dim |
| `inspect_execution_pressure` | 0.6398 | 0.7272 | 0.40 | 1 dim |
| `inspect_transport_status` | 0.4204 | **0.9978** | 0.42 | 1 dim |
| `watch_reliability` | 0.3097 | 0.9847 | 0.30 | 1 dim |

The top five have **identical** confidence and rank in **exact** `base_priority` order
(0.38 > 0.36 > 0.34 > 0.33 > 0.32). `inspect_transport_status` carries the highest confidence in the
policy (0.9978) and the highest `base_priority` (0.42), and ranks ninth.

Every candidate's floor (0.20–0.42) exceeds `min_priority` (0.10) before any state is consulted.

### The 2026-07-30 patch inverted the monoculture rather than fixing it

`docs/superpowers/specs/2026-07-30-proposal-priority-theory-anchor-design.md` replaced the
hand-picked `0.4/0.2/0.1` blend with precision weighting. It shipped 2026-07-30 and the substrate
went down 2026-07-31 through 2026-08-11. **Today is the first real data under that formula.**

`prepared_for_dispatch` counts by era:

| Jul 24–31 (old formula) | n | Aug 11 (new formula) | n |
| --- | --- | --- | --- |
| `watch_reliability` | 196,624 | `inspect_attended_target` | 140 |
| `inspect_execution_pressure` | 195,085 | `inspect_bus_channel_catalog` | 140 |
| `inspect_transport_status` | 188,212 | `summarize_transport_contract_drift` | 140 |
| `summarize_loaded_state` | 187,917 | `inspect_field_topology_catalog` | 109 |
| `inspect_node_resource_pressure` | 160,976 | `watch_transport_backpressure` | 66 |
| *(the other five)* | ~1% | *(the other five)* | ~15% |

Same winner-take-all shape, opposite membership. The five templates that declare a `dimensions:` map
used to take ~93% of slots; the five with `dimensions: {}` now take ~85%. This is the answer to that
doc's own open question 2, and it is not the answer it hoped for.

The mechanism is `_pressure_dimension_ids()`'s fallback (`orion/proposals/scoring.py:221`):

```python
return dims if dims else list(PRESSURE_DIMENSIONS)
```

A template declaring nothing is scored on the best of everything; a template declaring one dimension
is scored solely on that one. Declaring a dimension is strictly worse. **This is real and it is
live — but it is not the primary defect.** Deleting the fallback alone just flips the monoculture
back to the July membership, because `base_priority` still decides the ranking within whichever
group the formula happens to favor.

## The measurement

Raw max pressure across the three live dimensions, per tick, `substrate_proposal_frames`,
2026-07-29 → 2026-07-31, **60,237 real ticks**, no transform applied:

| threshold | % of ticks that would propose anything |
| --- | --- |
| > 0.05 | 92.77% |
| > 0.08 | 26.75% |
| > 0.10 | **6.37%** |
| > 0.12 | 1.02% |

Maximum ever observed: **0.1575**.

`min_priority: 0.10` — already in the file, already shipped — lands at 6.37%. It has never once
bound.

Dimension resolution over the same window (this is what makes an absolute threshold usable rather
than noise):

| dimension | n | distinct values | min | mean | stddev | max | % zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reliability_pressure` | 120,426 | 16,759 | 0.0000 | 0.0079 | 0.0264 | 0.1575 | 0.00 |
| `resource_pressure` | 120,382 | 388 | 0.0202 | 0.0620 | 0.0156 | 0.1125 | 0.00 |
| `execution_pressure` | 62,719 | 146 | 0.0000 | 0.0399 | 0.0143 | 0.0706 | 9.02 |
| `contract_pressure` | 207,610 | **1** | 0.0 | 0.0 | 0.0 | 0.0 | 100.00 |
| `field_intensity` | 162,372 | **1** | 0.0 | 0.0 | 0.0 | 0.0 | 100.00 |
| `uncertainty` | 3,017 | **1** | 0.0 | 0.0 | 0.0 | 0.0 | 100.00 |
| `agency_readiness` | 82 | **1** | 0.0 | 0.0 | 0.0 | 0.0 | 100.00 |

The bottom four are the dead dimensions removed from templates by the 2026-07-30 cleanup; they
appear here only in pre-cleanup frames and are absent from today's. The three live dimensions have
real resolution — 146 to 16,759 distinct values — which is what an absolute cut can act on.

### Two rejected alternatives, and why

**Rescale each dimension to [0,1] by its observed max.** Fixes range, not shape.
`reliability_pressure` is heavily right-skewed (mean 0.0079, max 0.1575 — 20x its mean) while
`resource_pressure` is compact and never approaches zero (min 0.0202). Divided by max, reliability
would almost never win and would win enormously when it did. It also introduces a new hand-typed
constant (the max) to replace the one being deleted.

**Rank/percentile-normalize each dimension against its own history.** Considered seriously — it is
parameter-free, and the parameter-free order statistic is this repo's proven winning pattern
(`orion/attention/field_attention/selectors.py:84-100` replaced 23 hand-picked weights with `max()`
and survived). **Rejected because a rank transform guarantees a winner on every tick.** Something is
always at the top of its own distribution, by construction. It is structurally incapable of
expressing "nothing is wrong right now" — the same failure class `CLAUDE.md` §0A already names by
example with `bus_synaptic_prediction_error()`'s permanent ~0.27 calm floor. A transform whose
purpose is to manufacture a winner out of quiet is the disease this patch exists to remove.

The pressures being small is not a scaling defect. It is a calm system correctly reporting that it
is calm.

## Proposed change

**Delete `base_priority` from all 12 templates and from `ProposalTemplateV1`.**

### The precedent: this exact operation already shipped one layer up

`config/attention/field_attention_policy.v1.yaml` held the same shape — 23 hand-picked weights
(`pressure*0.45 + novelty*0.20 + urgency*0.25 + confidence*0.10`, plus ~25 per-channel weights).
`docs/superpowers/specs/2026-07-21-attention-salience-cathedral-replacement-tentative-plan.md`
indicted it in terms that describe this file verbatim:

> three independent instances of the same shape — hand-typed linear-weighted-sum, zero citation,
> zero calibration, never outcome-validated

**What shipped (PR #1484/#1488): the weights were deleted from the YAML outright.** That file now
retains only `limits`, `thresholds`, and `observation_modes`. The generalized finding from the
survey of 49 specs: *what survived the kills was thresholds and limits, not weights.* Thresholds
carry a legible operator meaning ("below this, don't bother") and are cheap to reverse; weights
encode a causal claim about relative importance that nothing was ever going to validate.

`proposal_policy.v1.yaml` already has `limits` and `thresholds` blocks. They are defensible as they
stand. `base_priority` is the disease.

**Two requirements inherited from that precedent:**

1. **An in-file tombstone where the key was.** Dated, naming what was removed, why, and what
   replaced it. The census found tombstones are the mechanism that makes kills stick — they stop a
   later patch from "restoring" a key that looks missing. Model the wording on
   `field_attention_policy.v1.yaml`'s own: *"Killed, not left as dead config, per CLAUDE.md §0A's
   'kill means kill, no fallback to the thing being killed.'"*
2. **Audit the removal once per consumer function, not once globally.** A key's blast radius depends
   entirely on the aggregation operator downstream. This file already has all three failure modes on
   record from the 2026-07-30 dead-dimension purge: `max()` tolerated a dead entry
   (`template_match_score`), `mean()` was destroyed by it (`proposal_confidence` was *permanently
   halved*), and a string-suffix filter let it pass a gate while contributing nothing
   (`proposal_urgency`'s `_pressure` check suppressed the real fallback). `base_priority` is read by
   `proposal_priority()` only, but that must be confirmed by grep in the patch, not assumed.

### What is already healthy here, and does not need touching

The census of all 22 config YAMLs found `proposal_policy.v1.yaml` is structurally in the healthy
group: one Pydantic model with `extra="forbid"` and `model_validate` (`orion/proposals/policy.py:55`),
exactly one runtime loader (`services/orion-proposal-runtime/app/worker.py:21`), dated in-file
deletion comments, and — critically — **no Python mirror of its values anywhere.**

That last property is the one that matters. The census's headline finding:

> Duplication, not hand-authorship, is what kills these files.

Every hand-authored list that went stale (`action_ceiling_policy.v1.yaml`,
`grammar_producer_registry.v1.yaml`, `biometrics_lattice.yaml`, `orion_field_topology.v1.yaml`'s
`node_channels`) either has a duplicate elsewhere in Python or has no consumer at all. Every list
that stayed honest has exactly one loader and a live cross-check.

**So this is a values patch, not a structural one.** The plumbing is fine. Do not redesign the file's
shape while fixing its numbers.

```python
# orion/proposals/scoring.py
def proposal_priority(*, match_score: float, urgency: float, confidence: float) -> float:
    return clamp01(confidence * max(match_score, urgency))
```

Nothing authored survives in the expression. `thresholds.min_priority: 0.10` becomes the live gate
and gains a provenance block recording the 60,237-tick measurement above.

### Schema / API changes

- **Removed:** `ProposalTemplateV1.base_priority` (`orion/proposals/policy.py`), and the key from all
  12 templates in `config/proposals/proposal_policy.v1.yaml`. `extra="forbid"` means a stale config
  fails loudly at boot rather than silently ignoring the key — deliberate.
- **Signature changed:** `proposal_priority()` loses its `base_priority` parameter.
- **Unchanged:** `ProposalCandidateV1.priority_score` stays a `[0,1]` float. No table, column, bus
  channel, or event shape changes. `base_risk` is untouched (see Non-goals).

### Files likely to touch

- `config/proposals/proposal_policy.v1.yaml` — remove 12 keys; add provenance to `min_priority`
- `orion/proposals/policy.py` — remove the field
- `orion/proposals/scoring.py` — remove the term and the parameter
- `orion/proposals/builder.py` — call-site update
- `orion/proposals/evals/run_dispatch_rate_coupling_eval.py` — **new**, see Acceptance checks
- `services/orion-proposal-runtime/tests/` — regression tests
- `services/orion-proposal-runtime/README.md` — the stale "still static, hand-typed" note at :88-92

## Metric quality gate (`CLAUDE.md` §0A)

This changes what gates a live cognition loop, so the gate applies to `min_priority` in its new
load-bearing role.

1. **Provenance.** The gated quantity is `proposal_priority()`'s output, whose only remaining inputs
   are `template_match_score()`, `proposal_urgency()`, and `proposal_confidence()` — all in
   `orion/proposals/scoring.py`, all reading `field_pressures()` from
   `orion/field/pressure.py`. Traced to producing functions, not inferred from schema.
2. **Independence.** `min_priority` is a cut on a distribution, not a term added to it. It shares no
   upstream computation with what it gates. The *removed* term, `base_priority`, was not independent
   of anything — it was constant.
3. **Theory anchor.** Not a new instrument: the existing precision-weighted formula (Feldman &
   Friston 2010, already anchored at `scoring.py:253-266`) is unchanged. This patch removes an
   unanchored additive constant that was overwhelming it. The anchor argument is that a precision
   gate should *multiply* the signal it weights; an additive floor defeats that by construction.
4. **Live-data sanity.** 60,237 real ticks, above. Non-degenerate in both directions: 6.37% of ticks
   clear 0.10 and 93.63% do not, so the gate can both fire and rest. Explicitly checked for the
   "can it ever read calm" failure — a threshold on a bounded absolute quantity can; a rank
   transform cannot, which is why one was rejected.
5. **Existing mechanism.** `min_priority` already exists, is already loaded, and is already consumed.
   Nothing new is introduced. This is the search-before-building result: the mechanism was already
   there and disabled.
6. **Reversibility.** Re-adding a key to a YAML file and one term to one expression. No schema
   migration, no persisted artifact, no training default, no env key.

## Acceptance checks

1. **Red before green.** Against `origin/main`, `min_priority` rejects **zero** candidates across a
   replay of real historical field ticks. After the patch it rejects ~93.6%.
2. **The rate tracks state.** Over ≥24h of live post-deploy data, dispatches-per-hour correlates
   with mean field pressure at r ≥ 0.5. Today it is r = 0.249 with a coefficient of variation of
   2.2% — statistically flat.
3. **Quiet is reachable.** At least one real hour passes with zero dispatches. This is the check
   that distinguishes O1 from a rescaling; a system that can never be quiet has not achieved
   "capability varies with state."
4. **The ceiling is real.** O1 requires "a demonstrated, verified ceiling." Under load,
   dispatches-per-tick reaches `max_dispatches_per_tick` and is shown to be bounded by it.
5. **A flat rate fails CI.** `run_dispatch_rate_coupling_eval.py` replays recorded frames and fails
   if the dispatch rate's coefficient of variation drops below a floor derived from the replay
   itself. Per the survey finding that "revisit later" never happens across three live instances —
   **ship the check that fails, not the comment that asks.**
6. **The eval must actually be collectable.** Not a formality: `tests/test_field_topology_config.py`
   is a correct guard on a real invariant that has been silently failing (canonical lattice has 10
   edges, its alias 7, against an assertion that they are equal) because its
   `from app.graph.lattice import load_lattice` cannot resolve from repo root — the guard exists,
   is right, and never runs. Verify the new eval is collected by the command the repo actually
   invokes, not just that the file exists.

**Process discipline, stated in advance because the precedent broke it.** The attention-salience
replacement wired Candidate A live *before* its own head-to-head comparison ran, and this was caught
by code review rather than self-reported (that doc's own section header: *"Candidate A wired live
before this doc's own comparison ran"*). Checks 1–3 here run before the deletion is deployed to a
live runtime, not after.

## Non-goals

- **Not touching `base_risk`.** It feeds `sum_risk_dispatched_today()` and the self-calibrating daily
  risk cap, which exists specifically so "five trivial inspects no longer cost the same as five
  genuinely higher-risk candidates." Deriving risk uniformly from `required_policy_gate` +
  `reversibility` would collapse that back to an action count — a regression of a shipped
  improvement. Separate patch, if at all.
- **Not deleting the `_pressure_dimension_ids()` fallback in this patch.** Real, live, and
  documented above, but fixing it alone flips the monoculture rather than removing it, and fixing it
  *together* with `base_priority` makes the two effects impossible to separate in the post-deploy
  data. Sequence them.
- **Not fitting anything.** `dimension_weights` cannot be fit today: `config/feedback/feedback_policy.v1.yaml`
  hard-codes the outcome scores, making `FeedbackFrameV1.observations[].score` a deterministic
  function of config (real stddev ~1e-13), and 100% of `field_delta` observations are unattributed.
- **Not fixing the action.** See below. This makes the gating honest; it does not make the action
  real.
- **Not a template redesign.** The provenance/kill-criterion structure discussed alongside this is
  deferred pending the outcome question below.

## Missing questions

1. **Does `0.10` survive its own re-derivation?** It was chosen while `base_priority` inflated every
   score past it, so its landing at a defensible 6.37% is partly luck. It should be recorded as
   *measured against 60,237 ticks* — the disclosed-measured provenance state — not asserted as
   derived. Re-deriving is cheap; the query is in this document.
2. **Does the dispatch runtime behave correctly when idle 93% of the time?** The staleness-discard
   machinery, the fresh-priority fallback, and `staleness_discard_count_ewma` were all built against
   a permanently saturated queue. A near-empty queue is an untested regime.
3. **Does `resource_pressure` have a producer?** `docs/superpowers/specs/2026-07-31-o1-capability-state-coupling-brainstorm.md`
   (unmerged, on `origin/docs/o1-capability-state-coupling-brainstorm`) claims it has no producing
   edge. Live data contradicts this: 388 distinct values, mean 0.0620, 0% zero across 120,382
   observations. One of the two is wrong and it matters, because that doc is the only source of the
   per-dimension liveliness measurements.
4. **Is `reliability_pressure` a pressure at all?** `orion/substrate/transport_loop/extract.py`
   computes `observer_failure_pressure = 1.0 if state.observer_failure_count > 0 else 0.0`, then
   `reliability_pressure = max(observer_failure_pressure, 1.0 - delivery_confidence)`. It carries the
   highest weight in `dimension_weights` (0.35) and is partly a binary outage flag. That is a
   category error, not a mis-tuned weight, and no reweighting addresses it.

---

# Broader recommendations

Ordered by how much each moves O1, not by size.

## 1. The action does nothing — this is the largest single defect

`orion/cognition/verbs/substrate.inspect.yaml`, in its own words:

> Read-only; returns strict structured JSON, **no tool selection**, no side effects.

`services: [LLMGatewayService]`, `max_recursion_depth: 0`, one step. And
`orion/execution_dispatch/envelopes.py:30-42` documents what the model receives:

> `motivating_dimensions` is the actual `field_pressures()`/`template_match_score()` dimension scores
> … **the only real numbers the model gets** instead of a bare `target_id`.

So `inspect_transport_status` sends an LLM the pressure numbers that caused the proposal and asks it
to describe transport health. It never touches transport. **It cannot learn anything that was not
already in the prompt.** The action has neither causal reach nor epistemic reach, roughly 15,000
times a day.

This explains, at a stroke, why every candidate outcome signal is dead:

| signal | per-action? | measured verdict |
| --- | --- | --- |
| `field_delta` | no — one label per tick | improved≈worsened in every stratum (2.08/1.52, 5.94/5.64, 10.61/10.20). A read-only action cannot move the field. |
| `action_outcomes.surprise` | no | **95.26%** of multi-dispatch ticks have identical surprise across all concurrent actions; within-tick σ 0.0014 vs global σ 0.193. It is a field reading taken at emit time. |
| `substrate_dispatch_results.raw_len` | yes — 0.06% identical within tick | but it is LLM output length. Non-degenerate and unanchored: fails gate item 3. |
| `action_outcomes.success` | yes | `raw_len > 0`, true 99.9% of the time (54,540/54,597). No variance. |

**Recommendation:** route the substrate verbs at a real read-only skill.
`orion/cognition/skills_manifest.py:64-72` already classifies skills by risk, and the
`read_only: True` set is real reads of real state — docker ps, GPU, disk, biometrics, GitHub PRs.
This needs **none** of the mutating machinery: no new decision literal, no change to
`orion/execution_dispatch/builder.py:167`'s scope gate, no change to `envelopes.py`'s hardcoded
`read_only: True`, no touching `allow_mutating_dispatch` (whose only consumer, `builder.py:80-81`,
appends a warning string and does nothing else). It stays entirely inside `approved_read_only`.

It is also the cheap experiment that tells us which problem we actually have: if behavior starts
varying once real observations flow, the pipeline was sound and only the last inch was missing. If
it does not, the hand-authored template list is the problem and should be killed rather than tuned.

## 2. Close the loop — Layer 10 observes and nothing consumes it

`substrate_dispatch_results` has exactly three readers: the dispatch runtime's own idempotency
replay, its theater tripwire, and the feedback runtime's observation builder. **Nothing writes back
to the field.** The layer doc states this as intent — L10 is *"consequence made observable,"*
explicitly *"not learning yet"* (`docs/context-engineering/04_layer_1_to_11_pipeline.md`).

The repo already wrote down what should close it —
`docs/superpowers/specs/2026-07-17-field-native-motivational-substrate-design.md:189-192`, carried
forward in the charter's §8, still unbuilt:

> A dispatched action's real outcome **perturbs the same field channels that were in the winning
> coalition** — relief on success, sustained pressure on failure — at the granularity the coalition
> actually formed at, not smeared across a generic bucket.

Channel-scoped, which is exactly what fixes per-action attribution. Blocked on item 1: an action
that acquires no observation has no outcome to write back. Do not attempt before item 1 lands.

## 3. Retire the dead dimensions properly

`contract_pressure` appears in 207,610 recorded observations at exactly one distinct value.
`field_intensity`, 162,372. Both were removed from templates on 2026-07-30 but the producers were
never killed, so they continue to be computed and recorded. Per `CLAUDE.md` §0A: *"a narrow,
known-bad instrument that is merely excluded from one consumer but still ticking, still writing its
node, and still readable by every other consumer that iterates generically is not retired, it is
hiding."* Kill the producers.

## 4. Fix the outcome scores before attempting any calibration

`config/feedback/feedback_policy.v1.yaml` hard-codes 8 outcome scores, read at
`orion/feedback/builder.py:290,303`. This makes `FeedbackFrameV1.observations[].score` a
deterministic function of config — measured stddev ~1e-13 across the full real table. **A hand-typed
constant one layer downstream destroys the ability to derive anything upstream.** Any future attempt
to fit `dimension_weights` requires fixing this first, plus the 100%-unattributed `field_delta`
problem (the observation carries a *field tick id* where an action id belongs). Both are named
blockers, not near-term work.

## 5. Make provenance a boot failure, not a comment

Across 27 specs surveyed, every numeric that stayed real carries one of four provenances written
next to it: *measured* (naming the replay script), *theory-anchored* (constant eliminated by
functional form), *disclosed-uncalibrated*, or *refused*. Theater is the fifth state — a number with
no stated provenance. 6 of 22 config YAMLs cite a spec or a `scripts/analysis/measure_*` script;
`proposal_policy.v1.yaml` is not one of them.

Two findings make this actionable rather than aspirational:

- **`grep -rn "Kill criterion" config/ orion/ services/` returns exactly one hit** — in
  `proposal_policy.v1.yaml:205`, on `inspect_attended_target`, whose eval
  (`orion/autonomy/evals/run_attention_bound_proposal_eval.py`) exists and has never been run. The
  pattern is proven writable in this exact file and has never executed.
- **"Revisit later" never happens.** Three disclosed-uncalibrated constants shipped with explicit
  revisit instructions — `STALENESS_DISCARD_EWMA_ALPHA`/`_MIN_VARIANCE`, `max_dispatches_per_tick: 5`,
  `ORION_GOAL_PROVENANCE_MIN_STREAK=3` — all still sit exactly as shipped. The only constant that
  ever got re-derived was forced by a live incident.

**Disclosure is not one of the protective states.** The survey is blunt about this and it corrects an
earlier draft of this document: *"self-disclosure of uncalibratedness does not protect config; it
just documents the debt."* `LinearSalienceCombiner`'s `WEIGHTS_VERSION = "seed-v1"` explicitly
self-labeled as a placeholder awaiting a v2, and was killed anyway. Across 22 config YAMLs, **11
carry a `.v1.` and not one has ever produced a `v2`** — versioning here is decoration, not a lived
migration convention. A disclosed guess buys honesty, not correction.

**Recommendation:** `orion/proposals/policy.py` already uses `extra="forbid"`. Extend it to require a
`provenance` block naming a runnable check, and fail boot without one. Do **not** paste measured
values into the YAML — a `measured_zero_pct: 93.0` in a config file is a hand-typed snapshot of
runtime truth that goes stale in a day, which is the fifth state wearing a lab coat. Provenance
points at the check; the check computes the number.

The repo has already written this rule down, in the one hand-authored entity list that has stayed
honest — `config/field/field_channel_glossary.v1.yaml:25-30`:

> Deliberately does NOT include a "verdict" field: liveness verdicts are computed LIVE from
> `substrate_field_state` by `orion.field.channel_glossary.classify_channel_series()`, not
> hand-maintained here — a static verdict column is exactly what already went stale once

**Hold only the part a human must author; compute the part that can rot.** That is the whole recipe,
and it is also the argument for this patch: `base_priority` is an authored value that rots, sitting
in front of a pressure reading that is computed fresh every tick. Delete the one that rots.

That file also carries its own counter-proof, worth internalizing: its machine-readable body
correctly holds 38 channels while the hand-written prose header beside it still says 35. The
structured data stayed right; the sentence next to it did not.

## 6. Structural elimination beats recalibration

The two clean wins in the surveyed window both **deleted** constants rather than fitting them:
`proposal_priority()`'s `0.4/0.2/0.1` blend → precision weighting (3 constants → 0), and
`compute_salience()`'s 23 hand-picked weights → a parameter-free `max()`
(`orion/attention/field_attention/selectors.py:333-370`). Neither needed data; both survive.

Every attempt to *fit* a constant in the same window either returned STOP
(`2026-07-28-precision-weighted-proposal-scoring-design.md`) or is still uncalibrated. This patch is
in the first category, which is the reason to have confidence in it.

## 7. Unrelated defects surfaced by the config census

Found while grounding this design; none blocks it, all are real and independently shippable.

- **`tests/test_field_topology_config.py` is a guard that never runs.** `config/field/orion_field_topology.v1.yaml`
  declares 10 edges; its `config/field/biometrics_lattice.yaml` alias declares 7. The test asserts
  they are equal, and cannot be collected from repo root (`from app.graph.lattice import ...`).
  Missing from the alias: the entire `node:substrate.bus_synaptic → capability:transport` wiring,
  plus `capability:transport → capability:orchestration` and
  `capability:llm_inference → capability:orchestration`. `services/orion-field-digester/README.md:134`
  still advertises the alias as operator-selectable via `LATTICE_PATH`. Either resync it or delete
  it; either way upgrade the assertion from length equality to edge-set equality.
- **Two fully dead config files.** `config/substrate-lattice/action_ceiling_policy.v1.yaml` (42
  lines, zero loaders — its 7 labels are duplicated as `_CEILING_RANK` in
  `services/orion-hub/scripts/substrate_lattice_routes.py:434-437`, which holds 8) and
  `config/substrate-lattice/grammar_producer_registry.v1.yaml` (no loader; contradicts its own Python
  duplicate `_LANES` on `orion-cortex-exec`'s status).
- **`transport_lattice_policy.v1.yaml`'s Python mirror has drifted and won.**
  `substrate_lattice_routes.py:405-407` says "these values MIRROR the YAML — update this dict to
  match," and its first key is `stream_backlog_pressure`, the name the YAML retired on 2026-07-27 in
  favor of `bus_synaptic_pressure`. Roughly 40% of the best-documented config file in the repo is
  inert as a result. Delete the mirror; the route already loads the YAML two lines later.
- **A stale doc reference to a deleted directory.** `services/orion-field-digester/README.md:670`
  points at `config/self_state/self_state_policy.v1.yaml`, removed in the SelfStateV1 burn.

## Recommended next patch

This one — delete `base_priority`, record the threshold measurement, ship the coupling eval. It is
the smallest change that moves O1, it removes rather than adds, and its acceptance checks are
falsifiable against 60,237 ticks of real history that already exist on disk.

Then item 1 (real perception), which is the largest remaining defect and the gate on everything in
Layer 10.
