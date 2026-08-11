# Why the proposal arena dispatches 5 actions every tick, forever

Date: 2026-08-11
Status: **design mode, awaiting sign-off.** Nothing implemented. Cognition-loop-adjacent per
`CLAUDE.md` §0A ("Proposal mode before invasive cognition changes").

> **Revision note (2026-08-11, same day).** This document previously argued that a hand-typed
> `base_priority` was the primary defect and that deleting it would drop the proposal rate from 100%
> to ~6% of ticks. **That was wrong, and it was wrong because every measurement in it was taken from
> the wrong column.** The correction is in "What broke in the first draft" below. The earlier
> conclusion is retracted in full; the surveyed config/spec findings that grounded it are unaffected
> and retained.

Targets Work Outcome **O1** from `orion/sentience_striving_program/README.md:110-113`:

> **O1 — Capability actually varies with state.** Orion's autonomous-action budget demonstrably
> rises and falls with real internal pressure, not a flat per-cycle allowance, with a demonstrated,
> verified ceiling.

## Arsonist summary

The arena dispatches exactly 5 actions per tick, every tick, forever. It has never once varied with
state, and `thresholds.min_priority: 0.10` has never rejected a single candidate.

The cause is not one constant. It is **three nested `max()` operations stacked on top of each
other**, each one individually defensible, which together guarantee that something always looks
urgent:

```text
max over sensors      normalize_thermal(hottest core)      orion/telemetry/biometrics_pipeline.py:117
max over sources      collect_field_channel_pressures()    orion/field/pressure.py:108-142
max over channels     map_channels_to_dimensions()         orion/field/pressure.py:68-76
max over dimensions   _pressure_dimension_ids() fallback   orion/proposals/scoring.py:216-221
```

An extreme-value statistic of an extreme-value statistic of an extreme-value statistic cannot report
calm. Measured over **28,749 real ticks** (24h, `substrate_field_state`), the per-tick maximum across
all four scored dimensions never once falls below `min_priority`. Not rarely — **0.00% of ticks.**

`base_priority` is real and is worth deleting, but it is a third-order term. Removing it leaves
100.00% of ticks proposing.

## What broke in the first draft

Every measurement in the previous revision was taken from `substrate_proposal_frames`'s persisted
`motivating_dimensions`. That field looks like raw `field_pressures()` output, and
`orion/execution_dispatch/envelopes.py:30-42` describes it that way in its own comment:

> `motivating_dimensions` is the actual `field_pressures()`/`template_match_score()` dimension scores

It is not. `orion/proposals/scoring.py:196-198` returns the **weighted contribution**, after
multiplication by both the policy's `dimension_weights` (0.15–0.35) and the template's own dimension
weight (0.40–0.60):

```python
contributions[dim_id] = clamp01(
    dimension_score(field_pressures, dim_id) * float(weight) * abs(policy_weight)
)
match = max(contributions.values()) if contributions else 0.0
return clamp01(match), contributions   # <- second return value is WEIGHTED
```

So every pressure figure in the first draft was low by roughly 9x, and the headline "6.37% of ticks
would clear 0.10, max ever observed 0.1575" described a distribution that does not exist. The real
per-tick max is 0.9000 and the real floor is 0.3035.

Two lessons, both already repo policy:

- **The comment describing a field is not provenance.** `CLAUDE.md` §0A gate item 1 says trace to the
  producing function, not to a schema comment. This document violated its own gate section.
- **Short-window sampling produced three separate false claims** in the course of this work
  (`execution_pressure` "has 4 distinct values"; `thermal_pressure` "has 6, floored at 0.4571";
  `resource_pressure` "floored at 0.4857"). All three were artifacts of a few-hundred-tick sample and
  all three were contradicted by the full window. Distribution claims below state their tick count.

## Current architecture

```text
substrate_field_state
  → orion-proposal-runtime           (L7)  scoring.py::proposal_priority()
  → orion-policy-runtime             (L8)  evaluator.py
  → orion-execution-dispatch-runtime (L9)  max_dispatches_per_tick: 5
  → orion-cortex-exec                      substrate.inspect / summarize / observe
  → substrate_dispatch_results
  → orion-feedback-runtime           (L10) FeedbackFrameV1
  → ∅
```

- Field/dimension mapping: `orion/field/pressure.py`
- Config: `config/proposals/proposal_policy.v1.yaml` (12 templates, hand-authored)
- Loader: `orion/proposals/policy.py` (`extra="forbid"`)
- Scoring: `orion/proposals/scoring.py`
- Threshold consumer: `orion/proposals/builder.py`

## The measurement

All figures below are raw channel values read directly from `substrate_field_state.field_json`,
**24-hour window, 28,735–28,749 ticks depending on the join.** No weighting, no transform.

### Layer 1 — thermal capture

`resource_pressure` is `max(thermal_pressure, pressure)` (`orion/field/pressure.py:69,71`).
`thermal_pressure` is `(T − 50) / (85 − 50)` on the hottest core
(`orion/telemetry/biometrics_pipeline.py:34-40,117`).

| node | n | min | mean | max | distinct |
| --- | --- | --- | --- | --- | --- |
| `node:athena` | 28,726 | 0.0992 | 0.4864 | 0.7143 | 39 |
| `node:circe` | 28,726 | 0.0000 | 0.0059 | 0.4286 | 15 |
| `node:atlas` | 28,726 | 0.0000 | 0.0000 | 0.0286 | 52 |
| `node:prometheus` | 28,726 | 0.0000 | 0.0000 | 0.0000 | **1** |

The other input, capability `pressure`:

| capability | min | mean | max | distinct |
| --- | --- | --- | --- | --- |
| `capability:orchestration` | 0.0438 | 0.2479 | 0.9000 | 1,905 |
| `capability:llm_inference` | 0.0065 | 0.2197 | 0.8500 | 1,564 |
| `capability:graph` | 0.0340 | 0.1834 | 0.7000 | 1,910 |
| `capability:storage` | 0.0137 | 0.1047 | 0.7500 | 1,907 |
| `capability:transport` | 0.0039 | 0.0304 | 0.6906 | 369 |
| `capability:memory` | 0.0000 | 0.0000 | 0.0000 | **1** |
| `capability:vision` | 0.0000 | 0.0000 | 0.0000 | **1** |

**`thermal_pressure` wins the merge on 91.76% of ticks.** A 39-value quantized temperature reading
overwrites a 1,895-value composite of five independent live capabilities, nine times out of ten:

| `resource_pressure` | min | mean | max | distinct | variance | % < 0.10 |
| --- | --- | --- | --- | --- | --- | --- |
| today | 0.2571 | 0.5017 | 0.9000 | **128** | 0.00702 | 0.00% |
| without the thermal input | 0.0928 | 0.3171 | 0.9000 | **1,895** | 0.01840 | 0.03% |

**15x resolution loss.** This is the honest indictment of the thermal routing — not that it imposes
a floor (its floor is 0.0992, not the 0.4857 previously claimed), but that it destroys the signal it
is merged into.

There is a second reason it is the wrong input. `thermal_pressure` is in
`services/orion-field-digester/app/digestion/decay.py`'s `NODE_DECAY_CHANNELS`, so a low reading is
produced by the producer going quiet as readily as by the hardware being cool — the
decayed-to-zero-vs-genuinely-calm ambiguity `CLAUDE.md` §0A names by example. A dimension whose calm
state is indistinguishable from its producer dying cannot serve as a gate input.

### Layer 2 — the max-stacking, which is the actual O1 blocker

Removing thermal from the map is not sufficient. Per-tick maximum across all four scored dimensions,
which is what the five `dimensions: {}` templates are scored on via `_pressure_dimension_ids()`'s
fallback:

| | min | mean | % of ticks < `min_priority` (0.10) |
| --- | --- | --- | --- |
| today | 0.3035 | 0.5048 | **0.00%** |
| without the thermal input | 0.1265 | 0.3414 | **0.00%** |

Seven capability channels and four nodes, five of them live and moving independently, merged by
`max()`. The merged series can only read calm when *every* source is simultaneously calm, which over
28,749 ticks has never happened. This is structural, not a tuning accident: the probability that all
five independent live sources are simultaneously below threshold falls off geometrically in the
number of sources, so **adding a healthy sensor makes the system look more urgent.**

That is the defect. `min_priority` is not mis-set; it is being compared against a quantity that was
built to be large.

### Layer 3 — `base_priority`, third-order

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

The top five have identical confidence and rank in exact `base_priority` order. That finding stands:
**within a tick, a hand-typed float decides the ranking.** What does *not* stand is the claim that it
decides whether anything proposes at all. Replaying real history with `base_priority` removed:
19,050 candidates, mean priority **0.3947**, against a threshold of 0.10 — **100.00% still clear.**

### The 2026-07-30 patch inverted the monoculture rather than fixing it

`docs/superpowers/specs/2026-07-30-proposal-priority-theory-anchor-design.md` replaced the
hand-picked `0.4/0.2/0.1` blend with precision weighting. It shipped 2026-07-30; the substrate was
down 2026-07-31 → 2026-08-11. `prepared_for_dispatch` counts by era:

| Jul 24–31 (old formula) | n | Aug 11 (new formula) | n |
| --- | --- | --- | --- |
| `watch_reliability` | 196,624 | `inspect_attended_target` | 140 |
| `inspect_execution_pressure` | 195,085 | `inspect_bus_channel_catalog` | 140 |
| `inspect_transport_status` | 188,212 | `summarize_transport_contract_drift` | 140 |
| `summarize_loaded_state` | 187,917 | `inspect_field_topology_catalog` | 109 |
| `inspect_node_resource_pressure` | 160,976 | `watch_transport_backpressure` | 66 |
| *(the other five)* | ~1% | *(the other five)* | ~15% |

Same winner-take-all shape, opposite membership. Templates declaring a `dimensions:` map took ~93%
of slots; templates declaring `dimensions: {}` now take ~85%. The mechanism is
`_pressure_dimension_ids()` (`orion/proposals/scoring.py:216-221`):

```python
return dims if dims else list(PRESSURE_DIMENSIONS)
```

**A template that declares nothing is scored on the best of everything; a template that declares one
dimension is scored solely on that one. Declaring a dimension is strictly a penalty.** This is the
same max-stacking defect wearing a different hat, and it is the reason the fallback is now in scope
rather than deferred.

### Two rejected alternatives, and why

**Rescale each dimension to [0,1] by its observed max.** Fixes range, not shape, and introduces a new
hand-typed constant (the max) to replace the one being deleted.

**Rank/percentile-normalize each dimension against its own history.** Considered seriously — it is
parameter-free, and the parameter-free order statistic is this repo's proven winning pattern
(`orion/attention/field_attention/selectors.py:84-100`). **Rejected because a rank transform
guarantees a winner on every tick.** Something is always at the top of its own distribution, by
construction. It is structurally incapable of expressing "nothing is wrong right now" — the same
failure class `CLAUDE.md` §0A names via `bus_synaptic_prediction_error()`'s permanent ~0.27 calm
floor. It is also *precisely the disease already diagnosed above*: manufacturing a winner out of
quiet is what the max-stack already does.

## Proposed change: three patches, sequenced, not one

They are separated because their post-deploy effects are otherwise impossible to attribute — the
same sequencing argument the first draft made for `_pressure_dimension_ids()`, now applied to itself.

### Patch A — remove `thermal_pressure` from `CHANNEL_DIMENSION_MAP`

One line (`orion/field/pressure.py:69`). Same units, same direction, same interpretation downstream;
`resource_pressure` still means "resource load," it just stops being 92%-determined by an idle CPU's
temperature. The raw `thermal_pressure` channel is untouched everywhere else — grammar emit/extract,
`state_deltas`, decay, tensor channels, and the `strain` composite
(`biometrics_pipeline.py:180`) all read it directly and are unaffected.

**Status: IMPLEMENTED** on `fix/thermal-pressure-dimension-unmap`. Implementing it corrected three
claims this section made in its own draft. All three are recorded below rather than silently edited,
because two of them were the *reassuring* direction — the kind that survives review by being pleasant.

**Measured effect,** through the real production path
(`scripts/analysis/measure_proposal_dimension_variance.py`, which parses each historical row into a
real `FieldStateV1` and runs the unmodified `field_pressures()`; 6h / 10,562 ticks, before vs after
on the same window):

| | before | after |
| --- | --- | --- |
| variance | 0.0037973 | **0.0174205** (4.6x) |
| mean | 0.538208 | 0.32551 |
| min | 0.314286 | 0.144535 |
| distinct values (24h window) | 128 | **1,895** (15x) |
| real upward transitions | 270 | **431** (+60%) |
| classification | `NON_DEGENERATE_USABLE` | `NON_DEGENERATE_USABLE`, no decay artifact |

**Correction 1 — the precision floor was stale, and so are the other three.** This draft claimed the
constant is measured on weighted values and therefore "reconciles, it is not stale," reasoning from
`raw_variance × weight²`. That was wrong twice over:
`measure_proposal_dimension_variance.py` **reads raw `field_pressures()` output, not
`motivating_dimensions`** — it never had the misread this document attributed to it — and the
apparent reconciliation was arithmetic coincidence. Measured on *unpatched* code, same 6h window,
against the values recorded in `scoring.py`'s own comment from 2026-07-28:

| dimension | recorded 2026-07-28 | measured 2026-08-11 | ratio | floor is |
| --- | --- | --- | --- | --- |
| `execution_pressure` | 9.63e-4 | 1.987e-2 | 20.6x | **20x too low** |
| `resource_pressure` | 4.88e-4 | 3.797e-3 | 7.8x | 8x too low |
| `reasoning_pressure` | 2.11e-6 | 3.494e-7 | 0.17x | **6x too high** |
| `reliability_pressure` | 1.09e-2 | 5.520e-3 | 0.51x | 2x too high |

All four have drifted, in both directions, independently of this change. Only
`resource_pressure` is re-derived in Patch A (`5e-5` → `2e-3`, ~1/10th of the new post-patch
variance, hand-verified by replaying the real series through `compute_ewma_update`: p99 |z| 3.116,
inside the 1.3–4.3 band the originals were checked against; tail bounded from max |z| 17.195 to
10.023). **The other three are deliberately left stale** so this patch's post-deploy effect stays
attributable to the one map entry it removes. Re-deriving all four at once would make the resulting
data unreadable. Follow-up work, tracked here.

This is the *fifth* live instance of "a constant calibrated once, never re-derived" in this arc, and
the first where the drift was found by re-running the measurement script that produced it rather than
by an incident. **Recommendation: the script should be a scheduled check that fails on drift beyond a
stated ratio, not a script someone remembers to run.** That is the same "ship the check that fails,
not the comment that asks" finding as broader recommendation 5, applied to the one instrument that
already exists and works.

**Correction 2 — `measure_autonomy_gate.py` is not a trivially-passing gate; it is a dead one.** This
draft predicted it would "pass trivially (≈45% of rows)." Run live, it reports **`UNMEASURABLE`** for
both of its verdicts: `resource_pressure rows: 0`, `self_state loaded rows=0/0`. It reads
`self_state`, which the 2026-07-22 SelfStateV1 burn emptied. It has been reporting nothing for three
weeks. It therefore cannot be broken by Patch A — but it is a §0A "hiding, not retired" instrument
and should be fixed or killed.

**Correction 3 — the phi-encoder risk is discharged, not outstanding.** This draft said any fitted
artifact "is invalidated; check for a live artifact before shipping." Checked: the active encoder is
`/mnt/telemetry/models/phi/encoders/active` → `v20260712-seedv4-postfix`, and its `input_features`
are `agency_readiness, execution_pressure, reasoning_pressure, overall_intensity, recall_gate_fired,
reasoning_present, execution_load, reasoning_load`. **`resource_pressure` is not among them.** No
invalidation. (Noted in passing, out of scope: that active encoder trains on `agency_readiness` and
probes `field_intensity` — two dimensions this repo declared never-produced on 2026-07-30 — and on
`execution_load`, renamed in PR #1338. The live phi encoder is fit on dead features. Its own finding.)

Blast radius audited, five consumers of the `resource_pressure` dimension: `orion/proposals/scoring.py`
(the intended target), the precision floor (re-derived), `config/feedback/feedback_policy.v1.yaml:27,34`
(same units and direction — it stops crediting actions for a CPU cooling off),
`measure_autonomy_gate.py` (dead, correction 2), `scripts/fit_phi_encoder.py:75,151` (unaffected,
correction 3).

**This patch does not move O1 on its own** (0.00% calm before and after, table above). It is
justified by the 15x resolution recovery and by removing a decay-ambiguous input from a gate, not by
the rate.

### Patch B — remove the `_pressure_dimension_ids()` fallback

`orion/proposals/scoring.py:216-221`. A template that declares no dimensions should score 0.0 on
match, not the max of everything. Five templates currently carry `dimensions: {}` and are the current
monoculture; after this they either get a real declared dimension or they honestly score zero and
stop winning slots they were never motivated by.

This is the patch that can actually move the rate, because it removes the outermost `max()` — the one
over dimensions. It is also the one that requires a decision, not just a deletion: **do the five
`dimensions: {}` templates have a real motivating signal, or should they be deleted?** Three of them
are transport/bus templates whose original dimension (`contract_pressure`) was removed on 2026-07-30
as never-produced, and no replacement dimension exists in `CHANNEL_DIMENSION_MAP`. Per §0A, a
template with no real motivating signal is a keyword cathedral entry and should be killed, not given
a fallback.

### Patch C — delete `base_priority`

Unchanged from the first draft in mechanism, downgraded in claimed effect. It does not gate anything;
it decides ranking within a tick, which is still a hand-typed float deciding a cognitive outcome and
still worth removing. Ships last so its (small) effect is separable.

```python
def proposal_priority(*, match_score: float, urgency: float, confidence: float) -> float:
    return clamp01(confidence * max(match_score, urgency))
```

**Precedent and its two inherited requirements.** `config/attention/field_attention_policy.v1.yaml`
held the same shape — 23 hand-picked weights — and PR #1484/#1488 deleted them from the YAML
outright, leaving only `limits`, `thresholds`, and `observation_modes`. The generalized finding from
the 49-spec survey: *what survived the kills was thresholds and limits, not weights.*

1. **An in-file tombstone where the key was**, dated, naming what was removed and why. The census
   found tombstones are the mechanism that makes kills stick.
2. **Audit the removal once per consumer function, not once globally.** This file has all three
   failure modes on record from the 2026-07-30 purge: `max()` tolerated a dead entry, `mean()` was
   destroyed by one (`proposal_confidence` was permanently halved), and a string-suffix filter let one
   pass a gate while contributing nothing.

**Known blocker, unresolved.** `orion/reverie/proposal.py:72` and `orion/metacog/proposal.py:163` set
`priority_score` directly from `salience` (flat 0.75 live), bypassing `proposal_priority()` entirely.
Both flags are `true` in the live container. With `base_priority` gone, template candidates drop to a
mean of 0.3947 and these external candidates at 0.75 would outrank every template on every tick.
**Patch C cannot ship until these two producers are on the same scale.** This was missed in the first
draft.

### Schema / API changes

- **Removed (C):** `ProposalTemplateV1.base_priority` and the key from all 12 templates.
  `extra="forbid"` means a stale config fails loudly at boot — deliberate.
- **Signature changed (C):** `proposal_priority()` loses its `base_priority` parameter.
- **Unchanged:** `ProposalCandidateV1.priority_score` stays a `[0,1]` float. No table, column, bus
  channel, or event shape changes. `base_risk` untouched (see Non-goals).

## Metric quality gate (`CLAUDE.md` §0A)

Applied to `min_priority` in its new load-bearing role.

1. **Provenance.** Traced to producing functions this revision, which is what caught the error:
   `normalize_thermal` (`biometrics_pipeline.py:34-40`) → `collect_field_channel_pressures` /
   `map_channels_to_dimensions` (`pressure.py:68-142`) → `template_match_score` /
   `proposal_urgency` / `proposal_confidence` (`scoring.py`). The first draft trusted a schema
   comment and was wrong by 9x.
2. **Independence.** `min_priority` is a cut on a distribution, not a term added to it. The four
   scored dimensions are **not** independent of each other in the way the max-stack assumes — that
   non-independence is the defect this document reports, not a caveat on it.
3. **Theory anchor.** No new instrument. The precision-weighted formula (Feldman & Friston 2010,
   anchored at `scoring.py:253-266`) is unchanged. The argument is structural: a precision gate must
   *multiply* the signal it weights, and neither an additive floor nor an outer `max()` over
   unrelated dimensions preserves that.
4. **Live-data sanity.** 28,749 ticks, above. **This gate currently FAILS in the calm direction:**
   0.00% of ticks can read below threshold. That failure is the finding. Patches A+B exist to make it
   pass; the acceptance checks below are the re-test.
5. **Existing mechanism.** `min_priority` already exists, is loaded, and is consumed. Nothing new is
   introduced — the mechanism was already there and defeated.
6. **Reversibility.** One YAML map entry, one `return` line, one config key. No schema migration, no
   persisted artifact, no training default, no env key. The one irreversible-ish item is the
   re-derived precision constant, which is recorded with its query.

## Acceptance checks

1. **Red before green.** Against `origin/main`, `min_priority` rejects **zero** candidates across a
   replay of real historical field ticks. Verified: 19,050 candidates, 100.00% clear.
2. **Quiet is reachable.** After A+B, at least one real hour passes with zero dispatches. This is the
   check that distinguishes O1 from a rescaling, and it is the one the current system fails at 0.00%.
3. **The rate tracks state.** Over ≥24h of live post-deploy data, dispatches-per-hour correlates with
   mean field pressure at r ≥ 0.5. Today r = 0.249, CV 2.2% — statistically flat.
4. **The ceiling is real.** O1 requires "a demonstrated, verified ceiling." Under load,
   dispatches-per-tick reaches `max_dispatches_per_tick` (5) and is shown bounded by it.
5. **A flat rate fails CI.** `orion/proposals/evals/run_dispatch_rate_coupling_eval.py` replays
   recorded frames and fails on ALWAYS_ON / NEVER_ON / FLAT. **It must read raw pressures, not
   `motivating_dimensions`** — the draft version inherited the misread and returned PASS at a
   reported 0.7% against a true rate of 100%. An eval that reads the wrong column is worse than no
   eval. Per the survey finding that "revisit later" never happens across three live instances: ship
   the check that fails, not the comment that asks.
6. **The eval must actually be collectable.** Not a formality:
   `tests/test_field_topology_config.py` is a correct guard on a real invariant that has been
   silently failing (canonical lattice 10 edges, alias 7, asserted equal) because its
   `from app.graph.lattice import load_lattice` cannot resolve from repo root. Verify the new eval is
   collected by the command the repo actually invokes.
7. **`capability:memory` and `capability:vision` produce a value, or are removed.** Both are flat
   0.0 across all 28,638 observed ticks, one distinct value each. Two of seven capability channels
   have never produced a reading. Either is defensible; silently carrying them in a `max()` is not.

**Process discipline, stated in advance because the precedent broke it.** The attention-salience
replacement wired Candidate A live *before* its own head-to-head comparison ran, and this was caught
by code review rather than self-reported. Checks 1–2 run before any deletion reaches a live runtime.

## Non-goals

- **Not touching `base_risk`.** It feeds `sum_risk_dispatched_today()` and the self-calibrating daily
  risk cap, which exists so five trivial inspects no longer cost the same as five higher-risk
  candidates. Separate patch, if at all.
- **Not changing `thermal_pressure`'s own semantics.** Patch A removes it from one map. It is *not* a
  proposal to make it a deviation-from-baseline signal, which would change units under `strain`,
  grammar emit, `state_deltas`, and the biometrics corpus — four domains with no stake in this fix.
  Contain the change at the mapping layer.
- **Not fitting anything.** `dimension_weights` cannot be fit today:
  `config/feedback/feedback_policy.v1.yaml` hard-codes the outcome scores, making
  `FeedbackFrameV1.observations[].score` a deterministic function of config (real stddev ~1e-13), and
  100% of `field_delta` observations are unattributed.
- **Not fixing the action.** See broader recommendation 1. This makes the gating honest; it does not
  make the action real.

## Missing questions

1. **Is `max()` the right merge at all, at any of the three layers?** This document establishes that
   stacking three of them is fatal, and proposes removing the outermost two inputs. It does not
   establish what the right operator is. A mean would understate genuine single-source emergencies; a
   max overstates everything. This is the real open design question and it should not be answered by
   picking a blend weight, which would reintroduce exactly what was deleted.
2. **Do the five `dimensions: {}` templates deserve to exist?** Gate on Patch B. Three are transport
   templates orphaned by the 2026-07-30 `contract_pressure` removal with no replacement dimension in
   `CHANNEL_DIMENSION_MAP`.
3. **Does the dispatch runtime behave correctly when idle?** The staleness-discard machinery, the
   fresh-priority fallback, and `staleness_discard_count_ewma` were all built against a permanently
   saturated queue. A near-empty queue is an untested regime, and Patches A+B create it.
4. **Is `reliability_pressure` a pressure at all?** `orion/substrate/transport_loop/extract.py`
   computes `observer_failure_pressure = 1.0 if state.observer_failure_count > 0 else 0.0`, then
   `reliability_pressure = max(observer_failure_pressure, 1.0 - delivery_confidence)`. It carries the
   highest weight in `dimension_weights` (0.35) and is partly a binary outage flag — a fourth `max()`,
   and a category error no reweighting addresses.
5. **Resolved:** the first draft asked whether `resource_pressure` has a producer, citing an unmerged
   brainstorm that claimed it has no producing edge. It does: five live capability channels plus
   thermal, table above. That claim is wrong.

---

# Broader recommendations

Ordered by how much each moves O1, not by size. These were grounded in a survey of 49 specs and a
census of all 22 config YAMLs, and are unaffected by the measurement error above — they were sourced
from different tables.

## 1. The action does nothing — this is the largest single defect

`orion/cognition/verbs/substrate.inspect.yaml`, in its own words:

> Read-only; returns strict structured JSON, **no tool selection**, no side effects.

`services: [LLMGatewayService]`, `max_recursion_depth: 0`, one step. And `envelopes.py:30-42`
documents what the model receives: `motivating_dimensions`, *"the only real numbers the model gets"*
— which, per the correction above, are not even the raw pressures.

So `inspect_transport_status` sends an LLM the (weighted) pressure numbers that caused the proposal
and asks it to describe transport health. It never touches transport. **It cannot learn anything that
was not already in the prompt.** Neither causal nor epistemic reach, roughly 15,000 times a day.

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
This needs none of the mutating machinery: no new decision literal, no change to
`orion/execution_dispatch/builder.py:167`'s scope gate, no change to `envelopes.py`'s hardcoded
`read_only: True`. It stays entirely inside `approved_read_only`.

It is also the cheap experiment that tells us which problem we have: if behavior varies once real
observations flow, the pipeline was sound and only the last inch was missing. If not, the
hand-authored template list is the problem and should be killed rather than tuned.

## 2. Close the loop — Layer 10 observes and nothing consumes it

`substrate_dispatch_results` has exactly three readers: the dispatch runtime's idempotency replay,
its theater tripwire, and the feedback runtime's observation builder. **Nothing writes back to the
field.** The layer doc states this as intent — L10 is *"consequence made observable,"* explicitly
*"not learning yet"* (`docs/context-engineering/04_layer_1_to_11_pipeline.md`).

The repo already wrote down what should close it —
`docs/superpowers/specs/2026-07-17-field-native-motivational-substrate-design.md:189-192`, carried
into the charter's §8, still unbuilt:

> A dispatched action's real outcome **perturbs the same field channels that were in the winning
> coalition** — relief on success, sustained pressure on failure — at the granularity the coalition
> actually formed at, not smeared across a generic bucket.

Channel-scoped, which is exactly what fixes per-action attribution. Blocked on item 1: an action that
acquires no observation has no outcome to write back.

## 3. Retire the dead dimensions and dead channels properly

`contract_pressure` appears in 207,610 recorded observations at exactly one distinct value;
`field_intensity`, 162,372. Both were removed from templates on 2026-07-30 but the producers were
never killed. Add to that list `capability:memory` and `capability:vision` (flat 0.0, 28,638 ticks)
and `node:prometheus`'s `thermal_pressure` (flat 0.0, one distinct value). Per `CLAUDE.md` §0A: *"a
narrow, known-bad instrument that is merely excluded from one consumer but still ticking, still
writing its node, and still readable by every other consumer that iterates generically is not
retired, it is hiding."* Kill the producers.

## 4. Fix the outcome scores before attempting any calibration

`config/feedback/feedback_policy.v1.yaml` hard-codes 8 outcome scores, read at
`orion/feedback/builder.py:290,303`, making `FeedbackFrameV1.observations[].score` a deterministic
function of config — measured stddev ~1e-13 across the full real table. **A hand-typed constant one
layer downstream destroys the ability to derive anything upstream.** Any future attempt to fit
`dimension_weights` requires fixing this first, plus the 100%-unattributed `field_delta` problem (the
observation carries a *field tick id* where an action id belongs).

## 5. Make provenance a boot failure, not a comment

Across 27 specs surveyed, every numeric that stayed real carries one of four provenances written next
to it: *measured* (naming the replay script), *theory-anchored* (constant eliminated by functional
form), *disclosed-uncalibrated*, or *refused*. Theater is the fifth state — a number with no stated
provenance. 6 of 22 config YAMLs cite a spec or a `scripts/analysis/measure_*` script;
`proposal_policy.v1.yaml` is not one of them.

- **`grep -rn "Kill criterion" config/ orion/ services/` returns exactly one hit** — in
  `proposal_policy.v1.yaml:205`, on `inspect_attended_target`, whose eval
  (`orion/autonomy/evals/run_attention_bound_proposal_eval.py`) exists and has never been run. The
  pattern is proven writable in this exact file and has never executed.
- **"Revisit later" never happens.** `STALENESS_DISCARD_EWMA_ALPHA`/`_MIN_VARIANCE`,
  `max_dispatches_per_tick: 5`, and `ORION_GOAL_PROVENANCE_MIN_STREAK=3` all shipped with explicit
  revisit instructions and all still sit exactly as shipped. The only constant ever re-derived was
  forced by a live incident.

**Disclosure is not one of the protective states.** *"Self-disclosure of uncalibratedness does not
protect config; it just documents the debt."* `LinearSalienceCombiner`'s
`WEIGHTS_VERSION = "seed-v1"` self-labeled as a placeholder awaiting a v2 and was killed anyway.
Across 22 config YAMLs, **11 carry a `.v1.` and not one has ever produced a `v2`.**

**Recommendation:** extend `orion/proposals/policy.py`'s `extra="forbid"` to require a `provenance`
block naming a runnable check, and fail boot without one. Do **not** paste measured values into the
YAML — a `measured_zero_pct: 93.0` in a config file is a hand-typed snapshot of runtime truth that
goes stale in a day. Provenance points at the check; the check computes the number.

The repo already wrote this rule down, in the one hand-authored entity list that stayed honest —
`config/field/field_channel_glossary.v1.yaml:25-30`:

> Deliberately does NOT include a "verdict" field: liveness verdicts are computed LIVE from
> `substrate_field_state` by `orion.field.channel_glossary.classify_channel_series()`, not
> hand-maintained here — a static verdict column is exactly what already went stale once

**Hold only the part a human must author; compute the part that can rot.** That file also carries its
own counter-proof: its machine-readable body correctly holds 38 channels while the hand-written prose
header beside it still says 35. The structured data stayed right; the sentence next to it did not —
which is, in miniature, exactly the failure this document's first draft made at scale.

## 6. Structural elimination beats recalibration

The two clean wins in the surveyed window both **deleted** constants rather than fitting them:
`proposal_priority()`'s `0.4/0.2/0.1` blend → precision weighting (3 constants → 0), and
`compute_salience()`'s 23 hand-picked weights → a parameter-free `max()`. Neither needed data; both
survive.

Every attempt to *fit* a constant in the same window either returned STOP
(`2026-07-28-precision-weighted-proposal-scoring-design.md`) or is still uncalibrated. Patches A–C
are all in the first category.

**With one correction this revision forces:** `max()` is a parameter-free operator, which is why it
was the winning replacement one layer up — and it is *also* the defect here. Parameter-free is not
the same as correct. An operator with no constants to tune can still be structurally incapable of
reporting the state you need it to report, which is the same trap the rejected rank transform sets.
The survey's lesson should be read as *eliminate the constant*, not *reach for `max()`*.

## 7. Unrelated defects surfaced by the config census

None blocks this; all are real and independently shippable.

- **`tests/test_field_topology_config.py` is a guard that never runs.**
  `config/field/orion_field_topology.v1.yaml` declares 10 edges; its
  `config/field/biometrics_lattice.yaml` alias declares 7. The test asserts equality and cannot be
  collected from repo root. Missing from the alias: the entire
  `node:substrate.bus_synaptic → capability:transport` wiring, plus
  `capability:transport → capability:orchestration` and
  `capability:llm_inference → capability:orchestration`.
  `services/orion-field-digester/README.md:134` still advertises the alias as operator-selectable via
  `LATTICE_PATH`. Either resync or delete it; either way upgrade the assertion to edge-set equality.
- **Two fully dead config files.** `config/substrate-lattice/action_ceiling_policy.v1.yaml` (42 lines,
  zero loaders — its 7 labels duplicated as `_CEILING_RANK` in
  `services/orion-hub/scripts/substrate_lattice_routes.py:434-437`, which holds 8) and
  `config/substrate-lattice/grammar_producer_registry.v1.yaml` (no loader; contradicts its own Python
  duplicate `_LANES`).
- **`transport_lattice_policy.v1.yaml`'s Python mirror has drifted and won.**
  `substrate_lattice_routes.py:405-407` says "these values MIRROR the YAML — update this dict to
  match," and its first key is `stream_backlog_pressure`, the name the YAML retired on 2026-07-27 in
  favor of `bus_synaptic_pressure`. Roughly 40% of the best-documented config file in the repo is
  inert. Delete the mirror; the route already loads the YAML two lines later.
- **A stale doc reference to a deleted directory.**
  `services/orion-field-digester/README.md:670` points at
  `config/self_state/self_state_policy.v1.yaml`, removed in the SelfStateV1 burn.

## Recommended next patch

~~**Patch A**~~ — **done**, on `fix/thermal-pressure-dimension-unmap`. One map entry, blast radius
audited to five consumers, 4.6x variance and 15x distinct-value recovery measured through the
production path. It corrected three of this document's own claims; see the Patch A section.

**Next: the four precision floors.** Correction 1 found all four drifted from their single 2026-07-28
derivation, in both directions, by up to 20.6x — found by re-running the measurement script, not by
an incident. Patch A re-derived only `resource_pressure`, to keep its own effect attributable. The
other three, plus turning that script into a check that fails on drift, is the immediate follow-up
and is independent of everything below.

Then **Patch B**, which is the one that can actually move O1, gated on answering missing question 2.
Then broader recommendation 1 (real perception), which is the largest remaining defect and the gate
on everything in Layer 10. **Patch C last, and not until the reverie/metacog scale collision is
resolved.**
