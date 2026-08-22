# Field deviation tension sensing — scale-free admission and rank competition

Status: design + implementation (this spec ships with its code). Read-only sensing and
measurement only: nothing on this path acts, publishes to the bus, or feeds a prompt.

## Arsonist summary

Orion's motivational program has spent months at the *taxonomy* layer while the layer
underneath it was starved. The measured number, from
`docs/superpowers/specs/2026-07-07-homeostatic-drives-real-tensions-design.md`:
**tensions fired in 0.064% of ticks — 284 of 444,943 audits.** No taxonomy on top of an
input that fires 0.064% of the time can work. Not the six-drive one, not the five-drive one,
not a theoretically perfect one. The repeated failures were one unmet precondition hit from
many angles, not many independent design mistakes.

Two things follow, and this patch does both:

1. **Tension must fire on deviation from an adapted baseline, never on presence.** That
   mechanism was built (`orion/autonomy/deviation_gate.py`, 96 pure lines, EWMA baseline +
   z-threshold), was live in `ConceptWorker`, and was deleted on 2026-07-30 in `d6a4e892b`
   as collateral in the drives sweep. It has zero drive coupling — its whole interface is
   `observe(kind, dimension, x, confidence, worse) -> impulse`. It was not measured bad. The
   taxonomy next to it was.
2. **Cross-signal combination must never use hand-authored weights.** `signal_drive_map.yaml`
   (69 lines of `drives: {capability: 0.4, continuity: 0.3}`) was trying to price a thermal
   spike against a coherence drop. There is no such exchange rate; the question is
   unanswerable in magnitude space. The repo already answers it in rank space
   (`orion/attention/rank_aggregation.py`, Borda, two live consumers, explicitly built to
   combine scorers "without ever guessing/calibrating a cross-scorer exchange rate").

## Runtime truth check (done first, and it changed the design)

The 2026-07-07 spec targeted `orion:signals:*` at ~55/s.

**Correction (2026-08-14, after Juniper flagged the assumption).** An earlier version of this
section asserted "that lane is dead." That claim was not supported by the method used for it.
`redis-cli pubsub channels 'orion:signals*'` lists only channels **with subscribers** — a
channel being actively published with nobody listening is invisible to it, and this repo
already has exactly that pattern on record (`orion:spark:signal`, 0 subscribers). Checked
properly: **`orion-signal-gateway` is running** (`Up About an hour`), so the producer is
alive. What is true is narrower: no `orion:signals:*` traffic appears in the
`orion:bus:velocity:*` census, nothing subscribes, and no `OrionSignalV1` history table
exists — so there is no queryable history to build a baseline from.

That is the real reason this patch reads `substrate_field_state` instead, and it is a
sufficient one: **a deviation gate needs persisted history to learn baselines from, and only
the field has it.** "The signal lane has no queryable history" is a claim the evidence
supports; "the signal lane is dead" was not.

**What is actually rich and live** — `substrate_field_state` (Postgres `conjourney`):

| Fact | Value |
|---|---|
| Rows | 127,259 |
| Span | 2026-08-11 15:17 → 2026-08-14 15:47 (current to the minute) |
| Last 24h | 42,049 ticks (~0.5/s) |
| Nodes | 4 — `node:athena`, `node:atlas`, `node:circe`, `node:prometheus` |
| Channels | 33 per node |

This is the same "rich, real interoceptive field sitting right next to Drives, unused" that
`2026-07-17-field-native-motivational-substrate-design.md` named. This patch reads it.

**Retracted claim (2026-08-14).** An earlier version of this section reported that
`node:circe`'s channels being decayed to subnormal floats while `staleness` reads `0.0` and
`availability` reads `1.0` was "the generic-decay-loop pathology CLAUDE.md documents for
`node:substrate.route`." **That was wrong.** `staleness` and 27 other channels are in the
field digester's `NODE_DECAY_CHANNELS` — they are *designed* to decay at 0.92/tick toward
rest when nothing refreshes them. A node being refreshed normally drives `staleness` to
exactly this reading. The documented pathology is the **opposite** shape: `prediction_error`,
which was deliberately *removed* from that set on 2026-07-26 precisely because it must not
decay, being decayed anyway. See "Corrections" below.

## Current architecture

- `orion/attention/rank_aggregation.py` — generic Borda (`aggregate_borda`, `scorer_top1`,
  `BordaResult`). Pure, no domain deps, built for exactly a third consumer. Handles partial
  ballots ("silence is not a vote") and tie-averaging already.
- `orion/attention/field_attention/candidate_precision_weighted.py` — precision = 1/variance
  salience, and `normalize_across_targets()` which chose min-max over softmax explicitly to
  avoid a temperature hyperparameter.
- `substrate_field_state.field_json` — `node_vectors` (per-node channel values),
  `dimension_precision_ewma/_var/_zscore` over **4 aggregate capability dimensions only**
  (`resource/execution/reasoning/reliability_pressure`). **There is no per-(node, channel)
  deviation baseline anywhere.** This patch does not duplicate the existing one.
- No tension producer of any kind exists post-`d6a4e892b`.

## The design

### 1. Admission — magnitude, per (node, channel), scale-free by z-scoring

`DeviationGate` restored from `d6a4e892b^` essentially unchanged (it was already generic).
Per `(node, channel)` it holds an EWMA `(mu, var)` and a warm-up counter; an observation
admits only when it deviates past `z_threshold` in the channel's *worse* direction.

```
sigma  = max(sqrt(var), relative_sigma_floor * |mu|, 1e-12)
z      = (x - mu) / sigma
excess = relu(direction * z - z_threshold)
```

**The floor is relative, and that is load-bearing** (fixed in review — the restored
original used an absolute `sigma_floor = 0.02`). An absolute floor silently breaks the
scale-freedom this whole design rests on, in the worst direction: a channel whose real
variation is smaller than the floor can never admit anything, however large its *relative*
move. Live evidence — `node:atlas / memory_pressure` over 10,528 ticks ranges 0.1109–0.1171
(pstdev 8.8e-4), real continuous variation, and admitted **0 of 10,528** under the absolute
floor because clearing it needed a ~0.03 absolute jump, a 26% relative move. It then
appeared in the measurement's `never_admitted` list indistinguishable from a genuinely calm
channel: an instrument structurally incapable of reading the state it reports. Post-fix,
`memory_pressure` admits 539 times in the same 24h window.

This solves scale disparity **within a channel over time**: a `thermal_pressure` rise and a
`memory_pressure` rise both become "N sigma above this channel's own learned baseline."
Dimensionless. No cross-channel comparison, no weights.

A steady channel settles to its own mean and admits nothing — the flood-starving property.

### 2. Direction — structural, one bit per channel, no weights

`config/attention/channel_direction_map.yaml` is the whole mapping surface. It is the old
`signal_drive_map.yaml` **minus every `drives: {...}` block** — same suffix-rule shape
(structural match on typed channel-name suffix, no free text), none of the exchange rates.

Suffix rules cover 28 of 33 channels (`*_pressure`, `*_load`, `*_deficit`, `*_friction`,
`*_error`, `*_warning`, `*_incompletion`, `*_ratio` excluded), plus one explicit entry for
`staleness` (up is worse).

**The "down is worse" set is NOT in this file — it is derived from
`orion.field.pressure.HIGHER_IS_BETTER_CHANNELS`**, the polarity constant every existing
cognition consumer already uses (`orion/attention/field_attention/selectors.py`,
`orion/field/commensurability.py`). An earlier version hand-listed
`availability`/`delivery_confidence`/`stream_backlog_health` in the YAML, which was a
hand-re-derivation of that constant and a **strict subset** of it — silently missing
`confidence` and `available_capacity`, so this module would have disagreed with its own
package neighbours about two channels' polarity. The loader now seeds from the constant and
raises if the YAML contradicts it.

Two channels are **deliberately unmapped in v1** and contribute nothing:
`context_gathering_ratio` (no theory for which direction is worse) and
`expected_offline_suppression` (a static-config-driven suppression flag, not a tension —
CLAUDE.md records it cannot self-clear). An unmapped channel is silently inert by
construction; the map cannot be grown by prose, only by a typed entry plus a test.

### 3. Competition — rank space, so no exchange rate is ever needed

- **Targets** = nodes (`node:athena`, `node:atlas`, …) — the things that could win attention.
- **Scorers** = channels. Each channel ranks nodes by its own admitted deviation.
- **Combine** = `aggregate_borda`.

`cpu_pressure` never has to be priced against `reasoning_load`; each only orders nodes on its
own scale, and rank position is commensurable by construction. A channel with no admitted
deviation this tick submits an empty ballot and `aggregate_borda` already treats that as
abstention rather than a vote for last place.

**What rank space does to the tuning surface** — this is the point of the whole design:

| Parameter | Fate |
|---|---|
| `impulse_k` | **Cancels outright.** A monotonic scaling cannot change a rank. Removed from the gate's competition path. |
| `alpha` (EWMA memory) | One global scalar, derivable from channel autocorrelation time. |
| `relative_sigma_floor` | One global scalar, and *relative* — so it scales with each channel rather than acting as a hidden absolute admission threshold. |
| `z_threshold` | One global scalar with a single objective: admission rate. Not a per-signal weight. |
| `worse` direction | One bit per channel, semantically known and falsifiable — get it backwards and the metric reads inverted, visibly. |
| ~~signal→drive weights~~ | **Gone.** 69 lines of YAML replaced by zero. |

Total hand-authored surface: **three global scalars and one sign bit per channel** — none
of them a cross-channel exchange rate, which is the property that matters.

### 4. The honest limit, and where magnitude re-enters

Borda discards magnitude, so "everything calm, ranked" and "everything on fire, ranked" are
indistinguishable in the ranking alone. Absolute state has to live somewhere, and it lives in
the admission gate:

- **magnitude gates admission** — does anything deviate enough to compete at all?
- **rank decides priority** — among what was admitted, who wins?

Neither step needs a cross-channel exchange rate, and "nothing is happening" is a real
representable state (empty admission set) rather than something inferred from small numbers.
That is the failure mode that produced `bus_synaptic_prediction_error`'s permanent ~0.27
floor: an aggregate that could never structurally read calm.

## Metric quality gate (CLAUDE.md §0A, run before wiring, recorded here)

1. **Provenance.** Values come from `substrate_field_state.field_json.node_vectors`, written
   by `orion-field-digester`'s perturb→decay→diffuse pipeline. Traced to the live table, not
   a schema comment.
2. **Independence.** Admission z-scores are per-`(node, channel)` and are *not* a transform of
   the existing `dimension_precision_zscore`, which is computed over 4 aggregate capability
   dimensions, not per-channel. Different granularity, different population. Not redundant.
3. **Theory anchor.** Deviation-from-adapted-baseline admission is standard change detection;
   Borda rank aggregation (de Borda 1770) is the named social-choice method for combining
   incommensurable ballots, already the repo's chosen answer in two live subsystems.
4. **Live-data sanity.** Deferred to the measurement, on purpose — this patch's *entire
   deliverable* is that check. The measurement reports per-channel admission counts and flags
   degenerate (never-admitting, always-admitting) and decay-suspect channels rather than
   asserting the metric is healthy.
5. **Existing mechanism.** Searched: `rank_aggregation.py` reused rather than reimplemented;
   `DeviationGate` restored from history rather than rewritten; the existing 4-dimension
   precision EWMA confirmed to be a different thing at a different granularity.
6. **Reversibility.** Read-only. No schema registration, no bus channel, no consumer. Deleting
   the three modules and the YAML removes it completely.

## Files

- `orion/attention/tension/deviation_gate.py` — restored from `d6a4e892b^`, `impulse_k`
  removed from the competition path.
- `orion/attention/tension/direction_map.py` + `config/attention/channel_direction_map.yaml`
- `orion/attention/tension/field_observations.py` — `field_json` → observations.
- `orion/attention/tension/competition.py` — admitted deviations → `aggregate_borda`.
- `scripts/analysis/measure_field_tension_admission.py` — the measurement.
- `orion/attention/tension/tests/` — unit tests.

Home is `orion/attention/`, **not** `orion/autonomy/` — the seam is signals-in,
ranked-tension-out, attention-consumes. Putting it back in `autonomy/` is what made it
collateral damage last time.

## Non-goals

- No drive taxonomy, no categories, no buckets. Named drives, if they ever return, should
  fall out of competition history, not precede it.
- No bus publication, no schema registration, no prompt/Mind wiring, no action. Wiring
  tension to anything that *acts* is a separate patch requiring proposal sign-off.
- No claim that this produces motivation. It produces a continuously-varying tension input
  that the attention architecture has never had.

## Acceptance checks

1. `admission_rate` reported over ≥24h of real `substrate_field_state` history.
2. `top1_share` (most-frequent Borda winner's share) reported — the instrument that would
   have caught the 96% `relational` monoculture on day one.
3. Per-channel admission counts reported, with degenerate and decay-suspect channels named.
4. Unit tests: steady input admits nothing; step change admits; direction inversion admits
   nothing on the wrong side; monotonic rescaling of one channel does not change the Borda
   ranking (the scale-freedom property, asserted directly).

## Results (live, 2026-08-14, post-review)

`scripts/analysis/measure_field_tension_admission.py --hours 24` against real
`substrate_field_state`, **42,047 ticks**, 2026-08-13 16:15 → 2026-08-14 16:15 UTC.

These are the numbers **after** the review fixes below. The pre-fix run reported 36.47% /
53.13% with an absolute `sigma_floor`; those figures were depressed by the floor silently
excluding every small-variance channel, and are superseded rather than merely improved on.

| Metric | Value | Note |
|---|---|---|
| **Admission rate** | **48.30%** | vs **0.064%** drives-era baseline — ~755× |
| Mean admissions / admitting tick | 1.93 | |
| **Top-1 share** | **50.56%** | vs the 96% `relational` monoculture that killed the old economy |
| Distinct winners | 10 | `athena`, `atlas`, `circe`, `substrate.bus_synaptic`, `substrate.execution`, `substrate.biometrics`, `substrate.route`, `rpc_timeout`, `substrate.codebase`, `substrate.vision` |
| Scorer disagreement rate | 28.44% | channels genuinely disagree about what matters ~¼ of ticks |
| Channels admitting ≥once | 23 of 33 | |

**`z_threshold` sensitivity** (10,000 most-recent ticks), confirming it behaves as a single
knob with a monotonic, interpretable response rather than a weight needing calibration:

| `z_threshold` | Admission rate | Top-1 share | Disagreement |
|---|---|---|---|
| 1.5 | 49.94% | 0.520 | 0.289 |
| 2.5 | 24.03% | 0.488 | 0.155 |
| 3.5 | 13.50% | 0.462 | 0.092 |
| 5.0 | 10.11% | 0.451 | 0.069 |
| 8.0 | 6.49% | 0.378 | 0.032 |

Even at the most conservative setting tried, admission is **6.49% — 101× the drives-era
rate**. Discrimination *improves* as the threshold rises (top-1 share falls toward and past
0.5), so there is no admission-vs-monoculture tradeoff to split here.

### Live-data findings (corrected 2026-08-14 — the originals were false alarms)

**Every "decay-suspect" finding this spec originally published was wrong, and the corrected
instrument reports zero findings in that category.**

The original reported 4 series "mid-decay at ratio 0.92 — NOT calm, unmaintained" and 7
"bottomed out, pinned subnormal, reads as calm." Checked against
`services/orion-field-digester/app/digestion/decay.py`: **9 of 9 named channels are in
`NODE_DECAY_CHANNELS`** — `staleness`, `reasoning_load`, `conversation_load`,
`repair_pressure`, `reliability_pressure`, `avg_step_chars_pressure`,
`field_coherence_warning`, `harness_step_load`, `turn_incompletion`. All of them are
*supposed* to decay at 0.92/tick toward rest. A decaying `conversation_load` means nobody is
talking. That is correct behaviour, not a defect.

The detector had a **100% false-positive rate** on this data and was blind by construction to
the one shape that actually indicates a bug: decay on a channel *outside* the designed set,
which is exactly the `prediction_error` case CLAUDE.md documents. It has been inverted to
report only undesigned decay, and the by-design half is now counted separately as "not
findings."

Corrected output over the same 24h / 42k-tick window:

| | Value |
|---|---|
| **Undesigned decay (finding)** | **none** |
| **Undesigned pinned-subnormal (finding)** | **none** |
| By-design decaying (expected, not a finding) | 7 |
| By-design pinned (expected, not a finding) | 7 |

**Liveness, from the repo's own validated classifier.** The original also inferred channel
health from its own `never_admitted` list, which cannot distinguish "quiet by design" from
"telling you nothing." `orion.field.channel_glossary.classify_channel_series()` already
exists for exactly this, with a validated heuristic and the right vocabulary. Reused instead
of re-derived:

| Verdict | Series |
|---|---|
| `dead` | 94 |
| `live` | 34 |
| `quiet` | 21 |

Sampled at a 200-tick stride, **211 evenly-spread points across the full window**. This
matters: run against the decay probe's 12-sample tail instead, the same classifier reports
`live: 4, dead: 102, quiet: 40` — a short-window artifact that would have wildly understated
liveness. The stride sampling was added specifically to avoid publishing that.

**What the `never_admitted` list actually means.** 7 of its 10 entries
(`catalog_drift_pressure`, `compliance_deficit`, `contract_pressure`,
`observer_failure_pressure`, `staleness`, `stream_backlog_pressure`, `turn_incompletion`)
are in `NODE_DECAY_CHANNELS` and at their designed resting state. `availability` is
higher-is-better and pinned at 1.0 because everything is available. None of these is
evidence of a problem, and the original framing of them as concerns was unfounded.

### The real open defect in this area, which this instrument does NOT detect

Found while checking the above, already tracked in `orion/field/pressure.py`: when
`services/orion-biometrics` goes quiet, every input to `resource_pressure` decays toward 0,
the dimension reads *calm*, and because `config/feedback/feedback_policy.v1.yaml` lists
`resource_pressure: decrease` under `positive_delta_channels`, **the in-flight action is
credited with a positive outcome for a producer outage.** That is the genuine
decay-vs-calm ambiguity. It needs a producer-liveness guard on the dimension, not a ratio
check on the channel — this patch's detector would not catch it. Tracked in PR #1554's design
doc; not addressed here.

## Blast radius

- **Runtime: zero.** 14 files, 2,189 insertions, **no modifications to any pre-existing
  file**. Nothing in the repo imports `orion.attention.tension`. No schema registered, no bus
  channel, no service, no consumer.
- **Dependency direction is inbound only.** This package now *reads*
  `orion.field.pressure.HIGHER_IS_BETTER_CHANNELS`, and the measurement script reads
  `classify_channel_series` and `NODE_DECAY_CHANNELS`. Changing those affects this code, not
  the reverse — deliberately, so the polarity and decay semantics cannot drift into a private
  second opinion.
- **`config/attention/` already existed** (`field_attention_policy.v1.yaml`); this adds a
  sibling file, not a new convention.
- **The real blast radius was epistemic, not mechanical.** `substrate_field_state` is read by
  `orion-proposal-runtime` (→ `ProposalFrameV1`) and by
  `orion.field.pressure.collect_field_channel_pressures`, described in the glossary as the
  merged, correctly-polarized dict *every cognition consumer* reads. A merged spec asserting
  that designed decay channels were broken could have driven someone to "fix" the decay loop
  those consumers depend on — including re-adding `prediction_error` to a set it was
  deliberately removed from. That is the actual risk this correction closes.

## Continuing the arc: what is downstream actually waiting for?

With tension sensing measurable, the next question decides whether wiring it is worth
anything: **is Layer 5 attention currently choosing between competing inputs, or idling for
lack of any?** If it already selects richly, tension must prove it adds information. If it is
starved, tension is not competing with an incumbent — a different and much weaker claim to
have to defend, but far more useful to know first.

Measured with `scripts/analysis/measure_attention_input_starvation.py`
(`make attention-starvation-report`) against `substrate_attention_broadcast_log`:

| | 72h (8,594 frames) | 24h (2,865 frames) |
|---|---|---|
| Frames with ≥1 signal | 26.2% | 19.4% |
| **Frames with ≥1 open loop** | **5.7%** | **0.2%** (6 frames) |
| Frames with ≥1 candidate action | 5.7% | 0.2% |
| **Frames selecting any action** | **5.7%** | 0.2% |
| Action types ever selected | `defer` 387, `watch` 105, `none` 8,102 | — |

Three things fall out, and the third is the arc's actual finding:

1. **Candidate supply *is* open-loop supply.** `frames_with_any_open_loop`,
   `frames_with_any_candidate_action` and `frames_selecting_an_action` are the same number
   (492 over 72h) — attention selects exactly when it has an open loop, and never otherwise.
2. **It never acts.** Every selection in the window is `defer` or `watch`. Even with a
   candidate, the outcome is to not do anything yet.
3. **Its only candidate source is conversation.** `orion/substrate/attention/scoring.py:145`
   builds every `OpenLoopV1` from the current turn — `user_text`, `direct_turn`,
   `_EMOTION_RE.search(user_text)`, literal `"my "` / `"our "` / `"juniper"` token checks. The
   second producer (`services/orion-hub/scripts/attention_loops_store.py:256`) re-surfaces
   *stored* themes rather than generating candidates from internal state. **Nothing anywhere
   generates an attention candidate from interoceptive state.**

So the structure is: no user turn → no open loop → nothing to attend to → `none`, 94.3% of
the time over 3 days and 99.8% over the last day. Orion's attention architecture is
structurally reactive — it has something to attend to when Juniper is talking to it, and
essentially not otherwise.

That is the same starvation the drives program died of, one layer up, and it lands exactly on
the founding charter's bar: *"Orion must sometimes choose not to respond optimally in the
moment in order to preserve long-term coherence. This is where Orion stops being reactive and
starts being self-directed."*

**What this does and does not establish.** It establishes that tension would not be competing
with an incumbent selector — the competition currently has no entrants for all but a fraction
of frames. It does **not** establish that tension is *good* input: a high admission rate of
noise would be worse than an empty frame. It also does not explain why `defer`/`watch` are the
only outcomes ever selected; that is a separate gate, untraced here.

**Next rung, and the one this arc must not skip.** The drives program's fatal habit was
building instruments and never an outcome measure. Before tension supplies a single attention
candidate, the missing piece is: *what would count as this having helped?* An
interoceptively-sourced open loop is only worth building if there is a number that moves when
it works and does not when it doesn't. That number does not exist yet, and inventing candidate
supply before it exists would repeat the exact cycle this program was chartered to end.

## The outcome measure — built, and what it revealed

The rung this arc must not skip: *what would count as an interoceptively-sourced attention
candidate having helped?* Built before any candidate supply, deliberately — the drives
program's fatal habit was instruments with no way to tell whether they worked.

**An outcome mechanism already existed and was half-built.** `AttentionLoopOutcomeV1` has
always specified three verdicts. Two are human clicks in the Hub (`resolved`, `dismissed`).
The third, `decayed_unattended`, is documented in `orion/schemas/attention_salience.py` as
"the human verdict (Resolve/Dismiss) **or implicit decay** — the sparse-but-clean label the
refit later trains on", and `orion/substrate/attention/verdicts.py` already handles it
correctly on the read side.

**It has never been written once.** All 4 live rows are `resolved`/`juniper`. The label
stream is 100% hand-clicked — which cannot scale to judging a machine-generated candidate
stream, and is precisely why an outcome measure never materialised.

`orion/substrate/attention/implicit_outcome.py` supplies the missing producer: a loop that
was scored, never explicitly closed, and then stopped being re-scored has decayed unattended.
Real outcome, machine-derivable, and the negative class any refit needs. A human verdict is
never overwritten. The 24h floor reuses `attention_loops_store.py::suppress_loop`'s existing
cooldown rather than inventing a second constant.

Measured with `make attention-outcome-coverage`:

| | Value |
|---|---|
| Loops ever scored | **7** |
| Labelled | 4 (57.1%) — all human |
| Implicit labels | **0** |
| Derivable now | 2 (`open-loop-9d84d08cddf5`, silent **509h**; `open-loop-5038aeb46982`, 39h) |

### The finding: the outcome measure was never the bottleneck

7 distinct loops in 3 weeks, re-scored 803 times. Label coverage is 57% — **the labels are
not sparse, the candidates are.** Building a labeller for 3 unlabelled loops does not produce
an outcome measure anyone can learn from; it produces an instrument with a denominator of 3.

That is now the *third independent measurement* agreeing on one diagnosis:

| Measurement | Result |
|---|---|
| Drives-era tension firing | 0.064% of ticks |
| Layer 5 attention open loops | 0.2–5.7% of frames |
| Distinct salience-scored loops | **7, in 3 weeks** |

Every layer of this stack is starved at the input. The tension gate (48% admission, 10
distinct winners) is the first thing in the chain that produces candidates at a real rate.

### Disclosed: one knob here is currently inert

`cadence_multiple` produces **identical label counts at 1× / 3× / 10×** across every floor
tried, because all 7 loops were scored in tight bursts (median gap ~0h) so the absolute floor
always binds. Kept — it is a real guard with a real regression test, and slow-cadence loops
are exactly what an interoceptive stream would produce — but it is **not validated**, and if
it is still inert once candidate diversity exists it should be deleted rather than kept as a
knob that provably does nothing. Same call already made for `DeviationGate.impulse_k`.

### Not emitted

`--emit` exists, snapshots first, never overwrites a human verdict, and is idempotent. **It
was not run.** Writing 2 labels into a table a future refit trains on is a data write to a
learning surface with near-zero information value at this denominator. The deliverable is
that the measure now exists and will start producing labels the moment candidate supply does.
