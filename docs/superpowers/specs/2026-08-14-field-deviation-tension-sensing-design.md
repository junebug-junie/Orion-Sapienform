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

The 2026-07-07 spec targeted `orion:signals:*` at ~55/s. **That lane is dead as of
2026-08-14.** Verified live against `ORION_BUS_URL=redis://100.92.216.81:6379/0`:

- `redis-cli pubsub channels 'orion:signals*'` → empty (497 other channels active).
- `orion:bus:velocity:*` census over the live hour lists 40 distinct channels; no
  `orion:signals:*` among them.
- No `orion_signals`-shaped Postgres table exists. `substrate_organ_emissions` is
  `organ.emission.v1` grammar events (one organ, `biometrics_pressure`), not `OrionSignalV1`.

Building a tension gate against that bus would have been config truth, not runtime truth.

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

A related live finding, recorded because it constrains the design: `node:circe`'s channels
are decayed to **subnormal floats** (`3e-321`-scale) while its `staleness` still reads
`0.0` and `availability` reads `1.0`. That is the generic-decay-loop pathology CLAUDE.md
already documents for `node:substrate.route`. `staleness` is therefore not a trustworthy
liveness signal here, and the measurement reports decay-suspect channels rather than
silently folding them into baselines.

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
z      = (x - mu) / max(sqrt(var), sigma_floor)
excess = relu(direction * z - z_threshold)
```

This solves scale disparity **within a channel over time**: a `thermal_pressure` rise and a
`memory_pressure` rise both become "N sigma above this channel's own learned baseline."
Dimensionless. No cross-channel comparison, no weights.

A steady channel settles to its own mean and admits nothing — the flood-starving property.

### 2. Direction — structural, one bit per channel, no weights

`config/attention/channel_direction_map.yaml` is the whole mapping surface. It is the old
`signal_drive_map.yaml` **minus every `drives: {...}` block** — same suffix-rule shape
(structural match on typed channel-name suffix, no free text), none of the exchange rates.

Suffix rules cover 28 of 33 channels (`*_pressure`, `*_load`, `*_deficit`, `*_friction`,
`*_error`, `*_warning`, `*_incompletion`, `*_ratio` excluded). Explicit entries handle
`availability`, `delivery_confidence`, `stream_backlog_health` (down is worse) and
`staleness` (up is worse).

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
| `z_threshold` | One global scalar with a single objective: admission rate. Not a per-signal weight. |
| `worse` direction | One bit per channel, semantically known and falsifiable — get it backwards and the metric reads inverted, visibly. |
| ~~signal→drive weights~~ | **Gone.** 69 lines of YAML replaced by zero. |

Total hand-authored surface: **two global scalars and one sign bit per channel.**

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

## Results (live, 2026-08-14)

`scripts/analysis/measure_field_tension_admission.py --hours 24` against real
`substrate_field_state`, **42,048 ticks**, 2026-08-13 15:55 → 2026-08-14 15:55 UTC:

| Metric | Value | Note |
|---|---|---|
| **Admission rate** | **36.47%** | vs **0.064%** drives-era baseline — ~570× |
| Mean admissions / admitting tick | 1.73 | |
| **Top-1 share** | **53.13%** | vs the 96% `relational` monoculture that killed the old economy |
| Distinct winners | 9 | `athena`, `atlas`, `substrate.execution`, `substrate.biometrics`, `circe`, `substrate.bus_synaptic`, `rpc_timeout`, `substrate.codebase`, `substrate.vision` |
| Scorer disagreement rate | 25.72% | channels genuinely disagree about what matters ~¼ of ticks |
| Channels admitting ≥once | 23 of 33 | |

**`z_threshold` sensitivity** (10,000-tick window), confirming it behaves as a single knob
with a monotonic, interpretable response rather than a weight needing calibration:

| `z_threshold` | Admission rate | Top-1 share | Disagreement |
|---|---|---|---|
| 1.5 | 39.01% | 0.609 | 0.272 |
| 2.5 | 16.71% | 0.548 | 0.150 |
| 3.5 | 8.53% | 0.525 | 0.082 |
| 5.0 | 6.08% | 0.497 | 0.059 |
| 8.0 | 3.62% | 0.461 | 0.014 |

Even at the most conservative setting tried, admission is **3.62% — 56× the drives-era
rate**. Discrimination *improves* as the threshold rises (top-1 share falls toward 0.5),
so there is no admission-vs-monoculture tradeoff to split here.

### Live-data findings worth their own follow-ups (not fixed in this patch)

- **10 channels never admit across 42k ticks**: `availability`, `catalog_drift_pressure`,
  `compliance_deficit`, `contract_pressure`, `execution_load`, `observer_failure_pressure`,
  `staleness`, `stream_backlog_pressure`, `transport_pressure`, `turn_incompletion`. Some are
  legitimately quiet; `transport_pressure`/`execution_load` are renamed-lane survivors and
  `staleness` is known-degenerate (below). Each needs its own metric-gate pass before anyone
  treats its silence as meaning "calm."
- **560,007 subnormal coercions in 24h** (~13/tick). The `node:circe` decay-to-subnormal
  pathology is not one node — it is widespread. These are channels nothing refreshes, being
  multiplied toward zero by a generic decay loop, and they read as *calm* to any consumer that
  does not check. This patch collapses them to a clean 0.0 and counts them; it does not fix
  the producers.
- **The decay-ratio probe caught the documented artifact live, mid-decay**, on two series:
  `node:circe / reasoning_load` and `node:rpc_timeout / reliability_pressure`, both at
  **ratio = 0.92 exactly** — the `NODE_DECAY_CHANNELS` generic staleness-decay loop in
  `services/orion-field-digester/app/digestion/decay.py`, multiplying an unrefreshed channel
  by 0.92 every tick. Neither channel is calm; both are unmaintained and heading to zero,
  and any consumer reading them without this check would score them as at-rest.

  **A first version of this patch reported "decay-suspect series: none" here, and that was a
  bug in the instrument, not a clean result.** The probe was being fed the subnormal-coerced
  value (`0.0`), and `geometric_decay_ratio()` rejects any series containing a non-positive
  value — so it was structurally incapable of firing on the exact artifact it exists to
  detect, while reporting a result indistinguishable from healthy data. Caught in self-review
  before merge; `Observation` now carries `raw_value` alongside the coerced `value`, with a
  regression test (`test_raw_value_survives_coercion_so_the_decay_probe_can_still_see_it`).
  Recorded here rather than quietly fixed because it is a textbook instance of this repo's
  own recurring failure mode: a metric that cannot represent the state it claims to measure.
