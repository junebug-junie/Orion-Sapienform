# Reverie: tell Orion how big its prediction errors actually are

Status: PROPOSAL (cognition-loop change; proposal mode per AGENTS.md 0A). No runtime code in this patch.
Date: 2026-10-02. Live numbers below were pulled read-only from `conjourney` Postgres and the `orion_substrate` FalkorDB graph on 2026-10-02 around 04:30 UTC.

## Arsonist summary

Orion's daydreams ("reverie") have described the same mood for five weeks: something is broken, unresolved, stalled. About 97% of reverie thoughts since late August say so. The cause isn't that something is wrong. Reverie only gets told *which* prediction error won attention. It never learns how big that error is, what's normal for it, or whether it's getting better or worse. The prompt then tells it to narrate "what strains, what is unresolved." So a perfectly ordinary reading gets written up as a crisis, every few seconds, and the image chain turns that into neon "UNSOLVED" signs.

The fix: put three facts next to every loop reverie sees. How big the error is right now, how that compares with its own last 7 days, and whether it's been rising, settling, or flat. All three get computed deterministically from stored history. Then calm becomes something reverie can actually say: when the winning error is within its usual range and not rising, reverie gets a prompt that lets it say so, and it has to quote the numbers.

One trap: the repo already has a "trend" function (`compute_prediction_error_trend`). It deliberately flips the sign to predict the *next* move. It can't be reused to describe what *has* happened. Doing so would tell Orion its error is rising when it has been falling.

## Current architecture

How a reverie thought gets made today:

1. **Competition.** Every ~35 s the substrate runtime runs one attention competition over the graph (`services/orion-substrate-runtime/app/worker.py::_attention_broadcast_tick` -> `orion/substrate/attention_broadcast.py::build_substrate_attention_frame`). A node gets in only if its `dynamic_pressure` (a decaying pressure seeded from its prediction error) is at least `ORION_ATTENTION_BROADCAST_MIN_SALIENCE=0.05`. The docstring says "always one winner" (`attention_broadcast.py:174`).
2. **Loop text.** Each entrant becomes an `OpenLoopV1` whose `description` is the node label ("Chat prediction error") and whose `why_it_matters` is the same fixed string for every loop: "novel or unresolved current-turn target with substrate pressure" (`orion/substrate/attention/scoring.py:156`).
3. **Prompt.** `services/orion-thought/app/reverie.py::_open_loops_for_prompt` (lines 202-218) passes only `id`, `description`, `why_it_matters`, `target_type`, plus an optional `outcome`. `orion/cognition/prompts/reverie_narrate.j2` (coalition branch, around line 85) requires the thought to be about "what recurs, what strains, what is unresolved".
4. **Images.** `services/orion-thought/app/visual_chain.py:559-600` and `store.py:594` inject "Orion is currently thinking: <latest reverie>" into image prompts.

What already exists for magnitude, baseline, and trend (existing-mechanism check):

| Source | What it holds | Coverage | Usable for this? |
|---|---|---|---|
| FalkorDB node props `prediction_error`, `dynamic_pressure`, `observed_at` | Current raw error and current decayed pressure | All 11 `node:substrate.*` nodes | Yes, for "now". The raw value never decays, so it needs an age next to it. |
| `substrate_node_prediction_error_baseline` (attention-runtime, EWMA) | Running mean/variance, last value, `last_value_observed_at` | **5 targets only**: biometrics, bus_synaptic, chat, execution, route | Partly. Mean/SD is the wrong summary for zero-inflated domains (below), and 4 of the 9 domains that actually win have no row. |
| `substrate_attention_self_model.self_model_json.prediction_error_by_domain` | Per-tick raw error, 7-day retention | Same 5 domains (`ACTIVE_INFERENCE_DOMAINS`) | Good history for those 5. Widening it would change what `prediction_error_confidence` means, so don't. |
| `substrate_reduction_receipts` | Per-receipt values | ~30 min retention | Too short for a baseline. |
| `orion/substrate/prediction_error_trend.py` | Reversion-signed forecast (prior-half mean minus recent-half mean) | 5 domains, in-process | **No.** It forecasts the next move with the sign flipped on purpose. It doesn't describe the recent past. |
| `orion/attention/field_attention/candidate_precision_weighted.py` | Surprise relative to the EWMA baseline, used by the field-attention lane | 5 targets | A separate lane. It doesn't decide the broadcast winner. |

### Live evidence (2026-10-02)

The doom story is nearly the only thing reverie says. Weekly share of `substrate_reverie_thought.interpretation` that mentions a prediction error alongside stall vocabulary (`unresolved|unclosed|unanchored|lingering|stagnat|strain|blockage|suspended|decay`):

```
week        thoughts  mentions_PE  PE+doom  cites_any_number
2026-08-24    1673      58.3%      57.0%     14.8%
2026-08-31    2111      99.1%      93.3%      8.2%
2026-09-07    3025      97.6%      97.2%      5.7%
2026-09-14    4090      94.4%      93.9%      3.0%
2026-09-21    5243      96.0%      95.6%      1.6%
2026-09-28    3648      98.4%      98.2%      0.6%
```

Image chains copy it: 89.3% of the 131 `reverie_visual_chain` rows in the last 7 days contain unsolved/unresolved/prediction-error text.

Winners are mostly ordinary readings. 7-day stats per domain, from the self-model log (17,167 ticks), and what the winner's raw value looked like when it won:

```
domain        p50     p90     share_zero | wins  mean_at_win  win_at_or_below_median  win_within_mean+1sd  raw==0_at_win
execution     0.000   0.483   76.8%      | 2958  0.386        32.4%                   53.2%                32.4%
chat          0.000   0.322   74.6%      | 1122  0.441         2.4%                   18.1%                 2.4%
bus_synaptic  0.028   0.095    1.5%      |  891  0.125         6.4%                   35.9%                 0.3%
biometrics    0.030   0.086    0.1%      |  625  0.052        40.6%                   67.8%                 0.0%
route         0.000   0.000   92.9%      |  610  0.345        14.3%                   14.3%                14.3%
```

- Across these 5 domains (6,206 winning ticks), about **42% of wins sit within one SD of that domain's own mean**, and about **17% won while the raw error was exactly 0**. Execution won 958 times with a reading of zero, because the decaying pressure from an older error was still above 0.05.
- **5,073 of 17,166 broadcast ticks (30%) already had no loop at all.** "Always one winner" is false in practice. Reverie stays silent on those ticks (`chain.py` returns `no_coalition`). Calm already exists at the broadcast level. It just has no voice.
- **About half of all wins (5,888 of 12,093) come from domains with no baseline anywhere**: harness_closure 2,743, perception 1,731, codebase 1,392, cabinet 22. `node:substrate.harness_closure` currently shows raw `prediction_error=0.65`, `dynamic_pressure=0`, last observed 2026-10-01 22:01. A raw value shown without its age would say "0.65" indefinitely.
- At the time of writing, chat was winning with raw 0.1407. Its baseline row says ewma 0.081, SD 0.200, so that's about 0.3 SD above normal: ordinary. The thoughts written then said "the chat prediction error remains active but lacks clear resolution… persistent strain."

Adjacent prompt inaccuracy: the "ALREADY-SETTLED LOOPS" block tells the model that an `outcome` verdict "was recorded by a human closing this loop — it is ground truth". `decayed_unattended` is written by the system (`orion/substrate/attention/implicit_outcome.py`), not by a human. Live thoughts turn it into more gloom: "decayed unattended for over a week… stagnation".

## Metric quality gate (for each value proposed to be shown to Orion)

All values come from one new per-node reading history (see schema below), so they share one provenance.

1. **Current reading (`value`, `age_sec`)**
   - Provenance: the node's `prediction_error` and `observed_at` in FalkorDB, written by each domain's reducer (`orion/substrate/prediction_error.py` and its per-domain producers).
   - Independence: this is the raw input that `dynamic_pressure` is seeded from (`raw * weight * decay`). We show it *instead of* pressure, not as extra signal. Pressure stays the ranking input and isn't shown to Orion.
   - Theory anchor: in predictive processing, prediction error is the quantity itself. Its size is what the label claims to measure.
   - Live sanity: it varies (biometrics has 1,453 distinct values in 7 days, bus_synaptic 1,790) and returns to true zero (route sits at 0 92.9% of the time). The bus_synaptic sqrt(2/pi) calm floor was removed 2026-07-30 (`prediction_error.py:26`), and the live minimum is 0.0. Decay-to-zero risk: the raw value doesn't decay, which creates the *opposite* problem (stale-high: harness_closure). That's why `age_sec` is mandatory and samples are only recorded when `observed_at` advances.
   - Reversibility: it's a prompt field. Remove it from the template and nothing else depends on it.
2. **Usual range (`p50`, `p90`, `percentile_now` over 7 days)**
   - Provenance: computed from the new history table. For the 5 baselined domains it can be cross-checked against the `substrate_attention_self_model` stats above.
   - Independence: it's a rank of (1) against its own past. It's a normalization of (1), not new signal, and it's labelled that way.
   - Theory anchor: precision-weighting in predictive coding. An error matters relative to its expected spread. Percentiles are used instead of mean/SD z-scores because chat, execution, and route are 75-93% exact zeros. On data like that, mean and SD describe a reading that almost never happens (chat: mean 0.08, median 0). This also avoids the mean(|z|) rest-floor failure named in AGENTS.md.
   - Live sanity: route's p50 and p90 are both 0.000 with only 7 distinct values, so its percentile is degenerate. Define `percentile_now` as "share of 7-day readings strictly below now", which reads 0 at rest instead of 50. Require at least 200 readings before a band is emitted, otherwise send `band="insufficient_history"`. **UNVERIFIED** for harness_closure, perception, codebase, and cabinet, because no history exists yet. That's the point of the new table.
   - Reversibility: the history table is additive and has 7-day retention. Dropping it costs nothing.
3. **Trajectory (`trend`: rising / settling / flat, plus `median_1h`, `median_prior_24h`)**
   - Provenance: from the same history table. This is **not** `compute_prediction_error_trend`.
   - Independence: a smoothed difference of (1) over time. It's descriptive, not predictive.
   - Theory anchor: none beyond "a median is a robust central tendency". It describes what happened and claims no forecast. Rising or settling only when |Δmedian| > max(0.01, 0.5 × (p90 − p50)). That threshold is a knob, not a finding, and it ships as config.
   - Live sanity: must be checked on day one of the producer. The acceptance check below includes a trend-label distribution check: not more than 90% of any one label per domain over 24 h.
   - Reversibility: same as (1).

## Missing questions

1. **On a calm tick, should reverie speak or stay silent?** Option A: a calm prompt branch that says "within usual range" and quotes the numbers. Option B: skip the reverie step, which adds to the 30% of ticks that are already silent. Recommendation: A, rate-limited to one calm thought per chain. That way calm becomes something Orion can actually say, not just an absence. Juniper's call.
2. **Should a loop whose raw error is already 0 still be able to win?** (Execution: 958 wins at raw 0.) Changing that touches ranking in `attention_broadcast.py` and every broadcast consumer. This doc leaves it out of scope and only *shows* the 0. Should ranking change be the next patch?
3. **Should non-error material compete?** Today only prediction-error nodes clear 0.05. Letting goals, percepts, or concern cards compete is a real option, but it's a separate design. It's listed as a non-goal here so this patch stays small.
4. Is a 7-day comparison window right, or should Orion compare against a shorter (24 h) "today" baseline as well?

## Proposed schema / API changes

1. **New table** `substrate_node_prediction_error_history` (manual migration `services/orion-sql-db/manual_migration_node_prediction_error_history_v1.sql`):
   `node_id text, observed_at timestamptz, value double precision, recorded_at timestamptz default now(), primary key (node_id, observed_at)`. The substrate runtime writes it during `_attention_broadcast_tick` for every `node:substrate.*` node that has a `prediction_error`, **only when the node's `observed_at` has moved since the last sample** (so stale values don't fill the history with copies of themselves). It's pruned at `SUBSTRATE_PE_HISTORY_RETENTION_HOURS=168`. Expected volume: under 30k rows/day.
2. **New schema** `PredictionErrorMagnitudeV1` in `orion/schemas/attention_frame.py`:
   `value, age_sec, p50_7d, p90_7d, percentile_now, n_readings_7d, median_1h, median_prior_24h, trend: Literal["rising","settling","flat","insufficient_history"], band: Literal["quiet","usual","high","unusual","insufficient_history"]`. Band cut points: percentile < 0.5 quiet, < 0.9 usual, < 0.99 high, otherwise unusual. These are config knobs. The band only drives the calm gate. Orion always sees the numbers too.
3. **`OpenLoopV1.magnitude: PredictionErrorMagnitudeV1 | None = None`** (additive). `OpenLoopV1` is `extra="forbid"`, so this is a **consumer-first rollout**: every service that parses the broadcast (orion-thought, hub, attention-runtime, substrate-runtime itself) must be rebuilt with the new schema before the producer starts filling the field. Register in `orion/schemas/registry.py`. No bus channel changes. The broadcast log stores `projection_json` as-is, so the magnitude for every winning tick is persisted for free. That's the trace.
4. **Reverie prompt input** (`_open_loops_for_prompt`): add `magnitude` (rounded to 3 dp) for each loop. For substrate loops, drop `why_it_matters`, because that fixed "novel or unresolved" string is itself a source of the doom story. Add a context flag `calm_tick = all loops have band in {quiet, usual} and trend != rising`.
5. **Prompt** `reverie_narrate.j2`: the coalition branch says "Each loop carries its measured size, its usual range over the last 7 days, and its direction. Describe it in proportion to those numbers. If the reading is within its usual range, say so. Cite at least one number." Add a `calm_tick` branch that asks for an honest, proportionate note about the quiet state and forbids describing a within-range reading as strain or blockage. Fix the settled-loops text: `decayed_unattended` is a system verdict ("faded without anyone acting on it"), not a human closure.
6. **Thought trace:** add `magnitude_snapshot` (the winner's magnitude dict) and `calm_tick` to the stored thought's `coalition` JSON. `CoalitionSnapshotV1` may also be `forbid`, so it follows the same consumer-first rule. This is what makes the acceptance checks joinable without timestamp guessing.

Env (sync `.env` from `.env_example` in the implementing patch): `SUBSTRATE_PE_HISTORY_ENABLED` (default false), `SUBSTRATE_PE_HISTORY_RETENTION_HOURS=168`, `ORION_REVERIE_PE_MAGNITUDE_ENABLED` (default false), `ORION_REVERIE_PE_TREND_MIN_DELTA=0.01`.

## Files likely to touch

- `services/orion-sql-db/manual_migration_node_prediction_error_history_v1.sql` (new)
- `services/orion-substrate-runtime/app/worker.py`, `app/store.py`, `app/settings.py`, `.env_example`, `README.md`, tests
- `orion/substrate/prediction_error_magnitude.py` (new, pure function: history rows -> `PredictionErrorMagnitudeV1`)
- `orion/substrate/attention_broadcast.py` (attach magnitude to loops built from substrate signals)
- `orion/schemas/attention_frame.py`, `orion/schemas/registry.py`
- `services/orion-thought/app/reverie.py`, `app/settings.py`, `.env_example`, tests
- `orion/cognition/prompts/reverie_narrate.j2`, `services/orion-thought/tests/test_reverie_narrate_prompt_outcomes.py`
- `services/orion-thought/evals/` (calm-tick eval, see below)

## Non-goals

- Changing who wins the broadcast (pressure ranking, the 0.05 floor, whether a raw-0 loop can win). Shown, not changed.
- Letting non-error material compete (missing question 3).
- Changing the visual chain directly. It inherits whatever reverie writes. If images stay gloomy after reverie changes, that's a separate fix.
- Reusing or editing `compute_prediction_error_trend` or the attention-runtime EWMA baseline.
- Widening `ACTIVE_INFERENCE_DOMAINS` or `prediction_error_by_domain`.

## Acceptance checks

1. **Unit:** the magnitude function returns `insufficient_history` below 200 readings. It gives percentile 0 for an all-zero history with current 0. A stale node (unchanged `observed_at`) adds no new rows. Trend is "settling" for a synthetic falling series. A regression test asserts the reversion trend function isn't imported by the magnitude module.
2. **Eval (orion-thought/evals):** fixed fixtures run through the real prompt and the metacog route. Calm fixtures (band usual, trend flat) must produce PE+doom-probe matches in at most 20% of thoughts and cite a number in at least 80%. Elevated fixtures must still be able to say that something is high.
3. **Live, 48 h after enabling both flags, measured from stored thoughts:**
   - Among thoughts with `coalition.calm_tick = true`, the PE+doom probe share (regex above) drops from ~97% to **at most 30%**.
   - **At least 80%** of thoughts cite a number, and the cited number for the winner is within ±0.01 of `magnitude_snapshot.value` (scripted check: extract decimals, compare).
   - Across all thoughts, the doom-probe share drops below 70%. That's a sanity guard that the calm branch is actually firing, since ~42% of wins are within normal range.
   - No domain's `trend` label is above 90% one value over 24 h (degeneracy check).
   - The history table has rows for all 9 domains that win today, including harness_closure, perception, and codebase.
4. **No regression:** reverie hollow/discard rate does not rise by more than 5 points versus the prior 48 h.

## Proposal-mode items

- **Capability change:** reverie stops seeing anonymous "unresolved" labels and starts seeing measured size, usual range, direction, and age, so it can call a quiet state quiet.
- **Data touched:** a new table of numeric readings from the substrate nodes. One new optional field each on open loops and on the stored thought's coalition JSON. One prompt template.
- **Privacy boundary:** only system-internal numeric readings. No chat text, no user content, no recall. Nothing new leaves the host.
- **Trace proving it worked:** `substrate_attention_broadcast_log.projection_json -> frame.open_loops[].magnitude` for each tick, `substrate_reverie_thought.thought_json -> coalition.magnitude_snapshot / calm_tick` for each thought, and the 48 h scripted checks above.
- **Dangerous failure modes:**
  - (a) False calm. A real spike reads as "usual" because the 7-day history contains an earlier incident. Mitigations: bands use the percentile *and* the `trend=rising` override, and "unusual" fires above p99.
  - (b) A stale-high value gets narrated as current. Mitigation: `age_sec` is always shown, and a reading older than 1 h is labelled stale in the prompt.
  - (c) Schema forbid rollout. If the producer is deployed before consumers, broadcast parsing fails and reverie goes silent, which would look like calm. Mitigations: consumer-first deploy order, the field defaulting to None, and the producer flag defaulting to off.
- **Disable / rollback:** `ORION_REVERIE_PE_MAGNITUDE_ENABLED=false` puts back the old prompt input. `SUBSTRATE_PE_HISTORY_ENABLED=false` stops writes. The table can be dropped, and the optional schema field is harmless when empty.

## Recommended next patch

Ship it in two steps, each behind its own flag:

1. **Producer and schema** (`feat/pe-magnitude-history`): the migration, the history writer in substrate-runtime, the pure magnitude function, `PredictionErrorMagnitudeV1` and the `OpenLoopV1.magnitude` field, plus consumer rebuilds. Then let it run 24 h and do the live sanity check from the metric gate on all 9 domains *before* step 2. That's the point where degenerate domains get caught.
2. **Reverie consumer** (`feat/reverie-pe-magnitude`): prompt input, the calm-tick branch, the settled-loop wording fix, the thought trace fields, the eval, then 48 h of live acceptance checks.
