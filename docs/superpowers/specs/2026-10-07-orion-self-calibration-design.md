# Self-calibration: give every number Orion reads about itself a sense of scale

Status: **design + proposal mode. Not implemented.** This changes attention ranking and what Orion
reads about its own state, both cognition / self-model surfaces, so CLAUDE.md §0A requires an
explicit proposal before implementation. It subsumes the unshipped steps 2–3 of
`docs/superpowers/specs/2026-10-02-reverie-prediction-error-magnitude-proposal.md` (approved
2026-10-02; only step 1 shipped) and is the "reader" for `orion_metacog` that Juniper asked about
on 2026-10-07.

## Arsonist summary

Orion fixates on prediction error. It isn't choosing to. Every number it reads about itself comes
without a scale, so "a little" and "a crisis" look the same.

- **Attention always crowns a winner at maximum importance.** Over 3 days, live:
  - 124,181 of 124,181 attention frames had a winner at salience exactly 1.0.
  - 49% of those winners were calm (raw error < 0.05).
  - The bus node won 42% of all frames at a median raw error of 0.035, basically resting.
  - Cause: `normalize_across_targets` min-max rescales so the top candidate is always 1.0
    (`orion/attention/field_attention/candidate_precision_weighted.py:355`). The open-loop salience
    Orion reads every turn is a Borda rank, so the worst loop is always 0 and the best always about 1
    (`orion/substrate/attention/salience.py:170-175`).
  - So the least-calm of a calm set always reads as the most important thing in Orion's world.
- **Many self-signals are stuck, replayed or dead, and presented as if they were live:**
  - harness_closure prediction error has been exactly 0.65 for 5 days.
  - One repair-pressure spike (level 0.913) has been replayed into curiosity 5,251 times.
  - A curiosity candidate source is constant 1.0.
  - The spark valence/arousal/coherence/novelty rollups are all zero (`pct_missing = 1.0`).
  - The "flow" trigger fires on the resting state: it triggers at confidence ≥ 0.9, and its median
    is 0.98.
- **Some "scores" are an LLM's own guesses rendered as if they were measurements.** These are the
  Mind frontier score and the curiosity prior confidence.
- **Some aggregates hide the signal:**
  - `strain` is a mean, so one pegged channel is averaged away.
  - distress/zen rest at 0.025/0.975 and barely move (p95 0.037). That is why every metacog trigger
    says "zen".
- **The fix already exists in one place and is thrown away.**
  - `PredictionErrorMagnitudeV1` (`orion/substrate/prediction_error_magnitude.py`) computes each
    node's 7-day percentile, band (quiet / usual / high / unusual) and trend every tick, and attaches
    it to open loops (`attention_broadcast.py:258-272`).
  - No prompt renders it: reverie strips it (`services/orion-thought/app/reverie.py:202-218`), and
    the chat frame builds its loops separately.
  - The transport baseline gate (`orion/metacog/transport_baseline.py`) is the other working model:
    per-key learned normal, floor, materiality, regime-shift.

**The proposal.** One calibration layer that every self-signal passes through before Orion reads
it. Each reading gets three things:
1. **How unusual it is,** against its own history, never against its neighbours.
2. **Which way is bad.**
3. **What it tends to lead to.** This is learned, and comes in phase 2.

Readers come next: attention ranks by real unusualness, chat/stance and reverie get proportionate
readings, and metacog rows become "this signal left its normal range" / "came back". The stream of
consciousness narrates that calibrated state, not ranks.

Not building a cathedral: no signal enters without a producer, a live sanity check and a reader.

## Current architecture (verified 2026-10-07)

### What reaches Orion's prompts (code trace)

| Signal (frequency) | Reaches | Presented as today | Problem |
|---|---|---|---|
| Open-loop `salience` and features (every stance turn) | `stance_react.j2:75`, `chat_stance_brief.j2:24` | Borda rank 0–1 | **relative**: calm and crisis look identical |
| Attended prediction-error node labels ("Biometrics prediction error") (every stance turn, every reverie tick) | `coalition_projection`, `reverie_narrate.j2` | chosen by min-max, top = 1.0 | **relative**: calm node named as the focus |
| `mind_coloring.attention_frontier.score` / uncertainty (stance) | `stance_react.j2:44,51` | raw | an LLM's self-assessment shown as a measurement |
| Curiosity prior `confidence` (every chat turn, situation brief) | `orion/situational/context.py:2293` | raw | LLM-written belief, no history shown |
| `coalition_stability_score` / `dwell_ticks` (reverie) | `reverie.py:258-266` | 3 buckets (0.9/0.6/0.3) | a bucket dressed as a score |
| Node `prediction_error` in `recent_trend_signals_json` (metacog) | `executor.py:4144` | raw float | no range |
| `strain` / `peak` / `power` (metacog biometrics cue) | `executor.py:783-804` | mean / raw / EWMA band | the mean hides a pegged channel; absolute watts lost |
| Bus transport "% anomalous" (Mind) | `recall_signal_resolver.py:300-385` | raw % plus learned EWMA rungs | partly calibrated |
| Queue pressure "N/10 (band)" (hire turns) | `queue_contention_disclosure.py` | **EWMA 0–10 + band** | already calibrated (model) |
| Metacog transport severity | `evidence_map.py` | **EWMA-z episodes** | already calibrated (model) |

Fetched but never rendered: `OpenLoopV1.magnitude` (the per-node percentile/band/trend) and reverie
salience. Dead path: the `self_state {dim} score` hazard (nothing writes `self_dimension_id`), and
`autonomy_slice` drive/tension (always empty).

### Live ranges (read-only, 2026-10-07; full tables in the PR report)

**Prediction-error nodes** (`substrate_node_prediction_error_history`, about 5 days; raw error 0–1,
higher is worse):

| Node | Typical (median) | p95 | Rest / behaviour |
|---|---|---|---|
| biometrics | 0.025 | 0.096 | continuous; usable |
| bus_synaptic | 0.029 | 0.067 | ~0.01 floor, rarely exactly 0; usable |
| perception, cabinet, execution, route, codebase, chat | 0 | 0.22–0.83 | 70–95% exact zeros. Need a "no reading" state distinct from 0 |
| harness_closure | 0.65 | 0.65 | **stuck constant** |
| vision | — | — | 1 reading; dead |

**Other self-signals:**

| Signal | Typical (median) | p95 | Rest / behaviour |
|---|---|---|---|
| distress / zen | 0.025 / 0.975 | 0.037 / — | near-flat; distress can't express load |
| strain | 0.16 | 0.30 | 0.08–0.57; continuous, healthy; usable |
| thermal | 0.14 | 0.49 | usable |
| `recent_perturbation_zscore` | 0.33 | 1.07 | properly centred; usable |
| self-model confidence | — | — | 3 discrete values |
| `prediction_error_confidence` | 0.98 | — | at ceiling; the flow trigger fires on rest |
| `heartbeat_smear` | 2.6 | 70 | max 1.5M; needs clipping or a log scale |
| repair_pressure level / confidence | 0.087 / 0.65 | — | floor / pinned |
| `sustained_load_pressure` | — | — | bimodal 0 / ~0.81; a switch, not a measure |
| curiosity strength (repair source) | 0.913 | — | replayed spike |
| curiosity strength (no source) | 1.0 | — | constant |
| spark rollups | 0 | — | dead |
| transport gate | z ≈ 0 at rest | — | calibrated (reference) |

**Retention constraint.** `substrate_attention_frames` and `substrate_field_state` keep about 3
days, and prediction-error history about 5. A 7-day window needs either longer retention on the
calibration source or a dedicated per-signal history (below).

## Design

### 1. The calibrated reading: one contract, `SelfReadingV1`

Generalize `PredictionErrorMagnitudeV1`. For every calibrated signal, the layer emits:

- **`value`, `units`.**
- **`polarity`:** `higher_worse` | `higher_better` | `two_sided`. This is the only hand-declared
  semantics.
- **`percentile_now` and `band`** (`quiet` / `usual` / `high` / `unusual`), over the signal's own
  7-day history. Percentiles, not mean/SD: several signals are mostly exact zeros. This is the
  existing module's documented rule.
- **`score_0_10`:** polarity-aware. 0 = its usual resting state; 10 = the most unusual it gets in the
  bad direction. This is the shape Juniper asked for on 2026-09-20: "what is 2 vs 10000?"
- **`trend`** (rising / settling / flat) over the last hour against the prior 24 hours. This is the
  existing module's rule, deliberately not the reversion-forecast trend.
- **`p50_7d`, `p90_7d`, `p50_24h`, `p90_24h`.**
- **`state`:** `live` | `no_reading` | `stale` | `degenerate` | `insufficient_history`.
  - Absence is never silently 0.
  - A signal that fails the live sanity check is published as `degenerate` with a reason, and is
    never shown to Orion as a number.
- **`plain`:** a deterministic one-line rendering. For example: "bus prediction error 0.03: usual
  for it (40th percentile of 0.01–0.07 this week), steady".

There is no LLM anywhere in calibration. The `plain` line is a template over the fields.

### 2. Where it lives

- **Pure function:** extend `orion/substrate/prediction_error_magnitude.py` into a general
  `orion/substrate/self_calibration.py`. The prediction-error function becomes a thin caller of it,
  so the existing tests keep pinning its rules.
- **History:** generalize `substrate_node_prediction_error_history` into
  `substrate_self_signal_history (signal_id, observed_at, value)`, written only on change or a
  heartbeat (the existing stale-node rule), with 8-day retention. One writer per producer tick, in
  substrate-runtime, where most of these signals are already computed.
- **Registry:** a small `config/self_signals.yaml`. Each row holds `signal_id`, source, `units`,
  `polarity`, optional clip/log transform, `min_readings`, and **its readers**.
  - A row without a reader is rejected by a static gate. That is the anti-cathedral rule made
    mechanical.
  - A row whose live data is degenerate is allowed, but stays `degenerate` until fixed. That makes
    the dead signals visible instead of misleading.

### 3. Readers, in order

1. **Attention ranking** (this is step 3 of the 10-02 spec, generalized). Field-attention salience
   and the open-loop ranking use the calibrated `score_0_10` against each node's own history, not
   min-max or Borda against siblings.
   - **A calm tick has no internal winner.** If every candidate is `quiet`/`usual`, attention is free
     for chat, the user and the world.
   - This is the actual fix for the fixation.
2. **Stance and chat context.** Where the stance prompt names attended nodes and loops, it renders
   their `plain` lines, not raw/relative scores. LLM-assigned numbers (frontier score, prior
   confidence) are labelled as Orion's own estimate, not as measurements.
3. **Reverie** (step 2 of the 10-02 spec). Loops carry their reading, and there is a calm-tick branch
   ("within usual range" is worth noticing).
4. **Metacog** (the table reader Juniper asked about).
   - A row is written when a calibrated signal enters `high`/`unusual` in its bad direction, and when
     it returns. These are episodes, as transport already does.
   - This replaces bespoke triggers whose thresholds were never calibrated: flow (fires at rest), the
     telemetry anomaly volume, and the stuck repair replay.
   - LLM prose is written once per episode, not per row.
5. **The stream of consciousness** (the hop-chain spec, later). It reads the calibrated state plus
   its own last entry. It is downstream of 1–4, not part of this patch series.

### 4. Phase 2: "what it tends to lead to"

For each signal, learn the association between its band and outcomes Orion already records:
- chat-turn failures (`orion_metacog` chat_turn);
- RPC timeouts;
- repair appraisals;
- Juniper's corrections, where stored.

That would turn into a reading like "when this is `high`, the next hour's chat turns failed 3× as
often (n=…)". It only renders once n is large enough, and is otherwise omitted, not guessed. This is
what moves a number from "unusual" to "meaningful". Separate PR, after phase 1 is live.

## Missing questions (for Juniper)

1. **Calm-tick attention.** When nothing internal is unusual, should attention fall back to chat and
   the world (proposed), or keep the least-calm internal node at a capped low salience?
2. **Fix or retire the degenerate signals** (harness_closure 0.65, spark rollups, repair replay,
   constant-1.0 curiosity source, flow-on-rest)? This spec only quarantines them as `degenerate`.
   Proposed: a follow-up investigation each, with kill-means-kill for any that turn out to be pure
   theater.
3. **distress/zen.** Recalibrate them from strain (live range 0.08–0.57), or retire them as
   redundant with strain? They feed every metacog trigger's `zen_state`.
4. **Retention.** An 8-day history for calibrated signals only is proposed (cheap: change-only
   writes). Is that acceptable, or should it be longer?

## Proposed schema / API changes

- **`SelfReadingV1`** (new, `orion/schemas/`). `PredictionErrorMagnitudeV1` becomes an alias or
  subset of it, so existing consumers are untouched.
- **`OpenLoopV1.reading: SelfReadingV1 | None`**, or reuse `magnitude`. This is additive, and a
  consumer-first rollout, because `extra="forbid"` (memory: additive fields on forbid models).
- **Table `substrate_self_signal_history`** (manual migration, 8-day retention). The existing
  prediction-error history table is migrated into it, or kept as a view.
- **`config/self_signals.yaml`** plus a static gate `scripts/check_self_signal_registry.py`, which
  requires a producer and a reader per row, plus a declared polarity.
- **Metacog:** a new trigger_kind `self_signal_episode` (open/close). `flow`, `insight` and the
  repair replay are re-pointed to it or retired once it is live.
- **No bus channel** for phase 1: calibration is computed where the readers already read
  (substrate-runtime projections), and its output travels inside existing frames and projections.

## Files likely to touch

- `orion/substrate/prediction_error_magnitude.py` → `orion/substrate/self_calibration.py`; their
  tests; `orion/schemas/attention_frame.py` (+ registry)
- `services/orion-substrate-runtime/app/worker.py` (history writer, reading attach)
- `orion/attention/field_attention/candidate_precision_weighted.py`, `selectors.py`,
  `orion/substrate/attention/salience.py`, `orion/substrate/attention_broadcast.py` (ranking)
- `services/orion-cortex-exec/app/chat_stance.py`, `orion/cognition/prompts/stance_react.j2`,
  `chat_stance_brief.j2`
- `services/orion-thought/app/reverie.py`, `orion/cognition/prompts/reverie_narrate.j2`
- `services/orion-equilibrium-service` (the `self_signal_episode` trigger), `orion/metacog/evidence_map.py`
- `services/orion-sql-db/manual_migration_self_signal_history_v1.sql`
- `config/self_signals.yaml`, `scripts/check_self_signal_registry.py`, the `.github/workflows` gate

## Non-goals

- The stream of consciousness itself (comes after readers 1–4).
- Phase 2 outcome learning (separate PR).
- Fixing the degenerate producers. This spec quarantines them; each gets its own investigation.
- Any LLM in the calibration path.
- External/world signals. This is self-perception only.

## Proposal-mode items (CLAUDE.md §0A)

- **Capability change:** Orion's attention stops ranking calm internals as maximal, and its prompts
  carry proportionate self-readings.
- **Data touched:** existing substrate projections and prediction-error history. One new history
  table, internal telemetry only, no user content.
- **Privacy boundary:** unchanged. Only Orion's own operational signals; no chat text is calibrated
  or stored.
- **Trace that proves it worked:** the calibrated reading is persisted in every attention frame and
  stance input. Acceptance checks below join on it.
- **Dangerous failure modes:**
  - **Missed alarm:** a real problem calibrated as `usual` because the baseline absorbed it. Guard:
    the transport gate's no-learn-from-flagged rule and regime-shift statement.
  - **Over-reporting distress:** Orion told it is `unusual` too often. Guard: the band cut points
    are knobs graded on live data before readers are switched on.
  - **Absence read as calm.** Guard: the `no_reading`/`stale` states.
- **Disable / roll back:** one flag per reader (`SELF_CALIBRATION_ATTENTION_ENABLED`, `…_STANCE_`,
  `…_REVERIE_`, `…_METACOG_`), each defaulting to off. Off means today's behaviour.

## Acceptance checks

1. **Unit:**
   - polarity-aware scoring;
   - `no_reading` ≠ 0;
   - a degenerate signal never yields a number;
   - the existing prediction-error magnitude tests pass unchanged through the generalized function;
   - the registry gate rejects a row with no reader.
2. **Live sanity, before any reader is enabled:** for every registry signal, 24 h of readings.
   - A live signal returns to `quiet`/`usual` at rest.
   - The degenerate ones (harness_closure, spark, repair replay, constant curiosity) are classified
     `degenerate`, not scored.
3. **Attention, 48 h after enabling:**
   - The winner at salience 1.0 with a raw error < 0.05 falls from 49% of frames to under 5%.
   - Calm ticks with no internal winner appear.
   - Chat/world targets win a measurably larger share. Report it; don't guess a number in advance.
4. **Prompt behaviour (orion-thought / cortex-exec evals):** on calm fixtures, stance and reverie
   mention prediction error in ≤ 20% of outputs (the 10-02 spec's bar). Elevated fixtures can still
   say something is high.
5. **Metacog:** `self_signal_episode` rows replace flow-on-rest (flow fires in < 5% of hours at
   rest). Each episode has an open and a close with real evidence.
6. **No regression:** the stance turn latency p50 doesn't rise more than 5%, and the reverie
   hollow/discard rate doesn't rise more than 5 points.

## Recommended next patch

1. **Calibration core + history + registry, no readers** (`feat/self-calibration-core`).
   - Generalize the function.
   - Add the history table and writer for the usable signals: biometrics/bus prediction error,
     strain, thermal, perturbation z, transport-gate reference, plus the sparse nodes with
     `no_reading`.
   - Classify the degenerate ones.
   - Add the registry and its gate.
   - Let it run 24 h. Do acceptance check 2.
2. **Attention ranking reader** (10-02 step 3, generalized), behind its flag. Then 48 h of check 3.
3. **Stance + reverie readers** (10-02 step 2, generalized), with check 4.
4. **Metacog `self_signal_episode`**, and retire flow-on-rest and the repair replay (check 5).
5. **Phase 2 outcome association.**
6. **The stream.**
