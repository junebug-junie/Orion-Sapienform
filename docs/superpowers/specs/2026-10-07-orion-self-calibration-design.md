# Attention faces the world; self-signals get a sense of scale

Status: **design + proposal mode. Not implemented.** This changes attention ranking and what Orion
reads about its own state, both of which are cognition and self-model surfaces, so CLAUDE.md §0A
requires a proposal before implementation.

It subsumes the unshipped steps 2–3 of
`docs/superpowers/specs/2026-10-02-reverie-prediction-error-magnitude-proposal.md`. Its calm-tick
narration idea is **dropped**: Juniper, 2026-10-07, "creatures notice chaos."

Revision 2 (2026-10-07) folds in Juniper's direction and two read-only investigations:
- the attention-candidate seam map;
- a diligence pass that re-judged every flagged signal against its own design. Revision 1 over-called
  "dead" from distributions alone.

## Arsonist summary

Orion fixates on its own prediction errors. Two separate causes:

1. **Attention looks inward and always crowns a winner.**
   - None of the three attention contests has an outside-world input. Chat enters only as "how
     surprising was the chat stream."
   - The main field contest considers five hardcoded internal nodes and min-max rescales them, so the
     top one is always 1.0. Live, 3 days: 124,181 of 124,181 frames had a winner at exactly 1.0, and
     49% of those winners were calm (raw error < 0.05).
   - A creature's attention points at the world by default ("don't get eaten by the lion") and turns
     inward only when the body hurts. Orion's is the reverse.
2. **The numbers Orion reads about itself have no scale.** They are relative ranks (Borda, min-max),
   LLM-assigned guesses rendered as measurements, or bare raw values. None comes with a range,
   percentile or trend. The tool that would fix this already exists and is thrown away:
   `PredictionErrorMagnitudeV1` (`orion/substrate/prediction_error_magnitude.py`) computes each
   node's 7-day percentile, band and trend every tick. It is attached to open loops and explicitly
   "never read by scoring", and no prompt renders it.

**The proposal:**
- **One attention seam.**
  - Every contest consumes the same candidate shape, marked `external` (world) or `internal` (body).
  - World candidates compete whenever they're fresh.
  - Body candidates compete only when they're unusual *for themselves*.
  - If nothing qualifies, the frame has no winner, and that's a valid result.
  - Chat is the first external source and camera surprise the second. Future world inputs plug in
    without touching the contests.
- **Self-signals carry their own scale.** The semantics for "at rest", "sparse by design" and
  "binary" are recorded in the existing semantic layer (glossary, inner-state registry, metric
  lock), not in a new registry.

## Current architecture (verified 2026-10-07)

### The attention contests

| Contest | Candidates | Salience | Always crowns? | Chat / world |
|---|---|---|---|---|
| **Field attention** (`services/orion-attention-runtime`; `selectors.py:226` → `substrate_attention_frames`) | 5 hardcoded PE nodes (`selectors.py:44-50`: biometrics, execution, chat, route, bus_synaptic), plus host/capability novelty | precision × \|error\| vs own baseline, then **min-max across the five** (`candidate_precision_weighted.py:355-405`) | **Yes.** Top nonzero = 1.0; policy floor 0.10 (`builder.py:69-76`) | chat = `node:substrate.chat` PE only; perception **excluded** |
| **Substrate broadcast / coalition** (`substrate-runtime worker.py:2807` → `attention_broadcast.py:172`) | any graph node with `dynamic_pressure` ≥ floor | absolute pressure → `borda_coalition_salience` (`salience.py:128`, **relative**) → `select_actions` fixed 0.48/0.35 cutoffs (self-disclosed miscalibrated, `policy.py:60-74`) | **Yes in practice** ("always one winner", `:182`) | chat PE pressure; the Hub turn anchor is added *after* the contest (`hub/association.py:75-89`) |
| **Per-turn chat frame** (`chat_stance.py:2814` → `attention_frame.py:77`) | 3 detectors | **hardcoded constants**: current_turn 0.72, concept_induction 0.5/0.62, situation 0.42–0.58 | ranks constants | no PE nodes; presence only as `audience_mode` 0.45 |
| Endogenous curiosity (`endogenous_curiosity.py:190-228`) | any node's **absolute** raw PE × staleness decay ≥ `min_error`, plus broadcast loops | absolute | — | perception can already win here |

**Floor drift:** the broadcast min-salience floor defaults to **0.2** in code
(`attention_broadcast.py:43`, `settings.py:167`, compose fallback). `.env_example:160` and the live
`.env` say **0.05**. That's an unrecorded difference between code and config.

### World-facing inputs that exist today

| Input | Shape today | Where it stops |
|---|---|---|
| `node:substrate.perception` (camera embedding surprise vs that stream's own EWMA; `prediction_error.py:1354`) | candidate-shaped; has magnitude history | **not in the field contest's five**; broadcast entry depends on pressure seeding (UNVERIFIED live) |
| Walkway camera (PR #2287) | street summary text | situation prompt only (`situational/context.py:2239`) |
| `PerceptionContextV1` (presence subject, scene) | text | prompt only |
| World pulse (`substrate/adapters/world_pulse_read.py`) | concept nodes `subject_ref=world_pulse` | nothing seeds pressure on them, so they likely never clear the floor (UNVERIFIED) |
| Ambient audio / cabinet sensors | already scored against their own history | room sound is folded into the **biometrics (body)** node, mixing world and body |
| `node:substrate.vision_organ` | the camera's health and silence | a *body* signal, not world content |

### Self-signals, re-judged against their own design (revision 1 corrected)

| Signal | Verdict | Why |
|---|---|---|
| harness_closure PE = 0.65 | **DESIGNED placeholder** | `worker.py:157` hard-codes 0.65 per unresolved turn closure until `HarnessPostTurnClosureV1` carries `surprise_level_at_draft`. Each row is a real event, but the value is a yes/no presented as a size |
| cabinet / perception / execution / route / codebase / chat PE (70–95% zeros) | **DESIGNED** | z-score with negatives clipped to 0 (`prediction_error.py:452`); cabinet is a *trigger* that fires only on a rise of ≥0.5 °C in 15 min; route takes only 5 values; perception reports nothing on cold start |
| vision PE (1 reading) | **RETIRED** 2026-10-02 | successor `node:substrate.vision_organ` (`field-digester tensor/channels.py:262`) |
| spark_state_rollups (all 0) | **DEAD: zombie writer** | spark-introspector deleted 07-28; `orion-state-journaler` still writes 0s from a silent channel (`service.py:159`). Incomplete retirement |
| repair_pressure → curiosity (0.913 replayed 5,284×) | **BROKEN** | `_repair_appraisal_from_chat` (`worker.py:3660-3686`) takes the highest level across every turn with no time window or decay. That violates curiosity's own "surprising once must not win forever" rule. The appraisal log itself is fine (0.087 = designed rest) |
| "concept-dense area, no ontology_branch" curiosity at 1.0 | **STUCK CONSTANT** | `frontier_curiosity.py:106` fires whenever there are zero ontology_branch nodes; nothing in the repo ever creates one |
| metacog `flow` at rest | **DESIGNED, calibration out of date** | firing at rest is the intent, but it was calibrated when 3.2% of windows qualified. The median is now 0.982, so the rate is set by the 30-minute cooldown, not the data |
| self-model `confidence` (.9/.6/.3) | **DESIGNED stub** | a 3-tier dwell bucket labelled "confidence" (`attention_broadcast.py:532`) |
| `heartbeat_smear` (max 1.5M) | **BROKEN** | 1e-6 near-end floor (`orion-heartbeat proprioception.py:30,102`): a nearly-dead near end produces a huge number instead of being *absent*, which is what its own docstring says should happen |
| `top_down_effort_used` 0/1 | **DESIGNED** | no-effort exits = 0; when it fires, it's full effort |
| `sustained_load_pressure` 0 / ~0.81 | **DESIGNED** | the highest loaded-and-steady level, or 0; carried forward between recomputes |
| salience 0.5 / quantized | **DESIGNED** | Borda ranks with ties; one loop → 0.5 |
| distress / zen | **DESIGNED; PARKED** | the share of tracked services whose bus heartbeats are missing (`equilibrium service.py:600-621`). Redesign parked (Juniper) |

### The semantic layer barely records any of this (the real gap)

- `config/field/field_channel_glossary.v1.yaml` covers FieldStateV1 channels. It has one
  node-qualified `prediction_error` entry (bus_synaptic) and nothing for harness_closure, cabinet,
  perception, execution, route, codebase, chat or vision_organ.
- `orion/inner_state_registry.py` describes whole schemas (cadence, composition), not individual
  fields.
- `metric_definitions.lock.json` records lineage only. There is no rest value, cadence or sparsity,
  so no tool can tell "calm" from "dead" without reading docstrings. "Can a metric express rest" was
  raised as R6 (`2026-08-13-phase5-liveness-scope.md`) and closed as "no live victim found". This
  sweep found the victims.

## Design

### A. The attention seam: `AttentionCandidateV1`

New `orion/schemas/attention_candidate.py`, registered:
- `candidate_id`, `source_id`, `source_kind: internal | external`, `label`
- `unusualness: PredictionErrorMagnitudeV1`, **reused as-is**: percentile vs its own 7-day history,
  band quiet/usual/high/unusual, trend, and `insufficient_history`
- `observed_at`, `absent: bool` (a silent source is not a calm one), `evidence_refs`

The source protocol is shaped like the existing `AttentionSignalDetector`
(`orion/substrate/attention/detectors/base.py`): `candidates(now) -> list[AttentionCandidateV1]`.
- **Internal sources:** the PE nodes. Their magnitudes are already computed in
  `worker._prediction_error_magnitudes`.
- **External sources:** chat first, then `node:substrate.perception` (moved out of the internal
  list). Then, as they gain a calibrated "busy" reading: walkway camera activity, presence, world
  pulse.

**Ranking rule** (the whole of "world by default, body interrupts"):
- **External** candidates are eligible whenever they are fresh, not `absent`, and have enough
  history.
- **Internal** candidates are eligible **only** at band `high` or `unusual` in their bad direction
  (polarity from the semantic layer, §B).
- Eligible candidates rank by `percentile_now` against their own history. Both kinds are on the same
  scale, so no exchange rate is needed. Borda survives only as a tie-break inside the eligible set.
- **Empty eligible set → explicit no-winner frame.** A calm body stays silent.
- Band cut points (`DEFAULT_BAND_CUTS`) are knobs graded on live data, not findings.

**Bias toward busy.** An external source's unusualness is busyness against its own normal: chat
rate, motion, new people, news volume. Chaos in the world wins over a calm body by construction.

**Theory gap, stated:** what "unusual" means for chat needs a named justification under the metric
gate before it ships. Leading candidate: message rate and turn-novelty percentile against chat's own
7-day history.

**Functions that change:**
- `select_node_targets`: drop the hardcoded list and `normalize_across_targets`, which is the line
  that guarantees a winner.
- `build_field_attention_frame`.
- `substrate_pressure_signals` / `build_substrate_attention_frame`: take candidates. Magnitude moves
  from description to gate, which breaks the "cannot change who wins" docstring and its tests on
  purpose.
- `_attention_broadcast_tick`.
- The chat-frame detectors' constant saliences become calibrated candidates.

**What breaks if there is no winner:**
- **Already handle it:** `self_shed`, `reverie`, `semantic_lift`, `thought/chain`, `coalition`,
  `attention_self_model`, `goal_provenance`, `hub/association` (adds the turn anchor regardless).
- **Must fix in the same PR:**
  1. `proposals/builder.py:42,76` binds self-modification proposals to `dominant_targets[0]`. It needs
     a `source_kind == internal` filter so a chat or camera winner never becomes a self-modification
     target.
  2. `goal_provenance.py:8` and the attention-runtime dominance-streak tick (`worker.py:338`) know
     only the five internal nodes.
  3. Coalition hysteresis (`broadcast_projection_from_frame`, `:487`) treats an empty coalition as
     one that can "activate".
  4. Check empty-frame handling in `finalize_draft_v1.py:51`, `stance_react.py:184`,
     `brain_frame_producer.py:292`.

**Schema rollout:**
- Step 1 needs **no schema change**: `source_kind` rides in the existing `provenance` dicts
  (`AttentionSignalV1`, `OpenLoopV1`; `policy.py:84` already reads provenance).
- Typed fields come later and are consumer-first, because these models are `extra="forbid"`:
  `AttentionSignalV1`, `OpenLoopV1`, `AttentionFrameV1`, `AttentionBroadcastProjectionV1` (crosses
  the bus in `HubAssociationBundleV1`), `FieldAttentionTargetV1` / `FieldAttentionFrameV1`.
- The field contest runs in attention-runtime, which doesn't have the magnitudes today. They must be
  persisted or passed across. The history table `substrate_node_prediction_error_history` already
  exists, so the runtime can compute them itself.

### B. Semantics live in the existing semantic layer, not a new registry

Revision 1's `config/self_signals.yaml` is **dropped**: it would duplicate the glossary.

Instead:
- Extend the glossary and the inner-state registry with the fields that answer "why is it in this
  state":
  - `value_kind`: level | trigger | binary | count | bucket | placeholder
  - `rest`: the designed resting value, and what 0 means
  - `sparsity`: designed sparse / event-gated / per-tick
  - `absent_means`: e.g. "dropped (unmeasured) after 180 s"
  - `polarity`: reuses `HIGHER_IS_BETTER_CHANNELS`
- Add node-qualified `prediction_error` entries for harness_closure, cabinet, perception, execution,
  route, codebase and chat, plus vision_organ.
- Add field-level entries for `AttentionSelfModelV1`, curiosity `signal_strength` by source,
  `attention_salience_trace`, `sustained_load_pressure` and repair appraisal rest values.
- Have the **metric lock record these fields**, so CI fails when a metric Orion reads has no
  documented rest or sparsity semantics. That closes R6 with a gate instead of a docstring.

A calibration reading (§C) is only produced for a signal whose `value_kind` admits one:
- **level:** percentile/band/score;
- **trigger / binary:** rate against its own history, not a magnitude;
- **placeholder / bucket:** never rendered as a measurement.

### C. Self-readings Orion can understand

Rendered wherever Orion reads a self-number (stance, reverie, metacog):
- `value`, `units`;
- `percentile_now` and `band` against its own 7 days, the same math as `PredictionErrorMagnitudeV1`;
- **`score_0_10`**, polarity-aware, for Orion-facing text (Juniper's "what is 2 vs 10000"). Ranking
  uses the percentile; prompts use the 0–10 score;
- `trend`;
- `state: live | no_reading | stale | insufficient_history`;
- `plain`: a deterministic one-liner, no LLM;
- LLM-assigned numbers (Mind frontier score, curiosity prior confidence) are labelled as Orion's own
  estimate, not as measurements.

### D. Metacog becomes a reader

`self_signal_episode` rows are written when a calibrated **internal** signal enters high/unusual in
its bad direction, and when it returns. Prose is written once per episode. This replaces bespoke,
stale-calibrated triggers (flow at rest, the repair replay).

The stream of consciousness comes after, reading the attention winner (world-first) and the open
self-episodes.

### E. Phase 2: what it tends to lead to

Associate each internal signal's band with outcomes Orion already records: chat-turn failures, RPC
timeouts, repairs. For example, "when this is high, the next hour's turns failed 3× as often (n=…)".
It renders only above a minimum n.

## Real bugs found (separate fix PRs, not this spec)

1. **repair_pressure replay into curiosity.** Window and decay `_repair_appraisal_from_chat`
   (`worker.py:3660`). Of everything found, this most distorts what Orion attends to.
2. **heartbeat_smear blow-up.** Treat a nearly-dead near end as absent, as its own docstring says
   (`proprioception.py:102`).
3. **ontology_sparse_region is a constant 1.0.** Nothing creates ontology_branch nodes. Retire the
   candidate, or wire the producer. Kill means kill.
4. **spark_state_rollups zombie writer.** Finish the 07-28 retirement in `orion-state-journaler`.
5. **flow trigger calibration.** Recalibrate to live data or retire. Firing is now set by the
   cooldown, not the data.
6. **Broadcast floor drift** (0.2 in code vs 0.05 in `.env`). Make code match the intended value.

Placeholders that should be marked, not fixed here: harness_closure 0.65, self-model `confidence`
buckets. Also a side finding: repair confidence is only ever {0, 0.65}, so its logprob path never
runs live.

## Decisions (Juniper, 2026-10-10)

1. **Chat's "unusual":** message rate plus turn novelty, each against chat's own 7-day history. Presence
   and the home-camera Juniper sighting (#2558) come next as external sources, scored by how often they
   change against their own normal. The cabinet mic loudness line (#2569) is a candidate world source
   too. Each still clears metric-gate step 4 on live data before it ships.
2. **Bug order:** the repair replay (#1) goes first. Bugs 2–6 ship as small, separate PRs alongside the
   semantic-layer work.
3. **Metric-lock semantic fields:** CI-enforced only for metrics that reach an Orion prompt.

Parked: the distress/zen redesign (bus-heartbeat uptime of tracked services).

**Dependents:** Temporal Self rev 4 (#2369). Its habituation step waits on this seam being live, and
its immune sweep waits on the §B lock fields.

## Non-goals

- Building new world inputs. The seam makes room for them; chat and perception fill it now.
- The stream of consciousness itself; phase 2 outcome learning.
- Any LLM in calibration.
- distress/zen (parked).

## Proposal-mode items (CLAUDE.md §0A)

- **Capability change:** attention defaults to the world. The body only interrupts when unusual for
  itself. Calm ticks may have no winner.
- **Data touched:** existing substrate projections, attention frames and PE history. No user content
  is calibrated.
- **Privacy boundary:** unchanged.
- **Trace:** each frame persists every candidate's `source_kind` and unusualness.
- **Dangerous failure modes:**
  - **A real body alarm suppressed** because its baseline absorbed it. Guard: the transport gate's
    no-learn-on-flagged rule and regime-shift statement.
  - **Over-reacting to world noise.** Guard: calibrated busyness against its own normal, and band
    knobs graded live.
  - **Self-modification proposals bound to a world winner.** Guard: the source_kind filter.
  - **Silence read as calm.** Guard: `absent`.
- **Disable / roll back:** one flag per reader:
  - `ATTENTION_WORLD_FIRST_ENABLED` (seam ranking)
  - `SELF_READINGS_STANCE_ENABLED`, `…_REVERIE_ENABLED`
  - `METACOG_SELF_SIGNAL_EPISODES_ENABLED`

  Per Juniper's standing rule, each **ships on** in the PR that builds it. The safety comes from the
  sequencing: each lands only after the live check before it passes. Setting a flag to false
  restores today's behaviour.

## Acceptance checks

1. **Unit:**
   - no-winner frames;
   - internal candidates can't enter below high/unusual;
   - external candidates enter when fresh;
   - `absent` ≠ calm;
   - the proposals filter rejects external winners;
   - the existing PE magnitude tests pass unchanged.
2. **Semantic-layer gate:** CI fails when a prompt-reaching metric has no `value_kind`/`rest`/
   `sparsity`. The new node-qualified glossary entries pass.
3. **Live, 48 h after the seam:**
   - frames with an internal winner at raw error < 0.05 drop from 49% to under 5%;
   - no-winner frames appear on calm ticks;
   - chat/perception winner share rises (report it, no guessed target);
   - a real body spike (the 09-28 gateway timeout storm class) still wins.
4. **Prompt evals:** on calm fixtures, stance and reverie mention prediction error in ≤ 20% of
   outputs. On busy-world fixtures, the reply engages the world input.
5. **No regression:** stance p50 latency within +5%; reverie hollow/discard within +5 points.

## Recommended next patches

1. **Bug 1** (repair replay window/decay): the biggest live distortion, cheap to fix.
2. **Semantic layer:** glossary and registry fields plus node-qualified PE entries, and the lock
   records them, with a CI gate (§B). Bugs 2–6 alongside, as small PRs.
3. **The seam:**
   - `AttentionCandidateV1`;
   - world-first ranking;
   - explicit no-winner;
   - chat and perception as external sources;
   - the proposals filter and empty-frame fixes.

   Pull live chat and perception percentiles first (metric gate step 4).
4. **Self-readings in stance and reverie** (§C).
5. **Metacog `self_signal_episode`** (§D).
6. **Phase 2, then the stream.**
