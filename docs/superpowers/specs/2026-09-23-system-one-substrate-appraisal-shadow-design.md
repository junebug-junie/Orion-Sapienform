# System One substrate appraisal — shadow design

Date: 2026-09-23  
Status: shadow collection shipped; **curiosity_pull promoted** in
`docs/superpowers/specs/2026-09-23-system-one-curiosity-admission-design.md`.
`reverie_fit`, `attention_interrupt`, and `deliberation_need` remain observational.

## Arsonist summary

Do **not** add a new state ontology, "facet" graph, or replacement for substrate grammar.

The existing doctrine already says:

```text
event -> schema -> trace -> reducer -> projection -> eval -> UI/debug surface
```

Kev / TypeSafe System One fits at **reducer -> projection**. The reducer consumes bounded,
already-live attention artifacts and emits a typed compiled frame. A normal
`GrammarProjectionV1` records the causal shadow. No `StateDeltaV1` is emitted in this patch,
so the learned appraisal cannot mutate FieldState or behavior.

## Current architecture

Inputs already exist:

- `AttentionBroadcastProjectionV1` — current GWT/workspace selection, open loops,
  coalition stability, dwell, top-down effort and evidence refs.
- `FieldAttentionFrameV1` — current field salience and dominant field targets.

The new path is:

```text
AttentionBroadcastProjectionV1
          +
FieldAttentionFrameV1 (fresh only)
          |
          v
build_system_one_input_state()
          |
          v
POST /v1/systemone  (Kev-compatible)
          |
          v
SystemOneAppraisalFrameV1
          |
          +--> substrate_system_one_appraisal   (append-only shadow history)
          |
          +--> GrammarEventV1 / GrammarProjectionV1
               projection_type=system_one_shadow_appraisal
```

The reducer rides the existing ~30s attention-broadcast tick. It has no independent timer.

## Capability change

Orion can now ask several small, typed, independent-within-request decision questions about
the same bounded current attention state without invoking the full generative cognition path.

The initial shadow question set is:

- `reverie_fit`
- `curiosity_pull`
- `deliberation_need`
- `attention_interrupt`

Each is a three-level `score` question. The full returned probability distribution is
preserved. The implementation deliberately does **not** turn the 0..2 expected score into
a fake 0..1 "propensity."

`QUESTION_SET_ID=orion.system_one.shadow.v1` is part of frame identity and MUST be bumped if
question membership, instructions, or criteria change.

## Existing-mechanism check

This patch reuses rather than replaces:

- `GrammarEventV1`, `GrammarAtomV1`, and `GrammarProjectionV1`
- the substrate-runtime attention-broadcast cadence
- `AttentionBroadcastProjectionV1`
- `FieldAttentionFrameV1`
- the schema registry
- substrate-runtime's existing projection-debug pattern

It does **not** add:

- a new service;
- a new graph;
- a new bus channel;
- a new `StateDeltaV1` target kind;
- a replacement for FieldState, attention, autonomy, or substrate grammar.

## Data and privacy boundary

The request intentionally excludes raw chat turns, raw prompts/completions, and full graph
snapshots. It contains a bounded projection of current attention:

- selected action/open-loop ids and descriptions;
- up to six open-loop summaries plus source refs;
- up to five dominant FieldAttention targets;
- coalition/dwell/effort values;
- bounded live unknowns and deferred items.

The descriptions are still **derived private cognitive context** and may contain semantic
summaries originating from conversation. A local Kev endpoint keeps this inside Orion's own
deployment boundary. Pointing `SUBSTRATE_SYSTEM_ONE_BASE_URL` at a hosted provider explicitly
moves that derived context across the provider boundary. The feature is default-off so this
cannot happen by accident.

No API key is logged or persisted in the frame.

## Failure behavior

- Kev unreachable / timeout: log and fail open; the normal attention broadcast persists.
- Stale FieldAttentionFrame: omit that source rather than present stale field evidence as current.
- Partial/malformed provider response: reject the entire appraisal; do not persist a
  schema-valid empty shell.
- Grammar publication fails after frame persistence: frame remains inspectable; publish failure
  is logged. No cognition or attention path is blocked.
- Bad learned judgment: currently behavior-inert. This is the primary reason for shadow mode.

Rollback is one env change:

```text
SUBSTRATE_SYSTEM_ONE_APPRAISAL_ENABLED=false
```

The additive history table can remain in place without any consumer.

## Metric-quality gate

### 1. Provenance

No input metric is newly fabricated. Every numeric/text feature sent to System One is copied
from the two existing typed input artifacts. The stored frame records both source ids/timestamps.

### 2. Independence

The four appraisals are **not independent measurements**. They share:

- the same bounded state;
- the same model;
- the same forward request;
- overlapping latent semantics.

Therefore downstream code MUST NOT average, vote, or otherwise combine them as if they were
independent evidence. Each question is its own learned judgment.

### 3. Theory anchors

These anchors justify the questions as hypotheses, not the model's correctness:

- `reverie_fit`: task-unrelated thought / mind-wandering and intentional mind-wandering.
  The current input only captures Orion's attention state, so this is a suitability judgment,
  not a detector proving reverie should occur.
- `curiosity_pull`: information-gap accounts of curiosity (Loewenstein) and epistemic
  uncertainty. Current inputs do not directly estimate expected information gain, so the result
  remains a hypothesis until correlated with real investigation outcomes.
- `deliberation_need`: rational metareasoning / value-of-computation framing — whether more
  expensive cognition is warranted by current evidence.
- `attention_interrupt`: biased-competition / global-workspace framing — whether represented
  evidence deserves scarce access over current focus.

### 4. Live-data sanity

**UNVERIFIED until deployed.** No behavioral consumer may be wired yet.

After deployment, run:

```bash
python scripts/analysis/eval_system_one_appraisal.py --hours 24
python scripts/smoke_system_one_appraisal.py
```

The first report checks row presence, variance, saturation, confidence distribution, and
per-level argmax counts. It intentionally does not invent a behavioral threshold.

### 5. Existing mechanism

Satisfied by the existing grammar/reducer/projection doctrine; this patch avoids a new state
primitive.

### 6. Reversibility

High. Default-off flag, one isolated table, no behavioral consumers, no FieldState mutation,
no new bus contract.

## Promotion requirements

A question may move from shadow output into a behavioral consumer only in a separate patch that:

1. shows non-degenerate live distributions over a meaningful observation window;
2. defines a labeled or outcome-based evaluation for the actual downstream decision;
3. measures calibration / error, not just mean score;
4. identifies the exact transformation consumed (for example `P(level=2)`, not an implicit
   reinterpretation of expected score);
5. compares against the existing heuristic/control path;
6. preserves an immediate kill switch and causal trace.

Until then, these frames are observations about a candidate inference mechanism, not truth.

## Acceptance checks

- valid Kev-compatible response round-trips into `SystemOneAppraisalFrameV1`;
- full option probabilities survive persistence;
- incomplete/malformed answers are rejected;
- Kev outage cannot fail the attention-broadcast tick;
- stale field input is omitted;
- compiled frame persists append-only and is queryable at
  `GET /projections/system_one_appraisal`;
- each persisted frame creates a normal substrate grammar projection trace when the bus is up;
- no StateDelta, FieldState, reverie, curiosity, autonomy, scheduler, or FCC behavior changes.

## Deployment

1. Deploy a Kev-compatible endpoint.
2. Apply `manual_migration_system_one_appraisal_v1.sql`.
3. Sync `services/orion-substrate-runtime/.env` from `.env_example`.
4. Set the endpoint and flip `SUBSTRATE_SYSTEM_ONE_APPRAISAL_ENABLED=true`.
5. Rebuild/restart only `orion-substrate-runtime`.
6. Run the smoke and shadow-distribution report.
7. Leave behavior unchanged until the promotion requirements above pass.
