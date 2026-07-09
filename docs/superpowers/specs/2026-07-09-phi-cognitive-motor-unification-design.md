# φ cognitive motor unification — harness + cortex-exec → execution trajectory + seed-v3 encoder input

> **Status:** Approved (brainstorming 2026-07-09).  
> **Builds on:** `docs/superpowers/specs/2026-07-07-phi-inner-state-truthful-design.md`, `docs/superpowers/specs/2026-07-08-phi-encoder-plan2-design.md`.  
> **Sibling:** `docs/superpowers/specs/2026-07-08-phi-intrinsic-reward-value-learning-design.md` (Δφ consumer).  
> **Motivation:** Live corpus accrual and telemetry mount are fixed (PR #909), but strict encoder variance gates fail and unified Orion turns do not populate cognitive features.

## Arsonist summary

Plan 2 wired spark to read `ExecutionRunStateV1` from substrate-runtime — but only **cortex-exec** grammar reaches that projection. Juniper's primary motor is **Orion unified turn** (harness-governor → fcc → finalize), which publishes `harness_fcc_step` events the execution reducer **no-ops** (wrong `source_service`, wrong trace-id shape, wrong semantic roles). Meanwhile the encoder still trains on **saturated felt dims** the veracity audit proved flat (`field_intensity`, `resource_pressure`, `introspection_pressure`), so the variance gate can never pass honestly.

**Fix both in one slice:** (1) motor-agnostic execution grammar — harness emits the same lifecycle roles as cortex-exec into the existing execution trajectory reducer; (2) `seed-v3` encoder input contract — drop flat felt dims and move `reliability_pressure` to infra-only per Plan 1 hygiene.

## Locked decisions (brainstorming)

| Decision | Choice |
|---|---|
| Approach | **1 — Motor-agnostic execution grammar** (not dual projections, not new bus channel) |
| Motors | **Both first-class:** `orion-cortex-exec` + `orion-harness-governor` |
| Projection | Single `GET /projections/execution_trajectory` (unchanged HTTP seam for spark) |
| Trace ID | `cortex.exec:{node}:{correlation_id}` for harness turns (same parser as cortex-exec) |
| Encoder input | **`features_version=seed-v3`** — narrowed trainable subset; full felt retained in corpus for audit |
| Training | Offline `scripts/fit_phi_encoder.py`; encoder inference stays off until strict gates pass |

---

## Problem evidence (runtime, 2026-07-09)

| Finding | Evidence |
|---|---|
| Unified turns bypass trajectory | Hub `mode=orion` + `ORION_UNIFIED_TURN_ENABLED` → `run_unified_turn` → harness-governor; no cortex-exec plan |
| Harness grammar ignored | `grammar_extract.py` filters `source_service == "orion-cortex-exec"` only |
| Harness trace shape rejected | Reducer requires `trace_id` prefix `cortex.exec:`; harness uses raw `correlation_id` |
| `reasoning_present` always false | Last 2h trajectory: 180 runs, 0 with `reasoning_present=true`; mostly `chat_quick` |
| Variance gate stuck | Corpus ~3.6k rows: **11/15** scaled dims active, need **12**; 4 structurally flat |
| Saturated felt dims | `field_intensity` raw=1.0 on 100% of rows; `resource_pressure` ~100% at 1.0; `introspection_pressure` ~99% at 0.0 |

---

## Current architecture

```text
cortex-exec ──GrammarEventV1──► execution_trajectory reducer ──► projection
                                      ▲
harness-governor ─harness_fcc_step──►  │ NOOP (wrong source + trace)
                                      │
spark-introspector ◄── GET /projections/execution_trajectory
        │
        └── InnerStateFeaturesV1 (seed-v2) ──► corpus JSONL ──► fit_phi_encoder
```

**Plan 1 hygiene already shipped:** `policy_pressure` / `uncertainty` dropped from felt; infra channels quarantined in `infra[]` — but `reliability_pressure` still in felt vector (spec gap).

---

## Target architecture

```text
┌─────────────────┐     ┌──────────────────┐
│ cortex-exec     │     │ harness-governor │
│ grammar_emit    │     │ grammar_emit     │
│ (existing)      │     │ (new lifecycle)  │
└────────┬────────┘     └────────┬─────────┘
         │  GrammarEventV1       │
         │  orion:grammar:event  │
         └──────────┬────────────┘
                    ▼
         ┌──────────────────────┐
         │ execution_trajectory │
         │ reducer (widened)    │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ ExecutionRunStateV1  │
         │ projection (single)  │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ spark-introspector   │
         │ seed-v3 features     │
         │ + corpus JSONL       │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ fit_phi_encoder.py   │
         │ (strict gates)       │
         └──────────────────────┘
```

---

## Section A — Harness grammar contract

### New module

`orion/harness/grammar_emit.py` — collector pattern mirroring `services/orion-cortex-exec/app/grammar_emit.py`.

### Trace ID

```python
trace_id = f"cortex.exec:{node_name}:{correlation_id}"
```

Same `parse_execution_trace_id()` as cortex-exec. `provenance.source_service = "orion-harness-governor"`.

### Lifecycle events (semantic roles)

| Role | When emitted | ExecutionRunStateV1 field |
|---|---|---|
| `exec_request_received` | Harness run start | `verb`, `mode` |
| `exec_plan_started` | Before fcc loop | `step_count` (upper bound / 0) |
| `exec_step_started` | Each fcc step begin | `started_step_count++` |
| `exec_step_completed` | Each fcc step success | `completed_step_count++` |
| `exec_step_failed` | Each fcc step failure | `failed_step_count++` |
| `exec_recall_gate_observed` | `recall_debug` indicates memory used | `recall_observed=true` |
| `exec_result_assembled` | Motor complete (draft or fail) | `status`, `final_text_present`, `reasoning_present`, `thinking_source` |
| `exec_result_emitted` | After finalize produces `HarnessRunV1.final_text` | egress hint |

### Harness `verb` / `mode` defaults

| Field | Value |
|---|---|
| `verb` | `orion_unified` |
| `mode` | `orion` |

Classic cortex-exec runs keep their existing verb/mode (`chat_general`, `chat_quick`, etc.).

### `reasoning_present` (deterministic, no keyword lists)

```text
reasoning_present =
    step_count > 0
    OR (reflection is not None AND NOT quick_lane_skipped_5b)
    OR len(grammar_receipts) > 0
```

### `thinking_source`

| Condition | Value |
|---|---|
| `step_count > 0` | `harness_fcc` |
| reflection ran (5b, not quick-skip) | `finalize_reflect` |
| else | `none` |

### Backward compatibility

Keep `publish_harness_step_grammar()` emitting `harness_fcc_step` for Hub receipts and unified-turn debug. Reducer **ignores** `harness_fcc_step`; lifecycle events drive trajectory.

### Publish points

1. **`HarnessRunner.run()`** — request, plan, per-step started/completed/failed, `exec_result_assembled` at motor end.
2. **Harness governor finalize path** — `exec_result_emitted` when `HarnessRunV1.final_text` is published to Hub.

### Failure policy

Grammar publish is **fail-open** (match cortex-exec): log + metric on bus failure; never block user-visible turn completion.

---

## Section B — Substrate reducer widening

### Constants

```text
EXECUTION_SOURCE_SERVICES = frozenset({
  "orion-cortex-exec",
  "orion-harness-governor",
})
```

Replace single `EXECUTION_SOURCE_SERVICE` guard in `reducer.py`.

### `grammar_extract.py`

- Accept events from either source in `EXECUTION_SOURCE_SERVICES`.
- Same `semantic_role` → field mapping for both motors.
- Ignore `harness_fcc_step` (and any non-lifecycle harness roles).

### `grammar_truth`

No new degraded reason. Execution lane health = reducer cursor + heartbeat — motor-agnostic. A healthy lane with zero harness events is still degraded for **cognitive substance** only when spark reads stale/empty active runs (existing `execution_trajectory.none` path).

### API

**No new endpoint.** `GET /projections/execution_trajectory` unchanged.

---

## Section C — Spark `seed-v3` encoder input

### Rationale

Plan 1 veracity audit: only `policy_pressure` and `uncertainty` are structurally dead and dropped from felt. **Saturated** dims (`field_intensity`, etc.) were kept for standardization — but when IQR=0 the robust scaler correctly outputs 0 forever. They must not count against the variance gate or pollute encoder training.

### `features_version=seed-v3`

Corpus rows emit:
- **Full felt** in `features[]` for Hub audit / debug (unchanged names).
- **`reliability_pressure` moved to `infra[]` only** (Plan 1 spec: infra never read by φ).
- **Encoder / fit script** reads **trainable subset** only.

### Trainable encoder input (11 dims)

```text
felt (trainable):
  coherence, agency_readiness, execution_pressure, reasoning_pressure,
  continuity_pressure, social_pressure, overall_intensity

cognitive (4):
  recall_gate_fired, reasoning_present, exec_step_fail_rate, execution_friction
```

### Dropped from encoder input (still in corpus audit or infra)

| Feature | Disposition |
|---|---|
| `field_intensity` | Emit in felt for audit; **exclude from train** |
| `resource_pressure` | Emit in felt for audit; **exclude from train** |
| `introspection_pressure` | Emit in felt for audit; **exclude from train** |
| `reliability_pressure` | **infra only** — remove from felt `features[]` |
| `policy_pressure`, `uncertainty` | Already dropped (Plan 1) |

### Variance gate (seed-v3)

- **11 trainable dims** → need **≥9** with variance > `PHI_ENCODER_MIN_FEATURE_VAR` (80%).
- `fit_phi_encoder.py`: `input_features(features_version=...)` returns per-version lists.
- `seed-v2` path unchanged for backward compat / legacy corpus replay.

### Spark defaults (post-ship)

```text
INNER_FEATURES_VERSION=seed-v3
ORION_PHI_ENCODER_ENABLED=false   # until promote gates pass on seed-v3 corpus
```

---

## Section D — Files likely to touch

| Area | Files |
|---|---|
| Harness grammar | `orion/harness/grammar_emit.py` (new), `orion/harness/runner.py`, harness governor finalize seam |
| Substrate | `orion/substrate/execution_loop/constants.py`, `grammar_extract.py`, `reducer.py` |
| Spark | `services/orion-spark-introspector/app/inner_state.py`, `settings.py`, `.env_example` |
| Fit / promote | `scripts/fit_phi_encoder.py`, `fixtures/phi_encoder_promote_gate.jsonl` |
| Tests | `orion/harness/tests/`, `orion/substrate/execution_loop/tests/`, `services/orion-spark-introspector/tests/`, `services/orion-substrate-runtime/tests/` |
| Config | `services/orion-spark-introspector/docker-compose.yml`, `services/orion-harness-governor/` env if grammar channel needed |
| Docs | This spec → `docs/superpowers/plans/2026-07-09-phi-cognitive-motor-unification.md` (implementation plan, via writing-plans) |

---

## Non-goals

- Dual projections or spark-side merge of multiple trajectory sources.
- New bus channel / `CognitiveSubstanceV1` event (deferred unless Approach 1 fails acceptance).
- Massaging saturated self-state scores in `orion/self_state/builder.py` to fake variance.
- Enabling `ORION_PHI_ENCODER_ENABLED=true` in the same slice (corpus must accrue seed-v3 first).
- Changing MLP architecture (`mlp_shallow_v1` locked in Plan 2).

---

## Acceptance checks

1. **Unified turn smoke:** Hub `mode=orion` → within 120s, projection contains run with `verb=orion_unified`, `reasoning_present=true` when fcc `step_count > 0`.
2. **Classic cortex-exec smoke:** `chat_general` with reasoning trace → `reasoning_present=true` in projection (regression guard).
3. **Reducer unit test:** harness lifecycle fixture events → `ExecutionRunStateV1` with correct step counters and `thinking_source`.
4. **Spark seed-v3 test:** `reliability_pressure` in `infra`, not `features`; trainable dim count = 11.
5. **Variance gate:** strict `fit_phi_encoder.py` on seed-v3 clean corpus passes (≥9/11) after mixed unified + classic traffic window.
6. **grammar_truth:** execution reducer healthy; cognitive features non-zero when active harness or cortex runs exist.
7. **Fail-open:** grammar bus failure does not block harness motor or Hub final frame.

---

## Risks

| Risk | Mitigation |
|---|---|
| Harness `cortex.exec:` trace IDs collide with real cortex-exec if same correlation_id reused | Document: unified turn owns correlation_id; cortex-exec paths use distinct IDs today (UUID per WS message) |
| `reasoning_present` heuristic too loose | Locked deterministic rules + fixture tests; tune only with eval evidence |
| seed-v2 / seed-v3 corpus mix during migration | Fit script filters by `features_version`; promote requires seed-v3 manifest |
| Reliability_pressure removal shifts headline | `honest_headline` already excludes infra; verify `test_inner_state_features` |

---

## Recommended implementation order

1. Substrate reducer widening + harness lifecycle reducer tests (no Hub change yet).
2. Harness `grammar_emit.py` + wire into `HarnessRunner` and finalize emit.
3. Unified-turn smoke proving projection row.
4. Spark seed-v3 felt/infra split + tests.
5. `fit_phi_encoder.py` versioned input + promote fixture.
6. Operator: accrue seed-v3 corpus → strict train → promote (encoder stays off until gates pass).
