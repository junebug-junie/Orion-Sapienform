# Turn Change Appraisal v1 — Design Spec

**Date:** 2026-06-22  
**Status:** Approved for planning  
**Service scope:** `orion-memory-consolidation` (primary), `orion-spark-introspector`, `orion-llm-gateway`, substrate emit path  

---

## Problem

Orion’s turn-level “novelty” and `turn_effect` are driven by **v0 tissue** (`orion/spark/orion_tissue.py`): a 16×16×8 Laplacian smoother that encodes chat as fake waveforms, computes cosine-distance “novelty,” and stamps `phi_before` / `phi_after` deltas. Downstream systems treat these as inner-state signals:

- Spark telemetry (`novelty`, `turn_effect`)
- `introspect_spark` (narrates φ shifts)
- Memory consolidation classify prompt (reads `phi_after.novelty`)
- Collapse mirror meta (`build_collapse_mirror_meta` from tissue φ)

This is **theater**, not grounded change detection. Substrate `SelfStateV1` (grammar → field → attention) provides real **organism condition** but does not answer **semantic “what changed vs the prior turn?”**

## Goal

Replace fake smoothers with **logprob-calibrated turn change appraisal** that:

1. **A — Telemetry:** Supplies authoritative `novelty` / change signal in `spark_meta` (no tissue φ on the hot path).
2. **B — Substrate:** Emits a grammar/substrate perturbation when the model is highly confident the turn is novel, routed by shift kind.

**Non-goal:** Shadow or parallel-run tissue for comparison. **Tissue is removed from feed-forward paths**, not flagged behind a debug switch.

## Success criteria

- Every persisted chat turn receives `turn_change_appraisal` in `spark_meta` via the existing patch rail.
- All scores in the happy path derive from **logprobs only** (no LLM-generated numeric scores).
- Spark telemetry `novelty` reads appraisal when `turn_change_status=ok`.
- High-confidence novel turns publish a substrate `organ_signal` molecule (existing signal-bridge pattern).
- Degraded appraisal → `novelty` is **null**, never tissue fallback.
- LLM gateway stops stamping tissue `phi_before`/`phi_after` for turn semantics.

---

## Two-layer inner state model

| Layer | Question | Source | v1 role |
|-------|----------|--------|---------|
| **Turn change** | Did this exchange change something vs baseline? | Logprob classifier vs prior context | **This spec** |
| **Organism condition** | How loaded/aligned/uncertain is the field? | `SelfStateV1` from substrate runtime | Complementary; not replaced |

Together they provide inspectable functional signals: **semantic delta** + **field synthesis**. Neither claims phenomenology.

---

## Appraisal contract

### Prompt output (exactly four lines)

```
NOVEL: YES or NO
SHIFT: NONE or TOPIC or STANCE or REPAIR
MEMORY: YES or NO
BOUNDARY: YES or NO
```

`MEMORY` and `BOUNDARY` remain as today. `NOVEL` and `SHIFT` are new.

### Scoring rules (logprob-only happy path)

Reuse and extend patterns from `orion/memory/consolidation_classify.py` and `services/orion-memory-consolidation/app/boundary.py`.

| Field | Derivation |
|-------|------------|
| `novelty_score` | `binary_score_from_top_logprobs` on YES/NO at `NOVEL` line |
| `shift_scores` | Softmax over `top_logprobs` at `SHIFT` token for `{NONE, TOPIC, STANCE, REPAIR}` |
| `shift_kind` | Argmax of `shift_scores` |
| `confidence` | `min(2 * abs(p - 0.5))` across scored lines (NOVEL required; SHIFT when NOVEL=YES or always scored) |
| Parsed text | **Degraded fallback only** when logprobs missing |

**Forbidden:** Free-form model numbers (“novelty is 0.82”), prose ratings, or tissue φ as score input.

### `turn_change_appraisal` patch object

```json
{
  "baseline_mode": "prior_turn | session_window | none",
  "prior_correlation_id": "uuid-or-null",
  "novelty_score": 0.82,
  "shift_kind": "TOPIC",
  "shift_scores": {
    "NONE": 0.05,
    "TOPIC": 0.78,
    "STANCE": 0.12,
    "REPAIR": 0.05
  },
  "confidence": 0.91,
  "turn_change_status": "ok | degraded",
  "turn_change_ts": "2026-06-22T12:00:00+00:00"
}
```

When `baseline_mode=none` (first turn, empty window): `novelty_score` may be `null`; substrate does not fire.

---

## Baseline comparison rules

### Primary — prior turn (A)

Compare current User/Orion pair to the **immediately prior** pair from `WindowStore`.

### Fallback — session window (D)

Rolling text from `WindowStore`: concatenate last **N** prior turns (default **N=3**, env `TURN_CHANGE_WINDOW_TURNS`). Same transcript builder pattern as `build_window_transcript` in memory-consolidation.

### Fallback triggers (both)

1. **No prior turn** in window → use session window text if any; else `baseline_mode: none`.
2. **Low confidence on prior-turn comparison** → re-appraise using session window baseline. Low = NOVEL YES/NO margin below **`TURN_CHANGE_CONFIDENCE_MARGIN`** (default **0.15**).

Second appraisal call is **change-only** (NOVEL + SHIFT lines) via `reappraise_with_session_window()` helper.

---

## Architecture

### RPC strategy

**Extend existing `classify_turn`** (one quick LLM RPC per turn on happy path). Parsing lives in new module `orion/memory/turn_change_classify.py` so a future split to a separate RPC remains cheap.

Gateway options (unchanged pattern):

```python
{
  "return_logprobs": True,
  "logprobs_top_k": 4,  # cover SHIFT enum tokens
  "logprob_summary_only": False,
  "max_tokens": 16,
  "llm_route": "quick",
}
```

### Data flow

```
chat_history_log committed
  → sql-writer emits memory.turn.persisted.v1
  → memory-consolidation handle_memory_turn_persisted
      → classify_turn (extended prompt + logprobs)
      → [if fallback] reappraise_with_session_window
      → publish_spark_meta_patch
          turn_change_appraisal + memory_significance_score + conversation_boundary_score
      → [if novelty_score >= threshold] publish substrate organ_signal
      → window_store.append_turn (unchanged)
```

Consumers read patch asynchronously (same ~1s lag as existing `memory_classify_ts` pattern).

### Substrate feed (B)

When `turn_change_status=ok` and `novelty_score >= TURN_CHANGE_SUBSTRATE_THRESHOLD` (default **0.65**):

Publish `OrionSignalV1` → `signal_to_molecule` (see `orion/substrate/signal_bridge.py` tests) with dimensions keyed by `shift_kind`:

| `shift_kind` | Gradient emphasis |
|--------------|-------------------|
| `TOPIC` | `novelty`, `salience` |
| `STANCE` | `contradiction`, `salience` |
| `REPAIR` | `contradiction` (aligns with repair-pressure vocabulary) |
| `NONE` + high NOVEL | low `salience` perturbation (deadband) |

Include `source_event_id=correlation_id` and `causal_parents` linking to the chat turn.

---

## Deprecations (no shadow tissue)

| Component | v1 action |
|-----------|-----------|
| `OrionTissue.propagate` in spark-introspector `handle_trace` | **Remove** from hot path |
| LLM gateway `_spark_ingest_for_body` / `_spark_post_ingest_for_reply` φ stamping | **Remove** for turn semantics |
| `turn_effect_from_spark_meta` tissue `phi_*` evidence | **Replace** with `turn_effect_from_appraisal` or deprecate keys |
| `introspect_spark.j2` φ delta table | **Replace** input with `turn_change_appraisal` + optional `SelfStateV1` snapshot text |
| `build_classify_prompt` `phi_after.novelty` | **Remove**; use appraisal when present |
| Tissue shadow / charter comparison mode | **Not in scope** |

On `turn_change_status=degraded`: telemetry `novelty=null`. **Never** fall back to tissue.

---

## Module layout

| File | Action | Responsibility |
|------|--------|----------------|
| `orion/memory/turn_change_classify.py` | Create | Prompt builder, enum softmax, margin/confidence, baseline modes |
| `orion/memory/consolidation_classify.py` | Modify | Unified four-line prompt; delegate NOVEL/SHIFT parsing |
| `services/orion-memory-consolidation/app/classify.py` | Modify | Baseline selection, fallback second call, patch fields |
| `services/orion-memory-consolidation/app/worker.py` | Modify | Substrate emit post-patch |
| `services/orion-spark-introspector/app/worker.py` | Modify | Telemetry `novelty` from appraisal; remove tissue propagate |
| `services/orion-llm-gateway/app/llm_backend.py` | Modify | Stop tissue φ stamping for chat |
| `orion/schemas/telemetry/turn_effect.py` | Modify | Appraisal-based turn effect helper |
| `orion/memory/consolidation_classify.py` | Modify | Classify prompt without tissue φ |
| `tests/test_turn_change_classify.py` | Create | Logprob fixtures, fallback, margin |
| `services/orion-memory-consolidation/tests/test_classify_turn_change.py` | Create | Integration patch shape |

---

## Environment variables

Add to `services/orion-memory-consolidation/.env_example` (sync via `scripts/sync_local_env_from_example.py`):

| Key | Default | Meaning |
|-----|---------|---------|
| `TURN_CHANGE_CONFIDENCE_MARGIN` | `0.15` | Min YES/NO margin before session-window fallback |
| `TURN_CHANGE_SUBSTRATE_THRESHOLD` | `0.65` | Min `novelty_score` to emit substrate signal |
| `TURN_CHANGE_WINDOW_TURNS` | `3` | Prior turns in session-window baseline |

---

## Error handling

| Condition | Behavior |
|-----------|----------|
| LLM timeout / RPC failure | `turn_change_status=degraded`; no substrate emit |
| Missing logprobs | Degraded; text parse only if explicit fallback enabled |
| Low confidence after D retry | Patch with scores + low `confidence`; no substrate emit |
| First turn, empty window | `baseline_mode=none`; `novelty_score=null` |

---

## Testing

### Unit

- `binary_score_from_top_logprobs` on NOVEL line (existing tests extended)
- Four-way `shift_scores` from synthetic `top_logprobs`
- Confidence margin calculation
- Fallback trigger when margin < 0.15
- Prompt includes prior turn vs window transcript

### Integration

- `handle_memory_turn_persisted` patch contains `turn_change_appraisal`
- Routine follow-up fixture → low `novelty_score` with high confidence
- Topic-pivot fixture → high `novelty_score`, `shift_kind=TOPIC`
- Substrate emit fires only above threshold

### Manual acceptance

1. Hub chat turn → `spark_meta.turn_change_appraisal` within patch timeout
2. Telemetry `novelty` matches appraisal after patch (not tissue 0.0)
3. No `phi_before`/`phi_after` from gateway on new turns
4. Confident novel turn → bus message on substrate signal channel

---

## Known limits

1. **Async patch** — First telemetry tick may precede appraisal (same as `memory_classify_ts` today).
2. **Classifier bias** — Logprobs reflect model uncertainty on constrained tokens, not ground truth.
3. **SelfStateV1 poll lag** — Field condition may trail chat grammar; substrate emit on novel turns adds direct perturbation.
4. **Functional only** — No claim about phenomenal experience.

---

## Appendix A — Future exploration (not v1)

### C — Prior turn + substrate context hints

Add text-only hints to the appraisal prompt (no φ):

- `repair_pressure_level` / `level_label` from hub `substrate_effect_pipeline`
- `conversation_phase.phase_change`
- Latest `SelfStateV1.summary_labels` (text)

Evaluate after v1 calibration of logprob margins.

### Separate change-appraiser RPC

Split from unified `classify_turn` if prompts interfere or degrade rates diverge. Parsing module already extracted to enable this.

---

## Appendix B — Relationship to memory consolidation

This spec **extends** the pipeline in `docs/superpowers/plans/2026-06-16-memory-consolidation-pipeline.md`. Same hook (`memory.turn.persisted`), same patch channel (`chat.history.spark_meta.patch`). Memory significance and boundary scores unchanged in semantics; novelty becomes a first-class logprob signal instead of tissue φ contamination.
