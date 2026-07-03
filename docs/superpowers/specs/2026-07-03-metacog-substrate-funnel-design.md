# Metacog substrate funnel — design spec

**Date:** 2026-07-03  
**Status:** approved for implementation (operator request)

## Problem

Metacog fires on a timer with health-uptime pressure only. `is_causally_dense` is LLM-guessed in enrich (threshold 0.70) and never crosses on baseline ticks. Substrate `self_state` / execution projections exist but are not wired to triggers or publish scoring.

## Goal

Funnel substrate runtime to **trigger** dense/pulse metacog and **augment** publish with deterministic φ-grounded `causal_density`.

## Non-goals

- Tier-0 deterministic mirror (no LLM) on quiet baseline — follow-up
- Moving all felt-state lanes into prompts — compact cue only
- New bus channel or microservice

## Architecture

### Trigger (equilibrium)

Before baseline emit, read substrate Postgres (`self_state`, `execution_trajectory_projection`) via shared `hydrate_felt_state_ctx`. Compute `compute_substrate_eventfulness()`:

| Signal | Weight |
|--------|--------|
| `overall_surprise` ≥ 0.55 | +0.35 |
| `overall_condition` strained/unstable | +0.25 |
| `trajectory_condition` degrading | +0.20 |
| max `prediction_error_scores` ≥ 0.5 | +0.20 |
| any execution run `failed_step_count` > 0 | +0.25 |

- score ≥ 0.55 → `trigger_kind=dense`
- score ≥ 0.30 → `trigger_kind=pulse`
- else → fall through to existing baseline logic

Gated by `EQUILIBRIUM_METACOG_SUBSTRATE_TRIGGER_ENABLE` (default true when metacog on).

### Augment (cortex-exec)

1. **MetacogContextService:** `hydrate_felt_state_ctx(ctx)` + `metacog_substrate_cue` compact string in telemetry/context.
2. **MetacogPublishService:** `apply_causal_density_to_entry(entry, self_state)` before SQL publish (in-memory; no collapse store).
3. **Lineage fix:** `telemetry.trigger_kind` passes through `trigger.trigger_kind` (baseline/dense/pulse/manual).

### Scoring

Reuse `orion/collapse/service.py` blend: `0.35 × self_report + 0.65 × phi_evidence`. Dense flag: `score >= 0.6` (existing service threshold).

## Files

- `orion/substrate/felt_state_reader.py` (shared reader)
- `orion/substrate/metacog_trigger_signals.py`
- `orion/collapse/service.py` — `apply_causal_density_to_entry`
- `services/orion-cortex-exec/app/executor.py`
- `services/orion-cortex-exec/app/substrate_felt_state_reader.py` — re-export
- `services/orion-equilibrium-service/app/substrate_metacog_gate.py`
- `services/orion-equilibrium-service/app/service.py`
- env examples + tests

## Acceptance

- Unit tests: eventfulness scoring, lineage trigger_kind, apply_causal_density in-memory
- Baseline smoke: `telemetry.trigger_kind` matches trigger payload; publish applies substrate score when self_state present
