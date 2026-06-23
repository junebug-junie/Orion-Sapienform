# PR: Turn Change Appraisal v1

**Branch:** `feat/turn-change-appraisal-v1`  
**Base:** `main` (from `fe17da9c`)  
**Design:** `docs/superpowers/specs/2026-06-22-turn-change-appraisal-v1-design.md`  
**Plan:** `docs/superpowers/plans/2026-06-22-turn-change-appraisal-v1.md`

## Summary

Replaces tissue-derived turn novelty with logprob-calibrated `turn_change_appraisal` on persisted chat turns. Memory-consolidation classifies each turn (after the first in a window) against a prior-turn or session-window baseline, patches `spark_meta` via `orion:chat:history:spark_meta:patch`, and emits `OrionSignalV1` on `orion:signals:memory_consolidation` when novelty exceeds threshold. Spark telemetry reads novelty from appraisal only (no tissue fallback). LLM gateway stops φ stamping on chat turns.

## Motivation

Tissue φ deltas on the gateway hot path were a poor proxy for conversational shift: they mixed embedding geometry with turn semantics, stamped synchronously before consolidation, and leaked into telemetry even when appraisal was missing. This PR moves turn novelty to a dedicated classifier on the memory-consolidation rail, keeps tissue for semantic-upsert / snapshot paths, and wires substrate signals for high-confidence novel turns.

## What changed

### Core scoring (`orion/memory/`)
- **`turn_change_classify.py`** — enum softmax, confidence/margin, four-line and two-line prompt builders, appraisal dict assembly
- **`turn_change_signal.py`** — `OrionSignalV1` builder with shift_kind → dimension gradients and hub `causal_parents`
- **`consolidation_classify.py`** — unified four-line classify prompt (no `phi_after`)

### Memory consolidation service
- **`boundary.py`** — four-line logprob parsing (novelty, shift, memory, boundary); `scoring_source` on degraded paths
- **`classify.py`** — prior-turn baseline, session-window fallback on low margin, reappraisal guards, `turn_change_appraisal` patch
- **`worker.py`** — fetch prior turns before classify; best-effort substrate publish (won't block append_turn)
- **`settings.py` / `.env_example`** — `TURN_CHANGE_*`, `CHANNEL_SIGNALS_PREFIX`

### Downstream consumers
- **`orion-spark-introspector`** — telemetry novelty from `_novelty_from_spark_meta()`; tissue propagate removed from candidate/trace hot paths; `turn_effect_from_appraisal` preferred
- **`orion-llm-gateway`** — thin spark meta only (`latest_user_message`, `trace_verb`); no tissue ingest or φ stamping
- **`turn_effect.py`** — `turn_effect_from_appraisal`; φ fallback blocked when appraisal is `degraded`
- **`introspect_spark.j2`** — appraisal block replaces φ metric table

### Bus / signals
- **`orion/signals/registry.py`** — `memory_consolidation` organ + `turn_change` kind
- **`orion/substrate/signal_bridge.py`** — `(memory_consolidation, turn_change)` support
- **`orion/bus/channels.yaml`** — `orion:signals:memory_consolidation` channel

### Docs
- **`services/orion-memory-consolidation/README.md`** — turn change appraisal, env, channels
- **`services/orion-spark-introspector/README.md`** — appraisal novelty path, spark_meta patch channel
- **`services/orion-llm-gateway/README.md`** — thin spark_meta contract

### Review hardening (post-initial implementation)
- Window cap aligned with `TURN_CHANGE_WINDOW_TURNS`; prior-turn text clip at 300 chars
- First turn in window: `turn_change_status=skipped` (no LLM call)
- Substrate publish isolated from classify/patch path
- Reappraisal try/except; skip reappraisal when only one prior turn exists
- Degraded baseline preservation; confidence refresh on reappraisal
- Dead tissue helpers removed from gateway and spark worker

## Env (`.env_example` → local `.env`)

| Key | Default | Purpose |
|-----|---------|---------|
| `TURN_CHANGE_CONFIDENCE_MARGIN` | `0.15` | Re-appraise vs session window when novelty margin is below this |
| `TURN_CHANGE_SUBSTRATE_THRESHOLD` | `0.65` | Minimum novelty to emit substrate signal |
| `TURN_CHANGE_WINDOW_TURNS` | `3` | Prior turns in session-window baseline |
| `CHANNEL_SIGNALS_PREFIX` | `orion:signals` | Bus prefix for organ signal publish |

After merge, run `python scripts/sync_local_env_from_example.py` (prefixes `TURN_CHANGE_` and `CHANNEL_SIGNALS_` added to sync script). Restart memory-consolidation container if already running.

## Test plan

### Automated (done)

```bash
cd .worktrees/feat/turn-change-appraisal-v1  # or repo root on branch
PYTHONPATH=. ./orion_dev/bin/python -m pytest \
  tests/test_turn_change_classify.py \
  tests/test_consolidation_classify.py \
  tests/test_turn_effect.py \
  tests/test_substrate_signal_bridge.py \
  services/orion-memory-consolidation/tests/ -q
```

**Result:** 56 passed

```bash
PYTHONPATH=. python -m compileall orion/memory \
  services/orion-memory-consolidation/app \
  services/orion-spark-introspector/app \
  services/orion-llm-gateway/app
```

**Result:** exit 0

### Live stack (manual — UNVERIFIED)

- [ ] Hub chat turn → `spark_meta.turn_change_appraisal` within patch timeout
- [ ] Spark telemetry `novelty` matches appraisal when `turn_change_status=ok`; `null` when `skipped`/`degraded`
- [ ] No `phi_before`/`phi_after` from gateway on new turns
- [ ] Confident novel turn → bus message on `orion:signals:memory_consolidation` with `signal.memory_consolidation.turn_change`
- [ ] First turn in session window → `turn_change_status=skipped`, no consolidation LLM call

## Known v1 limits

- Async patch lag before appraisal visible in `spark_meta`
- Classifier bias / calibration not tuned on production traffic
- SelfStateV1 poll lag unchanged
- Legacy rows without `turn_change_appraisal` may still show φ-based `turn_effect` (intentional)
- Substrate context hints (Appendix A in design) deferred

## Commits (7)

1. `feat: turn change appraisal v1 with logprob scoring and substrate signals`
2. `fix: spark paths prefer appraisal turn_effect and drop dead tissue encoding`
3. `fix: address code review I1-I4 for turn change appraisal`
4. `fix: close review gaps I5-I6 and minor findings M1-M4`
5. `fix: harden reappraisal path and remove dead tissue helpers`
6. `fix: pre-merge review I1-I4 and test hardening`
7. `docs: document turn_change_appraisal in spark and gateway READMEs`

## Files changed (27)

`orion/memory/turn_change_classify.py`, `orion/memory/turn_change_signal.py`, `orion/memory/consolidation_classify.py`, `orion/schemas/telemetry/turn_effect.py`, `orion/signals/registry.py`, `orion/substrate/signal_bridge.py`, `orion/bus/channels.yaml`, `orion/cognition/prompts/introspect_spark.j2`, `services/orion-memory-consolidation/app/*`, `services/orion-memory-consolidation/.env_example`, `services/orion-memory-consolidation/README.md`, `services/orion-memory-consolidation/tests/*`, `services/orion-spark-introspector/app/worker.py`, `services/orion-spark-introspector/README.md`, `services/orion-llm-gateway/app/llm_backend.py`, `services/orion-llm-gateway/README.md`, `scripts/sync_local_env_from_example.py`, `tests/test_turn_change_classify.py`, `tests/test_consolidation_classify.py`, `tests/test_turn_effect.py`, `tests/test_substrate_signal_bridge.py`, `docs/superpowers/pr-reports/2026-06-22-turn-change-appraisal-v1-pr.md`
