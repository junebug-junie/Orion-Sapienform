# PR: Turn Change Appraisal v1

**Branch:** `feat/turn-change-appraisal-v1`  
**Design:** `docs/superpowers/specs/2026-06-22-turn-change-appraisal-v1-design.md`  
**Plan:** `docs/superpowers/plans/2026-06-22-turn-change-appraisal-v1.md`

## Summary

Replaces tissue-derived turn novelty with logprob-calibrated `turn_change_appraisal` on every persisted chat turn (after the first in a window). Spark telemetry reads novelty from appraisal (never tissue fallback). High-confidence novel turns emit passthrough `OrionSignalV1` on `orion:signals:memory_consolidation`, bridged to substrate `organ_signal` molecules. LLM gateway stops φ stamping on chat turns.

## What changed

### Core scoring (`orion/memory/`)
- **`turn_change_classify.py`** — enum softmax, confidence/margin, four-line and two-line prompt builders, appraisal dict assembly
- **`turn_change_signal.py`** — `OrionSignalV1` builder with shift_kind → dimension gradients
- **`consolidation_classify.py`** — unified four-line classify prompt (no `phi_after`)

### Memory consolidation service
- **`boundary.py`** — four-line logprob parsing (novelty, shift, memory, boundary)
- **`classify.py`** — prior-turn baseline, session-window fallback on low margin, `turn_change_appraisal` patch
- **`worker.py`** — fetch prior turns before classify; substrate signal emit above threshold
- **`settings.py` / `.env_example`** — `TURN_CHANGE_*`, `CHANNEL_SIGNALS_PREFIX`

### Downstream consumers
- **`orion-spark-introspector`** — telemetry novelty from appraisal; tissue propagate removed from candidate/trace hot paths
- **`orion-llm-gateway`** — non-tissue spark meta only (`latest_user_message`, no φ stamping)
- **`turn_effect.py`** — `turn_effect_from_appraisal` preferred over φ deltas
- **`introspect_spark.j2`** — appraisal block replaces φ metric table

### Bus / signals
- **`orion/signals/registry.py`** — `memory_consolidation` organ + `turn_change` kind
- **`orion/substrate/signal_bridge.py`** — `(memory_consolidation, turn_change)` support
- **`orion/bus/channels.yaml`** — `orion:signals:memory_consolidation` channel

## Env (`.env_example` → local `.env`)

| Key | Default |
|-----|---------|
| `TURN_CHANGE_CONFIDENCE_MARGIN` | `0.15` |
| `TURN_CHANGE_SUBSTRATE_THRESHOLD` | `0.65` |
| `TURN_CHANGE_WINDOW_TURNS` | `3` |
| `CHANNEL_SIGNALS_PREFIX` | `orion:signals` |

Run `python scripts/sync_local_env_from_example.py` after merge (prefixes `TURN_CHANGE_` and `CHANNEL_SIGNALS_` added to sync script).

## Test plan

- [x] `PYTHONPATH=. pytest tests/test_turn_change_classify.py tests/test_consolidation_classify.py tests/test_turn_effect.py tests/test_substrate_signal_bridge.py` — 24 passed
- [x] `pytest services/orion-memory-consolidation/tests/` — 20 passed
- [x] `python -m compileall orion/memory services/orion-memory-consolidation/app services/orion-spark-introspector/app services/orion-llm-gateway/app` — exit 0
- [ ] Live stack: hub chat turn → `spark_meta.turn_change_appraisal` within patch timeout
- [ ] Spark telemetry `novelty` matches appraisal when `turn_change_status=ok`
- [ ] No `phi_before`/`phi_after` from gateway on new turns
- [ ] Confident novel turn → `orion:signals:memory_consolidation` with `signal.memory_consolidation.turn_change`

## Known v1 limits

- Async patch lag before appraisal visible in spark_meta
- Classifier bias / calibration not tuned on production traffic
- SelfStateV1 poll lag unchanged
- Substrate context hints (Appendix A) deferred

## Files changed (24)

`orion/memory/turn_change_classify.py`, `orion/memory/turn_change_signal.py`, `orion/memory/consolidation_classify.py`, `orion/schemas/telemetry/turn_effect.py`, `orion/signals/registry.py`, `orion/substrate/signal_bridge.py`, `orion/bus/channels.yaml`, `orion/cognition/prompts/introspect_spark.j2`, `services/orion-memory-consolidation/app/*`, `services/orion-spark-introspector/app/worker.py`, `services/orion-llm-gateway/app/llm_backend.py`, `scripts/sync_local_env_from_example.py`, tests (6 files), `services/orion-memory-consolidation/README.md`, `.env_example`
