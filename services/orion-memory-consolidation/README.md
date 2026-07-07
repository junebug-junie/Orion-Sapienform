# Orion Memory Consolidation

Subscribes to `orion:memory:turn:persisted` (sql-writer post-commit outbox), classifies each chat turn via LLM gateway quick lane logprobs, patches `chat_history_log.spark_meta`, tracks consolidation windows, and on boundary closure runs a **deterministic consolidation gate** (default) or legacy graph suggest.

## Consolidation output modes

| `MEMORY_CONSOLIDATION_OUTPUT` | Behavior |
|-------------------------------|----------|
| `crystallization_propose` (default) | Run `consolidation_memory_gate`; **skip** low-signal windows (`consolidation_status=skipped`) or **propose** `MemoryCrystallizationV1` for governor review |
| `graph_draft` | Legacy path: LLM `memory_graph_suggest` + pending graph draft insert (manual bridge only) |
| `skip_only` | Run gate for traceability; always mark window skipped — no crystallization or graph draft |

Gate thresholds: `MEMORY_CONSOLIDATION_MIN_NOVELTY` (default `0.35`), `MEMORY_CONSOLIDATION_MIN_SIGNIFICANCE` (default `0.40`).

Grammar repair evidence (read-only): `MEMORY_CONSOLIDATION_FETCH_GRAMMAR_EVIDENCE=true` queries `grammar_events` by `hub.chat:{NODE_NAME}:{correlation_id}` trace. Optional override DSN: `MEMORY_CONSOLIDATION_GRAMMAR_DSN`.

**Note:** Proposed crystallization IDs are stored in `memory_consolidation_windows.draft_id` until a dedicated `crystallization_id` column migration lands.

## Channels

| Direction | Channel |
|-----------|---------|
| In | `orion:memory:turn:persisted` |
| Out | `orion:chat:history:spark_meta:patch` |
| Out (threshold) | `orion:signals:memory_consolidation` (`signal.memory_consolidation.turn_change`) |
| Out (propose) | `orion:memory:crystallization:proposed` (`memory.crystallization.proposed.v1`) |

## Turn change appraisal

Each persisted turn (after the first in a window) gets a logprob-calibrated `turn_change_appraisal` patch on `spark_meta`: novelty score, shift kind, confidence, and baseline mode (`prior_turn` or `session_window` fallback). The first turn in a window uses `turn_change_status=skipped` (no baseline, no LLM call). High-confidence novel turns also emit `OrionSignalV1` on `orion:signals:memory_consolidation`.

| Env | Default | Purpose |
|-----|---------|---------|
| `TURN_CHANGE_CLASSIFY_ROUTE` | `metacog` | Gateway route for classify RPC (`metacog` or `quick`) |
| `TURN_CHANGE_CONFIDENCE_MARGIN` | `0.15` | Re-appraise vs session window when novelty margin is below this |
| `TURN_CHANGE_SUBSTRATE_THRESHOLD` | `0.65` | Minimum novelty to emit substrate signal |
| `TURN_CHANGE_WINDOW_TURNS` | `3` | Prior turns in session-window baseline |
| `CHANNEL_SIGNALS_PREFIX` | `orion:signals` | Bus prefix for organ signal publish |

## Bring-up

```bash
docker compose --env-file ../../.env --env-file ../orion-bus/.env -f docker-compose.yml up -d --build
```

Apply Postgres migration: `services/orion-sql-db/manual_migration_memory_consolidation_v1.sql`

## Smoke

```bash
PYTHONPATH=. python scripts/smoke_memory_consolidation_pipeline.py
bash scripts/smoke_memory_consolidation_gate.sh
```

Gate smoke runs deterministic unit tests (greeting skip + substantive propose). Pipeline smoke requires live stack (bus, sql-writer, postgres, llm-gateway, cortex).
