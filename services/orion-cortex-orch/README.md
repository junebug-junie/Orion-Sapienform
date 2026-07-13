# Orion Cortex Orchestrator

The **Cortex Orchestrator** (Orch) is the entry point for the Cognitive Runtime. It accepts high-level client requests (via `orion-cortex:request`), manages the session state, and delegates execution planning to **Cortex Exec**.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-cortex:request` | `ORCH_REQUEST_CHANNEL` | `cortex.orch.request` | Client requests (Brain, Agent, Council modes). |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-cortex-exec:request` | `CORTEX_EXEC_REQUEST_CHANNEL` | `cortex.exec.request` | Delegation to Cortex Exec. |
| (Caller-defined) | (via `reply_to`) | `cortex.orch.result` | Final result sent back to client. |
| `orion:grammar:event` | `GRAMMAR_EVENT_CHANNEL` | `grammar.event.v1` | Shadow route-arbitration trace (lane pick, mind-gate decision, output mode) for the substrate-runtime `route_grammar` reducer. Off by default. |

### Route arbitration visibility

`call_verb_runtime()` computes, per turn: which execution lane was picked and why (`resolve_execution_lane`), whether "mind" projection fired or was skipped and why, and the output mode. These facts are:

1. Always attached to the returned `VerbResultV1.output["_route_metadata"]` (no flag — always on, zero schema/bus cost) and merged into `main.py`'s `final_meta["route_metadata"]` on the client-facing response.
2. Published as a `GrammarEventV1` trace (`trace_id` prefix `orch.route:{node}:{correlation_id}`, `source_service=orion-cortex-orch`) when `PUBLISH_CORTEX_ORCH_GRAMMAR=true` (default), for the substrate-runtime `route_grammar` reducer to materialize into `active_route_arbitration`. Requires `manual_migration_route_substrate_loop.sql` applied on substrate-runtime's DB and `ENABLE_ROUTE_GRAMMAR_REDUCER=true` there. See `docs/superpowers/specs/2026-07-12-orch-route-grammar-lane-design.md`.

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `ORCH_REQUEST_CHANNEL` | `orion-cortex:request` | Input channel. |
| `CORTEX_EXEC_REQUEST_CHANNEL` | `orion-cortex-exec:request` | Output channel to Exec. |
| `REDIS_URL` | ... | Redis connection. |
| `PUBLISH_CORTEX_ORCH_GRAMMAR` | `true` | Publish route arbitration as a `GrammarEventV1` trace. Fire-and-forget; a publish failure never affects the chat response. |
| `GRAMMAR_EVENT_CHANNEL` | `orion:grammar:event` | Channel used for the route-arbitration grammar trace above. |

## Compactor workflows

Orch executes two compactor cognition workflows in `app/workflow_runtime.py`. Both build their digest verb request through the shared `_build_compactor_digest_request` / `_compactor_digest_from_payload` helpers, and both use the shared budget/parse helpers in `orion/cognition/compactor/`.

### `chat_history_compactor_pass`

Pipeline: resolve window (`orion/cognition/chat_history_compactor/window.py`) → fetch turns via `skills.chat.discussion_window.v1` → digest via brain-lane verb `chat_history_compactor_digest_v1` → upsert indexed memory card → append journal entry.

Behavior contract:

- **Window bounds**: `window_mode` is `day` (yesterday, `America/Denver`, covers the full day to `time.max`) or `rolling` (default 24h). Request `lookback_hours` is capped at 14 days; unknown `window_mode` values fail loud (`chat_compactor_window_invalid`).
- **Digest route retry**: the digest verb runs on the `chat` LLM route first, retrying once on `quick`. Over-budget digests fail loud with no retry and no persistence (`compactor_output_over_budget:<field>`).
- **Quiet windows persist nothing**: zero turns or an empty transcript writes no card and no journal stub; the result reports the skip honestly.
- **Card persistence degrades, never discards**: one active card per `compactor_index` via `upsert_indexed_compactor_card` (enforced by the partial unique index `idx_mc_active_compactor_index`). If the card write fails for any reason, the workflow still appends the journal entry and reports `card_persist_skipped_reason` in workflow metadata.
- **Idempotent journal**: journal entry id is a stable UUIDv5 of `workflow_id|compactor_index`, so re-runs of the same window overwrite rather than duplicate.

Requires cortex-orch `RECALL_PG_DSN` for card writes and cortex-exec SQL access for the discussion window skill. Scheduling/bootstrap lives in `services/orion-actions` (daily 06:00 Denver; see that README).

### `github_compactor_pass`

Daily merged-PR digest: fetch via `skills.repo.github_recent_prs.v1`, digest via `github_compactor_digest_v1` (single attempt, fail loud), supersede-slot card (`compactor_slot`), journal append. Quiet days write a journal entry noting the card was left unchanged.

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-cortex-orch
```

### Tests and evals
```bash
pytest services/orion-cortex-orch/tests -q
pytest services/orion-cortex-orch/evals -q   # deterministic digest budget/quiet-honesty evals
```

### Smoke Test
Use the bus harness in "Brain" mode.
```bash
python scripts/bus_harness.py brain "hello world"
```
