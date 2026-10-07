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

### RPC-health snapshot publish (default off)

Step 3 of `docs/superpowers/specs/2026-07-23-rpc-health-signal-gateway-wiring-design.md`.
Same mechanism as `orion-cortex-exec`'s (see that service's README for the full write-up):
`RPC_HEALTH_PUBLISH_ENABLED=true` starts a periodic task draining this process's real
`OrionBusAsync.rpc_request()` outcome tally onto `orion:rpc_health:snapshot`, read via the
existing `_bus_for_rpc()` helper so it drains the real `_rpc_bus` fork (used by
`DecisionRouter`/`workflow_runtime`), not the idle `svc.bus`. Live-verified; on in
`.env_example`.

Per-hop breakdown (2026-09-24, gated by `RPC_HEALTH_CHANNEL_LATENCY_ENABLED`): the chat
lane's hand-rolled `orion:verb:request` round trip is recorded as hop `verb:<verb_name>`;
metacog dispatch (`dispatch_metacog_trigger`) passes `health_label="log_orion_metacognition"`,
so its hop is `<background exec channel>#log_orion_metacognition`. That call runs on the
equilibrium Hunter's own bus, which the publish loop folds in hop-only (per-hop stats merged,
pooled fields untouched). Publishes `instance="main"`.

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `ORCH_REQUEST_CHANNEL` | `orion-cortex:request` | Input channel. |
| `CORTEX_EXEC_REQUEST_CHANNEL` | `orion-cortex-exec:request` | Output channel to Exec. |
| `REDIS_URL` | ... | Redis connection. |
| `PUBLISH_CORTEX_ORCH_GRAMMAR` | `true` | Publish route arbitration as a `GrammarEventV1` trace. Fire-and-forget; a publish failure never affects the chat response. |
| `GRAMMAR_EVENT_CHANNEL` | `orion:grammar:event` | Channel used for the route-arbitration grammar trace above. |
| `RPC_HEALTH_PUBLISH_ENABLED` | `true` | Periodically publish this process's real RPC-health snapshot. See "RPC-health snapshot publish" above. |
| `RPC_HEALTH_PUBLISH_INTERVAL_SEC` | `30` | Publish cadence for the above. |
| `RPC_HEALTH_CHANNEL_LATENCY_ENABLED` | `true` | Include per-hop `channel_latency` in each snapshot. On since 2026-09-24, after signal-gateway and equilibrium shipped the field (consumer-first). |

## Compactor workflows

Orch executes two compactor cognition workflows in `app/workflow_runtime.py`. Both cover the full window (every merged PR / chat turn of the previous Denver day on a scheduled run) and store `journal_body` untrimmed. The split (2026-09-30):

- **Orch, synchronous**: resolve the window, fetch (GitHub PR walk / chat discussion window), build the chunk inputs (`orion/cognition/*_compactor/digest.py`). A quiet window needs no LLM and finalizes inline.
- **`compactor.digest` durable run** (orion-durable-runs, admitted on `llm.route.agent`, background): every chunk digest and the merge call is one checkpointed node holding a GPU pool hold (`options.gpu_lease`). Orch submits it through `durable_runs.dispatch_durable_run` and replies `status="accepted"` with `metadata.workflow.durable_run` (`run_id`, `deadline_at`, `generation`). There is no in-process digest call or retry.
- **Orch, finalize**: the durable run calls back with `workflow_request.durable_digest` (`CompactorDigestResultV1`); the same pass function writes the memory card + journal entry (stable ids) and the workflow result, and `execute_chat_workflow` notifies then (never on `accepted`).
- **Idempotency**: `run_id = compactor:<workflow>:<window>[:<repo>]:<sha256(brief+admission)[:12]>`, `correlation_id = uuid5(run_id)`, `deadline_at = window end + 24h` -- a re-dispatch of the same input finds the existing run. Receipt `completed` -> reported as already finalized (`status=success`, nothing re-run); `failed`/`cancelled` -> next generation `:g2`... (max `COMPACTOR_MAX_RUN_GENERATIONS`); no receipt -> the dispatch fails (`compactor_durable_submit_failed`) and the scheduler retries with backoff.
- Requires `CORTEX_DURABLE_ADMISSION_ENABLED=true` (the compactors fail loudly without it).

Result metadata evidence: `digest_chunk_count`, `digest_merge_mode`, `digest_merge_skipped_reason`, `digest_llm_route`, `digest_attempts` (call attempts; pool waits are not attempts), `digest_gpu_roles`, `durable_run_id`, coverage (`total_count`, `covered_count`, `input_truncated`), `journal_body_chars`.

### `chat_history_compactor_pass`

Pipeline: resolve window (`orion/cognition/chat_history_compactor/window.py`) → fetch turns via `skills.chat.discussion_window.v1` → digest via brain-lane verb `chat_history_compactor_digest_v1` → upsert indexed memory card → append journal entry.

Behavior contract:

- **Window bounds**: `window_mode` is `day` (yesterday, `America/Denver`, covers the full day to `time.max`) or `rolling` (default 24h). Request `lookback_hours` is capped at 14 days; unknown `window_mode` values fail loud (`chat_compactor_window_invalid`).
- **Digest calls**: `agent` route inside the `compactor.digest` durable run (above). Over-budget card prose is trimmed to its cap (reported in the run's `trimmed_fields`), never the journal body. A malformed digest is one bounded attempt; a chunk that exhausts its attempts fails the run.
- **Quiet windows persist nothing**: zero turns or an empty transcript writes no card and no journal stub; the result reports the skip honestly.
- **Card persistence degrades, never discards**: one active card per `compactor_index` via `upsert_indexed_compactor_card` (enforced by the partial unique index `idx_mc_active_compactor_index`). If the card write fails for any reason, the workflow still appends the journal entry and reports `card_persist_skipped_reason` in workflow metadata.
- **Idempotent journal**: journal entry id is a stable UUIDv5 of `workflow_id|compactor_index`, so re-runs of the same window overwrite rather than duplicate.

Requires cortex-orch `RECALL_PG_DSN` for card writes and cortex-exec SQL access for the discussion window skill. Scheduling/bootstrap lives in `services/orion-actions` (daily 06:00 Denver; see that README).

### `github_compactor_pass`

Daily merged-PR digest: fetch via `skills.repo.github_recent_prs.v1` (paginated, window-bounded), digest via `github_compactor_digest_v1` in the `compactor.digest` durable run (map-reduce over every merged PR of the day), supersede-slot card (`compactor_slot`), journal append. Quiet days write a journal entry noting the card was left unchanged (inline, no durable run).

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

## Resource-admitted Curiosity runs

`context.metadata.durable_run.admission` selects the resource-admission path.
`CORTEX_DURABLE_ADMISSION_ENABLED=true` is the operator-template default and
requires the migrated durable runner and Gateway authority to be ready first.
Cortex waits at most `CORTEX_DURABLE_RECEIPT_TIMEOUT_SEC` (10 by default)
for a matching `durable.run.receipt.v1` confirming committed registration, then
returns `status=accepted` with the run ID, workflow, resource and durable status
in `metadata.durable_run`. Resource waiting never holds this RPC open. A missing
or invalid receipt returns `AdmissionUnconfirmed`; retry the same run ID or
inspect it at the runner, because registration may already have committed.
Requests without admission retain the existing synchronous/legacy behavior.
See `docs/architecture/durable-resource-admission.md` for ownership and rollout.
