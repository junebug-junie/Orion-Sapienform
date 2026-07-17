# Orion Cortex Exec

**Cortex Exec** is the execution engine for the Cognitive Runtime. It receives a `PlanExecutionRequest` from the Orchestrator, decomposes it into steps (using Planner/Agents), and coordinates workers (LLM, Recall, Tools) to fulfill the request.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-cortex-exec:request` | `CHANNEL_EXEC_REQUEST` | `cortex.exec.request` | Direct-RPC request from Orchestrator (bare channel, one per `EXEC_LANE` container -- see below). Handled by `handle()`/`svc` (`Rabbit` chassis, single-consumer). |
| `orion:verb:request` | (fixed) | `verb.request` | Shared verb-dispatch broadcast (`Hunter` chassis -- Redis pub/sub, every subscriber runs every message). Only the `chat`-lane container (and unset/empty `EXEC_LANE`, which falls back to `chat`) subscribes to this -- see "Exec lane containers" below. |
| `orion:cortex:pre_turn_appraisal:request` | `CHANNEL_PRE_TURN_APPRAISAL_REQUEST` | `pre_turn_appraisal.request.v1` | Pre-turn appraisal RPC from Hub (logprob repair pressure). |

### Exec lane containers

Four containers run in production, distinguished by `EXEC_LANE`: `legacy` (base, unset defaults here), `chat`, `spark`, `background` -- each with its own `CHANNEL_EXEC_REQUEST[_LANE]` bare RPC channel. Only `chat` (and empty-string `EXEC_LANE`, which `app/main.py` maps to `chat`) additionally subscribes to the shared `orion:verb:request` broadcast (`app/main.py`, `_lane in {"chat", ""}`) -- this is where chat-lane verbs (`chat_general`/`chat_quick`) actually land, since orion-cortex-orch's lane routing structurally excludes chat from ever using the direct per-lane RPC path.

`legacy` previously also matched that set, meaning both `legacy` and `chat` independently ran every chat-lane verb request in full -- confirmed live, real duplicate LLM gateway calls, fixed 2026-07-13. `legacy` still serves its own real, distinct direct-RPC traffic (e.g. `orion-thought`'s `stance_react`) on its bare `CHANNEL_EXEC_REQUEST` channel -- that's unrelated and unaffected by this fix. See `docs/superpowers/pr-reports/2026-07-13-chat-lane-duplicate-exec-fix-pr.md`.

### Pre-turn appraisal RPC (repair pressure v2)

Second Rabbit listener (`app/pre_turn_appraisal.py`) serves Hub pre-turn appraisal requests. Paradigms are resolved from `orion/substrate/appraisal/paradigms/registry.py` (`PARADIGM_REGISTRY`); today `repair_pressure` runs logprob YES/NO probes via LLM Gateway and returns a `TurnAppraisalBundleV1`.

**Flow**

1. Hub publishes `PreTurnAppraisalRequestV1` on `orion:cortex:pre_turn_appraisal:request`.
2. Handler loops `paradigms_requested`, builds each paradigm from `PARADIGM_REGISTRY`, runs with timeout from `req.options.timeout_ms`.
3. `repair_pressure` paradigm calls LLM Gateway with `return_logprobs=true` (`REPAIR_PRESSURE_PROBE_ROUTE`, default `quick`).
4. Kind scores + weights (`REPAIR_PRESSURE_WEIGHTS_V2_PATH`) feed `assemble_repair_contract_delta()`; contract metadata attached when mode changes.
5. Reply on `orion:cortex:pre_turn_appraisal:result:{correlation_id}` as `TurnAppraisalBundleV1`.

**Operator enable** — Hub: `ENABLE_PRE_TURN_APPRAISAL=true`. Cortex-exec: **only one replica** may register the pre-turn listener (`ENABLE_PRE_TURN_APPRAISAL_HANDLER=true` on main `cortex-exec`; lane containers `chat` / `spark` / `background` must set `false` in compose to avoid RPC reply races).

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENABLE_PRE_TURN_APPRAISAL_HANDLER` | `true` (main only) | Register Rabbit listener on `orion:cortex:pre_turn_appraisal:request`. Lane replicas: `false`. |
| `REPAIR_PRESSURE_WEIGHTS_V2_PATH` | `config/substrate/repair_pressure_weights.v2.yaml` | Kind weights for level reduction. |
| `REPAIR_PRESSURE_PROBE_ROUTE` | `quick` | LLM Gateway route for logprob probes. |
| `CHANNEL_PRE_TURN_APPRAISAL_REQUEST` | `orion:cortex:pre_turn_appraisal:request` | Request channel. |
| `CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX` | `orion:cortex:pre_turn_appraisal:result` | Reply channel prefix. |

**Fail-closed notes**

- Missing weights file → slice notes include `weights_file_missing`, level stays 0.
- Unknown paradigm name → listed in `failed_paradigms`.
- Probe timeout / LLM failure → paradigm in `failed_paradigms`, Hub wiring returns no bundle (fail-closed when bus missing: log + skip).

**Tests**

```bash
cd services/orion-cortex-exec
pytest tests/test_pre_turn_appraisal_rpc.py -q

# Substrate paradigm + registry (repo root)
pytest tests/test_paradigm_registry.py tests/test_repair_pressure_v2_paradigm.py -q
```

**Related:** Hub `ENABLE_REPAIR_PRESSURE_SPEECH_WIRING` + executor `compile_speech_contract()` merge `repair_pressure_contract` into TURN CONTRACT text on the same turn.

### Grammar substrate (shadow observability)

| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:grammar:event` | `GRAMMAR_EVENT_CHANNEL` | `grammar.event.v1` | Execution trajectory trace (one per plan run). |

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PUBLISH_CORTEX_EXEC_GRAMMAR` | `false` | Enable grammar publish after plan execution. |
| `GRAMMAR_EVENT_CHANNEL` | `orion:grammar:event` | Grammar bus channel. |

Trace id format: `cortex.exec:{NODE_NAME}:{correlation_id}`. Execution semantics use `GrammarAtomV1.semantic_role` (e.g. `exec_step_started`), not custom `GrammarEventKind` values.

### Published Channels (Worker RPC)
Exec sends requests to these channels and listens for replies on ephemeral `reply_to` channels.

| Target Worker | Channel (Env Var) | Default Value | Kind |
| :--- | :--- | :--- | :--- |
| **LLM Gateway** | `CHANNEL_LLM_INTAKE` | `orion-exec:request:LLMGatewayService` | `llm.chat.request` |
| **Recall** | `CHANNEL_RECALL_INTAKE` | `orion-exec:request:RecallService` | `recall.query.request` |
| **Context Exec** | `CHANNEL_CONTEXT_EXEC_INTAKE` | `orion:exec:request:ContextExecService` | `context.exec.request` |
| **Council** | `CHANNEL_COUNCIL_INTAKE` | `orion:agent-council:intake` | `council.request` |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_EXEC_REQUEST` | `orion-cortex-exec:request` | Main input. |
| `CHANNEL_LLM_INTAKE` | ... | LLM worker channel. |
| `CHANNEL_RECALL_INTAKE` | ... | Recall worker channel. |
| `CHANNEL_CONTEXT_EXEC_INTAKE` | ... | Context-exec delegate channel (replaces planner-react + agent-chain). |
| `CHANNEL_COUNCIL_INTAKE` | ... | Council worker channel. |
| `ORION_REPO_ROOT` | auto-detected | Optional override for self-study repo-root resolution when the container layout differs from local dev. |
| `ORION_ACTION_OUTCOME_DB_URL` | `postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney` | Shared SQL store for autonomous action outcomes. When set, chat-stance `load_action_outcomes` reads the `action_outcomes` table (written by sql-writer from `action.outcome.emit.v1`); when blank it falls back to the per-container JSON file `ORION_ACTION_OUTCOME_STORE_PATH`. |

### AutonomyStateV2 on chat stance (default off)

`app/chat_stance.py` can run the AutonomyStateV2 reducer after social/reasoning locals are built and **before** writing `ctx["chat_social_bridge_summary"]`.

| Variable | Default (`.env_example`) | Description |
| :--- | :--- | :--- |
| `AUTONOMY_STATE_V2_REDUCER_ENABLED` | empty / not `true` | When `true`, compile typed evidence from stance locals → `reduce_autonomy_state` → `ctx["chat_autonomy_state_v2"]` + delta + debug keys. |

**Flow (flag on)**

1. `_project_social_from_beliefs` / `_compile_reasoning_summary` produce locals.
2. `compile_autonomy_evidence(...)` (omit-when-empty) stamps optional `signal_kind` / `dimension` / `value`.
3. `reduce_autonomy_state` mints tensions via `chat_evidence_to_tension` + `signal_drive_map` (no keyword pressure).
4. Debug: `chat_autonomy_evidence_debug`, `chat_autonomy_tension_debug` (from reducer `tensions_minted`), `chat_autonomy_movement_debug`.

**Do not enable** until the movement eval is green:

```bash
PYTHONPATH=. python orion/autonomy/evals/run_autonomy_v2_movement_eval.py
```

Operator notes: [docs/autonomy_state_v2_reducer.md](../../docs/autonomy_state_v2_reducer.md). Package README: [orion/autonomy/README.md](../../orion/autonomy/README.md).

### Recent dispatch actions in chat evidence (P2, always on)

`_project_recent_dispatch_actions` (`app/chat_stance.py`) queries
`load_action_outcomes(subject="orion")` directly — **not** gated by
`AUTONOMY_STATE_V2_REDUCER_ENABLED` and not read from `ctx`, on purpose:
reading `ctx["chat_autonomy_state_v2"]["last_action_outcomes"]` instead would
silently go blank whenever the reducer resolved a different subject for that
turn (e.g. `"relationship"` during autonomy contextual fallback), even though
real Layer 9 (`orion-execution-dispatch-runtime`) dispatch outcomes exist
under `subject="orion"`. Result lands in `ctx["chat_recent_dispatch_actions"]`
(at most 3, newest-first, projected to exactly `{kind, summary, success,
observed_at}` — never `action_id`/`query`/`articles`/`salience`) and renders
in `orion/cognition/prompts/chat_general.j2`'s EVIDENCE-GATED CLAIMS section.
Fail-open: `[]` on any DB/parse failure, never raises.

```bash
pytest services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py -q
```

### drive_state.v1 visibility on chat stance (default off)

| Variable | Default (`.env_example`) | Description |
| :--- | :--- | :--- |
| `CHAT_STANCE_DRIVE_STATE_VISIBLE` | empty / not `true` | When `true`, `build_chat_stance_inputs` surfaces drive measurement as a **sibling** `inputs["drive_state"]` key. |
| `CHAT_STANCE_DRIVE_STATE_FETCH_TIMEOUT_SEC` | `0.4` | Bounded fail-open timeout for the latest `drive_audits` row fetch. |
| `ORION_ACTION_OUTCOME_DB_URL` | conjourney DSN | Same Postgres sql-writer writes; chat stance reads latest `drive_audits` (`subject='orion'`) here. |

Measurement SoR is Postgres `drive_audits` (bus → sql-writer), not substrate graph snapshots. `ctx["chat_drive_state"]` is filled unconditionally when a meaningful row exists; the visibility flag only gates prompt `inputs["drive_state"]`.

```bash
pytest services/orion-cortex-exec/tests/test_drive_state_postgres.py -q
pytest services/orion-cortex-exec/tests/test_chat_stance_drive_state_projection.py -q
```

### Turn effect and drive tensions (bus)

When the executor computes `turn_effect` / `turn_effect_evidence`, they are included in `PlanExecutionResult.metadata`. Hub merges that metadata into each chat turn `spark_meta`, so `orion-spark-concept-induction` can derive `extract_tensions` from `spark_meta.turn_effect`. For graph-backed tension kinds on `DriveAudit`, `orion-rdf-writer` must still consume `memory.drives.audit.v1` after concept-induction publishes it.

**Stack verification (after deploy):** subscribe to the Hub chat turn channel (see Hub `chat_history_turn_channel`) and confirm envelope payloads include `spark_meta.turn_effect` on turns where phi telemetry ran. Confirm `orion-spark-concept-induction` logs `handle_envelope` for that channel and `orion-rdf-writer` logs RDF for `memory.drives.audit.v1`. In GraphDB, query the latest `DriveAudit` for your subject and check `orion:derivedFromTension` / `orion:tensionKind` bindings.

### Collapse mirror verbs and φ-gated causal density

Cortex Exec registers collapse verbs in `app/collapse_verbs.py` (`orion.collapse.log`, `orion.collapse.enrich`, `orion.collapse.score`). Scoring logic lives in `orion/collapse/service.py` and is invoked by `orion.collapse.score`.

**Lanes** (`mirror_kind()` in `orion/schemas/collapse_mirror.py`):

| Lane | Typical observer / origin | Scoring |
| :--- | :--- | :--- |
| `strict` | Juniper-observed, or `source_service` → `collapse_mirror_service` | Self-reported numeric signals only (unchanged). |
| `metacog` | Orion-observed, or `source_service` → `metacog` | Blends self-report with computed `SelfStateV1` φ evidence when available. |
| `unknown` | Everything else | Self-reported numeric signals only (unchanged). |

For **metacog-lane** entries, `ScoreCausalDensityVerb` hydrates the latest fresh `self_state` via `hydrate_felt_state_ctx()` (`app/substrate_felt_state_reader.py`, same reader used by `chat_stance.py`) and passes it to `score_causal_density_with_self_state()`. The blend is `0.35 × self_report_score + 0.65 × phi_evidence_score` (clamped `[0,1]`). φ evidence combines max `prediction_error_scores`, `overall_condition` severity rank, and a bump when `trajectory_condition == "degrading"`. Narrative fields (`summary`, `mantra`, `trigger`, etc.) are untouched — only `causal_density.score` and the derived `is_causally_dense` boolean change.

If `self_state` is absent, stale, or fails to parse, metacog-lane scoring falls back to pure self-report (fail-open, same as strict/unknown). `score_causal_density(event_id)` remains a no-`self_state` entry point for callers that do not need the blend.

**Relevant env (no new keys for this feature):**

| Variable | Default (`.env_example`) | Role |
| :--- | :--- | :--- |
| `ENABLE_SUBSTRATE_FELT_STATE_CTX` | `true` | Gates Postgres reads in `SubstrateFeltStateReader`. When `false`, metacog scoring uses self-report only. |
| `SUBSTRATE_FELT_STATE_DATABASE_URL` | `postgresql://…/conjourney` | Postgres URL for `substrate_self_state` (written by `orion-self-state-runtime`). |
| `SUBSTRATE_FELT_STATE_MAX_AGE_SEC` | `120` | Freshness window; stale rows are not injected. |

**Tests:** `tests/test_collapse_service_causal_density.py` (repo root) covers strict-lane parity, metacog pull-down/pull-up, dict-shaped `self_state` coercion, and malformed-input fallback.

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-cortex-exec
```

### Self-study repo root override
If self-study runs in a flattened container layout, set `ORION_REPO_ROOT` to the mounted repo root to override auto-detection; otherwise the service searches upward for repo markers and falls back to the container app root.

### Self-study GraphDB readback
`self_retrieve` can read persisted self-study RDF back from GraphDB when either `RECALL_RDF_ENDPOINT_URL` or `GRAPHDB_URL` / `GRAPHDB_REPO` are present in the environment. If that persisted backend is unavailable, Cortex Exec falls back explicitly to the in-process self-study snapshot path instead of silently widening trust semantics.

### Backfill Journal PageIndex indices
```docker exec -i orion-athena-sql-writer python - <<'PY'
from app.db import get_session, remove_session
from app.models import JournalEntrySQL, JournalEntryIndexSQL
from orion.journaler import JournalEntryWriteV1, build_journal_entry_index_payload

sess = get_session()
created = 0
skipped = 0

try:
    existing_ids = {row[0] for row in sess.query(JournalEntryIndexSQL.entry_id).all()}

    rows = (
        sess.query(JournalEntrySQL)
        .order_by(JournalEntrySQL.created_at.asc())
        .all()
    )

    for row in rows:
        if row.entry_id in existing_ids:
            skipped += 1
            continue

        payload = JournalEntryWriteV1(
            entry_id=row.entry_id,
            created_at=row.created_at,
            author=row.author,
            mode=row.mode,
            title=row.title,
            body=row.body,
            source_kind=row.source_kind,
            source_ref=row.source_ref,
            correlation_id=row.correlation_id,
        )

        # Generates enriched index metadata (trigger/stance/facets) used by orion-pageindex export.
        index_payload = build_journal_entry_index_payload(
            payload,
            trigger=None,
            chat_stance=None,
            stance_metadata=None,
        )

        sess.merge(JournalEntryIndexSQL(**index_payload))
        created += 1

        if created % 100 == 0:
            sess.commit()
            print(f"committed {created} rows...")

    sess.commit()
    print(f"done: created={created} skipped={skipped}")

finally:
    try:
        sess.close()
    finally:
        remove_session()
PY
```

### Smoke Test
Exec is tested via the Orchestrator flow.
```bash
python scripts/bus_harness.py brain "plan a party"
```
