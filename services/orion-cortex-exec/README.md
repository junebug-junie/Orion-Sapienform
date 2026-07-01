# Orion Cortex Exec

**Cortex Exec** is the execution engine for the Cognitive Runtime. It receives a `PlanExecutionRequest` from the Orchestrator, decomposes it into steps (using Planner/Agents), and coordinates workers (LLM, Recall, Tools) to fulfill the request.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-cortex-exec:request` | `CHANNEL_EXEC_REQUEST` | `cortex.exec.request` | Request from Orchestrator. |

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
| **Planner** | `CHANNEL_PLANNER_INTAKE` | `orion-exec:request:PlannerReactService` | `agent.planner.request` |
| **Agent Chain** | `CHANNEL_AGENT_CHAIN_INTAKE` | `orion-exec:request:AgentChainService` | `agent.chain.request` |
| **Council** | `CHANNEL_COUNCIL_INTAKE` | `orion:agent-council:intake` | `council.request` |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_EXEC_REQUEST` | `orion-cortex-exec:request` | Main input. |
| `CHANNEL_LLM_INTAKE` | ... | LLM worker channel. |
| `CHANNEL_RECALL_INTAKE` | ... | Recall worker channel. |
| `CHANNEL_PLANNER_INTAKE` | ... | Planner worker channel. |
| `CHANNEL_AGENT_CHAIN_INTAKE` | ... | Agent Chain worker channel. |
| `CHANNEL_COUNCIL_INTAKE` | ... | Council worker channel. |
| `ORION_REPO_ROOT` | auto-detected | Optional override for self-study repo-root resolution when the container layout differs from local dev. |

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
