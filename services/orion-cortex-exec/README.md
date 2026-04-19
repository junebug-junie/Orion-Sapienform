# Orion Cortex Exec

**Cortex Exec** is the execution engine for the Cognitive Runtime. It receives a `PlanExecutionRequest` from the Orchestrator, decomposes it into steps (using Planner/Agents), and coordinates workers (LLM, Recall, Tools) to fulfill the request.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion-cortex-exec:request` | `CHANNEL_EXEC_REQUEST` | `cortex.exec.request` | Request from Orchestrator. |

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

## Running & Testing

### Run via Docker
```bash
docker-compose up -d orion-cortex-exec
```

### Self-study repo root override
If self-study runs in a flattened container layout, set `ORION_REPO_ROOT` to the mounted repo root to override auto-detection; otherwise the service searches upward for repo markers and falls back to the container app root.

### Self-study GraphDB readback
`self_retrieve` can read persisted self-study RDF back from GraphDB when either `RECALL_RDF_ENDPOINT_URL` or `GRAPHDB_URL` / `GRAPHDB_REPO` are present in the environment. If that persisted backend is unavailable, Cortex Exec falls back explicitly to the in-process self-study snapshot path instead of silently widening trust semantics.

### Backfill Journal PageIndex-indcies
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
