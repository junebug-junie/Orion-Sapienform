# Orion Spark Introspector

The **Orion Spark Introspector** is a headless worker service that listens for "spiky" or emotionally significant turns detected by the Spark Engine and turns them into structured, internal introspection notes.

It sits between **Orion Brain**, **Cortex Orchestrator**, and **SQL Writer**, wiring them together as a small metacognition loop:

1. **Brain** publishes "introspection candidates" (spiky turns) on a Redis channel.
2. **Spark Introspector** consumes those candidates and asks Cortex to run a `spark_introspect` verb.
3. **Cortex Orchestrator** fans the request out to `LLMGatewayService` over the exec bus.
4. **Brain** reflects on its own state and returns an internal note.
5. **Spark Introspector** publishes a normalized row to SQL Writer, which persists to Postgres.

This gives Orion a growing table of its own internal reflections keyed by `trace_id` and enriched with Spark metadata (φ, SelfField, tags, etc.).

---

## High-Level Data Flow

**Channels & services involved:**

- **Input (from Brain):**
  - `CHANNEL_SPARK_INTROSPECT_CANDIDATE = "orion:spark:introspect:candidate"`
- **Cortex Orchestrator (bus):**
  - `CORTEX_ORCH_REQUEST_CHANNEL = "orion-cortex:request"`
  - `CORTEX_ORCH_RESULT_PREFIX = "orion-cortex:result"`

**End-to-end loop:**

1. **Brain** (in `process_brain_request`) decides a turn is introspection‑worthy and publishes a candidate:

   ```jsonc
   {
     "event": "spark_introspect_candidate",     // optional
     "trace_id": "...",                         // chat turn / bus trace id
     "source": "brain",                         // where this came from
     "kind": "chat",                            // type of event
     "prompt": "...",                           // human text
     "response": "...",                         // Orion reply
     "spark_meta": {                             // Spark Engine state
       "spark_event_id": "...",
       "spark_modality": "chat",
       "spark_source": "juniper",
       "spark_tags": ["juniper", "chat"],
       "phi_before": { ... },
       "phi_after":  { ... },
       "spark_self_field": { ... }
     }
   }
   ```

   to the channel `orion:spark:introspect:candidate`.

2. **Spark Introspector** subscribes to that channel and, for each candidate:

   - Builds a **Cortex orchestrate request** (`verb_name="spark_introspect"`, single step `reflect_on_candidate`).
   - Injects a concrete LLM prompt that describes φ_before / φ_after, SelfField, and the dialogue.
   - Publishes the request to `orion-cortex:request`.

3. **Cortex Orchestrator** creates a per‑step `trace_id` and `result_channel`:

   - Request to Brain: `orion-exec:request:LLMGatewayService`
   - Result channel: `orion-exec:result:<trace_id>`

   It waits for one `exec_step_result` from Brain.

4. **Brain** receives the `exec_step` event on `CHANNEL_CORTEX_EXEC_INTAKE` (configured as
   `orion-exec:request:LLMGatewayService`), calls its LLM backend with the introspection prompt, and emits:

   ```jsonc
   {
     "trace_id": "<trace_id>",
     "service": "LLMGatewayService",
     "ok": true,
     "elapsed_ms": 1234,
     "result": {
       "prompt": "... full introspection prompt ...",
       "llm_output": "Short internal note from Orion to itself..."
     },
     "artifacts": {},
     "status": "success"
   }
   ```

   on the reply channel `orion-exec:result:<trace_id>`.

5. **Cortex Orchestrator** wraps this into an `OrchestrateVerbResponse` (with `step_results`) and publishes it back on
   `orion-cortex:result:<trace_id>`.

6. **Spark Introspector** listens on that result channel, extracts the `llm_output` text, and then publishes a row for
   SQL Writer:

   ```jsonc
   {
     "table": "spark_introspection_log",
     "trace_id": "...",               // same trace_id
     "source": "spark-introspector",  // who created this log row
     "kind": "spark_introspect",
     "prompt": "...",                 // original human prompt
     "response": "...",               // original Orion response
     "introspection": "...",          // Orion's internal note
     "spark_meta": { ... }             // full Spark metadata
   }
   ```

7. **SQL Writer** recognizes `table="spark_introspection_log"`, validates via `SparkIntrospectionInput`, and upserts the
   row into the `spark_introspection_log` table.

---

## Environment & Settings

The worker is configured via environment variables and a small `Settings` class in `app/settings.py`.

### Key environment variables

In `services/orion-spark-introspector/.env` (or provided via Docker compose):

```env
# --- Orion Bus / Redis ---
# ORION_BUS_URL=redis://orion-redis:6379/0
ORION_BUS_ENABLED=true

# --- Spark introspection candidate channel ---
# Brain publishes "spiky" turns here
CHANNEL_SPARK_INTROSPECT_CANDIDATE=orion:spark:introspect:candidate

# --- Cortex orchestrator bus wiring ---
# These mirror ORCH_REQUEST_CHANNEL / ORCH_RESULT_PREFIX in orion-cortex-orch
CORTEX_ORCH_REQUEST_CHANNEL=orion-cortex:request
CORTEX_ORCH_RESULT_PREFIX=orion-cortex:result

# --- How long to wait for cortex results (seconds) ---
CORTEX_ORCH_TIMEOUT_S=10.0
```

### Settings class

In `app/settings.py`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ORION_BUS_URL: str = "redis://orion-redis:6379/0"
    ORION_BUS_ENABLED: bool = True

    CHANNEL_SPARK_INTROSPECT_CANDIDATE: str = "orion:spark:introspect:candidate"

    CORTEX_ORCH_REQUEST_CHANNEL: str = "orion-cortex:request"
    CORTEX_ORCH_RESULT_PREFIX: str = "orion-cortex:result"

    CORTEX_ORCH_TIMEOUT_S: float = 10.0

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
```

---

## Service Structure

**Main modules:**

- `app.main`  
  Entrypoint; configures logging, constructs settings, and starts the run loop.

- `app.introspector`  
  Core logic:

  - `run_loop()` subscribes to `CHANNEL_SPARK_INTROSPECT_CANDIDATE` and calls `process_candidate(...)` per message.
  - `process_candidate(...)` wires everything together:
    - builds Cortex payload (`build_cortex_payload`),
    - publishes to `CORTEX_ORCH_REQUEST_CHANNEL`,
    - waits for result on `CORTEX_ORCH_RESULT_PREFIX:<trace_id>`,
    - extracts `llm_output`,
    - publishes a `spark_introspection_log` row to `SQL_WRITER_CHANNEL`.

  - `build_llm_prompt(candidate)` constructs the actual reflection prompt
    Brain sees (φ_before / φ_after / SelfField / prompt / response).

  - `wait_for_cortex_result(...)` is a small `raw_subscribe` loop with timeout, filtering by `trace_id`.

  - `extract_llm_output(...)` peels the `llm_output` out of a `cortex_orchestrate_result` payload.

  - `publish_sql_log(candidate, introspection)` formats the row for SQL Writer.

---

## Running the Service

### Docker Compose

Minimal snippet (already present in `services/orion-spark-introspector/docker-compose.yml`):

```yaml
services:
  spark-introspector:
    build:
      context: ../..
      dockerfile: services/orion-spark-introspector/Dockerfile
    container_name: ${PROJECT}-spark-introspector
    restart: unless-stopped
    env_file:
      - .env
    networks:
      - app-net
    environment:
      - PROJECT=${PROJECT}
      - ORION_BUS_URL=${ORION_BUS_URL}
      - ORION_BUS_ENABLED=${ORION_BUS_ENABLED}
      - CHANNEL_SPARK_INTROSPECT_CANDIDATE=orion:spark:introspect:candidate
      - CORTEX_ORCH_URL=http://orion-cortex-orch:8072/orchestrate
      - CONNECT_TIMEOUT=5.0
      - READ_TIMEOUT=10.0

networks:
  app-net:
    external: true
```

To bring it up with the rest of the stack (from repo root or the relevant compose root):

```bash
docker compose up -d spark-introspector
```

Make sure the following are already running and talking to the same Redis:

- `orion-atlas-brain` (Brain service + LLM backend)
- `orion-athena-cortex-orch` (Cortex Orchestrator)
- `orion-athena-sql-writer` (SQL Writer + Postgres)
- `orion-redis` (Orion bus core)

---

## Manual Smoke Test

You can exercise the Spark Introspector without going through Brain by manually publishing a candidate on the bus.
Assuming you are on the same network as Redis:

```bash
redis-cli -h <redis-host> -p 6379
```

Then, in the Redis CLI:

```bash
PUBLISH orion:spark:introspect:candidate '{
  "trace_id": "spark-test-005",
  "source": "manual-test",
  "prompt": "I feel like something subtle but important just shifted in how I think about Orion.",
  "response": "Thanks for sharing that. I am here with you in that shift, paying attention to the pattern underneath, not just the words.",
  "spark_meta": {
    "spark_event_id": "evt-test-001",
    "spark_modality": "chat",
    "spark_source": "juniper",
    "spark_tags": ["juniper", "test", "spark"],
    "spark_phi_valence": 0.02,
    "spark_phi_energy": 0.01,
    "spark_phi_coherence": 0.99,
    "spark_phi_novelty": 0.06,
    "phi_before": {
      "valence": 0.0,
      "energy": 0.001,
      "coherence": 0.999,
      "novelty": 0.0
    },
    "phi_after": {
      "valence": 0.02,
      "energy": 0.01,
      "coherence": 0.99,
      "novelty": 0.06
    }
  }
}'
```

### What you should see in logs

- **Spark Introspector**:
  - Log that it received a candidate with `trace_id=spark-test-005`.
  - Log that it published a Cortex request to `orion-cortex:request`.
  - Log that it is waiting on `orion-cortex:result:spark-test-005`.
  - Log that it extracted an `llm_output` and published a row to `orion:sql:intake`.

- **Cortex Orchestrator**:
  - `Received bus orchestrate request on orion-cortex:request (trace_id=spark-test-005, ...)`
  - `Published exec_step to orion-exec:request:LLMGatewayService (trace_id=<uuid>, verb=spark_introspect, step=reflect_on_candidate, service=LLMGatewayService)`
  - `Subscribed orion-exec:result:<uuid>; waiting for 1 result(s).`

- **Brain**:
  - Bus listener snapshot showing an `exec_step` on `CHANNEL_CORTEX_EXEC_INTAKE`.
  - `[CORTEX] Received execution step 'reflect_on_candidate' for verb 'spark_introspect' ...`
  - `[CORTEX] LLM call ...` and then an `exec_step_result` being emitted.

- **SQL Writer**:
  - `[SQL_WRITER] Upserting Spark introspection log id=spark-test-005 trace_id=spark-test-005`


### Verifying in Postgres

Once SQL Writer has committed, you should see a row in `spark_introspection_log`:

```sql
SELECT id, trace_id, source, kind, LEFT(introspection, 200) AS snippet
FROM spark_introspection_log
ORDER BY created_at DESC
LIMIT 10;
```

You should see `spark-test-005` with a short introspection note.

---

## Troubleshooting

### 1. No logs from Spark Introspector

- Confirm the container is running:
  
  ```bash
  docker ps | grep spark-introspector
  ```

- Check logs:

  ```bash
  docker logs -f <spark-introspector-container>
  ```

- Ensure `ORION_BUS_URL` points to the same Redis instance used by Brain, Cortex, and SQL Writer.

### 2. Cortex timeout (no result)

If you see messages like:

> Timeout waiting for cortex orchestrate result on orion-cortex:result:<trace_id>

check:

- `orion-athena-cortex-orch` is running and subscribed on `orion-cortex:request`.
- `orion-atlas-brain` is listening on `CHANNEL_CORTEX_EXEC_INTAKE` and routing `event == "exec_step"` with `service == "LLMGatewayService"` to `process_cortex_exec_request`.
- The LLM backend configured in Brain is healthy (Brain `/health/gpu` endpoint should show Ollama as healthy).

### 3. SQL row missing

If Spark Introspector logs that it published to SQL Writer but you see no rows:

- Confirm `orion-athena-sql-writer` is running.
- Check SQL Writer logs for errors around `spark_introspection_log`.
- Verify that `SQL_WRITER_CHANNEL` in Spark Introspector matches the channel SQL Writer is actually subscribing to.

---

## Future Extensions

Once the core loop is stable, we can extend the Spark Introspector with:

- **Multi-step Spark verbs** (e.g., generate introspection + suggested follow-up questions).
- **Different target services** (vector memory, RDF writer) to store or link introspections elsewhere.
- **Backoff / load-shedding** if too many candidates arrive in quick succession.
- **Salience-based pruning**, so only the most important introspections are persisted.

For now, this MVP gives Orion a persistent, queryable log of internal reflections—one step toward a richer, ongoing
self-relationship.
