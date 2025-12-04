# Orion Cortex Orchestrator

Phase 2 · Sprint 2c · `orion-cortex-orch`

The **Orion Cortex Orchestrator** is the multi-step cognition "program runner" in the Orion Sapienform mesh.

- **Semantic Layer** → defines verbs and pipelines (steps)
- **Cortex Exec** → executes a single step against one or more cognitive services via the Orion bus
- **Cortex Orchestrator** → runs the full multi-step pipeline, wiring context and prior results between steps
- **Agents (Phase 3)** → call the Orchestrator with verbs to get complex cognitive work done

This service exposes:

- An HTTP API (`/orchestrate`) for synchronous multi-step runs
- A bus worker (optional) listening on `CORTEX_ORCH_REQUEST_CHANNEL` for bus-driven orchestration

It connects to the **Orion bus (Redis)** and orchestrates fan-out / fan-in via:

- `EXEC_REQUEST_PREFIX:<ServiceName>`
- `EXEC_RESULT_PREFIX:<trace_id>`

---

## Directory Layout

Within the repo (root = `/mnt/scripts/Orion-Sapienform`):

```text
services/
  orion-cortex-orch/
    app/
      __init__.py
      main.py           # FastAPI entrypoint
      settings.py       # Pydantic settings
      orchestrator.py   # Core multi-step orchestration logic
      tests/
        __init__.py
        smoke_bus_worker.py   # Bus smoke worker (TestService)
    Dockerfile
    docker-compose.yml
    Makefile
    .env                # Service-local overrides (optional)
```

> Note: The service is run via `docker compose` from the **repo root**, using
> `services/orion-cortex-orch/docker-compose.yml`.

---

## Configuration

### Environment Variables

At minimum, you should have the following defined (typically in the root `.env` and optionally in `services/orion-cortex-orch/.env`):

```env
# Node identity
NODE_NAME=athena-cortex-orchestrator

# Redis bus
ORION_BUS_URL=redis://orion-redis:6379/0
ORION_BUS_ENABLED=true

# Exec routing
EXEC_REQUEST_PREFIX=orion-exec:request
EXEC_RESULT_PREFIX=orion-exec:result

# Orchestrator bus routing (optional)
CORTEX_ORCH_REQUEST_CHANNEL=orion-cortex:request
CORTEX_ORCH_RESULT_PREFIX=orion-cortex:result

# Per-step timeout (ms)
ORION_CORTEX_STEP_TIMEOUT_MS=8000

# HTTP API
API_HOST=0.0.0.0
API_PORT=8072
LOG_LEVEL=INFO
```

The **service container** reads these via `app/settings.py`.

---

## Docker / Compose

The service is managed by a service-specific compose file:

```text
services/orion-cortex-orch/docker-compose.yml
```

This file is executed **from the repo root**.

### Bring up the service stack

From repo root:

```bash
cd /mnt/scripts/Orion-Sapienform

# Use both the root and service .env files, and the service-specific compose
docker compose \
  --env-file .env \
  --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml up -d
```

This will build and start the orchestrator (and any dependencies defined in that compose file, such as Redis if you chose to include it there).

You should then see a container similar to:

```bash
docker ps

CONTAINER ID   IMAGE                                         COMMAND                  PORTS
...            orion-cortex-orch-orion-cortex-orchestrator   "uvicorn app.main:ap…"  0.0.0.0:8072->8072/tcp
```

---

## HTTP API

### Health Check

```bash
curl http://localhost:8072/health | jq .
```

Example output:

```json
{
  "status": "ok",
  "node_name": "athena-cortex-orchestrator",
  "bus_enabled": true,
  "exec_request_prefix": "orion-exec:request",
  "exec_result_prefix": "orion-exec:result",
  "cortex_orch_request_channel": "orion-cortex:request",
  "cortex_orch_result_prefix": "orion-cortex:result"
}
```

### Orchestrate Endpoint

`POST /orchestrate`

Request body schema (simplified):

```json
{
  "verb_name": "introspect",
  "origin_node": "athena-smoke",
  "context": { "foo": "bar" },
  "steps": [
    {
      "verb_name": "introspect",
      "step_name": "echo-test",
      "description": "Simple echo test step.",
      "order": 0,
      "services": ["TestService"],
      "prompt_template": "Base instructions...",
      "requires_gpu": false,
      "requires_memory": false
    }
  ],
  "timeout_ms": 8000
}
```

The orchestrator:

1. Sorts steps by `order`.
2. For each step:
   - Builds a contextual prompt (including `context` + prior step results).
   - Publishes an `exec_step` message to the bus for each service in `services`:
     - Channel: `EXEC_REQUEST_PREFIX:<ServiceName>` (e.g. `orion-exec:request:LLMGatewayService`).
   - Subscribes to `EXEC_RESULT_PREFIX:<trace_id>` and waits for results.
3. Returns a structured `OrchestrateVerbResponse` containing all step and service results.

---

## Smoke Tests

To make it easy to verify the full Semantic → Exec → Orchestrator path, this service includes:

- A **bus smoke worker** that acts as a fake cortex service (`TestService`)
- A **Makefile** that wraps the orchestration smoke test

### 1. Makefile Overview

`services/orion-cortex-orch/Makefile` defines a few convenience targets:

```bash
# From within the service dir:
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-orch

make help
```

You should see:

- `make build` – build the service image
- `make up` – bring up the stack via `services/orion-cortex-orch/docker-compose.yml`
- `make down` – stop the stack
- `make logs` – tail orchestrator logs
- `make smoke-worker` – run the bus smoke worker inside the service image
- `make smoke-http` – run a one-step HTTP orchestrate smoke test

### 2. Start the service stack

From the **service directory**:

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-orch
make up
```

This is equivalent to running from root:

```bash
cd /mnt/scripts/Orion-Sapienform

docker compose \
  --env-file .env \
  --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml up -d
```

Ensure the orchestrator container is running:

```bash
docker ps | grep orion-cortex-orchestrator
```

### 3. Run the smoke bus worker

The smoke worker is located at `app/tests/smoke_bus_worker.py` **inside the image**. It:

- Subscribes to `EXEC_REQUEST_PREFIX:TestService`
- Receives `exec_step` messages
- Publishes a synthetic result to the provided `result_channel`

To run it inside the service image (using the same compose config and env):

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-orch

make smoke-worker
```

Under the hood, this executes (from repo root):

```bash
docker compose --env-file .env --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml run --rm \
    --entrypoint python \
    orion-cortex-orchestrator \
    app/tests/smoke_bus_worker.py
```

You should see logs similar to:

```text
[smoke-worker] Connecting to bus: redis://orion-redis:6379/0
[smoke-worker] Subscribing to orion-exec:request:TestService
```

Leave this running in **Terminal A**.

### 4. Run the HTTP smoke test

In **Terminal B**, from the same service directory:

```bash
cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-orch
make smoke-http
```

This runs a `curl` command that:

- POSTs a one-step `introspect` verb to `/orchestrate`
- Targets `TestService` as the service for the step

The `smoke-http` target sends this payload (one-line JSON):

```json
{
  "verb_name": "introspect",
  "origin_node": "athena-smoke",
  "context": {"foo": "bar", "test": "cortex-orch smoke"},
  "steps": [
    {
      "verb_name": "introspect",
      "step_name": "echo-test",
      "description": "Simple echo test step.",
      "order": 0,
      "services": ["TestService"],
      "prompt_template": "This is a smoke test of the Cortex Orchestrator and bus wiring.",
      "requires_gpu": false,
      "requires_memory": false
    }
  ],
  "timeout_ms": 8000
}
```

The response should look like:

```json
{
  "verb_name": "introspect",
  "origin_node": "athena-smoke",
  "steps_executed": 1,
  "step_results": [
    {
      "verb_name": "introspect",
      "step_name": "echo-test",
      "order": 0,
      "services": [
        {
          "service": "TestService",
          "trace_id": "...",
          "ok": true,
          "elapsed_ms": 100,
          "payload": {
            "note": "smoke test response from TestService",
            "prompt_preview": "This is a smoke test of the Cortex Orchestrator and bus wiring.\n\n# Orion Cortex Orchestrator Context\n- Verb: introspect\n- Step: echo-test (order=0)\n- Target Service: TestService\n- Origin Node: athena-smoke"
          }
        }
      ],
      "prompt_preview": "This is a smoke test of the Cortex Orchestrator and bus wiring."
    }
  ],
  "context_echo": {
    "foo": "bar",
    "test": "cortex-orch smoke"
  }
}
```

This confirms that:

1. `/orchestrate` accepts the multi-step verb request.
2. The Cortex Orchestrator publishes an `exec_step` to `EXEC_REQUEST_PREFIX:TestService`.
3. The smoke worker receives that message and publishes a response to `EXEC_RESULT_PREFIX:<trace_id>`.
4. The Orchestrator fan-ins the result and returns a structured `OrchestrateVerbResponse`.

---

## Next Steps

With this service and smoke test working, the Phase 2 stack is effectively in place:

- **Semantic Layer** – defines verbs and pipelines
- **Cortex Exec** – single-step executor
- **Cortex Orchestrator** – multi-step runner (this service)

The next workstream (Phase 3) is to implement **Agents** that:

- Decide which verb to call (e.g., `introspect`, `summarize.session`, `dream.synthesize`)
- Fetch the pipeline from the Semantic Layer
- Call the Cortex Orchestrator with `verb_name + steps + context`
- Interpret and act on the orchestrated results.
