# Orion LLM Gateway Service

## 1. Overview

`orion-llm-gateway` is the abstraction layer between the Orion bus and one or more LLM backends.

It:
- Listens for LLM work items on a Redis / OrionBus channel.
- Normalizes requests into a common internal format (model, backend, messages, options, trace id).
- Dispatches the call to a configured LLM backend (Ollama today; vLLM / LangChain-ready by design).
- Normalizes the response back into a consistent payload.
- Publishes the result on a reply channel so upstream services (Brain, Agent Council, Cortex, Dream, Recall, Hub) don’t care which LLM stack is actually running.

Think of this as the **LLM I/O card** for the organism.

---

## 2. Topology & Dependencies

### Placement in the Mesh

- **Node:** Currently running on `athena` (so it’s close to the Redis bus and SQL stack).
- **Backends:** Talks over HTTP to your LLM host(s). Today:
  - Primary: Ollama (Atlas)
  - Future: vLLM (NVLink V100 carrier board), LangChain / LangGraph orchestrations.
- **Upstream callers:**
  - `cortex-exec` (brain + agent flows)
  - `planner-react` (planner loop)
  - `agent-chain` (indirect within tools)
  - Legacy: `orion-brain`, `orion-dream`, `orion-recall`

The entire point is that **all of those callers only need to know how to talk to the bus**, not how to talk to Ollama vs vLLM vs LangChain.

---

## 3. Configuration (.env)

The service is configured via its own `.env` and Docker `environment` block.

Core identity:

```env
SERVICE_NAME=llm-gateway
SERVICE_VERSION=0.1.0
PORT=8222
```

Bus configuration:

```env
ORION_BUS_ENABLED=true
ORION_BUS_URL=redis://100.92.216.81:6379/0

CHANNEL_LLM_INTAKE=orion-exec:request:LLMGatewayService
# Replies always go to the caller-provided reply_to (exec uses orion-exec:result:LLMGatewayService:<uuid>)
```

Backend selection / routing (high-level sketch):

```env
# Default logical backend name (must exist in BACKEND_REGISTRY in code)
LLM_DEFAULT_BACKEND=ollama

# Comma-separated list of enabled backends
LLM_ENABLED_BACKENDS=ollama

# Ollama backend
OLLAMA_BASE_URL=http://100.xx.xx.xx:11434

# Optional future vLLM backend
VLLM_BASE_URL=

# Optional future LangChain/LangGraph router
LANGCHAIN_ROUTER_URL=
```

The exact variable names should match what’s defined in `app/settings.py`; this doc is deliberately high-level so you can evolve the settings without rewriting the README every time.

---

## 4. Bus Contracts

### 4.1 Intake Channel

**Channel:** `${CHANNEL_LLM_INTAKE}` (default: `orion-exec:request:LLMGatewayService`)

**Inbound envelope (from Exec / Planner / others):**

```jsonc
{
  "kind": "llm.chat.request",
  "correlation_id": "uuid-or-upstream-trace",
  "reply_to": "orion-exec:result:LLMGatewayService:<uuid>",
  "payload": {
    "model": "llama3.1:8b-instruct-q8_0",
    "profile": null,
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "options": {
      "temperature": 0.6,
      "top_p": 0.9,
      "num_ctx": 4096,
      "max_tokens": 512
    },
    "session_id": "abc",
    "user_id": "abc"
  }
}
```

**Reply routing:** LLM Gateway responds on whatever `reply_to` the caller provides:

- Exec RPC: `orion-exec:result:LLMGatewayService:<uuid>`
- Planner loop: `orion:llm:reply:<uuid>`

### 4.2 Reply Payload

**Channel:** Caller-provided reply channel, e.g. `orion-exec:result:LLMGatewayService:<uuid>` or `orion:llm:reply:<uuid>`.

**Outbound payload:**

```jsonc
{
  "trace_id": "uuid-or-upstream-trace",
  "source": "llm-gateway",
  "backend": "ollama",
  "model": "llama3.1:8b-instruct-q8_0",
  "latency_ms": 324.5,
  "ok": true,
  "error": null,
  "completion": {
    "role": "assistant",
    "content": "final assistant text here"
  },
  "raw": {
    "backend_payload": { /* backend’s native response, truncated if needed */ }
  }
}
```

If the backend fails, `ok=false` and `error` is populated; the caller decides how to degrade.

---

## 5. Backend Routing

The actual routing logic lives in `app/llm_backends.py` (or similar), but the design is:

- A **registry** of backends keyed by name (e.g. `"ollama"`, `"vllm"`, `"langchain"`).
- Each backend implements a simple interface:

```python
class LLMBackend(Protocol):
    name: str

    async def chat(self, request: LLMRequest) -> LLMResult:
        ...
```

- `LLMRequest` is built from the bus payload and has normalized fields.
- `LLMResult` is what gets transformed into the reply payload.

### Ollama Backend

- Talks to `${OLLAMA_BASE_URL}/api/chat`.
- Maps `messages` + `options` into Ollama’s body.
- Extracts `.message.content` as the canonical `completion.content`.

### vLLM Backend (future)

- Will talk to an OpenAI-compatible HTTP endpoint (vLLM’s server mode).
- Same `LLMBackend` interface, different HTTP payload.
- Key point: **Agent Council / Brain don’t change at all** when you add vLLM; you just add a new backend implementation and maybe flip `LLM_DEFAULT_BACKEND`.

### LangChain / LangGraph Router (future)

- Optional: wrap complex graph-based flows behind the same interface.
- Example use: special models, tool-calling, long-context workflows, or graph-based agents.

---

## 6. HTTP API (Optional / Thin)

The service reserves port `${PORT}` for:

- `/health` — basic liveness / config echo.
- Future: `/debug/echo`, `/debug/invoke`, etc. (useful when debugging without the bus).

Example `GET /health`:

```json
{
  "ok": true,
  "service": "llm-gateway",
  "version": "0.1.0",
  "bus_enabled": true,
  "bus_url": "redis://100.92.216.81:6379/0",
  "default_backend": "ollama",
  "enabled_backends": ["ollama"]
}
```

---

## 7. Running Locally (Docker)

From the Orion root:

```bash
cd /mnt/scripts/Orion-Sapienform

# Build
docker compose \
  --env-file .env \
  --env-file services/orion-llm-gateway/.env \
  -f services/orion-llm-gateway/docker-compose.yml build

# Run
docker compose \
  --env-file .env \
  --env-file services/orion-llm-gateway/.env \
  -f services/orion-llm-gateway/docker-compose.yml up -d
```

You should then see the container (e.g. `orion-athena-llm-gateway`) attach to `app-net` and log its connection to the Redis bus.

---

## 8. Debugging & Smoke Tests

Suggestions:

1. **Health check**
   - `curl http://localhost:8222/health`
2. **Manual bus test**
   - Publish a small `llm.chat.request` envelope to `orion-exec:request:LLMGatewayService` with `reply_to=orion-exec:result:LLMGatewayService:smoke` (or your own prefix) and watch that channel for `llm.chat.result`.
3. **Watch logs**
   - `docker logs -f orion-athena-llm-gateway`

If calls are slow or failing, the logs should show whether it’s the bus, routing, or backend HTTP.

---

## 9. Future Extensions

- Add vLLM backend wired to your 2×V100 SMX2 carrier board.
- Add optional LangChain / LangGraph router backend for specialized flows.
- Expose simple `/invoke` HTTP endpoint for debugging prompts without going through the bus.
- Track simple per-backend telemetry (call count, mean latency, error rate) and publish to a metrics channel for Dream / BI / dashboards.
