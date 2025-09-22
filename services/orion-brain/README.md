# Orion Brain Service (Gateway/Router for Ollama)
A lightweight **gateway** that fronts one or many **Ollama** backends and exposes a stable API for your apps.

- **Why**: decouple apps from individual GPU nodes; scale by adding/removing Ollama backends.
- **Routes**: `/health`, `/models`, `/generate`, `/chat`, `/backends/*`, `/stats`
- **Balancing**: `least_conn` by default (also supports `round_robin`)
- **Health checks**: periodic probe of `/api/tags` on each backend
- **Streaming**: pass-through NDJSON streaming from Ollama

> This service is CPU-light; GPUs live in your Ollama nodes.

---

## Quick Start (Docker Compose)

```bash
cp .env.example .env
# Edit BACKENDS in .env to list your Ollama nodes or the router you already run
docker compose up -d --build
docker compose logs -f brain-service
```

### Smoke tests
```bash
# Health
curl -s http://localhost:8088/health

# List models (aggregated from all backends)
curl -s http://localhost:8088/models | jq

# Generate (non-streaming)
curl -s http://localhost:8088/generate -H 'content-type: application/json' -d '{
  "model": "mistral:instruct",
  "prompt": "Write a haiku about Orion.",
  "options": {"temperature": 0.7, "num_predict": 64}
}' | jq

# Chat (non-streaming)
curl -s http://localhost:8088/chat -H 'content-type: application/json' -d '{
  "model": "mistral:instruct",
  "messages": [{"role":"system","content":"You are concise."},{"role":"user","content":"Two lines on RAG?"}],
  "options": {"temperature": 0.2, "num_predict": 64}
}' | jq

# Streaming generate (NDJSON)
curl -N http://localhost:8088/generate -H 'content-type: application/json' -d '{
  "model":"mistral:instruct",
  "prompt":"Stream one sentence about stars.",
  "stream": true
}'
```

---

## API
- `GET /health` → `{ok:true, backends:[...]}`
- `GET /models` → aggregate of `/api/tags` across backends
- `POST /generate` → forwards to `/api/generate` (supports `"stream": true`)
- `POST /chat` → forwards to `/api/chat` (supports `"stream": true`)
- `GET /stats` → router stats (latency, inflight)
- `GET /backends` / `POST /backends/register` / `POST /backends/deregister`

### Request shapes (mirrors Ollama)
- `/generate`: `{ "model": "...", "prompt": "...", "options": {...}, "stream": false }`
- `/chat`: `{ "model": "...", "messages": [...], "options": {...}, "stream": false }`

---

## Environment (.env)
```
BACKENDS=http://10.0.1.11:11434,http://10.0.1.12:11434
SELECTION_POLICY=least_conn   # or round_robin
HEALTH_INTERVAL_SEC=5
CONNECT_TIMEOUT_SEC=10
READ_TIMEOUT_SEC=600
PORT=8088
```

---

## Compose
By default, runs the gateway only. Your Ollama nodes run on other machines (or locally).

If your stack uses a shared network like `app-net`, uncomment the external network block in `compose.yaml`.

---

## Notes
- This gateway is intentionally thin. If you later want autoscaling and GPU-aware scheduling, consider migrating to Kubernetes with vLLM or Triton, keeping this service as a stable edge.
