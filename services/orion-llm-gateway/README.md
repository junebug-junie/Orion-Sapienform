# Orion LLM Gateway

The **LLM Gateway** provides a unified interface to various LLM backends (OpenAI, Anthropic, Local, etc.). It accepts standard `ChatRequestPayload` messages and returns normalized `ChatResultPayload` responses.

It now supports **latent vector emission** for vLLM/llama-cola responses (when the backend returns a spark vector), publishing those latents to the vector writer while leaving semantic embeddings to orion-vector-host.

## Contracts

### Consumed Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:exec:request:LLMGatewayService` | `CHANNEL_LLM_INTAKE` | `llm.chat.request` | Chat requests. |
| `orion:spark:introspect:candidate` | `CHANNEL_SPARK_INTROSPECT_CANDIDATE` | `spark.introspect` | Spark introspection requests. |

### Published Channels
| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| (Caller-defined) | (via `reply_to`) | `llm.chat.result` | Chat completion result. |

### Environment Variables
Provenance: `.env_example` → `docker-compose.yml` → `settings.py`

| Variable | Default (Settings) | Description |
| :--- | :--- | :--- |
| `CHANNEL_LLM_INTAKE` | `orion:exec:request:LLMGatewayService` | Primary intake. |
| `CHANNEL_VECTOR_LATENT_UPSERT` | `orion:vector:latent:upsert` | Latent vector upsert channel. |
| `ORION_VECTOR_LATENT_COLLECTION` | `orion_latent_store` | Latent vector collection. |
| `ORION_LLM_VLLM_URL` | `None` | URL for vLLM host. |
| `ORION_LLM_LLAMACPP_URL` | `None` | Legacy single-endpoint llama.cpp URL; route-table mode is primary. |
| `LLM_GATEWAY_ROUTE_TABLE_JSON` | `None` | Preferred JSON route table for explicit single-subscriber routing. |
| `LLM_ROUTE_DEFAULT` | `chat` | Default routing key when none provided. |
| `LLM_ROUTE_CHAT_URL` | `None` | Fallback URL for `route=chat` (if JSON not set). |
| `LLM_ROUTE_METACOG_URL` | `None` | Fallback URL for `route=metacog` (if JSON not set). |
| `LLM_ROUTE_LATENTS_URL` | `None` | Fallback URL for `route=latents` (if JSON not set). |
| `LLM_ROUTE_SPECIALIST_URL` | `None` | Fallback URL for `route=specialist` (if JSON not set). |
| `LLM_GATEWAY_HEALTH_PORT` | `8210` | Local HTTP health port. |

Important routing note:

- `LLM_GATEWAY_ROUTE_TABLE_JSON` is the primary mechanism for Atlas.
- `served_by` is metadata returned for observability and smoke checks; it does
  not drive routing.
- The legacy per-route env aliases only cover `chat`, `metacog`, `latents`,
  and `specialist`.
- The current Atlas `agent` lane therefore requires `LLM_GATEWAY_ROUTE_TABLE_JSON`.

## Running & Testing

### Run via Docker
```bash
docker compose -f services/orion-llm-gateway/docker-compose.yml up -d llm-gateway
```

> Note: Only run a single `orion-llm-gateway` subscriber on the shared request topic.
> Route isolation should be expressed through `LLM_GATEWAY_ROUTE_TABLE_JSON`, not by running multiple gateways.
> Atlas default merged mode keeps logical `chat` and `agent` routes separate while mapping both to the same chat worker URL.

### Route table example (default merged mode)
```bash
LLM_GATEWAY_ROUTE_TABLE_JSON='{
  "chat":{"url":"http://100.121.214.30:8011","served_by":"atlas-worker-1","backend":"llamacpp"},
  "agent":{"url":"http://100.121.214.30:8011","served_by":"atlas-worker-1","backend":"llamacpp"},
  "metacog":{"url":"http://100.121.214.30:8012","served_by":"atlas-worker-2","backend":"llamacpp"},
  "helper":{"url":"http://100.121.214.30:8013","served_by":"atlas-worker-helper-1","backend":"llamacpp"},
  "quick":{"url":"http://100.121.214.30:8015","served_by":"atlas-worker-quick-1","backend":"llamacpp"}
}'
```

`helper` is an internal lane for bounded substeps; `quick` is a user-facing chat lane.

### Route table example (optional split agent mode)
```bash
LLM_GATEWAY_ROUTE_TABLE_JSON='{
  "chat":{"url":"http://100.121.214.30:8011","served_by":"atlas-worker-1","backend":"llamacpp"},
  "agent":{"url":"http://100.121.214.30:8014","served_by":"atlas-worker-agent-1","backend":"llamacpp"},
  "metacog":{"url":"http://100.121.214.30:8012","served_by":"atlas-worker-2","backend":"llamacpp"},
  "helper":{"url":"http://100.121.214.30:8013","served_by":"atlas-worker-helper-1","backend":"llamacpp"},
  "quick":{"url":"http://100.121.214.30:8015","served_by":"atlas-worker-quick-1","backend":"llamacpp"}
}'
```

### Smoke Test
```bash
PYTHONPATH=/workspace/Orion-Sapienform python -m scripts.smoke_llm_gateway_routes \
  --redis "${ORION_BUS_URL:-redis://localhost:6379/0}" \
  --request-channel "${CHANNEL_LLM_INTAKE:-orion:exec:request:LLMGatewayService}"
```

### Health Check
```bash
curl http://localhost:8210/health
```
