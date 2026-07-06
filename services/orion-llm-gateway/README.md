# Orion LLM Gateway

The **LLM Gateway** provides a unified interface to various LLM backends (OpenAI, Anthropic, Local, etc.). It accepts standard `ChatRequestPayload` messages and returns normalized `ChatResultPayload` responses.

It now supports **latent vector emission** for vLLM/llama-cola responses (when the backend returns a spark vector), publishing those latents to the vector writer while leaving semantic embeddings to orion-vector-host.

For **llama.cpp** and **llama-cola** backends, `ChatRequestPayload.options` may include **`chat_template_kwargs`** (e.g. `{"enable_thinking": false}`). The gateway forwards that object to `/v1/chat/completions` so Qwen3-style thinking can be toggled **per request** without restarting the model host.

### Spark metadata (v1)

The gateway no longer runs tissue ingest on chat turns. Result `spark_meta` is thin metadata only:

- `latest_user_message`, `latest_assistant_message` (clipped)
- `trace_verb`, `spark_phase`, `spark_used_raw_user_text`

Turn novelty and shift classification live in `spark_meta.turn_change_appraisal`, patched asynchronously by `orion-memory-consolidation` on `orion:chat:history:spark_meta:patch`. See `services/orion-memory-consolidation/README.md`.

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
| `LLM_GATEWAY_ANTHROPIC_PASSTHROUGH_ENABLED` | `true` | Enable Anthropic Messages passthrough for Claude Code / FCC. |
| `LLM_GATEWAY_ANTHROPIC_PASSTHROUGH_TIMEOUT_SEC` | `900` | Read timeout for `/v1/messages` upstream proxy (tool calls can be long). |
| `LLM_ROUTE_HEALTH_TIMEOUT_SEC` | `1.5` | Upstream `/health` probe timeout for route catalog. |
| `LLM_LOGPROB_SUMMARY_ENABLED` | `false` | Global gate for summary-only `llm_uncertainty` on chat results. |
| `LLM_LOGPROB_TOP_K_DEFAULT` | `5` | Default `top_logprobs` / `n_probs` depth when `return_logprobs` is set. |
| `LLM_LOGPROB_LOW_MARGIN_THRESHOLD` | `0.5` | Low top-1 margin token threshold. |
| `LLM_LOGPROB_LOW_LOGPROB_THRESHOLD` | `-2.0` | Low logprob token threshold. |
| `LLM_LOGPROB_UNSTABLE_SPAN_MIN_LEN` | `3` | Consecutive low-margin run length for unstable spans. |
| `LLM_LOGPROB_NATIVE_COMPLETION_ENABLED` | `false` | Allow aligned `POST /apply-template` + `POST /completion` path. |
| `LLM_LOGPROB_NATIVE_COMPLETION_MAX_TOKENS` | `256` | Default `n_predict` when native path omits `max_tokens`. |

### HTTP endpoints

| Path | Description |
| :--- | :--- |
| `GET /health` | Service liveness and configured route keys. |
| `GET /routes` | Route catalog from `LLM_GATEWAY_ROUTE_TABLE_JSON` with `default_route=chat` and per-route `id`, `served_by`, `backend`, `status`, `latency_ms`, `last_checked_at`. |
| `GET /v1/models` | Anthropic-compatible model list from configured route keys (FCC / Claude Code). |
| `GET /v1/messages` | Anthropic Messages endpoint liveness (same as HEAD). |
| `POST /v1/messages` | Anthropic Messages passthrough to upstream llama.cpp `/v1/messages` via route table. |
| `POST /v1/chat/completions` | OpenAI chat passthrough to upstream `/v1/chat/completions` via route table (AI Town, OpenAI clients). |
| `POST /v1/embeddings` | OpenAI embeddings passthrough to `orion-vector-host` `POST /embedding`. |
| `HEAD /v1/messages` | Liveness probe for Anthropic Messages endpoint. |
| `OPTIONS /v1/messages` | CORS/method discovery for Anthropic clients. |

### Claude Code / free-claude-code (FCC) passthrough

The gateway exposes an Anthropic Messages-compatible HTTP membrane for Claude Code and FCC. Traffic uses the same `LLM_GATEWAY_ROUTE_TABLE_JSON` lanes (`agent`, `chat`, `quick`, `metacog`, etc.) but **does not** go through the bus-native `run_llm_chat()` path.

Topology:

```text
Claude Code / FCC -> http://athena:8210/v1/messages -> route table -> Atlas llama.cpp /v1/messages
```

Optional per-route upstream model alias in the route table:

```json
{
  "agent": {
    "url": "http://100.121.214.30:8011",
    "served_by": "atlas-worker-1",
    "backend": "llamacpp",
    "model": "qwen-coder-local"
  }
}
```

FCC example config:

```bash
LLAMACPP_BASE_URL=http://127.0.0.1:8210/v1
MODEL=llamacpp/agent
MODEL_OPUS=llamacpp/agent
MODEL_SONNET=llamacpp/agent
MODEL_HAIKU=llamacpp/quick
ANTHROPIC_AUTH_TOKEN=freecc
ENABLE_MODEL_THINKING=false
PROVIDER_MAX_CONCURRENCY=1
HTTP_READ_TIMEOUT=600
VOICE_NOTE_ENABLED=false
MESSAGING_PLATFORM=none
```

Smoke:

```bash
curl -s http://127.0.0.1:8210/v1/models | jq
curl -s http://127.0.0.1:8210/v1/messages \
  -H 'content-type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"llamacpp/agent","max_tokens":64,"stream":false,"messages":[{"role":"user","content":"Say OK."}]}' | jq
```

### Logprob / `llm_uncertainty` (language surface stability)

Summary-only metrics (`confidence_semantics=language_surface_stability_not_truth`). Not factual confidence.

**OpenAI-compatible path (default):** per-request `options.return_logprobs=true` on `/v1/chat/completions` when `LLM_LOGPROB_SUMMARY_ENABLED=true`. Source label: `{backend}_openai_chat`.

**Native aligned path (llama.cpp only):** additionally set `options.logprob_probe_mode=native_completion` and `LLM_LOGPROB_NATIVE_COMPLETION_ENABLED=true`. The gateway runs `/apply-template` → `/completion` with `n_probs` on the **same** text returned to callers. Source label: `llamacpp_native_completion`.

```json
{
  "return_logprobs": true,
  "logprob_probe_mode": "native_completion",
  "logprobs_top_k": 5,
  "logprob_summary_only": true
}
```

Mind (`MIND_LLM_RETURN_LOGPROBS_SEMANTIC` + `MIND_LLM_LOGPROB_PROBE_MODE`) and cortex metacog draft (`CORTEX_METACOG_RETURN_LOGPROBS` + `CORTEX_METACOG_LOGPROB_PROBE_MODE`) can set these options when enabled in their service `.env` files.

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
  "quick":{"url":"http://100.121.214.30:8013","served_by":"atlas-worker-fast-1","backend":"llamacpp"}
}'
```

`quick` is the FAST lane route used by user-facing quick chat and chat_general pass-1.

### Route table example (optional split agent mode)
```bash
LLM_GATEWAY_ROUTE_TABLE_JSON='{
  "chat":{"url":"http://100.121.214.30:8011","served_by":"atlas-worker-1","backend":"llamacpp"},
  "agent":{"url":"http://100.121.214.30:8014","served_by":"atlas-worker-agent-1","backend":"llamacpp"},
  "metacog":{"url":"http://100.121.214.30:8012","served_by":"atlas-worker-2","backend":"llamacpp"},
  "quick":{"url":"http://100.121.214.30:8013","served_by":"atlas-worker-fast-1","backend":"llamacpp"}
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
