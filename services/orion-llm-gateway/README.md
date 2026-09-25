# Orion LLM Gateway

The **LLM Gateway** provides a unified interface to various LLM backends (OpenAI, Anthropic, Local, etc.). It accepts standard `ChatRequestPayload` messages and returns normalized `ChatResultPayload` responses.

It now supports **latent vector emission** for vLLM/llama-cola responses (when the backend returns a spark vector), publishing those latents to the vector writer while leaving semantic embeddings to orion-vector-host.

For **llama.cpp** and **llama-cola** backends, `ChatRequestPayload.options` may include **`chat_template_kwargs`** (e.g. `{"enable_thinking": false}`). The gateway forwards that object to `/v1/chat/completions` so Qwen3-style thinking can be toggled **per request** without restarting the model host.

### Spark metadata (v1)

The gateway no longer runs tissue ingest on chat turns. Result `spark_meta` is thin metadata only:

- `latest_user_message`, `latest_assistant_message` (clipped)
- `trace_verb`, `spark_phase`, `spark_used_raw_user_text`

Turn novelty and shift classification live in `spark_meta.turn_change_appraisal`, patched asynchronously by `orion-memory-consolidation` on `orion:chat:history:spark_meta:patch`. See `services/orion-memory-consolidation/README.md`.

### Model identity (`model_used`, 2026-08-14)

`ChatResultPayload.model_used` is meant to be the model that actually served the request. Before this date it was silently wrong for every route: `run_llm_chat()` stamped it from the **requested route-table label** (e.g. `"Active-GGUF-Model"`), not the served weights -- confirmed live, that placeholder even leaked into a log line claiming the metacog route ran `llama-3-8b-instruct-q4_k_m` when it was actually running Qwen3-8B.

Fix: `llm_backend.py`'s `_served_model()` now prefers the backend's own echoed model id (present in `result["raw"]["model"]` for both the OpenAI-compat and Ollama-native response shapes) and only falls back to the requested label when the backend didn't echo one (llama.cpp's native `/completion` endpoint, or any error path). This is a live-serving-time fix, not a schema change -- `model_used` was already a first-class field, it just held the wrong value.

`route_catalog.py`'s `GET /routes` got the equivalent point-in-time fix: `_probe_model()` does a live `/v1/models` read against each route's backend (only when its `/health` probe is up) and surfaces the real id as the `model` field, cached with the same 15s TTL as route health. `services/orion-cortex-exec`'s situation brief reads this endpoint to tell Orion what it's currently running on -- see that service's README, "Situation brief" section.

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
| `GPU_POOL_CONFIG_PATH` | `/app/config/gpu_pool.yaml` | Route -> pool class/priority map (`routes:`). A route not listed is refused (`route_not_in_gpu_pool`). |
| `LLM_GATEWAY_POOL_WAIT_SEC` | `300` | Max wait for a pool grant (interactive/system); capped by the caller's own `gateway_read_timeout_sec`. |
| `LLM_GATEWAY_POOL_BACKGROUND_WAIT_SEC` | `900` | Same, for background-priority routes. |
| `LLM_GATEWAY_EXECUTOR_WORKERS_PER_ROLE` | `8` | Threads per granted role URL (one executor per GPU role). |
| `LLM_ROUTE_DEFAULT` | `quick` | Default routing key when none provided. |
| `LLM_LANE_ROUTING_ENABLED` | `true` | Honor trusted logical lane metadata when resolving a route name (`chat`, `agent`, `quick`, `metacog`, ...). Set `false` only as a rollback to body-route-only behavior. |
| `LLM_GATEWAY_HEALTH_PORT` | `8210` | Local HTTP health port. |
| `LLM_GATEWAY_ANTHROPIC_PASSTHROUGH_ENABLED` | `true` | Enable Anthropic Messages passthrough for Claude Code / FCC. |
| `LLM_GATEWAY_ANTHROPIC_PASSTHROUGH_TIMEOUT_SEC` | `900` | Read timeout for `/v1/messages` upstream proxy (tool calls can be long). |
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
| `GET /routes` | **Compatibility view generated from orion-gpu-pool state** (removed in stage 6). Same shape as before: per-route `id`, `served_by`, `backend`, `status` (`up`/`down`/`operator_closed`/`unknown`), `model` (discovered model file), `n_ctx` (discovered ctx per slot), `vision`, `upstream`, `gate_open`. `unknown` for every route when the pool cannot be reached -- never a fabricated `up`. |
| `GET /v1/models` | Anthropic-compatible model list from configured route keys (FCC / Claude Code). |
| `GET /v1/messages` | Anthropic Messages endpoint liveness (same as HEAD). |
| `POST /v1/messages` | Anthropic Messages passthrough to the pool-granted llama.cpp role's `/v1/messages` (lease holder `http:anthropic`). |
| `POST /v1/chat/completions` | OpenAI chat passthrough to the pool-granted role's `/v1/chat/completions` (lease holder `http:openai`; AI Town, OpenAI clients). |
| `POST /v1/embeddings` | OpenAI embeddings passthrough to `orion-vector-host` `POST /embedding`. |
| `HEAD /v1/messages` | Liveness probe for Anthropic Messages endpoint. |
| `OPTIONS /v1/messages` | CORS/method discovery for Anthropic clients. |

### Claude Code / free-claude-code (FCC) passthrough

The gateway exposes an Anthropic Messages-compatible HTTP membrane for Claude Code and FCC. Traffic uses the same route names (`config/gpu_pool.yaml` `routes:` -- `agent`, `chat`, `harness`, `quick`, `metacog`, etc.) and takes a GPU pool lease like the bus path, but **does not** go through the bus-native `run_llm_chat()` path.

Claude session hooks can append `role=system` context inside `messages` after a
user turn. Gateway moves those blocks into Anthropic's top-level `system` field
before forwarding to llama.cpp, whose model template requires system context
first. Existing system blocks, cache metadata, and conversation/tool ordering
are preserved. Durable-lease validation and a GPU pool lease still apply to the request.

Topology:

```text
Claude Code / FCC -> http://athena:8210/v1/messages -> route table -> Circe llama.cpp /v1/messages
```

Optional per-route upstream model alias in the route table:

```json
{
  "agent": {
    "url": "http://100.112.254.99:8015",
    "served_by": "circe-worker-agent-1",
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

Mind (`MIND_LLM_RETURN_LOGPROBS_SEMANTIC` + `MIND_LLM_LOGPROB_PROBE_MODE`) can set these options when enabled in its service `.env` file. (The cortex metacog draft probe that also used them was removed 2026-09-24.)

Important routing note:

- Which GPU serves a call is **orion-gpu-pool's** decision. The gateway maps the route name to a
  pool class + priority (`config/gpu_pool.yaml` `routes:`), takes a lease, and sends the call to
  the granted role's URL. `served_by` in results is the grant's (`circe-worker-<role>`), so a
  spilled call is attributed to the card that actually ran it.

### GPU pool placement (2026-09-24, GPU pool spec stage 3)

Every LLM call -- bus RPC, `/v1/chat/completions`, `/v1/messages` -- goes through
`app/pool_placement.py`:

1. **Route -> class.** `config/gpu_pool.yaml` `routes:` gives the pool class and priority. A route
   not listed is refused with `route_not_in_gpu_pool`; the gateway never guesses a GPU.
2. **Lease.** `orion.gpu_pool.client.gpu_lease(work_class, priority, holder, min_ctx_tokens,
   deadline_sec, turn_correlation_id)`. `holder` is the calling service's name on the bus path,
   `http:openai` / `http:anthropic` on the passthroughs. `min_ctx_tokens` =
   ceil(prompt chars / 4) + `max_tokens`, so the pool never places a prompt on a role whose
   per-slot context is smaller. The wait is capped by `LLM_GATEWAY_POOL_[BACKGROUND_]WAIT_SEC`
   and by the caller's own budget; what is left after the grant becomes the upstream read timeout.
3. **Run on the grant.** The call goes to `grant.url` on a per-role thread pool
   (`LLM_GATEWAY_EXECUTOR_WORKERS_PER_ROLE`), never one executor shared across lanes. Streams
   hold the lease until the stream ends, errors, or the client leaves.
4. **Release.** `ok` on success; `upstream_error` when the upstream failed (an exception, an HTTP
   error, or a returned `[Error: ...]`/`raw.error` result).
5. **Context overflow.** If the granted role rejects the prompt as too long, the lease is released
   and re-acquired **once** with `min_ctx_tokens = grant.ctx_per_slot + 1`. If that also
   overflows, or no role is big enough, the overflow error is returned. (Replaces the old
   escalation ladder, which POSTed straight to other routes' URLs.)

Failure shape (bus): empty text with `raw.error = "gpu_pool_unavailable"` and
`raw.details = {reason, route, work_class}` -- `reason` is the pool's (`deadline`,
`no_serviceable_role`, ...) or `pool_bus_unavailable` / `pool_unreachable:<Error>`. HTTP: 503
with `error.type = "gpu_pool_unavailable"`.

Deleted with this cutover: `capacity.py` (durable-runs `/capacity` permits),
`upstream_admission.py` (per-upstream semaphores), `priority_admission.py` (background `/slots`
polling -- pool priority replaces it), `lane_gate.py` + `GET/PUT /routes/{id}/gate` (the pool's
gpu0 lend flag replaces the chat-burst gate), `admission_ledger.py` + `GET /admission`, the
route table and startup route probes, and the context-overflow ladder.

## Running & Testing

### Run via Docker
```bash
docker compose -f services/orion-llm-gateway/docker-compose.yml up -d llm-gateway
```

> Note: Only run a single `orion-llm-gateway` subscriber on the shared request topic.
> Route isolation is expressed through `config/gpu_pool.yaml` (classes and roles), not by running multiple gateways.
> **Updated 2026-08-14**: `agent` split off from `chat` as the default. It used to alias
> `chat`'s worker (merged mode, below) because no distinct agent-lane model existed yet.
> Now that Muse Glimmer is live on Circe's dedicated agent-lane worker (port 8014),
> `agent` points there instead by default.
>
> **CORRECTED 2026-09-02**: the agent-lane worker's port moved 8014 -> 8015
> (confirmed live: 8014 collides with orion-circe-diffusion-host's own port
> mapping on circe, an unrelated service that happened to claim the same
> host port -- not something to fix on diffusion-host's side). The worker
> also now serves `qwen3.8-27b-udq4kxl-v100-32gb-circe-agent-flex`
> (`config/llm_profiles.yaml`), not Muse Glimmer -- that profile is still
> defined in the file, just no longer wired to `ATLAS_AGENT_PROFILE_NAME`
> by default.
>
> **Do not infer physical host from the `atlas-*` naming** anywhere in this file --
> `ATLAS_AGENT_*` env vars and the `atlas-agent` compose service/container name are a
> fixed naming convention for this worker *pattern*, reused across whichever physical
> host runs it (the same reason `orion-atlas-llamacpp-chat`, below, runs on Circe
> hardware despite its name). An earlier version of this doc pointed `agent` at
> Atlas's IP based on that naming alone -- wrong; nothing was listening there, and the
> gateway correctly reported `agent` as down until this was corrected the same day.
>
> **2026-08-21: Atlas is retired for good** (chassis reused for other hardware; the
> Atlas Tailscale node itself is offline for good). `metacog`/`quick`/`quick_background`
> below have been repointed from Atlas's old IP (`100.121.214.30`) to Circe -- confirmed
> live before the fix that every call through those three routes was hanging up to 700s
> against a dead host instead of failing. Both replacement workers are now deployed and
> confirmed live via `GET /routes` (all six routes report `status: "up"`):
> `quick`/`quick_background` -> `circe-worker-fast-1` (port 8013, GPU 4, Qwen3-8B Q4_K_M,
> `qwen3-8b-q4km-v100-16gb-balanced`); `metacog` -> `circe-worker-2` (port 8012, GPU 3,
> Qwen3-8B Q5_K_M, `qwen3-8b-q5km-v100-16gb-atlas-metacog-16k` -- single-GPU, not the
> 2xGPU qwen3-30b profile that shares the "atlas-metacog" name prefix).

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
## Optional durable resource leases

`LLM_GATEWAY_LEASE_VALIDATION_ENABLED=true` is the operator-template default.
Requests carrying a typed `resource_lease` (bus) or the bounded
`X-Orion-Resource-Lease` header (Anthropic HTTP) must validate against
`LLM_GATEWAY_LEASE_VALIDATION_URL` (default
`http://durable-runs:8121/leases/validate`). Checks occur before dispatch,
periodically during execution/streaming, and before accepting the final result.
The interval defaults to 5 seconds and validation timeout to 2 seconds. Missing
tokens remain valid for existing synchronous traffic; malformed or stale tokens
are rejected. Tokens never reach the model prompt or backend headers.

Since the GPU pool cutover the durable lease is an **admission token only**:
the lane must match and the broker must still consider the generation current,
but placement always comes from a GPU pool lease (a burst route is agent work:
`agent-burst`/`chat-burst` -> class `agent`). The lease's `backend_key` is not
compared with the granted URL, because the pool may legitimately place the call
on another role. A cancelled blocking Python HTTP thread keeps its pool lease
until the thread exits; stale results are rejected, but physical inference
cannot be forcibly stopped by cancelling that thread.

See [resource admission ownership and rollout](../../docs/architecture/durable-resource-admission.md).

## Lending chat's card

The old `chat-burst` operator gate (`lane_gate.py`, `PUT /routes/chat-burst/gate`) is gone. Lending
Juniper's chat card is the GPU pool's `lent` flag on `gpu0` (Hub GPU pool panel -> control RPC).
Until durable-runs reads pool state (stage 4), `GET /routes` still reports `chat-burst` as
`operator_closed` (with `gate_open: false`) unless gpu0 is lent, and `agent-burst` as `up` only
while the `agent-gpu2` swap seat is confirmed.

## Optional GPU2 elastic admission

GPU2 diffusion/agent-burst borrowing is additive and defaults off. See the
[ownership ADR](../../docs/architecture/gpu2-elastic-admission.md),
[pre-edit repository/live evidence](../../docs/architecture/gpu2-elastic-evidence.md),
and [consumer-first rollout and rollback](../../docs/runbooks/gpu2-elastic-admission.md)
for this service's exact flags, HTTP contracts and operator commands.
No production env sync, migration, GPU transition or deployment was performed.
