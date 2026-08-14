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
> **Updated 2026-08-14**: `agent` split off from `chat` as the default. It used to alias
> `chat`'s worker (merged mode, below) because no distinct agent-lane model existed yet.
> Now that Muse Glimmer is live on Atlas's dedicated agent-lane worker
> (`atlas-worker-agent-1`, port 8014), `agent` points there instead by default.

### Route table example (default: split agent mode)
```bash
LLM_GATEWAY_ROUTE_TABLE_JSON='{
  "chat":{"url":"http://100.112.254.99:8011","served_by":"circe-worker-1","backend":"llamacpp"},
  "agent":{"url":"http://100.121.214.30:8014","served_by":"atlas-worker-agent-1","backend":"llamacpp"},
  "metacog":{"url":"http://100.121.214.30:8012","served_by":"atlas-worker-2","backend":"llamacpp"},
  "quick":{"url":"http://100.121.214.30:8013","served_by":"atlas-worker-fast-1","backend":"llamacpp"}
}'
```

`quick` is the FAST lane route used by user-facing quick chat and chat_general pass-1.

> **Update (2026-07-30): circe IS wired into `chat`/`agent` now, and that's**
> **reserved capacity -- AI Town must never land on it.** The note that used
> to live here (dated 2026-07-18) described circe as not-yet-wired-in and
> deprioritized; that's stale. Circe came online, `chat`/`agent` route to it
> (`served_by: "circe-worker-N"`, correctly labeled since `0e3cae4d`), and it
> now appears as a real `execution_run` producer as this note originally
> anticipated. What that old note didn't anticipate: circe is meant to be
> **reserved for Juniper's direct deep/FCC turns**, not shared with AI Town's
> NPC dialogue (`2026-07-10`, `cfcb3126`, "Route AI Town and default LLM
> consumers to quick" -- deliberately pointed AI Town at the `quick` lane
> instead). That fix only changed a *script default*; it never touched the
> already-provisioned AI Town world's persisted Convex `LLM_MODEL`, which
> silently stayed on `chat` (-> circe) for weeks, undetected, until circe
> went offline and the whole town's NPC dialogue silently stalled for 10+
> hours (confirmed live 2026-07-30). Fixed by re-pointing that world's
> `LLM_MODEL` at `quick` and adding
> `services/orion-ai-town/scripts/check_llm_route_not_circe.py` -- a gate
> that reads AI Town's *live* configured model, resolves it through this
> gateway's own `/v1/models`, and refuses to pass if the resolved worker is
> circe-hosted. It's wired into both `wire_llm_gateway.sh` (hard-fails a
> fresh wire-up that points at circe) and `compact_convex_data.sh` (which
> replays whatever `LLM_MODEL` was already set, so it self-corrects instead
> of quietly re-preserving the same drift on every future compaction). If
> AI Town's `served_by` for its configured route ever needs to change,
> update `AITOWN_LLM_CHAT_ROUTE` deliberately -- don't just add a route
> entry that happens to point at circe.

### Route table example (legacy: merged mode, `agent` aliases `chat`)

Use this only if no distinct agent-lane model is deployed yet on your box — it
re-merges `agent` back into `chat`'s worker, the pre-2026-08-14 default:
```bash
LLM_GATEWAY_ROUTE_TABLE_JSON='{
  "chat":{"url":"http://100.112.254.99:8011","served_by":"circe-worker-1","backend":"llamacpp"},
  "agent":{"url":"http://100.112.254.99:8011","served_by":"circe-worker-1","backend":"llamacpp"},
  "metacog":{"url":"http://100.121.214.30:8012","served_by":"atlas-worker-2","backend":"llamacpp"},
  "quick":{"url":"http://100.121.214.30:8013","served_by":"atlas-worker-fast-1","backend":"llamacpp"}
}'
```

### Background-priority routes

A route entry can carry `"priority":"background"` and an optional
`"reserved_free_slots"` (default `1` if unset) alongside the normal
`url`/`served_by`/`backend` fields, sharing the exact same upstream as a
regular route:

```json
"quick_background": {
  "url": "http://100.121.214.30:8013",
  "served_by": "atlas-worker-fast-1",
  "backend": "llamacpp",
  "priority": "background",
  "reserved_free_slots": 2
}
```

Any request resolved to a background route waits for the upstream's own
`/slots` endpoint to report at least `reserved_free_slots` idle slots before
dispatching -- it never competes evenly with foreground traffic sharing the
same llama.cpp process. It's a fail-open gate, not a hard block: if `/slots`
is unreachable, or the upstream is permanently busy past
`LLM_GATEWAY_BACKGROUND_MAX_WAIT_SEC` (default 30s), the request forwards
anyway with a logged warning -- a background caller never gets its request
silently dropped.

Two entry points, two implementations in `priority_admission.py` (same
contract, different concurrency model -- see that module's docstring for
why): `handle_chat_completions_post` (`openai_passthrough.py`, async, used by
AI Town's native Convex `chatCompletion()` calls) awaits `wait_for_slack` via
the `background_admission` context manager, which also caps concurrent
background dispatches per route key with an `asyncio.Semaphore`
(`LLM_GATEWAY_BACKGROUND_CONCURRENCY`, default 1) so a burst can't itself
claim more slots than intended even when nominally free. `run_llm_chat`
(`llm_backend.py`, sync -- dispatched via `asyncio.to_thread`, used by
orion-cortex-exec's bus-native RPC path) calls `wait_for_slack_sync` directly,
with no concurrency cap: its only caller in that context (orion-embodiment's
speech path) is bounded to ~1, rarely 2, concurrent calls in practice (one
`active_conversation` at a time), not a strictly serialized guarantee, but
nowhere near the burst the async path's cap exists for.

Plain routes (no `priority` field) are completely unaffected on both paths --
the gate is never invoked for them, zero added latency, zero behavior change.

**Pilot instance (2026-07-30):** `quick_background` above started as AI
Town's native NPC dialogue route (via the async passthrough) and was
extended the same day to Orion's own in-town speech (via the sync bus path,
`EMBODIMENT_SPEECH_QUICK_LLM_ROUTE`) -- both now share GPU1's
`atlas-worker-fast-1` process without competing evenly with `orion-mind`
(`MIND_SEMANTIC_MODEL_ROUTE`/`MIND_STANCE_MODEL_ROUTE`) and `orion-hub`
(`MEMORY_GRAPH_SUGGEST_PRIMARY_ROUTE`), both still pointed at plain `quick`
and unaffected. Reaching Orion's own speech required two more fixes,
confirmed live: orion-cortex-exec's `llm_route` override only accepted
`{chat, quick, metacog}` (now includes `quick_background`), and
orion-embodiment's `_request_utterance_quick` only ever passed
`extra={"lane": ...}` to cortex-exec, never `extra={"llm_route": ...}` --
meaning `EMBODIMENT_SPEECH_LANE`/`EMBODIMENT_SPEECH_HUB_LLM_ROUTE` had never
actually controlled the live gateway route at all; the observed "quick"
behavior came entirely from cortex-exec's own per-verb default mapping,
coincidentally matching. No second GPU was available to give AI Town its own
dedicated small model (confirmed live: both V100s already host one model
each), so this pattern exists specifically so AI Town's dialogue can share
the same GPU/model without ever making those other, snappier consumers wait
behind it. This is meant as a reusable seam -- any other lane can add its
own `<lane>_background` entry pointed at an existing route's `url`/`served_by`
to get the same behavior, no gateway code changes required.

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
