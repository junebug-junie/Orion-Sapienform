# Orion Atlas LLM Postflight

This note packages the current Atlas multi-worker onboarding and config-alignment instructions into one shareable document.

## Scope

This covers the current intended Atlas route layout:

- `chat` -> existing Atlas chat worker endpoint (`atlas-worker-1`)
- `metacog` -> Atlas dedicated metacog worker (`atlas-worker-2`)
- `agent` -> Atlas heavy Qwen worker (`atlas-worker-agent-1`)

It assumes the current Orion architecture:

- routing is gateway-centered
- `LLM_GATEWAY_ROUTE_TABLE_JSON` is the primary routing mechanism
- `served_by` is observability metadata, not routing authority
- `orion-llamacpp-host` is effectively one profile per container / one worker endpoint per container
- isolation is achieved by explicit route -> URL mapping, unique ports, unique service identity, unique GPU binding, and unique `LLM_PROFILE_NAME`

---

## 1. Atlas worker model

Atlas should run **3** `orion-llamacpp-host` containers for the current layout:

1. `atlas-chat`
2. `atlas-metacog`
3. `atlas-agent`

The repo models this directly in:

- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`
- `config/llm_profiles.yaml`

### What makes each worker distinct

Each worker should have a distinct:

- compose service name
- container name
- `SERVICE_NAME`
- published host port
- `LLM_PROFILE_NAME`
- `CUDA_VISIBLE_DEVICES_OVERRIDE`

The provided compose file uses one shared compose project prefix (`${PROJECT:-orion}`), so project name is **not** the main source of isolation between the 3 workers. The isolation comes from the explicit per-service identity, port, profile, and GPU binding.

### Current worker inventory

| Route | Compose service | Worker `SERVICE_NAME` | Host port | Profile | GPU binding |
| --- | --- | --- | --- | --- | --- |
| `chat` | `atlas-chat` | `atlas-worker-1` | `8011` | operator-supplied via `ATLAS_CHAT_PROFILE_NAME` | `ATLAS_CHAT_CUDA_VISIBLE_DEVICES` |
| `metacog` | `atlas-metacog` | `atlas-worker-2` | `8012` | `llama3-8b-instruct-q4km-atlas-metacog` | `ATLAS_METACOG_CUDA_VISIBLE_DEVICES=2` |
| `agent` | `atlas-agent` | `atlas-worker-agent-1` | `8014` | `qwen3-30b-a3b-q4km-atlas-agent` | `ATLAS_AGENT_CUDA_VISIBLE_DEVICES=1` |

---

## 2. Source-of-truth files

Primary files to inspect and edit:

- `services/orion-llamacpp-host/.env_example`
- `services/orion-llamacpp-host/README.md`
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`
- `services/orion-llm-gateway/.env_example`
- `services/orion-llm-gateway/README.md`
- `services/orion-llm-gateway/app/settings.py`
- `services/orion-llm-gateway/app/llm_backend.py`
- `services/orion-llm-gateway/app/main.py`
- `config/llm_profiles.yaml`
- `scripts/smoke_llm_gateway_routes.py`

---

## 3. Env contract map

## 3.1 Shared Atlas llama.cpp worker vars

These are normally shared across all Atlas worker containers:

```dotenv
SERVICE_VERSION=0.1.0
ORION_BUS_URL=redis://100.92.216.81:6379/0
LLM_PROFILES_CONFIG_PATH=/app/config/llm_profiles.yaml
LLM_CACHE_DIR=/mnt/telemetry/llm-cache
HF_TOKEN=
hf_token=
```

## 3.2 Per-worker unique Atlas llama.cpp vars

These should differ per worker:

```dotenv
ATLAS_CHAT_SERVICE_NAME=atlas-worker-1
ATLAS_CHAT_PROFILE_NAME=llama3-8b-instruct-q4km-athena-p4
ATLAS_CHAT_CUDA_VISIBLE_DEVICES=0
ATLAS_CHAT_HOST_PORT=8011

ATLAS_METACOG_SERVICE_NAME=atlas-worker-2
ATLAS_METACOG_PROFILE_NAME=llama3-8b-instruct-q4km-atlas-metacog
ATLAS_METACOG_CUDA_VISIBLE_DEVICES=2
ATLAS_METACOG_HOST_PORT=8012

ATLAS_AGENT_SERVICE_NAME=atlas-worker-agent-1
ATLAS_AGENT_PROFILE_NAME=qwen3-30b-a3b-q4km-atlas-agent
ATLAS_AGENT_CUDA_VISIBLE_DEVICES=1
ATLAS_AGENT_HOST_PORT=8014
```

Uniqueness requirements:

- unique `SERVICE_NAME`
- unique host port
- unique `LLM_PROFILE_NAME`
- unique GPU visibility binding unless intentional sharing is desired

## 3.3 Atlas gateway vars

Minimum gateway vars to set explicitly:

```dotenv
PROJECT=orion
SERVICE_NAME_LLM_GATEWAY=llm-gateway
SERVICE_VERSION_LLM_GATEWAY=0.1.1
ORION_BUS_URL=redis://100.92.216.81:6379/0
ORION_BUS_ENABLED=true
ORION_BUS_ENFORCE_CATALOG=true
CHANNEL_LLM_INTAKE=orion:exec:request:LLMGatewayService
LLM_ROUTE_DEFAULT=chat
LLM_GATEWAY_HEALTH_PORT=8210
CONNECT_TIMEOUT_SEC=10
READ_TIMEOUT_SEC=700
```

### Route-table JSON (primary)

For the current Atlas merged-default layout, use route-table JSON as the authoritative mapping:

```dotenv
LLM_GATEWAY_ROUTE_TABLE_JSON='{
  "chat":{"url":"http://100.121.214.30:8011","served_by":"atlas-worker-1","backend":"llamacpp"},
  "agent":{"url":"http://100.121.214.30:8011","served_by":"atlas-worker-1","backend":"llamacpp"},
  "metacog":{"url":"http://100.121.214.30:8012","served_by":"atlas-worker-2","backend":"llamacpp"},
  "helper":{"url":"http://100.121.214.30:8013","served_by":"atlas-worker-helper-1","backend":"llamacpp"},
  "quick":{"url":"http://100.121.214.30:8013","served_by":"atlas-worker-helper-1","backend":"llamacpp"}
}'
```

Optional split mode keeps the logical `agent` route but points it at `8014`.

`quick` is a distinct logical route that intentionally shares the `helper` physical worker lane.

### Legacy / secondary envs

These still exist but should be treated as secondary:

- `ORION_LLM_LLAMACPP_URL`
- `LLM_ROUTE_CHAT_URL`
- `LLM_ROUTE_METACOG_URL`
- `LLM_ROUTE_LATENTS_URL`
- `LLM_ROUTE_SPECIALIST_URL`

Important: there is **no** legacy `LLM_ROUTE_AGENT_URL`, so the current `agent` route depends on `LLM_GATEWAY_ROUTE_TABLE_JSON`.

---

## 4. Operator bring-up runbook

## 4.1 Atlas workers

### Create the env file

```bash
cp services/orion-llamacpp-host/.env_example services/orion-llamacpp-host/.env.atlas
```

### Edit by hand

Edit in `services/orion-llamacpp-host/.env.atlas`:

- shared vars:
  - `ORION_BUS_URL`
  - `LLM_CACHE_DIR`
  - `HF_TOKEN` / `hf_token` if needed
- per-worker vars:
  - all `ATLAS_CHAT_*`
  - all `ATLAS_METACOG_*`
  - all `ATLAS_HELPER_*`
  - optional `ATLAS_AGENT_*` (split mode only)

### Ensure network exists

```bash
docker network create app-net >/dev/null 2>&1 || true
```

### Launch workers

```bash
docker compose \
  --env-file services/orion-llamacpp-host/.env.atlas \
  -f services/orion-llamacpp-host/docker-compose.atlas-workers.yml \
  up -d --build atlas-chat atlas-metacog atlas-helper
```

### Verify worker health

```bash
curl http://localhost:8011/health
curl http://localhost:8012/health
curl http://localhost:8013/health
```

## 4.2 Gateway

### Create the env file

```bash
cp services/orion-llm-gateway/.env_example services/orion-llm-gateway/.env
```

### Edit by hand

At minimum edit:

- `PROJECT`
- `ORION_BUS_URL`
- `ORION_BUS_ENFORCE_CATALOG`
- `CHANNEL_LLM_INTAKE` if your bus contract differs
- `LLM_GATEWAY_ROUTE_TABLE_JSON`
- `LLM_GATEWAY_HEALTH_PORT` if desired

### Launch gateway

```bash
docker compose \
  --env-file services/orion-llm-gateway/.env \
  -f services/orion-llm-gateway/docker-compose.yml \
  up -d llm-gateway
```

### Verify gateway health

```bash
curl http://localhost:8210/health
```

---

## 5. Validation / smoke tests

### Route smoke test

Use the module invocation form:

```bash
PYTHONPATH=/workspace/Orion-Sapienform python -m scripts.smoke_llm_gateway_routes \
  --redis "${ORION_BUS_URL}" \
  --request-channel "${CHANNEL_LLM_INTAKE}"
```

Expected outcome:

- `chat` returns `served_by=atlas-worker-1`
- `agent` returns `served_by=atlas-worker-1` (default merged mode)
- `metacog` returns `served_by=atlas-worker-2`
- `helper` returns `served_by=atlas-worker-helper-1`
- `quick` returns `served_by=atlas-worker-helper-1`

---

## 6. Known drift / caveats

### Still important to know

1. `LLM_GATEWAY_ROUTE_TABLE_JSON` is the correct primary routing mechanism.
2. `served_by` should be treated as metadata for observability and smoke checks, not as routing authority.
3. The legacy fallback path is incomplete for Atlas, because `agent` has no dedicated fallback alias env.
4. In legacy alias mode, `metacog` fallback metadata in runtime still defaults to `athena-worker-1` rather than `atlas-worker-2`.
5. `services/orion-llamacpp-host/docker-compose.yml` still passes some compatibility vars (`MODELS_MOUNT_ROOT`, `ENSURE_MODEL_DOWNLOAD`, `WAIT_FOR_MODEL_SECONDS`) that the current `orion-llamacpp-host` runtime does not read directly.
6. `services/orion-llm-gateway/docker-compose.yml` still contains a comment that says Ollama is required today; that comment is stale relative to the current Atlas llama.cpp route-table setup.

---

## 7. Quick checklists

### Bring-up checklist

- [ ] Create `services/orion-llamacpp-host/.env.atlas`
- [ ] Set shared Atlas worker vars
- [ ] Set all `ATLAS_*` worker vars
- [ ] Confirm unique ports / profiles / GPU bindings
- [ ] Create `app-net`
- [ ] Launch `atlas-chat`, `atlas-metacog`, `atlas-helper` (plus optional `atlas-agent` in split mode)
- [ ] Verify worker health on `8011`, `8012`, `8013` (and `8014` for split mode)
- [ ] Create `services/orion-llm-gateway/.env`
- [ ] Set `PROJECT`, bus vars, and route-table JSON
- [ ] Launch `llm-gateway`
- [ ] Verify gateway health on `8210`

### Smoke checklist

- [ ] `chat` route responds
- [ ] `agent` route responds
- [ ] `metacog` route responds
- [ ] `helper` route responds
- [ ] `quick` route responds
- [ ] each route returns the expected `served_by`

---

## 8. Bottom line

If you only remember five things:

1. Run **3** Atlas `orion-llamacpp-host` containers in merged mode (`chat`, `metacog`, `helper`) while routing logical `quick` through `helper`.
2. Keep each worker unique by **port**, **profile**, **GPU binding**, and **service identity**.
3. Use **`LLM_GATEWAY_ROUTE_TABLE_JSON`** as the source of truth for `chat`, `agent`, `metacog`, `helper`, and `quick` routing.
4. Treat **`served_by` as metadata**, not routing logic.
5. Use the repo smoke test to verify the exact route map before declaring Atlas ready.
