# orion-llamacpp-host

`orion-llamacpp-host` is Orion's **profile-driven llama.cpp wrapper**.

It exists to keep GGUF serving deterministic:

- one container boots **one active profile**
- `LLM_PROFILE_NAME` selects that profile
- `config/llm_profiles.yaml` is the source of truth for model path / download spec / llama.cpp runtime knobs / GPU hints
- `.env` should mostly contain **selection and safe overrides**, not model definitions

If you are standing up Atlas workers for the current `chat` / `agent` (merged physical lane), `metacog`, and `quick` (Hub-facing) layout, start with the Atlas quickstart below and then use the repo-root [`postflight.md`](../../postflight.md) as the shareable operator runbook.

---

## TL;DR for Atlas operators

For the default merged gateway route layout, Atlas should run **3 always-on `orion-llamacpp-host` containers** plus an optional split-agent worker (serving 4 logical routes in merged mode):

| Route | Compose service | Worker `SERVICE_NAME` | Host port | Profile source | GPU binding env |
| --- | --- | --- | --- | --- | --- |
| `chat` | `atlas-chat` | `atlas-worker-1` | `8011` | `ATLAS_CHAT_PROFILE_NAME` | `ATLAS_CHAT_CUDA_VISIBLE_DEVICES` |
| `agent` (logical) | routes to `chat` backend by default | `atlas-worker-1` | `8011` | same as chat | same as chat |
| `metacog` | `atlas-metacog` | `atlas-worker-2` | `8012` | `ATLAS_METACOG_PROFILE_NAME` | `ATLAS_METACOG_CUDA_VISIBLE_DEVICES` |
| `quick` (Hub-visible) | `atlas-fast` | `atlas-worker-fast-1` | `8013` | `ATLAS_FAST_PROFILE_NAME` | `ATLAS_FAST_CUDA_VISIBLE_DEVICES` |
| `agent` (optional split mode) | `atlas-agent` | `atlas-worker-agent-1` | `8014` | `ATLAS_AGENT_PROFILE_NAME` | `ATLAS_AGENT_CUDA_VISIBLE_DEVICES` |

`quick` is a distinct **logical gateway route** that uses the dedicated FAST physical lane (`atlas-fast`).

Use:

- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml`
- `services/orion-llamacpp-host/.env_example`
- `config/llm_profiles.yaml`
- `../../postflight.md`

Key rule: each worker must keep a unique **service identity**, **published port**, **`LLM_PROFILE_NAME`**, and **GPU visibility binding**.

---

## Atlas quickstart

### 1. Create the Atlas worker env file

```bash
cp services/orion-llamacpp-host/.env_example services/orion-llamacpp-host/.env.atlas
```

### 2. Edit the shared values

At minimum set:

```dotenv
ORION_BUS_URL=redis://100.92.216.81:6379/0
LLM_CACHE_DIR=/mnt/telemetry/llm-cache
HF_TOKEN=
hf_token=
```

### 3. Edit the per-worker unique values

```dotenv
ATLAS_CHAT_SERVICE_NAME=atlas-worker-1
ATLAS_CHAT_PROFILE_NAME=llama3-8b-instruct-q4km-athena-p4
ATLAS_CHAT_CUDA_VISIBLE_DEVICES=0
ATLAS_CHAT_HOST_PORT=8011

ATLAS_METACOG_SERVICE_NAME=atlas-worker-2
ATLAS_METACOG_PROFILE_NAME=
ATLAS_METACOG_CUDA_VISIBLE_DEVICES=
ATLAS_METACOG_HOST_PORT=8012

ATLAS_FAST_SERVICE_NAME=atlas-worker-fast-1
ATLAS_FAST_PROFILE_NAME=
ATLAS_FAST_CUDA_VISIBLE_DEVICES=
ATLAS_FAST_HOST_PORT=8013

ATLAS_AGENT_SERVICE_NAME=atlas-worker-agent-1
ATLAS_AGENT_PROFILE_NAME=
ATLAS_AGENT_CUDA_VISIBLE_DEVICES=
ATLAS_AGENT_HOST_PORT=8014
```

Notes:

- `ATLAS_CHAT_PROFILE_NAME` is intentionally operator-supplied; keep it pointed at the current Atlas chat profile.
- `ATLAS_METACOG_PROFILE_NAME`, `ATLAS_FAST_PROFILE_NAME`, and `ATLAS_AGENT_PROFILE_NAME` are intentionally operator-supplied.
- FAST (`ATLAS_FAST_*`) is shared by full-chat first pass and user-facing quick mode.
- Do **not** let the default workers collide on host ports (`8011/8012/8013`).
- Optional `atlas-agent` split mode is behind compose profile `agent-split`.

### 4. Ensure the Docker network exists

```bash
docker network create app-net >/dev/null 2>&1 || true
```

### 5. Launch the Atlas workers

```bash
  docker compose \
  --env-file services/orion-llamacpp-host/.env.atlas \
  -f services/orion-llamacpp-host/docker-compose.atlas-workers.yml \
  up -d --build atlas-chat atlas-metacog atlas-fast
```

### 6. Verify each worker directly

```bash
curl http://localhost:${ATLAS_CHAT_HOST_PORT}/health
curl http://localhost:${ATLAS_METACOG_HOST_PORT}/health
curl http://localhost:${ATLAS_FAST_HOST_PORT}/health
```

Optional split agent worker:

```bash
docker compose \
  --env-file services/orion-llamacpp-host/.env.atlas \
  --profile agent-split \
  -f services/orion-llamacpp-host/docker-compose.atlas-workers.yml \
  up -d --build atlas-agent
curl http://localhost:${ATLAS_AGENT_HOST_PORT}/health
```

### 7. Then validate through the gateway

Atlas worker validation is only half the story. The route map is gateway-centered, so once the workers are healthy, continue with:

- `services/orion-llm-gateway/.env_example`
- `services/orion-llm-gateway/README.md`
- `../../postflight.md`

Use the route smoke command documented in `postflight.md` / gateway README:

```bash
PYTHONPATH=/workspace/Orion-Sapienform python -m scripts.smoke_llm_gateway_routes \
  --redis "${ORION_BUS_URL}" \
  --request-channel "${CHANNEL_LLM_INTAKE}"
```

---

## What this service actually does

On boot, the wrapper:

1. loads `config/llm_profiles.yaml`
2. selects one profile via `LLM_PROFILE_NAME`
3. resolves the GGUF model path
4. downloads the GGUF from Hugging Face if the file is missing and the profile includes a download spec
5. launches `llama-server` with the profile's runtime values plus any explicitly-set overrides

This service is intentionally **not** an orchestrator. It runs **exactly one active profile per container**.

---

## Source-of-truth files

When working on this service, these are the main files that matter:

- `config/llm_profiles.yaml` — model/profile registry
- `services/orion-llamacpp-host/.env_example` — selection + override contract, plus Atlas `ATLAS_*` examples
- `services/orion-llamacpp-host/docker-compose.yml` — single-worker compose
- `services/orion-llamacpp-host/docker-compose.atlas-workers.yml` — Atlas merged-mode workers (`chat`, `metacog`, `fast`) where logical `quick` and chat pass-1 share the FAST physical lane, plus optional `agent-split` profile
- `services/orion-llamacpp-host/app/settings.py` — env contract actually parsed by the wrapper
- `services/orion-llamacpp-host/app/main.py` — model resolution, GPU binding, and llama-server launch logic
- `../../postflight.md` — operator-facing Atlas runbook

---

## Configuration model

## 1. Profiles are the source of truth

The wrapper expects a top-level `profiles:` key in `config/llm_profiles.yaml`.

Minimal shape:

```yaml
profiles:
  llama3-8b-instruct-q4km-athena-p4:
    backend: llamacpp
    model_id: llama-3-8b-instruct-q4_k_m
    gpu:
      num_gpus: 1
      tensor_parallel_size: 1
      device_ids: [0]
    llamacpp:
      model_root: /models/gguf
      hf_repo_id: PawanKrd/Meta-Llama-3-8B-Instruct-GGUF
      hf_filename: llama-3-8b-instruct.Q4_K_M.gguf
      host: 0.0.0.0
      port: 8080
      ctx_size: 4096
      n_gpu_layers: 35
      threads: 8
      n_parallel: 1
      batch_size: 256
```

Rules:

- `backend` must be `llamacpp` for this service.
- Prefer keeping model/runtime definition in the profile, not in `.env`.
- For Atlas multi-worker use, the profile name is what differentiates chat/metacog/agent behavior.

## 2. `.env` is for selection + overrides

The wrapper directly cares about these envs:

### Required or practically required

```dotenv
SERVICE_NAME=llamacpp-host
SERVICE_VERSION=0.1.0
ORION_BUS_URL=redis://100.92.216.81:6379/0
LLM_PROFILES_CONFIG_PATH=/app/config/llm_profiles.yaml
LLM_PROFILE_NAME=llama3-8b-instruct-q4km-athena-p4
LLM_CACHE_DIR=/mnt/telemetry/llm-cache
```

### Optional

```dotenv
HF_TOKEN=
hf_token=
LLAMACPP_MODEL_PATH_OVERRIDE=
LLAMACPP_HOST_OVERRIDE=
LLAMACPP_PORT_OVERRIDE=
LLAMACPP_CTX_SIZE_OVERRIDE=
LLAMACPP_N_GPU_LAYERS_OVERRIDE=
LLAMACPP_THREADS_OVERRIDE=
LLAMACPP_N_PARALLEL_OVERRIDE=
LLAMACPP_BATCH_SIZE_OVERRIDE=
CUDA_VISIBLE_DEVICES_OVERRIDE=
```

### Compatibility / compose passthroughs

These still appear in `.env_example` because the compose wiring expects them or sibling services use them, but the current `orion-llamacpp-host` runtime does not read them directly:

```dotenv
MODELS_MOUNT_ROOT=/models
ENSURE_MODEL_DOWNLOAD=true
WAIT_FOR_MODEL_SECONDS=0
ORION_BUS_ENFORCE_CATALOG=true
```

Rule of thumb: if a value already exists in `config/llm_profiles.yaml`, do not duplicate it in `.env` unless you are deliberately overriding it.

---

## Compose usage

## Single-worker compose

Use `services/orion-llamacpp-host/docker-compose.yml` when you want one standalone worker:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-llamacpp-host/.env \
  -f services/orion-llamacpp-host/docker-compose.yml \
  up -d --build
```

This is the simpler path for one worker / one profile.

## Atlas multi-worker compose

Use `services/orion-llamacpp-host/docker-compose.atlas-workers.yml` for the current Atlas topology.

Important characteristics of this compose file:

- one compose service per active lane (`atlas-chat`, `atlas-metacog`, `atlas-fast`) plus optional `atlas-agent` split profile
- one active profile per container
- one published host port per container
- one explicit GPU binding per container
- all workers mount the same model cache root at `/models`
- it assumes `app-net` already exists

This is the repo's best current Atlas pattern, but it is still **operator-edited compose**, not a full deployment system.

---

## Runtime endpoints

Each worker exposes:

- `GET /health`
- `POST /v1/chat/completions`

Typical internal container URL:

```text
http://orion-llamacpp-host:8080
```

Typical host-facing URL:

```text
http://<host-ip>:<host-port>
```

For Atlas multi-worker bring-up, the relevant host ports are normally `8011`, `8012`, and `8014`.

---

## Model loading behavior

## Automatic download

If the resolved GGUF file does not exist and the selected profile includes:

- `llamacpp.hf_repo_id`
- `llamacpp.hf_filename`

then the wrapper will download the file into the configured model root.

## Manual population

You can also stage the GGUF manually under:

```text
${LLM_CACHE_DIR}/gguf/
```

The compose files mount that cache into the container at `/models`.

---

## GPU pinning model

There are two layers:

1. **Profile hint** in `config/llm_profiles.yaml`
   - example: `gpu.device_ids: [2]`
2. **Actual container runtime binding**
   - `CUDA_VISIBLE_DEVICES_OVERRIDE`
   - or fallback to the profile's `gpu.device_ids`

Recommended practice:

- keep `gpu.device_ids` correct in the profile
- use `CUDA_VISIBLE_DEVICES_OVERRIDE` only when the operator needs to rebalance GPU placement
- for Atlas multi-worker setups, make the binding explicit for all 3 workers

---

## Common failures

### `llm_profiles.yaml not found`

- confirm the image contains `/app/config/llm_profiles.yaml`
- confirm `LLM_PROFILES_CONFIG_PATH=/app/config/llm_profiles.yaml`

### `LLM profile 'X' not found in registry`

- confirm the profile exists under the top-level `profiles:` key
- confirm `LLM_PROFILE_NAME` exactly matches the registry key

### `Model not found and no download spec available`

- the selected profile resolved to a GGUF path that does not exist
- and the profile does not provide a usable `hf_repo_id` / `hf_filename`

### `missing tensor ...`

Usually one of:

- incompatible llama.cpp build vs GGUF version
- incomplete or corrupted GGUF
- wrong quant/file selected

### CLI flag errors such as `--n-parallel`

The wrapper currently launches llama.cpp using `--parallel`. If your engine build changes behavior, confirm the exact CLI supported by that image.

---

## Debug commands

### Confirm the profile exists inside the image

```bash
docker run --rm --entrypoint sh orion-llamacpp-host:0.1.0 -lc \
  "grep -n 'llama3-8b-instruct-q4km-athena-p4' /app/config/llm_profiles.yaml && echo OK"
```

### Confirm the GGUF exists on the host

```bash
ls -lah ${LLM_CACHE_DIR}/gguf | head
```

### Watch VRAM while loading

```bash
watch -n 1 nvidia-smi
```

---

## Recommended operating pattern

- treat `config/llm_profiles.yaml` as the authoritative model registry
- switch workers by changing `LLM_PROFILE_NAME`, not by rewriting runtime flags
- keep Atlas isolation explicit with route -> URL mapping, unique worker identity, unique host port, and unique GPU binding
- use `../../postflight.md` when handing the procedure to another operator or another GPT session

---

## Pinned llama.cpp CUDA base image

`services/orion-llamacpp-host/Dockerfile` now pins to:

- `ghcr.io/ggerganov/llama.cpp:server-cuda-b5401` (via `LLAMACPP_IMAGE_TAG` build arg)

Why this pin:

- Qwen docs list `b5092` as the minimum for Qwen3/Qwen3MoE support.
- `b5401` is above that floor while remaining a fixed non-HEAD build.
- Existing Qwen2/Qwen2.5 GGUF workflows remain in the same llama.cpp runtime family and OpenAI-compatible `llama-server` surface.

### Build with the pinned target

```bash
docker compose \
  --env-file services/orion-llamacpp-host/.env_example \
  -f services/orion-llamacpp-host/docker-compose.yml \
  build orion-llamacpp-host
```

### Optional explicit build tag override

```bash
LLAMACPP_IMAGE_TAG=server-cuda-b5401 docker compose \
  --env-file services/orion-llamacpp-host/.env_example \
  -f services/orion-llamacpp-host/docker-compose.yml \
  build orion-llamacpp-host
```

### Rollback

If regression appears, rollback to prior known-good image tag:

```bash
LLAMACPP_IMAGE_TAG=server-cuda-b4719 docker compose \
  --env-file services/orion-llamacpp-host/.env_example \
  -f services/orion-llamacpp-host/docker-compose.yml \
  build --no-cache orion-llamacpp-host
```

### Validation harness

Run the repo-local validator:

```bash
MODEL_QWEN25_PATH=/mnt/telemetry/llm-cache/gguf/Qwen2.5-32B-Instruct-abliterated.Q6_K.gguf \
MODEL_QWEN3_PATH=/mnt/telemetry/llm-cache/gguf/Qwen3-30B-A3B-Q4_K_M.gguf \
IMAGE=orion-llamacpp-host:0.1.0 \
services/orion-llamacpp-host/scripts/validate_llamacpp_upgrade.sh
```
