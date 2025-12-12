# orion-llamaccp

A **bus-native, profile-driven llama.cpp server wrapper** for Orion.

This service exists to make **GGUF model serving deterministic**:
- A single env var (`LLM_PROFILE_NAME`) selects the active model.
- Everything else (model path, HF download spec, llama-server knobs, GPU pinning hints) lives in **`config/llm_profiles.yaml`**.
- `.env` is for **selection + overrides only**.

---

## What it runs

- **Engine:** `ghcr.io/ggerganov/llama.cpp:server-cuda-<build>`
- **Wrapper:** `services/orion-llamaccp/app` (Python)
- **Model format:** GGUF
- **Network:** `app-net`

The wrapper:
1. Loads `config/llm_profiles.yaml`
2. Selects the profile by `LLM_PROFILE_NAME`
3. Resolves the GGUF file path
4. If missing and a download spec is present, downloads from Hugging Face
5. Launches `llama-server` with CLI flags derived from the profile (plus any env overrides)

---

## Repo layout (root-driven)

From the Orion repo root:

```
config/
  llm_profiles.yaml

services/
  orion-llamaccp/
    app/
    docker-compose.yml
    Dockerfile
    requirements.txt
    .env
```

Important: `config/llm_profiles.yaml` is a **global registry** (many models). The active one is selected by `LLM_PROFILE_NAME`.

---

## Configuration model

### 1) Profiles (source of truth)

The wrapper expects a top-level `profiles:` key.

Example shape:

```yaml
profiles:

  deepseek-70b-gguf-atlas:
    display_name: "DeepSeek R1 70B – GGUF via llama.cpp (Atlas)"
    task_type: chat
    backend: llamacpp
    model_id: "deepseek-r1-70b-q4_k_m"

    supports_tools: false
    supports_embeddings: false
    supports_vision: false

    gpu:
      num_gpus: 4
      tensor_parallel_size: 4
      device_ids: [0,1,2,3]
      max_model_len: 8192
      max_batch_tokens: 512
      max_concurrent_requests: 1
      gpu_memory_fraction: 0.9

    llamacpp:
      model_root: "/models/gguf"
      hf_repo_id: "unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF"
      hf_filename: "DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf"

      host: "0.0.0.0"
      port: 8080
      ctx_size: 8192
      n_gpu_layers: 80
      threads: 16
      parallel: 1
      batch_size: 512
```

Notes:
- `backend` **must** be `llamacpp` for this service.
- `model_id` is Orion’s logical label.
- The actual file is derived from:
  - `llamacpp.model_root + llamacpp.hf_filename` (or)
  - a direct GGUF path (if you set it up that way in your wrapper).

### 2) Service `.env` (selection + overrides)

This file should be minimal. It chooses the profile and optionally overrides a few knobs.

Recommended minimal `.env`:

```bash
# identity
SERVICE_NAME=orion-llamaccp
SERVICE_VERSION=0.1.0

# profile selection
LLM_PROFILE_NAME=deepseek-70b-gguf-atlas
LLM_PROFILES_CONFIG_PATH=/app/config/llm_profiles.yaml

# model cache mount inside container (compose uses this)
LLM_CACHE_DIR=/mnt/telemetry/llm-cache

# optional HF auth (only if gated/private)
HF_TOKEN=

# optional overrides (blank = not set)
LLAMACPP_CTX_SIZE_OVERRIDE=
LLAMACPP_N_GPU_LAYERS_OVERRIDE=
LLAMACPP_THREADS_OVERRIDE=
LLAMACPP_PARALLEL_OVERRIDE=
LLAMACPP_BATCH_SIZE_OVERRIDE=
CUDA_VISIBLE_DEVICES_OVERRIDE=
```

Rule: if a value is already in the profile, **don’t duplicate it** in `.env` unless you’re overriding.

---

## docker-compose

This service is run from the repo root with **two env files**:
- root `.env` (shared stack)
- service `.env` (selection + overrides)

Example command:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-llamaccp/.env \
  -f services/orion-llamaccp/docker-compose.yml \
  up -d --build
```

### Key compose behaviors

- Mounts the host GGUF cache into the container at `/models`.
- Exposes llama-server on `LLAMACPP_HOST_PORT`.
- Wrapper reads `/app/config/llm_profiles.yaml` (baked into the image).

---

## Runtime endpoints

- **Health:** `GET /health`
- **OpenAI-style:** `POST /v1/chat/completions`

Typical internal URL (from app-net):

```
http://orion-llamaccp:8080
```

If you map it to the host:

```
http://<host-ip>:<host-port>
```

---

## How the model gets onto disk

### Automatic (preferred)
If the GGUF file is missing under:

```
/models/gguf/<hf_filename>
```

…and your profile provides:
- `llamacpp.hf_repo_id`
- `llamacpp.hf_filename`

…the wrapper will download the file using `huggingface_hub`.

### Manual
You can also copy models into the host cache directory:

```
${LLM_CACHE_DIR}/gguf/
```

…and the service will pick them up.

---

## GPU pinning

Two layers exist:

1. **Profile hint** (Orion semantics):
   - `gpu.device_ids: [0,1,2,3]`

2. **Actual container pinning** (what NVIDIA runtime enforces):
   - `CUDA_VISIBLE_DEVICES` (or your wrapper’s override)

Recommendation:
- Use the profile `device_ids` as the default.
- Only override via env when you’re actively rebalancing GPUs.

---

## Common failures

### `llm_profiles.yaml not found`
- Ensure the Dockerfile **copies** `config/` into the image at `/app/config`.
- Ensure `LLM_PROFILES_CONFIG_PATH=/app/config/llm_profiles.yaml`.

### `LLM profile 'X' not found in registry`
- Your YAML is probably:
  - missing the top-level `profiles:` key, or
  - the profile name doesn’t match `LLM_PROFILE_NAME`.

### `missing tensor ...`
- Usually indicates one of:
  - incompatible llama.cpp build vs GGUF version
  - corrupted/incomplete download
  - wrong file (not the expected quant)

Fix:
- delete the GGUF and re-download
- confirm the exact filename in the profile matches what exists under `/models/gguf`

### CLI flag errors (e.g. `invalid argument: --n-parallel`)
- llama.cpp CLI flags change over time.
- Confirm the correct flag name for your build (`--parallel` vs `--n-parallel`).

---

## Recommended operating pattern

- Keep `config/llm_profiles.yaml` as the authoritative model registry.
- Switch models by changing **only** `LLM_PROFILE_NAME`.
- Use env overrides only when troubleshooting stability (ctx, gpu layers, parallel, batch).

---

## Quick debug commands

### Verify profile exists inside image

```bash
docker run --rm --entrypoint sh orion-llamaccp:0.1.0 -lc \
  "grep -n 'deepseek-70b-gguf-atlas' -n /app/config/llm_profiles.yaml && echo OK"
```

### Verify GGUF exists on the host

```bash
ls -lah ${LLM_CACHE_DIR}/gguf | head
```

### Watch VRAM while loading

```bash
watch -n 1 nvidia-smi
```

---

## Versioning

- `SERVICE_VERSION` should track wrapper changes.
- Engine build is pinned by the llama.cpp image tag.

---

## Notes

This service is intentionally **not** an orchestration layer. It loads **exactly one active profile** per container instance.

To run multiple models at once:
- start multiple `orion-llamaccp` instances (separate service names/ports)
- each with a different `LLM_PROFILE_NAME`
- and ideally different GPU pinning
