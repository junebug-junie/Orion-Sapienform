# orion-llamacpp-host

A **bus-native, profile-driven llama.cpp server wrapper** for Orion.

This service exists to make **GGUF model serving deterministic**:
- A single env var (`LLM_PROFILE_NAME`) selects the active model.
- Everything else (model path, HF download spec, llama-server knobs, GPU pinning hints) lives in **`config/llm_profiles.yaml`**.
- `.env` is for **selection + overrides only**.

## Neural Projection (Dual-Lobe Architecture)

This service now runs **two** independent `llama-server` processes ("lobes") in the same container to support Orion's Neural Projection architecture:

1.  **Chat Lobe (Port 8000):** The primary chat model (e.g., Llama-3, DeepSeek). Defined by the profile.
2.  **Embedding Lobe (Port 8001):** A secondary lightweight model (e.g., `nomic-embed-text`) dedicated to generating vector embeddings for every chat output.

**Why?** This allows Orion to "feel" the semantic weight of its own output without slowing down the main chat loop or requiring a separate container.

---

## What it runs

- **Engine:** `ghcr.io/ggerganov/llama.cpp:server-cuda-<build>`
- **Wrapper:** `services/orion-llamacpp-host/app` (Python)
- **Model format:** GGUF
- **Network:** `app-net`

The wrapper:
1. Loads `config/llm_profiles.yaml`.
2. Selects the profile by `LLM_PROFILE_NAME`.
3. Resolves the Chat GGUF file path (downloads if missing).
4. Launches the **Chat Lobe** on Port 8000 (standard OpenAI API).
5. Launches the **Embedding Lobe** on Port 8001 (if configured via env vars).
6. Monitors both processes and restarts them if they crash.

---

## Repo layout (root-driven)

From the Orion repo root:

```
config/
  llm_profiles.yaml

services/
  orion-llamacpp-host/
    app/
    docker-compose.yml
    Dockerfile
    requirements.txt
    .env
```

Important: `config/llm_profiles.yaml` is a **global registry** (many models). The active one is selected by `LLM_PROFILE_NAME`.

---

## Configuration model

### 1) Profiles (Chat Lobe)

The wrapper expects a top-level `profiles:` key for the main chat model.

Example shape:

```yaml
profiles:

  deepseek-70b-gguf-atlas:
    display_name: "DeepSeek R1 70B – GGUF via llama.cpp (Atlas)"
    task_type: chat
    backend: llamacpp
    model_id: "deepseek-r1-70b-q4_k_m"

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

### 2) Embedding Lobe Configuration (Env Vars)

The Embedding Lobe is configured purely via environment variables in `.env` (or docker-compose).

| Variable | Default | Description |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL_PATH` | `None` | Path to the embedding GGUF model (e.g., `/models/gguf/nomic-embed-text-v1.5.Q4_K_M.gguf`). |
| `EMBEDDING_HOST` | `0.0.0.0` | Bind address. |
| `EMBEDDING_PORT` | `8001` | Port for the embedding API. |
| `EMBEDDING_CTX_SIZE` | `2048` | Context size for embeddings. |
| `EMBEDDING_N_GPU_LAYERS` | `99` | GPU layers to offload (usually small model fits entirely). |

### 3) Service `.env` (selection + overrides)

This file should be minimal. It chooses the profile and optionally overrides a few knobs.

Recommended minimal `.env`:

```bash
# identity
SERVICE_NAME=orion-llamacpp-host
SERVICE_VERSION=0.1.0

# profile selection (Chat Lobe)
LLM_PROFILE_NAME=deepseek-70b-gguf-atlas
LLM_PROFILES_CONFIG_PATH=/app/config/llm_profiles.yaml

# Embedding Lobe
EMBEDDING_MODEL_PATH=/models/gguf/nomic-embed-text-v1.5.Q4_K_M.gguf
EMBEDDING_PORT=8001

# model cache mount inside container (compose uses this)
LLM_CACHE_DIR=/mnt/telemetry/llm-cache

# optional HF auth (only if gated/private)
HF_TOKEN=

# optional overrides (blank = not set)
CUDA_VISIBLE_DEVICES_OVERRIDE=
```

---

## docker-compose

This service is run from the repo root with **two env files**:
- root `.env` (shared stack)
- service `.env` (selection + overrides)

Example command:

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-llamacpp-host/.env \
  -f services/orion-llamacpp-host/docker-compose.yml \
  up -d --build
```

### Key compose behaviors

- Mounts the host GGUF cache into the container at `/models`.
- Exposes Chat Lobe on `LLAMACPP_HOST_PORT` (mapped to 8000 internally by default, though config says 8080 often).
- Exposes Embedding Lobe on `EMBEDDING_PORT` (mapped to 8001 internally).
- Wrapper reads `/app/config/llm_profiles.yaml` (baked into the image).

---

## Runtime endpoints

- **Chat Lobe:** `POST /v1/chat/completions` (Port 8000/8080)
- **Embedding Lobe:** `POST /v1/embeddings` (Port 8001)

Typical internal URL (from app-net):

```
Chat:      http://orion-llamacpp-host:8000
Embedding: http://orion-llamacpp-host:8001
```

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

---

## Recommended operating pattern

- Keep `config/llm_profiles.yaml` as the authoritative model registry.
- Switch models by changing **only** `LLM_PROFILE_NAME`.
- Use env overrides only when troubleshooting stability (ctx, gpu layers, parallel, batch).

---

## Quick debug commands

### Verify profile exists inside image

```bash
docker run --rm --entrypoint sh orion-llamacpp-host:0.1.0 -lc \
  "grep -n 'deepseek-70b-gguf-atlas' -n /app/config/llm_profiles.yaml && echo OK"
```

### Watch VRAM while loading

```bash
watch -n 1 nvidia-smi
```

---

## Versioning

- `SERVICE_VERSION` should track wrapper changes.
- Engine build is pinned by the llama.cpp image tag.
