# Orion Llama-CoLA Host

This service hosts the Llama-3.1-CoLA model with a dual-path response:
text completions plus latent action indices extracted from the model's
intention head. The service mirrors the Orion neural host plumbing and
exposes the same HTTP endpoints used by the rest of the Orion stack.

## Quick Start

1. Configure a profile in `config/llm_profiles.yaml` (the repo already ships
   a `llama3-1-cola` profile pointing at the official Hugging Face repo).
2. Set your environment variables (see `.env_example` for a template).
3. Start the service:

```bash
docker compose -f services/orion-llama-cola-host/docker-compose.yml up
```

## Configuration

### Required

* `LLM_PROFILE_NAME` — profile key from `config/llm_profiles.yaml`
* `LLM_PROFILES_CONFIG_PATH` — defaults to `/app/config/llm_profiles.yaml`

### Optional (common)

* `HF_TOKEN`/`hf_token` — only required for gated or private Hugging Face repos
* `LLAMA_COLA_MODEL_PATH_OVERRIDE` — override the model path with a local directory
* `LLAMA_COLA_REVISION_OVERRIDE` — override the revision/tag for snapshot downloads
* `ENSURE_MODEL_DOWNLOAD` — set `false` to skip `snapshot_download`
* `WAIT_FOR_MODEL_SECONDS` — delay model loading on boot

### Profile fields

The `llama3-1-cola` profile references the Hugging Face repository directly:

```yaml
  llama3-1-cola:
    backend: llama-cola
    model_id: "LAMDA-RL/Llama-3.1-CoLA-10B"
    llama_cola:
      repo_id: "LAMDA-RL/Llama-3.1-CoLA-10B"
```

When `repo_id` is set, the host downloads the model into:

```
/models/cola/LAMDA-RL--Llama-3.1-CoLA-10B
```

You can instead provide a local directory via `model_path` (in the profile)
or `LLAMA_COLA_MODEL_PATH_OVERRIDE`.

## Endpoints

### `POST /v1/chat/completions`

Returns an OpenAI-style chat completion plus `action_indices` (latent actions):

```json
{
  "id": "cola-...",
  "object": "chat.completion",
  "model": "llama3-1-cola",
  "choices": [
    { "index": 0, "message": { "role": "assistant", "content": "..." }, "finish_reason": "stop" }
  ],
  "action_indices": [[1, 4, 9, ...]],
  "text_output": ["..."]
}
```

### `POST /v1/embeddings`

Generates embeddings by running a short generation and extracting action indices.

### `GET /health`

Returns `{ "status": "ok" }` once the model is loaded.


# Troubleshooting CoLA Model Loading

## Issue Description
When attempting to load the `LAMDA-RL/Llama-3.1-CoLA-10B` model, the inference service may fail with one of the following errors:

**Error 1:**
```
FileNotFoundError: No such file or directory: .../model-00001-of-00010.safetensors
```
**Cause:** Loader is expecting an older version of the model split into **10 shards**, but the actual weights on disk are a newer version split into **184 shards**.

**Error 2:**
```
Initialization Warning: Some weights of IntentionForCausalLM were not initialized ... newly initialized: ['lm_head.weight', ...]
```
**Cause:** A broken or empty index file causes the model to "load" using random initialization — resulting in unusable model output.

## Root Cause
The Hugging Face repository `LAMDA-RL/Llama-3.1-CoLA-10B` is in an inconsistent state:
- **Weights:** Updated and split into **184 `*.safetensors` files**
- **Index:** `model.safetensors.index.json` still points to a 10‑file layout

Since the index acts as the loader's map, the mismatch prevents proper weight resolution.

---

## The Fix — Regenerate the Index File
You must rebuild the index locally to match the actual `.safetensors` shards.

### Step 1 — Locate the Model Directory
Ensure the directory contains the 184 shards. Example:
```
/mnt/telemetry/llm-cache/cola/LAMDA-RL--Llama-3.1-CoLA-10B
```

### Step 2 — Create the Repair Script
Save as `force_fix.py` inside that directory:

```python
import os, json
from safetensors import safe_open

print("Starting Index Generation...")
files = sorted([f for f in os.listdir(".") if f.endswith(".safetensors") and "model-" in f])
print(f"Found {len(files)} shards.")

if len(files) == 0:
    print("ERROR: No .safetensors found! Check your mount.")
    exit(1)

index_map = {}
total_size = 0

for filename in files:
    try:
        with safe_open(filename, framework="pt", device="cpu") as f:
            total_size += os.path.getsize(filename)
            for key in f.keys():
                index_map[key] = filename
    except Exception as e:
        print(f"FAILED to read {filename}: {e}")

output = {
    "metadata": {"total_size": total_size},
    "weight_map": index_map
}

if os.path.exists("model.safetensors.index.json"):
    os.rename("model.safetensors.index.json", "model.safetensors.index.json.bak")

with open("model.safetensors.index.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"SUCCESS: Index file rebuilt with {len(index_map)} keys.")
```

---

### Step 3 — Run the Script via Docker
Running inside the inference container ensures dependencies exist.

```
cd /mnt/telemetry/llm-cache/cola/LAMDA-RL--Llama-3.1-CoLA-10B

docker run --rm \
  -v "$(pwd)":/work \
  -w /work \
  llama-cola-host:0.1.0 \
  python3 force_fix.py
```

---

### Step 4 — Verify + Restart
Expected output:
```
Found 184 shards.
SUCCESS: Index file rebuilt with 410 keys.
```
Then:
```
docker restart orion-athena-orion-llama-cola-host
```
Expected logs:
```
INFO: llama-cola-host: Model loaded successfully.
INFO: Uvicorn running on http://0.0.0.0:8005
```

---

## Result
The model now loads using the correct shard map and inference returns meaningful outputs instead of random noise.

