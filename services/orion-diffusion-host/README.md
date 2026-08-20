# Orion Diffusion Host

**Status: skeleton only, no real diffusion model wired yet (Patch 1 of
`docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md`).** This
service currently does nothing but boot a FastAPI process and answer
`/health`. There is no `diffusers`/`torch` dependency, no model loading, and
no bus consumer in this patch — those land in a later patch.

**Node/port/GPU assignment (2026-08-20, per Juniper):** this service is
assigned to Circe, `HOST_PORT=8014`, `CUDA_VISIBLE_DEVICES=2` — the existing
`orion-llamacpp-host` "agent lane" GPU worker slot (`services/orion-llamacpp-host/README.md`'s
chat/metacog/quick/agent 4-slot convention), not the "new, unprovisioned 4th
physical V100" the design doc originally assumed (design doc §3). The
llama.cpp `atlas-agent` worker (`muse-glimmer-30b`) that previously occupied
that port/GPU must be stopped by the operator before this service comes up
on Circe, or the two will fight over the same card — see `.env_example` for
the exact keys and the stop/start sequence below. `docker-compose.yml` now
reserves the GPU (`gpus: all`) so the card is visible ahead of Patch 2's
real model load; no model reads it yet in this patch.

## What this is for

The visual reverie chain (design doc §1/§3) is a second, parallel reverie
chain alongside the existing text chain owned by `orion-thought`: on a slow,
capacity-gated cadence, generate an image about gathered context via a
diffusion model, then feed the image back through `orion-vision-host`'s
captioning path to get a description, and feed *that* description into the
next reverie's context. This service is the "generate an image" half of that
loop — modeled directly on `services/orion-vision-host`'s shape (FastAPI +
GPU-resident model on a raw CUDA base image), since vision-host is the one
existing non-llama.cpp GPU-resident inference service in this repo
(design doc §3, existing-mechanism check).

## Current skeleton status

What exists in this patch:

- `app/main.py` — FastAPI app, `/health` only.
- `app/settings.py` — pydantic-settings config (port, node name, bus flag
  placeholders, model cache dir). No field here is read by anything that
  actually loads a model yet.
- `docker-compose.yml` — container skeleton; reserves the GPU (`gpus: all`)
  now that this service has a real node/port/GPU assignment (see above), but
  no model volume mount doing real work yet (the model cache dir is declared
  and mounted so the *path convention* is settled now, not so anything is
  downloaded there).
- `requirements.txt` — FastAPI + uvicorn only. Deliberately **no**
  `diffusers`/`torch` — that is real GPU weight, not a skeleton dependency,
  and gets added in the patch that wires real generation.

What does **not** exist in this patch (do not assume otherwise from the
service directory existing or from the node/port/GPU assignment above):

- Any diffusion model, any image generation, any actual GPU compute (the
  card is visible to the container now; nothing in this patch's code reads
  or allocates it).
- Any bus consumer/producer wiring (no intake channel, no reply channel — the
  channel names in `orion/bus/channels.yaml` have not been touched by this
  patch and this service is not in that file yet).
- Any chain orchestration (`visual_chain.py` in `orion-thought` — Patch 2).
- Any context-seeding (Patch 3).

## Model cache dir convention: `/mnt/storage-warm/models/diffusion`

`orion-vision-host` caches its (much smaller, ~2.7B-class caption) models
under `/mnt/telemetry/models/vision` — a disk this repo also uses for
telemetry/logs/vision-frame capture, i.e. a mixed-purpose disk that happens
to also hold vision-host's weights. `orion-vllm-host` and `orion-recall`
instead put model weights under `/mnt/storage-warm/...` — a disk whose whole
purpose is model storage (see `orion-recall`'s
`RECALL_TENSOR_RANKER_MODEL_PATH=/mnt/storage-warm/orion/recall/
tensor-ranker.pt` and `orion-vllm-host`'s `/mnt/storage-warm/models:/models`
compose mount).

Diffusion model weights (SDXL-class checkpoints are several GB per component
— UNet, VAE, text encoder(s)) are a large, single-purpose download much
closer in kind to an LLM checkpoint than to vision-host's caption model, so
this service follows the `storage-warm` convention:
`/mnt/storage-warm/models/diffusion`, not vision-host's `/mnt/telemetry`
path. See `.env_example` for the exact keys.

## Bringing this up on Circe

Must be run from a worktree on Circe itself (`scripts/safe_docker_build.sh`
refuses the shared checkout), after syncing `.env` from `.env_example`:

```bash
# 1. Stop the llama.cpp agent-lane worker currently on this port/GPU.
docker compose -f services/orion-llamacpp-host/docker-compose.atlas-workers.yml \
  stop atlas-agent

# 2. Bring up diffusion-host on the now-free port 8014 / GPU index 2.
scripts/safe_docker_build.sh orion-diffusion-host up -d --build

# 3. Verify.
curl -fsS http://localhost:8014/health
```

## Probes

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness only. Reports service name/version and that no model is loaded (this patch never loads one). |

There is no `/ready` in this patch — readiness (VRAM floors, warm-on-start,
etc., following vision-host's `/ready` shape) is meaningful only once a real
model is being loaded, which is a later patch.

## Tests

```bash
cd services/orion-diffusion-host
PYTHONPATH=.:../.. python3 -m pytest tests/ -q
```
