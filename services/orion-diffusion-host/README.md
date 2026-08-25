# Orion Diffusion Host

**Status: real model wired.** Patch 1 of
`docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md` shipped a
FastAPI process that only answered `/health`. This follow-up patch loads
`stabilityai/sdxl-turbo` at startup and exposes `POST /generate`, returning
raw PNG bytes. Still no bus consumer/producer and no chain orchestration —
those are Patch 2 (`visual_chain.py` in `orion-thought`) and Patch 3
(context-seeding).

**Node/port/GPU assignment (2026-08-20, per Juniper):** this service is
assigned to Circe, `HOST_PORT=8014`, `CUDA_VISIBLE_DEVICES=2` — the existing
`orion-llamacpp-host` "agent lane" GPU worker slot
(`services/orion-llamacpp-host/README.md`'s chat/metacog/quick/agent 4-slot
convention), not the "new, unprovisioned 4th physical V100" the design doc
originally assumed (design doc §3). `docker-compose.yml` reserves the GPU
(`gpus: all`); `CUDA_VISIBLE_DEVICES` scopes the container to physical
device 2.

**GPU 2 has had two collisions since this assignment, both resolved live on
Circe as of 2026-08-25 — check for a third before assuming the card is
free:**

1. The llama.cpp `atlas-agent` worker (`muse-glimmer-30b`) that originally
   occupied this port/GPU. Already stopped as of 2026-08-25 (confirmed via
   `docker ps` / `nvidia-smi` on Circe — not present).
2. `services/orion-affectgpt-worker`'s compose pinned the *same* physical
   GPU 2026-08-22 — two days after this assignment — calling it "Juniper's
   own designated slot for prototyping an affective model." Evicted live on
   Circe 2026-08-25 (`docker compose stop affectgpt-worker`). **That
   service's `docker-compose.yml` comment still claims GPU 2 as its slot —
   this is now a real contradiction between the two files, not fixed by
   this patch.** Whoever revisits `orion-affectgpt-worker` needs to either
   give it a different physical card or update its comment to say GPU 2 is
   diffusion-host's now.

Before bringing this service up anywhere, confirm nothing is pinned to
physical GPU 2: `nvidia-smi --query-compute-apps=pid,used_memory,process_name
--format=csv` should show nothing, and `docker ps` should show no container
whose compose reserves `device_ids: ["2"]`.

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

## Model: `stabilityai/sdxl-turbo`

A distilled, single-step SDXL variant, not full SDXL (25-50 steps). Picked
because this card serves a slow, capacity-gated cadence (design doc §4) with
no batching or queueing to absorb a slow generation — a few seconds per
image is fine, tens of seconds is not. `DIFFUSION_NUM_INFERENCE_STEPS=1` and
`DIFFUSION_GUIDANCE_SCALE=0.0` are this model's documented operating point,
not arbitrary tuning — a non-turbo checkpoint dropped in via
`DIFFUSION_MODEL_ID` without also revisiting those two values will produce
degraded output, not just slower output. Trained at 512x512
(`DIFFUSION_DEFAULT_WIDTH`/`HEIGHT`).

## Current status

What exists:

- `app/main.py` — FastAPI app. Loads the model once at startup (in a worker
  thread, so `/health` stays responsive during a cold weight download —
  see `app/main.py`'s `_load_pipeline` docstring). `GET /health` (liveness,
  always 200), `GET /ready` (200 once the model is loaded, 503 otherwise),
  `POST /generate` (see below).
- `app/settings.py` — pydantic-settings config: port, node name, model
  cache dir, and the `DIFFUSION_*` model/generation block.
- `docker-compose.yml` — reserves the GPU (`gpus: all`), mounts the real
  model cache dir, passes through the full `DIFFUSION_*` env block.
- `requirements.txt` / `Dockerfile` — `diffusers`/`transformers`/
  `accelerate`/`pillow` in requirements; CUDA torch wheels installed
  explicitly in the Dockerfile (same pattern as
  `services/orion-vision-host/Dockerfile` — a plain `pip install torch`
  from requirements.txt pulls a CPU-only wheel).

What does **not** exist yet:

- Any bus consumer/producer wiring (no intake channel, no reply channel —
  the channel names in `orion/bus/channels.yaml` have not been touched and
  this service is not in that file yet). A caller reaches this service over
  plain HTTP.
- Any chain orchestration (`visual_chain.py` in `orion-thought` — Patch 2).
- Any context-seeding (Patch 3).
- Concurrency beyond a single in-process lock: this service assumes it owns
  its GPU exclusively and serializes generation requests one at a time
  (`app/main.py`'s module docstring explains why there is no
  cancel-on-timeout).

## `POST /generate`

```json
{
  "prompt": "a calm orion, soft light",
  "negative_prompt": null,
  "width": 512,
  "height": 512,
  "num_inference_steps": 1,
  "guidance_scale": 0.0,
  "seed": null
}
```

Only `prompt` is required; every other field defaults to the
`DIFFUSION_*` settings above. Response is raw `image/png` bytes on success
(200), `503` if the model hasn't finished loading, `422` if the prompt
exceeds `DIFFUSION_MAX_PROMPT_CHARS`, `500` on a generation failure (OOM,
etc. — the service stays up, the request fails).

A caller should pass the response body straight into
`orion.reverie.visual_storage.store_visual_artifact` — that function sniffs
the mime from magic bytes and does not trust a declared content type, so
this endpoint's job is just to return real image bytes, not to assert what
they are.

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
refuses the shared checkout), after syncing `.env` from `.env_example` and
confirming physical GPU 2 is free (see the collision list above):

```bash
scripts/safe_docker_build.sh orion-diffusion-host up -d --build

# Model load happens in the background after the container starts --
# /health returns immediately, model_loaded flips to true once the
# download/load finishes (several GB on a cold cache).
curl -fsS http://localhost:8014/health
curl -fsS http://localhost:8014/ready

curl -fsS -X POST http://localhost:8014/generate \
  -H 'content-type: application/json' \
  -d '{"prompt": "a calm orion, soft light"}' \
  -o /tmp/orion-diffusion-smoke.png
file /tmp/orion-diffusion-smoke.png   # should say PNG image data
```

## Probes

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness. Always 200 if the process is up; reports whether the model has finished loading. |
| `GET /ready` | Readiness. 200 once the model is loaded, 503 otherwise. |
| `POST /generate` | Generate one image. See above. |

## Tests

```bash
cd services/orion-diffusion-host
PYTHONPATH=.:../.. python3 -m pytest tests/ -q
```

`tests/test_generate.py` injects a fake pipe (no GPU, no torch, no network)
and asserts the response is real, sniffable PNG bytes matching
`orion.reverie.visual_storage`'s contract — the real diffusion model itself
is only exercised by the live curl smoke above, not by the test suite.
