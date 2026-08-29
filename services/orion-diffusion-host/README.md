# Orion Diffusion Host

**Status: real model wired -- currently `YuCollection/FLUX.1-schnell-Diffusers`
(swapped from `stabilityai/sdxl-turbo` 2026-08-28, see "## Model" below for
why).** Patch 1 of
`docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md` shipped a
FastAPI process that only answered `/health`. A follow-up patch first loaded
a real model at startup and exposed `POST /generate`, returning raw PNG
bytes; this file's own git history has both the original sdxl-turbo wiring
and the later FLUX swap. Still no bus consumer/producer and no chain
orchestration — those are Patch 2 (`visual_chain.py` in `orion-thought`) and
Patch 3 (context-seeding).

**Node/port/GPU assignment (2026-08-20, per Juniper):** this service is
assigned to Circe, `HOST_PORT=8014`, `CUDA_VISIBLE_DEVICES=2` — the existing
`orion-llamacpp-host` "agent lane" GPU worker slot
(`services/orion-llamacpp-host/README.md`'s chat/metacog/quick/agent 4-slot
convention), not the "new, unprovisioned 4th physical V100" the design doc
originally assumed (design doc §3). `docker-compose.yml` reserves the GPU
(`gpus: all`); `CUDA_VISIBLE_DEVICES` scopes the container to physical
device 2.

**GPU 2 has had two collisions since this assignment. Both were checked and
cleared live on Circe on 2026-08-25 via direct `docker ps`/`nvidia-smi`
inspection over SSH during this patch's own deploy session — that is a
point-in-time confirmation, not a standing guarantee. Re-run the check below
before assuming the card is still free:**

1. The llama.cpp `atlas-agent` worker (`muse-glimmer-30b`) that originally
   occupied this port/GPU. Confirmed absent from `docker ps` / `nvidia-smi`
   on Circe on 2026-08-25.
2. `services/orion-affectgpt-worker`'s compose pinned the *same* physical
   GPU 2026-08-22 — two days after this assignment — calling it "Juniper's
   own designated slot for prototyping an affective model." Confirmed
   running on GPU 2 (18.4GB, `nvidia-smi --query-compute-apps`) on
   2026-08-25 and stopped via `docker compose stop affectgpt-worker`,
   confirmed freed immediately after (0 MiB). **That service's
   `docker-compose.yml` comment still claims GPU 2 as its slot — this is
   now a real contradiction between the two files, not fixed by this
   patch.** Whoever revisits `orion-affectgpt-worker` needs to either give
   it a different physical card or update its comment to say GPU 2 is
   diffusion-host's now.

Before bringing this service up anywhere, confirm nothing is pinned to
physical GPU 2: `nvidia-smi --query-compute-apps=pid,used_memory,process_name
--format=csv` should show nothing, and `docker ps` should show no container
whose compose reserves `device_ids: ["2"]`.

**A third, more insidious bug found on this exact deploy: `/health`
reporting success does not prove the model landed on the intended physical
card.** `CUDA_VISIBLE_DEVICES=2` was set correctly, but torch's default CUDA
device enumeration is "fastest first," not `nvidia-smi`'s PCI-bus-id order
-- on this host they disagree. The first real deploy of this patch loaded
`sdxl-turbo` onto physical GPU **3** (a busy V100 already serving a
llama.cpp worker) while `CUDA_VISIBLE_DEVICES=2` was intended to mean the
empty PG500-216 at physical index 2, and `/health`/`/ready` both reported
clean success throughout -- nothing about the HTTP contract exposed the
mismatch. Same root cause, same day, same host as
`services/orion-world-model`'s identical bug (see that service's README
"Operator checklist" item 1). Fixed here by setting
`CUDA_DEVICE_ORDER=PCI_BUS_ID` in the Dockerfile, which forces torch's
enumeration to match `nvidia-smi`'s.

**Verify index parity after every deploy — don't trust `/health` alone:**

```bash
docker exec orion-circe-diffusion-host python3 -c \
  "import torch; print(torch.cuda.get_device_name(0))"
# must print "Tesla PG500-216" -- if it prints a V100/P100/other name,
# the model landed on the wrong physical card even though /health is green.
```

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

## Model: `YuCollection/FLUX.1-schnell-Diffusers` (was `stabilityai/sdxl-turbo`)

**Swapped 2026-08-28** — real root-cause fix, not a preference change.
sdxl-turbo's CLIP text encoder truncates at 77 tokens, silently (see
`app/main.py`'s `_log_prompt_token_budget`) — confirmed live that a real
reverie-visual-chain prompt hit 191 tokens, and everything past 77 never
reached the model. FLUX.1-schnell carries a T5-XXL second text encoder
with a real ~256-token budget (`DIFFUSION_MAX_SEQUENCE_LENGTH`) for actual
cross-attention content, not CLIP's 77 — CLIP-L is still loaded here but
only contributes one pooled embedding in this architecture, so its
77-token limit is far less consequential than it was for SDXL.

`DIFFUSION_MODEL_ID` points at `YuCollection/FLUX.1-schnell-Diffusers`,
**not** the official `black-forest-labs/FLUX.1-schnell` — the official
repo is gated (requires accepting a license via the HF web UI before any
token can download it). Confirmed live: neither `HF_TOKEN` already
configured elsewhere in this repo (`orion-vllm`, `orion-llama-cola-host`)
has accepted that gate — both return a real `403` on the actual weight
files (the model-metadata API endpoint alone returns a misleading `200`
even when gated access has not been granted; don't trust that endpoint
alone). The mirror is a verified-ungated (`gated: false`), full
diffusers-format re-upload of the identical Apache-2.0-licensed weights
(same `model_index.json`: `FluxPipeline`/`CLIPTextModel`+
`T5EncoderModel`/`FluxTransformer2DModel`/`AutoencoderKL`) — downloadable
with no token at all, and Apache 2.0 explicitly permits this kind of
redistribution.

Picked because this card serves a slow, capacity-gated cadence (design
doc §4) with no batching or queueing to absorb a slow generation.
`DIFFUSION_NUM_INFERENCE_STEPS=4` (was `1`) is schnell's documented sweet
spot, not sdxl-turbo's. `DIFFUSION_GUIDANCE_SCALE=0.0` is unchanged --
schnell is also guidance-distilled, same operating point.
`DIFFUSION_DEFAULT_WIDTH`/`HEIGHT=1024` (was `512`) — FLUX was trained
primarily at 1024x1024; 512 is off-distribution and produces worse output
for no VRAM saving worth the quality loss.

**`DIFFUSION_DTYPE=fp16`, not bf16 — a real correction, caught before
deploy.** FLUX's own docs recommend bf16 (avoids a known fp16 overflow
risk), but that assumes Ampere-or-later hardware. This service's actual
card, physical GPU 2 ("Tesla PG500-216" per "Node/port/GPU assignment"
above), is **Volta architecture with first-generation Tensor Cores** —
bf16 tensor-core acceleration is an Ampere+ (compute capability ≥ 8.0)
feature this card does not have; other PyTorch-based projects on this
exact GPU class report hard failures ("Bfloat16 is only supported on
GPUs with compute capability of at least 8.0"). Volta's tensor cores were
built for fp16, which is why sdxl-turbo (a different model, same GPU)
already ran fp16 correctly. `_DTYPE_MAP` in `app/main.py` still supports
`bf16` as a configurable option (`_load_pipeline` reads whichever dtype
`DIFFUSION_DTYPE` names) for a future deployment on newer hardware — fp16
is this specific deployment's correct choice given its actual card, not
a hardcoded assumption.

**VRAM**: FLUX.1-schnell fully GPU-resident at 2 bytes/param (fp16 or
bf16, same cost) needs up to ~33GB (12B-param transformer + 4.5B-param
T5-XXL encoder) — over budget on this card's 32GB regardless of which of
the two dtypes is used. `DIFFUSION_ENABLE_MODEL_CPU_OFFLOAD=true` (new)
calls `pipe.enable_model_cpu_offload()`, keeping components on CPU until
their turn in the forward pass — cuts peak GPU residency to ~24GB, same
real weights and math, not reduced precision, just staged residency, at
some generation-latency cost this service's slow cadence already
tolerates. sdxl-turbo never needed this (comfortably fits fully resident)
— the flag defaults `True` here because this deployment's default model
needs it; a future sdxl-turbo-shaped model could set it `False` again.

`_run_generation`/`_pipe_accepts` (`app/main.py`) build the actual
`/generate` call kwargs by inspecting the loaded pipeline's real `__call__`
signature rather than hardcoding "if Flux" branches -- `FluxPipeline` has
neither `negative_prompt` (schnell has no true classifier-free-guidance
path) nor accepts one silently; the old code passed it unconditionally,
which would have raised `TypeError` on every single request against this
model, a full outage caught in review before deploy.

## Current status

What exists:

- `app/main.py` — FastAPI app. Model load runs as a background task kicked
  off from `lifespan()` and is never awaited before startup completes, so
  `/health`/`/ready` are live immediately, not just "eventually" (see the
  module docstring's "Startup" section — an earlier draft of this patch got
  this wrong: awaiting the load before `yield` blocked *every* route,
  caught in review). Load retries up to 3 times with backoff before giving
  up permanently. `GET /health` (liveness, always 200), `GET /ready` (200
  once the model is loaded, 503 otherwise, stays 503 forever after a
  permanent load failure — restart the container to retry), `POST
  /generate` (see below). Also wires the standard `HeartbeatOnly` bus
  chassis (off by default, `ORION_BUS_ENABLED=false`, matching Patch 1's
  decision to defer all bus wiring — flip the env flag to get
  `SystemHealthV1` liveness on `orion:system:health` with no further code
  changes).
- `app/settings.py` — pydantic-settings config: port, node name, model
  cache dir, and the `DIFFUSION_*` model/generation block.
- `docker-compose.yml` — reserves the GPU (`gpus: all`), mounts the real
  model cache dir, passes through the full `DIFFUSION_*` env block with
  `:-default` fallbacks matching `settings.py` (a present-but-empty env var
  otherwise crashes pydantic-settings' int/float validation at import
  time, not falls back).
- `requirements.txt` / `Dockerfile` — `diffusers`/`transformers`/
  `accelerate`/`pillow` in requirements; CUDA torch wheels installed
  explicitly in the Dockerfile (same pattern as
  `services/orion-vision-host/Dockerfile` — a plain `pip install torch`
  from requirements.txt pulls a CPU-only wheel).

What does **not** exist yet:

- Any bus consumer/producer wiring beyond the heartbeat above (no intake
  channel, no reply channel — the channel names in
  `orion/bus/channels.yaml` have not been touched and this service is not
  in that file yet). A caller reaches this service over plain HTTP.
- Any chain orchestration (`visual_chain.py` in `orion-thought` — Patch 2).
- Any context-seeding (Patch 3).
- Concurrency beyond a dedicated single-worker executor: this service
  assumes it owns its GPU exclusively; model load and every generation call
  run on one dedicated thread, and a second `/generate` call while one is
  already running gets an immediate `429`, not a queued wait
  (`app/main.py`'s module docstring explains why there is no
  cancel-on-timeout instead).

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
`DIFFUSION_*` settings above. `width`/`height` are bounded `(0, 1536]`,
`num_inference_steps` `(0, 50]`, `guidance_scale` `[0, 20]` — out-of-range
values get a clean `422`, not an opaque failure from inside diffusers.
`negative_prompt` is only effective when the *effective* `guidance_scale`
is above `1.0`; diffusers does not apply classifier-free guidance (and
therefore ignores the negative prompt) at or below that, which is exactly
this model's own default (`0.0`) — passing a `negative_prompt` at the
default guidance is accepted but silently has no effect (logged as a
warning server-side, not surfaced in the response).

Response is raw `image/png` bytes on success (200), `503` if the model
hasn't finished loading, `422` on a validation failure, `429` if another
generation is already in flight, `500` on a generation failure (OOM, etc.
— the service stays up, the request fails; the response detail is a
generic message plus the exception class name only, never the raw
exception text, which can embed local paths or driver internals).

A caller should pass the response body straight into
`orion.reverie.visual_storage.store_visual_artifact` — that function sniffs
the mime from magic bytes and does not trust a declared content type, so
this endpoint's job is just to return real image bytes, not to assert what
they are.

**Real token-budget visibility (2026-08-27, live incident):** `DIFFUSION_
MAX_PROMPT_CHARS` (default 2000) only bounds total character count -- it
says nothing about SDXL-turbo's actual CLIP text encoders, which truncate
at 77 tokens each, completely silently (no exception, no response header).
Confirmed live: a caller's real prompt hit 191 tokens; the model only ever
saw the first 77. `_log_prompt_token_budget` (called from `_run_generation`
before every real generation) tokenizes the prompt with the ACTUAL loaded
pipeline's own tokenizer(s) (`tokenizer`/`tokenizer_2` -- SDXL carries two
encoders, both checked) and logs a WARNING with real numbers whenever
either budget is exceeded. Visibility only -- never changes what gets
generated, and never raises (a tokenizer-check failure degrades to a DEBUG
log, not a blocked request). Zero new dependency: `transformers` is
already a hard requirement of this service.

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
| `GET /ready` | Readiness. 200 once the model is loaded, 503 otherwise (permanently, after a load failure exhausts its retries — restart to retry again). |
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

The FLUX swap added `SdxlLikeFakePipe`/`FluxLikeFakePipe` -- fakes with
EXPLICIT `__call__` signatures (unlike the plain `FakePipe`'s `**kwargs`
catch-all, which `inspect.signature` reports as having no named
parameters at all, so it can't exercise `_pipe_accepts`'s real logic) --
proving `_run_generation` builds the correct kwargs for each pipeline
shape: `negative_prompt` reaches an SDXL-shaped pipe but is dropped (with
a visible warning, not silently) against a Flux-shaped one that doesn't
accept it, and `max_sequence_length` reaches the Flux-shaped pipe using
`settings.DIFFUSION_MAX_SEQUENCE_LENGTH`. `_log_prompt_token_budget` has
a dedicated test proving `tokenizer_2`'s check uses the passed-in
`max_sequence_length`, not the tokenizer's own raw `model_max_length`
(a T5-style tokenizer often reports an effectively-unbounded placeholder
there, not the pipeline's real effective limit).
