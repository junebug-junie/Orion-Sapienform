# Orion Vision Host

GPU-backed vision inference service (FastAPI + Redis bus). Tasks arrive on `CHANNEL_VISIONHOST_INTAKE` with payload schema **`VisionTaskRequestPayload`** (`orion/schemas/registry.py`, `orion/bus/channels.yaml`). Replies are published to `reply_to` (channel pattern `orion:vision:reply:*`) as envelope kind **`vision.task.result`** with payload **`VisionTaskResultPayload`**, including **`error_code`**, **`timings`**, and optional **`meta`** (e.g. `warnings`) on failures. Optional **`vision.artifact`** broadcast on `CHANNEL_VISIONHOST_PUB`.

Also publishes a bus-native `SystemHealthV1` heartbeat to `orion:system:health` every
`HEARTBEAT_INTERVAL_SEC` (default 10s), independent of `service.bus` above.

## Probes

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | **Liveness:** process up; static scheduler limits and bus flag. |
| `GET /ready` | **Readiness:** HTTP **503** when not ready — profiles loaded, bus connected if enabled, at least one CUDA device passes VRAM hard floor, and **warm-on-start** profiles loaded without error. Body lists `degraded_reasons` and `warm_failed_profiles`. |
| `GET /profiles` | Loaded YAML version, enabled profiles, routing. |

## Operator checklist

1. **Caches:** Set `MODEL_CACHE_DIR`, `HF_HOME`, and `TRANSFORMERS_CACHE` to fast, persistent disk (see `.env_example` / deployment env).
2. **GPU visibility:** `CUDA_VISIBLE_DEVICES` and `VISION_DEVICES` must list indices the container can see. NVML (`nvidia-ml-py`) must work for VRAM-aware scheduling.
3. **Bus:** When `ORION_BUS_ENABLED=true`, Redis must be reachable before `/ready` goes green.
4. **VRAM floors:** Tune `VISION_VRAM_RESERVE_MB`, `VISION_VRAM_SOFT_FLOOR_MB`, `VISION_VRAM_HARD_FLOOR_MB` to match co-hosted workloads.
5. **Concurrency:** `VISION_MAX_INFLIGHT`, `VISION_MAX_INFLIGHT_PER_GPU`, `VISION_QUEUE_WHEN_BUSY`, `VISION_MAX_QUEUE` — queue full returns `error_code=queue_full` on the bus reply and in structured logs.
6. **Timeouts:** `VISION_TIMEOUT_S` wraps the threaded `VisionRunner.execute` path (wall-clock); logs include `scheduler_total_s`, estimated `queue_wait_est_s`, and `inference_s` when available.
7. **Models:** Override `VISION_VLM_MODEL_ID` for your VRAM budget (default is BLIP2 per grounded pipeline plan; use `Salesforce/blip-image-captioning-base` on P100 cohabitation — see `.env_example` comment). Enable only profiles you need via `VISION_ENABLED_PROFILES`.
8. **Caption quality:** VLM captions use a factual prompt and `caption_sanitize` rejects prompt-echo and stoplist garbage before artifacts are stored. Rejected captions append `caption_rejected:{reason}` to task meta warnings.

## Caption sanitizer

`retina_fast` caption path (`_run_caption_frame`):

- Prompt: list visible objects/people only — no activity guesses.
- Post-decode: `sanitize_caption()` rejects `prompt_echo`, `too_short`, and high stoplist ratio (YouTube/google/video slop).
- Rejected captions are cleared; warnings surface in artifact meta.

Env: `VISION_VLM_MODEL_ID`, `VISION_VLM_TEMPERATURE` (default `0.2`).

## Identity hypothesis (`identity_face`)

`docs/superpowers/specs/2026-08-21-seeing-juniper-identity-and-situated-observation-design.md`
section 4. Real face detection + embedding (MTCNN + InceptionResnetV1,
VGGFace2-pretrained, `facenet-pytorch`) matched against **one enrolled
subject** -- not a growing gallery, not a stranger tracker.

**Non-negotiables, enforced in code, not just here:**

- **One enrolled subject. Gallery does not grow at runtime.**
  `app/identity_gallery.py::save_gallery_embedding` is only ever called
  from `scripts/enroll_identity_face.py` (a human-run CLI) --
  `tests/test_identity_gallery_never_grows_at_runtime.py` structurally
  pins that no file under `app/` references it.
- **Non-matches are never stored.** A query embedding is compared in
  memory and discarded; it is never written to disk or returned to the
  caller, matched or not.
- **`unsure` is a real, reachable third state.** `match_embedding` returns
  `{"subject", "similarity", "state"}` with `state` one of
  `probable` / `possible` / `unsure` -- never a binary match/no-match. A
  `subject` other than `"unknown"` is returned only for `probable`/`possible`.

**Enrollment** (must be run by hand, with real photos -- ships with zero
enrolled by default):

```bash
cd services/orion-vision-host
python3 scripts/enroll_identity_face.py --subject juniper photo1.jpg photo2.jpg photo3.jpg
```

Writes one JSON file (mean embedding across all usable photos) to
`IDENTITY_GALLERY_DIR` (default `/mnt/telemetry/orion-vision-host/identity_gallery`,
the existing bind mount -- no new volume). Re-running with the same
`--subject` overwrites the entry (re-enrollment, not accumulation).

**Unenrolled behavior:** `task_type=identity_face` still runs face
detection; every candidate comes back `{"subject": "unknown", "similarity":
null, "state": "unsure", "reason": "not_enrolled"}` and the artifact's
`gallery_enrolled` flag is `false` -- not an error.

**Thresholds** (`config/vision_profiles.yaml`'s `identity_face.params`):
`match_threshold` (below = `unsure`) and `probable_threshold` (at/above =
`probable`; between the two = `possible`). Both are a reasoned starting
point, **not yet live-validated against real enrolled photos** -- AGENTS.md's
metric-quality-gate item 4 (live-data sanity check) genuinely cannot run
until someone enrolls. Revisit once real camera data exists.

**Deviation from the design doc's literal prose, noted honestly:** the doc
describes running face detection on the person crop GroundingDINO already
produces. `_run_identity_face` runs MTCNN directly on the full frame
instead -- `runner.py`'s pipeline steps do not currently pass one step's
artifacts into the next step's request (`_run_pipeline` hands every step
the same original request dict), and building that plumbing is separate,
larger, riskier work than this patch's scope. MTCNN already does its own
face localization on a full image without needing a person crop first, so
this is a real simplification, not a silently dropped requirement.

**Not built here:** wiring the resulting hypothesis into
`orion-vision-council`'s evidence-grounding context as the design doc's
section 4 describes ("passed to the gateway call as grounding context,
exactly like hard labels") -- that is the next integration step, in a
different service, once this capability is live-validated with a real
enrollment. Also not built: `vision_events` retention policy (design doc
section 6.5: "identity-bearing rows are the most sensitive this system
will hold. Ships WITH section 4, not after") -- a real, separate gap this
patch did not close; flagged, not silently skipped.

## Observability (logs-first)

Each finished task emits one structured line prefix `[VISION_TASK]` with JSON containing at least: `correlation_id`, `task_type`, `ok`, `device`, `error`, `error_code`, `queue_depth_at_submit`, `scheduler_total_s`, `inference_s`, `queue_wait_est_s`.

**Dashboard hints:** aggregate `error_code`, p95 `scheduler_total_s` / `inference_s` by `task_type`, and `queue_depth_at_submit` at submit time.

## Config drift: `adaptive_degrade`

The repository root `config/vision_profiles.yaml` may describe `runtime.adaptive_degrade`. **That block is not implemented in `orion-vision-host`** — scheduling uses env-backed floors and `VisionScheduler` only. Do not assume resolution drops or profile disables happen automatically.

## Smoke scripts

- `scripts/publish_test_task.py` — publish a task over the bus.
- `scripts/tap_artifacts.py` — subscribe to artifact channel.

## Tests

```bash
cd services/orion-vision-host
PYTHONPATH=. python3 -m pytest tests/ -q --tb=short
```

Scheduler tests do not require CUDA or model weights.
