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
7. **Models:** Override `VISION_VLM_MODEL_ID` per node for your VRAM budget — default (`Salesforce/blip-image-captioning-base`) is sized for a shared/small card (e.g. athena's P4). `model_manager.py`'s `load_vlm_captioner` also supports BLIP2, Qwen2-VL, and Qwen2.5-VL model_ids (selected by substring match — see `.env_example` comment); Qwen2-VL-class models need real headroom (~4-5GB fp16) and route through a chat-template prompt, unlike BLIP's plain image+text call. Enable only profiles you need via `VISION_ENABLED_PROFILES`.
8. **Caption quality:** VLM captions use a factual prompt and `caption_sanitize` rejects prompt-echo and stoplist garbage before artifacts are stored. Rejected captions append `caption_rejected:{reason}` to task meta warnings.

## Circe Qwen2-VL lane (`docker-compose.circe-qwen.yml`)

A second, independent instance of this same service, deployed on **circe's
physical GPU index 4 (Tesla P100-PCIE-16GB)** running `Qwen/Qwen2-VL-2B-Instruct`
instead of the shared athena instance's BLIP-base. Not a replacement for
the athena instance above — that one keeps doing retina/detection/embedding
work on athena's P4. This lane exists for exactly one job: re-observing a
generated image for `orion-thought`'s reverie visual chain
(`services/orion-thought/app/visual_chain.py`), which needs a real,
non-degenerate caption that BLIP-base's quality ceiling cannot reliably
produce (live-evidenced 3/3 ticks on 2026-08-25/26 — every real GPU call
rejected by `sanitize_caption` as too-short/empty) and that athena's P4 has
no VRAM headroom to fix in place (2.4GB free measured live; Qwen2-VL-2B
needs ~4-5GB fp16).

**Why a second instance, not just a bigger model on athena:** checked and
ruled out — see the paragraph above. **Why its own channel, not the shared
one:** two vision-host instances racing the same shared
`orion:exec:request:VisionHostService` channel already caused a real
incident (PR #1859/#1860, 2m13s of dropped presence updates) — every
dedicated instance gets its own isolated channel, no exceptions
(`orion/bus/channels.yaml`'s `orion:exec:request:VisionHostService:*`
convention).

**No shared filesystem with athena** (`/mnt/telemetry` is local ext4, not
NFS-exported) — this lane never reads a frame by path. The caller
(`orion-thought`) uploads the generated image's bytes to
`orion-percept-store` first and hands this lane a `percept_sha256`; this
service's own `runner.py::_load_image_from_percept_store` is what fetches
those bytes back, server-side, to run inference on. That means
`VISION_PERCEPT_STORE_URL` **must be the real tailscale IP**
(`http://100.92.216.81:8021/percepts`), never the docker-internal service
name (`orion-athena-percept-store`) the shared athena instance uses — that
hostname only resolves on athena's own docker network. Live-caught
deploying this lane the first time: `Temporary failure in name resolution`.

**Bring up (from a worktree ON CIRCE, never the shared checkout):**

```bash
# Confirm the GPU is actually free RIGHT NOW -- do not trust an earlier
# session's snapshot, this host's GPU assignments have moved before:
nvidia-smi --query-gpu=index,name,memory.free --format=csv

docker compose \
  --env-file .env \
  --env-file services/orion-vision-host/.env \
  -f services/orion-vision-host/docker-compose.circe-qwen.yml \
  up -d --build

curl -fsS http://localhost:${CIRCE_QWEN_HOST_PORT:-6602}/health
curl -fsS http://localhost:${CIRCE_QWEN_HOST_PORT:-6602}/ready
```

Every `CIRCE_QWEN_*` key has a safe default baked into the compose file
itself, so the second `--env-file` above is optional for a first bring-up
-- but include it anyway (AGENTS.md section 8's dual-env-file convention):
an override made through the normal `sync_local_env_from_example.py`
workflow lands in `services/orion-vision-host/.env`, and a bring-up that
only passes the root `.env` would silently ignore it (review finding).

Env keys: see `.env_example`'s "Circe Qwen2-VL lane" section.

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
- **Never reaches the general artifact broadcast.** `CHANNEL_VISIONHOST_PUB`
  (`orion:vision:artifacts`) is consumed by `orion-security-watcher`,
  `orion-vision-window`, and `orion-vision-council` (`orion/bus/
  channels.yaml`) -- none identity-aware or retention-gated. `app/main.py`'s
  `should_broadcast_artifact()` excludes `identity_face` from both real
  broadcast call sites (found live, 2026-08-26: the bus-first path and the
  `/v1/vision/task` HTTP endpoint each had their own, and the second one
  was still unguarded after the first fix landed --
  `test_every_publish_artifact_broadcast_call_site_is_guarded` pins both
  now). Only the direct RPC/HTTP reply to whoever explicitly requested
  `task_type=identity_face` carries the real result.

**Enrollment** (real photos required -- ships with zero enrolled by
default; `scripts/` is copied into the image, confirmed live 2026-08-26
after an earlier version of this Dockerfile omitted it entirely). The
`/mnt/telemetry/orion-vision-host` host mount is not reliably writable by
an operator's own shell user (confirmed live: `athena`'s own user got
`Permission denied` there) -- `docker cp` into the container's own
filesystem is the tested, working path:

```bash
# Locally, against a dev checkout with facenet-pytorch installed:
cd services/orion-vision-host
python3 scripts/enroll_identity_face.py --subject juniper photo1.jpg photo2.jpg photo3.jpg

# Against the real running container -- docker cp, not the host mount:
docker exec orion-athena-vision-host mkdir -p /tmp/enrollment_photos
docker cp photo1.jpg orion-athena-vision-host:/tmp/enrollment_photos/
docker cp photo2.jpg orion-athena-vision-host:/tmp/enrollment_photos/
docker cp photo3.jpg orion-athena-vision-host:/tmp/enrollment_photos/
docker exec orion-athena-vision-host python3 scripts/enroll_identity_face.py \
  --subject juniper \
  /tmp/enrollment_photos/photo1.jpg \
  /tmp/enrollment_photos/photo2.jpg \
  /tmp/enrollment_photos/photo3.jpg
docker exec orion-athena-vision-host rm -rf /tmp/enrollment_photos  # cleanup
```

Writes one JSON file (mean embedding across all usable photos) to
`IDENTITY_GALLERY_DIR` (default `/mnt/telemetry/orion-vision-host/identity_gallery`,
the existing bind mount -- no new volume). Re-running with the same
`--subject` overwrites the entry (re-enrollment, not accumulation). The
script only ever reads its source images, never copies or retains them
itself -- the `rm -rf` above is on the operator, matching the real
enrollment this repo shipped with (see below).

**Where the real enrollment's photos came from:** no photos of Juniper
were sourced or requested directly. `orion-juniper-affective-state`
already runs a real, already-approved, already-live capture path (Hub's
own "Check now" button) -- `POST /v1/juniper/affect/capture_and_assess`
triggers a real ~8s clip from Juniper's own carbon webcam and returns
`capture.video_sha256`. That video is fetchable from `orion-percept-store`
(`GET {PERCEPT_STORE_BASE_URL}/{sha256}`, hash re-verified on receipt) the
same way `orion-juniper-affective-state`'s own `_fetch_percept` does.
`ffmpeg -i clip.mp4 -vf "select='not(mod(n\,40))'" -vsync vfr frame_%02d.jpg`
extracted 6 evenly-spaced frames, enrolled via `docker cp` + the script
above, then deleted from the container (`rm -rf /tmp/enrollment_photos`).
No new capture infrastructure -- this reuses a capability that already
exists and is already consented to for a different purpose.

**Unenrolled behavior:** `task_type=identity_face` still runs face
detection; every candidate comes back `{"subject": "unknown", "similarity":
null, "state": "unsure", "reason": "not_enrolled"}` and the artifact's
`gallery_enrolled` flag is `false` -- not an error.

**Thresholds** (`config/vision_profiles.yaml`'s `identity_face.params`):
`match_threshold` (below = `unsure`) and `probable_threshold` (at/above =
`probable`; between the two = `possible`). Live-validated 2026-08-26 against
real enrolled photos (a real `orion-juniper-affective-state` capture,
6 frames enrolled, a held-out 7th frame from the same clip queried):
`similarity=0.9585, state="probable"` -- comfortably clears
`probable_threshold` (0.55) with real margin. Negative control (the empty
room's own ceiling-camera frame) correctly returned zero candidates
(`no_face_detected`), not a false positive. Both directions checked, not
just the positive case.

**Deviation from the design doc's literal prose, noted honestly:** the doc
describes running face detection on the person crop GroundingDINO already
produces. `_run_identity_face` runs MTCNN directly on the full frame
instead -- `runner.py`'s pipeline steps do not currently pass one step's
artifacts into the next step's request (`_run_pipeline` hands every step
the same original request dict), and building that plumbing is separate,
larger, riskier work than this patch's scope. MTCNN already does its own
face localization on a full image without needing a person crop first, so
this is a real simplification, not a silently dropped requirement.

**Known, accepted residual exposure:** `orion-vision-frame-router` (and any
other `orion:vision:reply:*` wildcard subscriber) fully Pydantic-
deserializes every `identity_face` reply -- including the real
subject/similarity/state hypothesis -- before checking whether the
correlation id belongs to a request it made, then discards it. Confirmed
live in `dispatcher.py`. Real, but transient: never logged, persisted, or
forwarded downstream. `CHANNEL_VISIONHOST_PUB`'s broadcast is closed (see
the non-negotiable above, content-based not just task-type-keyed); this
reply-channel fan-out is a separate, narrower, still-open surface --
closing it properly needs either a non-wildcarded reply lane for this
task_type or a redesign of frame-router's own subscribe-then-filter
pattern, both out of scope here. Accepted as a known trade-off (Juniper's
explicit call, 2026-08-26) rather than left unmentioned.

**Not built here:** wiring the resulting hypothesis into
`orion-vision-council`'s evidence-grounding context as the design doc's
section 4 describes ("passed to the gateway call as grounding context,
exactly like hard labels") -- a real next integration step, in a
different service. Also not built: a `vision_events` table-level retention
policy (design doc section 6.5: "identity-bearing rows are the most
sensitive this system will hold. Ships WITH section 4, not after"). The
immediate leak path that requirement was guarding against -- identity data
reaching every consumer of the general artifact broadcast unfiltered -- is
closed; what remains open is retention for identity data that reaches
`vision_events` through some *future* real integration (e.g. the
council-grounding wiring above), which does not exist yet either. Flagged,
not silently skipped.

### VLM model families

`app/vlm_family.py` is the single source of truth both `model_manager.py`
(which transformers class to instantiate) and `runner.py` (how to build the
prompt / decode the reply) read for "which family does this model_id
belong to":

| Family | `model_id` match | Prompt / decode |
|--------|------------------|------------------|
| BLIP2 | contains `blip2` | `processor(images=, text=)`, full-sequence decode |
| BLIP | contains `blip` (not `blip2`) | same as above |
| Qwen2-VL | contains `qwen2-vl`/`qwen2_vl` | `apply_chat_template` + image, decode sliced by input token length |
| Qwen2.5-VL | contains `qwen2.5-vl`/`qwen2_5_vl` | same chat-template path as Qwen2-VL |
| anything else | — | generic `AutoModelForVision2Seq`, BLIP-style call shape |

The chat-template path exists because a chat-tuned VLM echoes its whole
templated prompt back through `generate()` — decoding the full sequence (as
BLIP correctly can) would hand the caller the prompt glued onto the real
answer. `_generate_vlm_text` slices by real input token length instead of
string-matching a prefix off the decoded text, which cannot reliably strip
chat special tokens.

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
