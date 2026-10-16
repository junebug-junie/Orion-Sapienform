# Orion Vision Frame Router

Policy bridge between **Retina** (continuous frame capture) and **Vision Host** (on-demand GPU inference). Subscribes to frame pointers, applies YAML baseline/triggered dispatch rules, and dispatches `VisionTaskRequestPayload` tasks while preserving envelope causality. **Triggered tier TTL is refreshed from host task replies** (GroundingDINO person detections), not from edge activity.

This service does **not** capture frames, run inference, or touch GPU models.

## Role in the vision pipeline

```text
  [camera / folder / rtsp]
           │
           ▼
   orion-vision-retina  ──publish──►  orion:vision:frames
           │                              (vision.frame.pointer)
           │                                      │
           │                                      ▼
           │                         orion-vision-frame-router
           │                    (baseline vs triggered policy + sampling)
           │                                      │
           │                                      ▼
           │              orion:exec:request:VisionHostService
           │                              (vision.task.request)
           │                                      │
           │                                      ▼
           │                         orion-vision-host (GPU)
           │                                      │
           │                    host reply (person in artifact)
           │                         refreshes trigger TTL
           │                                      │
           │                                      ▼
           │              orion:vision:reply:<correlation_id>
           │                              (vision.task.result)
           │
           ▼
   /mnt/telemetry/vision/frames/*.jpg   ← shared disk (image_path)
```

| | **orion-vision-retina** | **orion-vision-frame-router** (this) | **orion-vision-host** |
|---|-------------------------|--------------------------------------|------------------------|
| Role | Capture + persist JPEGs | Policy bridge + trigger gating + backpressure | GPU inference |
| Trigger | Internal loop (`RETINA_FPS`) | Bus: frames; host replies refresh trigger TTL | Bus: host intake channel |
| Output | Frame pointers | Host task requests (baseline or triggered tier) | Artifacts + task replies |
| Needs GPU | No | No | Yes |

## Dispatch tiers

Policy file: `config/vision_frame_router.yaml`. Merge order: `defaults` → `streams[stream_id]` → `cameras[camera_id]`.

The router maintains per-`stream_id` trigger TTL from **host task replies** (person labels in GroundingDINO output). Each frame dispatch selects one tier:

| Tier | When | Host request | Purpose |
|------|------|--------------|---------|
| **baseline** | No active trigger labels within TTL | `want_caption: false`, `want_embeddings: false` | Detect-only `retina_fast` without VLM on every frame |
| **triggered** | Host recently detected configured `trigger_labels` (default: `person`) within `trigger_ttl_seconds` | `want_caption: true`, `want_embeddings: true` | Caption + embed when a person was detected on the host pipe |

Task meta includes `dispatch_tier` (`baseline` or `triggered`) for observability, and
`triggered_by` (`labels` or `expectation`) on the triggered tier.

Trigger labels recorded from host replies are the defaults' plus the stream's own
(`policy.trigger_labels_for`), so a stream that adds a label (walkway's `dog`) can
actually trigger on it.

### Expectation steering (Redis key contract)

Walkway spec idea 8 (`docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md`).

| Key | Writer | Reader | Meaning |
|-----|--------|--------|---------|
| `orion:vision:expect:<stream_id>` (bus Redis, `ORION_BUS_URL`) | orion-sql-writer rhythm loop | this router (`app/expectation.py`) | Exists while an expectation window for that camera is open; TTL = the window. Value is informational; only presence is read. |

While the key exists, a stream that has trigger labels gets its **triggered** tier
(`triggered_by=expectation`), so Orion spends GPU where it predicts something will
happen. `decide()` never touches Redis: a background task refreshes the set of open
streams every `ROUTER_EXPECTATION_REFRESH_SEC` (default 5s) for streams the router has
seen. Any Redis error clears the set (reads as "no expectation", never stuck
triggered). Disable with `ROUTER_EXPECTATION_STEERING_ENABLED=false`. `/healthz` shows
`expectation_open_streams` and `expectation_refresh_failures`.

Manual check: `redis-cli -u $ORION_BUS_URL SET orion:vision:expect:walkway 1 EX 60`,
then the next walkway host task's meta carries `triggered_by: expectation`.

### Walkway stream

`streams.walkway` (captured by the `orion-vision-retina-walkway` instance, `stream_id=walkway`, full-resolution main stream): prompts
`person, dog, bicycle, stroller, vehicle, package, mail truck`, `trigger_labels:
[person, dog]`, and `want_crop_embeddings: true` on both tiers (host zones + embeds
tracked boxes, patio included since 2026-10-15) and `want_crop_faces` (a face
vector + face thumbnail on each embedded person crop). No identity dispatch:
Juniper's whole-frame identity check stays on the room cameras.

**Walkway captions come from Qwen2-VL, not BLIP (2026-10-15).** The walkway
primary task asks the athena host for no caption (`want_caption: false`). Its
`caption_dispatch` block makes the router, after a *triggered*-tier detect
reply that found at least one of `labels` at `>= min_score`, send the same
frame as a `caption_frame` task to `CHANNEL_CAPTION_INTAKE`
(`orion:exec:request:VisionHostService:circe-vl`, the Qwen2-VL-2B lane) with
`request.caption_prompt` ("In one short sentence, describe what is happening
in this image."), `request.caption_crop_boxes` + `request.caption_crop_pad`
(the frame's gate boxes; the host captions a padded crop around them, see
`crop_to_detections` / `crop_pad`) and `meta.source_objects` (the same
detections, read by orion-vision-window's caption gate). **2026-10-16:** the
10-15 prompt ("Are there any people, animals or vehicles ...? If there are
none, reply with only: nothing notable.") got "nothing notable" on every
walkway task (24/24 on 10-16, 44/50 on 10-15) because Qwen2-VL-2B answers
"no" to an existence question even on a clear crop of a car, and the whole
2560x1920 frame shrinks to ~1 MP for the model. No null answer is configured
now; `caption_null_answers` still works if one is set. It is
chained after the detect reply, so detection never waits on the ~3 s VLM; it
has its own in-flight cap (`max_inflight`) and per-camera clock
(`min_seconds_between_dispatch`) and never takes an athena host slot
(`RouterState.inflight_total()` counts host-lane tasks only). A frame whose
gate boxes all match (same label, IoU >= `same_scene_iou`) the last captioned
frame's within `same_scene_ttl_sec` is skipped as `same_scene` -- the patio
skeleton reads as `person` all night. Busy, timed out (`timeout_sec`, equal to
the window's `WINDOW_CAPTION_MAX_AGE_SEC`) or down = no caption; nothing falls
back to BLIP. The vision organ counts the lane's `caption_requested` when a
task is sent and `captions` when one comes back, leaving `replies_ok`/`failed`
to the athena host. Outcomes are counted in the health heartbeat:
`caption_dispatched_total`, `caption_captioned_total`, `caption_skip_counts`
(`no_detection`, `rate_limited`, `same_scene`, `lane_inflight_limit`), `caption_no_caption_counts`
(`null_answer`, sanitizer reasons), `caption_failure_counts`. Tests:
`tests/test_walkway_caption_dispatch.py`.

**Walkway tiled detection (2026-10-15).** The stream-level `tiled_detect`
block (`enabled`, `tiers`, `min_seconds_between`) makes
`FrameDispatchPolicy.apply_tiled_detect` set `request.detect_tiles: true` on
at most one primary walkway task per `min_seconds_between` (own per-camera
clock, `CameraState.last_tiled_detect_ts`), on any listed tier. The athena
host then also runs the detector on full-resolution tiles (see
`services/orion-vision-host/README.md` "Tiled detection"). Opt-in per stream:
cam0/carbon never get tiles, and only detect-bearing task types (`retina_fast`,
`retina_dense`, `detect_open_vocab`) are marked. The dispatch log line shows
`tiled=True|False`. Consequence for consumers: a far person only the tiles can
see shows up on tiled frames (one per `min_seconds_between`) and not on the
frames in between, so its trigger TTL (8 s) lapses between sightings and the
individuals reducer gets sparser sightings for it than for a near person.
Tests: `tests/test_walkway_tiled_detect.py`.

Per-stream overrides use the `streams:` block (e.g. `streams.cam0`). Legacy per-camera overrides remain under `cameras:` keyed by `camera_id`.

## Bus channels

| Channel | Direction | Payload |
|---------|-----------|---------|
| `orion:vision:frames` | Subscribe | `VisionFramePointerPayload` (`vision.frame.pointer`) |
| `orion:exec:request:VisionHostService` | Publish | `VisionTaskRequestPayload` (`vision.task.request`) |
| `orion:exec:request:VisionHostService:circe-vl` | Publish | `VisionTaskRequestPayload` `caption_frame` for streams with `caption_dispatch` (walkway) |
| `orion:vision:reply:*` | PSUBSCRIBE | `VisionTaskResultPayload` (`vision.task.result`) |
| `orion:system:health` | Publish | `SystemHealthV1` router metrics |
| `orion:grammar:event` | Publish | `GrammarEventV1` vision organ window (`vision.organ:`), when `VISION_ORGAN_GRAMMAR_ENABLED` |

## Vision organ self-report

`app/grammar_emit.py`. Every `VISION_ORGAN_WINDOW_SEC` (60 s) the router publishes one
grammar trace `vision.organ:<router>:<window_id>` on `orion:grammar:event`: one
`vision_stream_window_observed` atom per stream and a closing
`vision_organ_window_completed` atom, sent even when no frame arrived.

Per stream: frames received, age of the newest frame (`none` = nothing since the
router started), tasks dispatched (primary and identity), ok replies, failures by
class (`timeout`, `invalid_reply`, host `error_code` or `host_error`), policy skip
reasons, and yield (detect replies, objects, captions requested vs produced).
Reported streams = every enabled entry under `streams:` in the policy file, plus
any other stream seen -- so a configured camera that never sends is reported as
absent, never as calm. Counts and ages only: no paths, labels, captions or
identities.

Consumer: orion-substrate-runtime's `vision_organ` reducer
(`orion/substrate/vision_organ_loop/`) -> `node:substrate.vision_organ` ->
`capability:vision`. Off in code, on in `.env_example`.

## Policy file

Default policy: `config/vision_frame_router.yaml` (mounted at `/app/config/vision_frame_router.yaml` in Docker).

Policy defines `defaults.baseline` and `defaults.triggered` tiers (see **Dispatch tiers** above). Per-stream overrides control `streams.cam0` etc.; per-camera overrides remain under `cameras:`. Set `ROUTER_POLICY_PATH` to point at a custom file.

## Configuration

```bash
cd services/orion-vision-frame-router
cp .env_example .env   # edit ORION_BUS_URL, DRY_RUN, etc.
```

Key env vars: `ROUTER_ENABLED`, `DRY_RUN`, `MAX_INFLIGHT_TOTAL`, `TASK_TIMEOUT_SECONDS`, `REQUIRE_IMAGE_PATH_EXISTS`, `VISION_ORGAN_GRAMMAR_ENABLED`, `VISION_ORGAN_WINDOW_SEC`.

**Deployment:** Vision Host should consume **only** `orion:exec:request:VisionHostService` when this router is enabled — do not also wire Host to auto-subscribe `orion:vision:frames`, or GPU work will bypass policy.

**`DRY_RUN`:** Records dispatch metrics and inflight state without publishing to Host. Inflight limits still apply, so sustained dry-run can hit `global_inflight_limit` until replies/timeouts clear (or restart).

**`drop_when_busy`:** Router has no task queue. At `max_inflight_total`, new frames are skipped. `drop_when_busy: true` uses skip reason `global_inflight_limit`; `false` uses `global_inflight_backpressure` (same behavior, different metric label).

## Run locally

```bash
cd services/orion-vision-frame-router
docker compose build
docker compose up -d
docker logs -f orion-dev-vision-frame-router
curl -s localhost:8010/healthz | jq .
```

Build context is the **repo root** (Dockerfile copies `orion/`, `config/vision_frame_router.yaml`, and `app/`).

The frame volume mount (`/mnt/telemetry/vision/frames:ro`) must match Retina output paths so `REQUIRE_IMAGE_PATH_EXISTS` checks succeed inside the container.

## Smoke test (redis-cli)

With Retina publishing frames and the router running on shared `app-net`:

```bash
# Frame intake (from Retina)
redis-cli -u "$ORION_BUS_URL" SUBSCRIBE orion:vision:frames

# Host task dispatch (from router)
redis-cli -u "$ORION_BUS_URL" SUBSCRIBE orion:exec:request:VisionHostService

# Host replies (router clears pending + refreshes trigger TTL on person detect)
redis-cli -u "$ORION_BUS_URL" PSUBSCRIBE 'orion:vision:reply:*'

# Router health telemetry
redis-cli -u "$ORION_BUS_URL" SUBSCRIBE orion:system:health
```

Expect sampled `vision.task.request` envelopes on host intake only when policy allows dispatch (not every frame). Replies arrive on `orion:vision:reply:<correlation_id>` with the same `correlation_id` as the source frame envelope.

Dry-run mode (`DRY_RUN=true`) records dispatch metrics without publishing to host intake.

## Tests

From repo root (worktree):

```bash
PYTHONPATH=.:services/orion-vision-frame-router pytest services/orion-vision-frame-router/tests -q
```
