# Orion Vision Retina

Canonical **visual intake** for Orion: sample frames from a camera or folder, persist JPEGs to shared storage, publish `VisionFramePointerPayload` on the Redis bus, and emit `SystemHealthV1` telemetry.

Retina is the **eye** — it does not run YOLO, face detection, motion, captions, or GPU inference. Those live downstream (`orion-vision-host`, `orion-vision-window`, etc.).

## Relationship to `orion-vision-host`

**Retina does not run inside or on top of vision-host.** They are separate services.

| | **orion-vision-retina** (this service) | **orion-vision-host** |
|---|----------------------------------------|------------------------|
| Role | Continuous **capture** + frame evidence | On-demand **GPU inference** (embed, detect, caption) |
| Trigger | Internal loop (`RETINA_FPS`) | Bus RPC: `VisionTaskRequestPayload` on `CHANNEL_VISIONHOST_INTAKE` |
| Output | `orion:vision:frames` (`vision.frame.pointer`) | `orion:vision:artifacts`, task replies |
| Needs GPU | No | Yes |
| “retina” naming | Service name `vision-retina` | Task profile `retina_fast` / `pipeline_retina_fast` (inference pipeline, not this container) |

```text
  [camera / folder / rtsp]
           │
           ▼
   orion-vision-retina  ──publish──►  orion:vision:frames
           │                              (VisionFramePointerPayload)
           │                                      │
           │                                      ├──► orion-vision-edge (detector worker; legacy)
           │                                      └──► (future bridges / window / host subscribers)
           │
           ▼
   /mnt/telemetry/vision/frames/*.jpg   ← shared disk path in image_path

   Separate path (on-demand GPU):
   Client / cortex ──► orion-vision-host  (task_type=retina_fast, image_path=...)
                              └──► artifacts, captions, detections
```

**Integration today:** Retina writes `image_path` on disk and publishes pointers. **Vision-host** runs when something sends it a **task** with an `image_path` (see `services/orion-vision-host/scripts/publish_test_task.py`). Nothing in host currently auto-subscribes to `orion:vision:frames`; wiring “every new frame → host task” would be a follow-up bridge. The bus catalog lists host as a consumer of `orion:vision:frames` for contract alignment.

**Shared dependencies:** Redis (`ORION_BUS_URL`), `orion` Python package, and typically the same frame directory layout as edge (`/mnt/telemetry/vision/frames`).

## What this service does

```text
source (folder | mock | rtsp | webcam)
  → sample at RETINA_FPS
  → save JPEG → FRAME_STORAGE_DIR
  → VisionFramePointerPayload
  → BaseEnvelope (kind=vision.frame.pointer)
  → CHANNEL_RETINA_PUB (default orion:vision:frames)
  → periodic orion:system:health
```

**Non-goals:** detectors, substrate emitters, SQL/RDF/vector writes, vision council/scribe.

## Configuration

Copy env and edit for your mesh:

```bash
cp .env_example .env
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `ORION_BUS_URL` | `redis://100.92.216.81:6379/0` | Redis pub/sub — shared mesh bus (same as sibling vision services) |
| `RETINA_SOURCE_TYPE` | `folder` | `folder`, `mock`, `rtsp`, `webcam` |
| `RETINA_SOURCE` | `/mnt/telemetry/vision/intake` | Folder path, RTSP URL, or webcam index |
| `RETINA_SOURCE_PATH` | — | Legacy alias for `RETINA_SOURCE` |
| `RETINA_CAMERA_ID` | `retina-cam-01` | Pointer metadata |
| `RETINA_STREAM_ID` | `retina-stream-01` | Pointer metadata |
| `RETINA_FPS` | `1.0` | Capture rate |
| `FRAME_STORAGE_DIR` | `/mnt/telemetry/vision/frames` | JPEG output |
| `FRAME_RETENTION_SECONDS` | `300` | Cleanup age |
| `CHANNEL_RETINA_PUB` | `orion:vision:frames` | Publish channel |

Settings live in `app/settings.py` (Pydantic). Schemas: `orion/schemas/vision.py` (`VisionFramePointerPayload`) — already in `orion/schemas/registry.py`; this service does not register new schema types.

## Run locally

```bash
cd services/orion-vision-retina
cp .env_example .env   # set ORION_BUS_URL to your Redis
docker compose build
docker compose up -d
docker logs -f orion-vision-retina
```

Build context is the **repo root** (Dockerfile copies `orion/` + `services/orion-vision-retina/app`).

### Folder smoke test

```bash
mkdir -p /mnt/telemetry/vision/intake /mnt/telemetry/vision/frames
cp /path/to/test.jpg /mnt/telemetry/vision/intake/

export RETINA_SOURCE_TYPE=folder
export RETINA_SOURCE=/mnt/telemetry/vision/intake
export RETINA_FPS=1
```

Tap the bus:

```bash
redis-cli -u "$ORION_BUS_URL" SUBSCRIBE orion:vision:frames
```

Expect envelopes with `kind=vision.frame.pointer` and payload fields `image_path`, `camera_id`, `stream_id`, `width`, `height`, `frame_ts`.

### Optional: drive vision-host from a saved frame

After retina has written a JPEG:

```bash
cd services/orion-vision-host
python scripts/publish_test_task.py --image /mnt/telemetry/vision/frames/<file>.jpg --task retina_fast
```

That exercises **host inference**, not retina itself.

## On-demand video+audio clip capture (AffectGPT)

`POST /capture/clip` (new 2026-08-22, `RETINA_CLIP_ENABLED=false` by
default). Records `RETINA_CLIP_DURATION_SEC` of video (v4l2) and audio
(pulse) **concurrently** via ffmpeg subprocess, uploads both to
`orion-percept-store`, returns their sha256 refs. Nothing is written to
carbon's disk beyond a `TemporaryDirectory` that's always cleaned up --
same privacy discipline as `upload_frame`'s percept-store path. Gated by
`RETINA_CLIP_TOKEN` (header `X-Orion-Retina-Token`) once enabled -- unlike
every other route on this service, a POST here triggers a live recording,
so set this. **The HTTP route's refs alone are still not consumable
end-to-end from a bare curl**: `orion-affectgpt-worker` requires local file
paths, not a percept-store ref directly. The bus RPC path below (via
`orion-juniper-affective-state`) IS the built fetch-by-hash bridge -- see
that service's README.

**Live-verified against real hardware, 2026-08-22** (real device names, real
PulseAudio backend, real timing) -- the disclaimer that used to sit here was
written without access to carbon and has been superseded by an actual run.
`tests/test_vision_retina_clip_capture.py` (fake ffmpeg) still covers the
subprocess construction/error-handling layer in CI, where no camera exists.
To re-verify after a change:

```bash
# on carbon (Docker path), after deploying -- RETINA_HTTP_PORT defaults to
# 8027, not 8022, to avoid colliding with other services on a shared host
# (see docker-compose.yml comment); check your actual .env value.
curl -X POST "http://localhost:${RETINA_HTTP_PORT:-8027}/capture/clip?target_stream_id=${RETINA_STREAM_ID}" \
  -H "X-Orion-Retina-Token: ${RETINA_CLIP_TOKEN}"
# expect: {"ok": true, "video_sha256": "...", "audio_sha256": "...", ...}
# target_stream_id is REQUIRED and must equal THIS instance's own
# RETINA_STREAM_ID (2026-08-22) -- omitted or wrong returns
# {"ok": false, "error_code": "wrong_camera"}, same guarantee as the bus
# RPC path (see "Camera-identity check" below).
# then fetch both back from percept-store and confirm they're a real,
# audible/viewable clip -- sha256 round-tripping correctly does not by
# itself prove the *content* is a valid recording of anything.
```

If `PULSE_SERVER`/the audio socket mount is wrong for carbon's actual setup
(see docker-compose.yml comments), the symptom will be an ffmpeg audio
failure specifically -- video capture (v4l2) doesn't depend on that
mount and should work independently, which narrows down which half broke.

**Live-verified 2026-08-22** (real hardware, no longer just "should work") --
see `docs/operations/carbon-webcam.md`'s "Live-verified on carbon" section
for the actual evidence (sha256s, ffprobe output, byte counts), not just the
claim. The `/dev/video0` device-contention bug (`pause_device()` /
`_device_lock`, see below) was found and fixed from this first real run.

### Bus-reachable twin: `orion:exec:request:RetinaClipCaptureService`

Same capture, triggered over the bus instead of HTTP -- for a caller with no
network path to this node at all. carbon accepts no inbound HTTP whatsoever
(see "Nothing needs to reach carbon inbound" above); Hub's "Affect check"
button reaches it through `orion-juniper-affective-state` (circe), which
does the bus RPC. Request payload is `RetinaClipCaptureRequestPayload`
(empty -- no caller-tunable fields, see that schema's docstring in
`orion/schemas/vision.py`), reply is `RetinaClipCaptureResultPayload` on
`orion:retina:clip:reply:<corr_id>`. Gated the same way as the HTTP route
(`RETINA_CLIP_ENABLED`), minus the HTTP token check -- the bus itself is the
trust boundary here, same as every other channel this service already
publishes/consumes with `ORION_BUS_ENFORCE_CATALOG`. Both entry points share
one implementation (`RetinaService.capture_and_upload_clip()`) and one
`_clip_capture_lock`, so a capture started through either path excludes the
other.

No toggle, no scheduling logic lives here -- that's Hub's job
(`services/orion-hub/scripts/api_routes.py`'s `/api/vision/affect-capture`,
the Vision panel's "Affect check" button). See
`services/orion-juniper-affective-state/README.md` for the rest of the
chain (percept-store fetch, worker hand-off).

**Known, accepted risk (disclosed, not silently accepted, review finding
2026-08-22):** trusting the bus as the sole boundary means ANY bus-connected
service can trigger a live webcam+mic recording of Juniper via this channel
-- there is no per-caller identity or credential check, only "reachable the
bus." This is not unique to this channel (every RPC channel in this codebase
works the same way), but it is qualitatively more sensitive here: the action
IS a live recording, not a data query. `RETINA_CLIP_MIN_INTERVAL_SEC`
(default 30s, `ClipCaptureCooldownError`) bounds the worst case to a known
rate rather than de facto continuous recording -- it does not add real
authentication. Real per-caller auth on this channel (or a signed capability
token) is legitimate follow-up work, not done here; note this in any future
threat-model pass on the bus.

**Camera-identity check (Juniper's explicit instruction, 2026-08-22): "I want
this to only run on my carbon webcam."** This channel has no built-in
per-instance routing at all -- confirmed live 2026-08-22: no second retina
deployment is actually configured anywhere in this repo today (the office/
room camera, "Eye-Ball-1"/`cam0` in Hub's Vision panel, is a completely
separate service, `orion-vision-edge`, sharing zero code path with this
one), but the docs for THIS service already anticipate a future room-camera
retina deployment. Without a check, that future deployment (or any retina
instance with `RETINA_CLIP_ENABLED=true`) would silently race this one for
every request on the shared channel. `RetinaClipCaptureRequestPayload.target_stream_id`
(required, no default -- see `orion/schemas/vision.py`) closes that: every
instance compares the incoming request's `target_stream_id` against its own
`RETINA_STREAM_ID` BEFORE checking `RETINA_CLIP_ENABLED` or anything else,
and refuses (`error_code="wrong_camera"`) on a mismatch. Both entry points
enforce this the same way (`_handle_clip_request` for the bus RPC path,
`capture_clip_endpoint`'s required `?target_stream_id=` query param for
HTTP -- review finding, 2026-08-22: the HTTP route shipped this check
LATER than the bus path did, so the guarantee was fully bypassable via a
plain curl for one commit).

**Precise about what this actually guards against (review finding,
2026-08-22 -- an earlier version of this note overclaimed):** it's a
value-equality check on an unauthenticated channel/route, not
authentication. It closes the *accidental-misconfiguration* failure mode
this section opens with -- a second retina instance quietly left with
`RETINA_CLIP_ENABLED=true` no longer silently races this one for every
request. It does NOT stop a caller who deliberately knows or guesses the
right `target_stream_id` (the documented default, `"carbon"`), and it does
NOT stop two instances that are BOTH misconfigured with the same
`RETINA_STREAM_ID` from racing each other -- that's the same
unauthenticated-channel risk the "Known, accepted risk" paragraph above
already discloses, not a new gap this check claims to close.

**Deploy order:** `target_stream_id` is required with no default on
`RetinaClipCaptureRequestPayload` (`extra="forbid"`), so a version mismatch
between this service and `orion-juniper-affective-state` fails safely in
either direction -- a request missing the field, or an old retina that
doesn't recognize it, both land on `error_code="invalid_request"` (payload
parse failure), never a silent wrong-camera capture. No coordinated
rollout is required; a stale request just gets a clean rejection until both
sides are updated.

## Tests

From repo root (worktree or main):

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_vision_retina_*.py -v
```

## Modules

| File | Role |
|------|------|
| `app/settings.py` | Env contract |
| `app/sources.py` | Frame source adapters |
| `app/frame_store.py` | Save + retention |
| `app/clip_capture.py` | On-demand video+audio clip capture (AffectGPT, live-verified 2026-08-22) |
| `app/envelopes.py` | `BaseEnvelope` builder |
| `app/health.py` | `SystemHealthV1` helper |
| `app/main.py` | `RetinaService`, FastAPI lifespan |

## Docs

- Pipeline overview: `docs/vision_services.md`
- Bus channel: `orion/bus/channels.yaml` → `orion:vision:frames`
