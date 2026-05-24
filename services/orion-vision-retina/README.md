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
| `ORION_BUS_URL` | `redis://localhost:6379/0` | Redis pub/sub (use mesh URL in prod) |
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
| `app/envelopes.py` | `BaseEnvelope` builder |
| `app/health.py` | `SystemHealthV1` helper |
| `app/main.py` | `RetinaService`, FastAPI lifespan |

## Docs

- Pipeline overview: `docs/vision_services.md`
- Bus channel: `orion/bus/channels.yaml` → `orion:vision:frames`
