# Orion Vision Frame Router

Policy bridge between **Retina** (continuous frame capture) and **Vision Host** (on-demand GPU inference). Subscribes to frame pointers, applies YAML sampling/backpressure rules, and dispatches `VisionTaskRequestPayload` tasks while preserving envelope causality.

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
           │                              (policy + sampling)
           │                                      │
           │                                      ▼
           │              orion:exec:request:VisionHostService
           │                              (vision.task.request)
           │                                      │
           │                                      ▼
           │                         orion-vision-host (GPU)
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
| Role | Capture + persist JPEGs | Policy bridge + backpressure | GPU inference |
| Trigger | Internal loop (`RETINA_FPS`) | Bus: `orion:vision:frames` | Bus: host intake channel |
| Output | Frame pointers | Host task requests | Artifacts + task replies |
| Needs GPU | No | No | Yes |

## Bus channels

| Channel | Direction | Payload |
|---------|-----------|---------|
| `orion:vision:frames` | Subscribe | `VisionFramePointerPayload` (`vision.frame.pointer`) |
| `orion:exec:request:VisionHostService` | Publish | `VisionTaskRequestPayload` (`vision.task.request`) |
| `orion:vision:reply:*` | PSUBSCRIBE | `VisionTaskResultPayload` (`vision.task.result`) |
| `orion:system:health` | Publish | `SystemHealthV1` router metrics |

## Policy file

Default policy: `config/vision_frame_router.yaml` (mounted at `/app/config/vision_frame_router.yaml` in Docker).

Per-camera overrides control `task_type`, `every_n_frames`, rate limits, and request fields (e.g. open-vocab prompts). Set `ROUTER_POLICY_PATH` to point at a custom file.

## Configuration

```bash
cd services/orion-vision-frame-router
cp .env_example .env   # edit ORION_BUS_URL, DRY_RUN, etc.
```

Key env vars: `ROUTER_ENABLED`, `DRY_RUN`, `MAX_INFLIGHT_TOTAL`, `TASK_TIMEOUT_SECONDS`, `REQUIRE_IMAGE_PATH_EXISTS`.

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

# Host replies (router clears pending on match)
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
