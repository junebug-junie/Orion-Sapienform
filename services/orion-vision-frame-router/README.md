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

Task meta includes `dispatch_tier` (`baseline` or `triggered`) for observability.

Per-stream overrides use the `streams:` block (e.g. `streams.cam0`). Legacy per-camera overrides remain under `cameras:` keyed by `camera_id`.

## Bus channels

| Channel | Direction | Payload |
|---------|-----------|---------|
| `orion:vision:frames` | Subscribe | `VisionFramePointerPayload` (`vision.frame.pointer`) |
| `orion:exec:request:VisionHostService` | Publish | `VisionTaskRequestPayload` (`vision.task.request`) |
| `orion:vision:reply:*` | PSUBSCRIBE | `VisionTaskResultPayload` (`vision.task.result`) |
| `orion:system:health` | Publish | `SystemHealthV1` router metrics |

## Policy file

Default policy: `config/vision_frame_router.yaml` (mounted at `/app/config/vision_frame_router.yaml` in Docker).

Policy defines `defaults.baseline` and `defaults.triggered` tiers (see **Dispatch tiers** above). Per-stream overrides control `streams.cam0` etc.; per-camera overrides remain under `cameras:`. Set `ROUTER_POLICY_PATH` to point at a custom file.

## Configuration

```bash
cd services/orion-vision-frame-router
cp .env_example .env   # edit ORION_BUS_URL, DRY_RUN, etc.
```

Key env vars: `ROUTER_ENABLED`, `DRY_RUN`, `MAX_INFLIGHT_TOTAL`, `TASK_TIMEOUT_SECONDS`, `REQUIRE_IMAGE_PATH_EXISTS`.

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
