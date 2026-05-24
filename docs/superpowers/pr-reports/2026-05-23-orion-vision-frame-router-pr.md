# PR: Orion Vision Frame Router — policy bridge (Retina → Host)

**Branch:** `feat/orion-vision-frame-router-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/orion-vision-frame-router`

## Summary

Adds `services/orion-vision-frame-router` — a bounded policy bridge that subscribes to `orion:vision:frames`, validates `VisionFramePointerPayload`, applies YAML/env sampling and backpressure, and dispatches `VisionTaskRequestPayload` to `orion:exec:request:VisionHostService` with preserved causality (`derive_child`) and reply tracking on `orion:vision:reply:<correlation_id>`.

**Router is policy + bus wiring only** — no capture, cv2, YOLO, or inference imports.

## Architecture

```text
orion:vision:frames
  → VisionFramePointerPayload validate
  → FrameDispatchPolicy.decide (YAML + env)
  → derive_child → vision.task.request
  → orion:exec:request:VisionHostService (reply_to=orion:vision:reply:<corr>)

orion:vision:reply:* (PSUBSCRIBE)
  → VisionTaskResultPayload → clear pending

timeout sweeper + SystemHealthV1 → orion:system:health
```

| Module | Role |
|--------|------|
| `settings.py` | Channels, limits, policy path, `DRY_RUN` |
| `policy.py` | YAML load, `decide()`, `build_task_request()` |
| `state.py` | Per-camera counters, pending tasks, inflight |
| `envelopes.py` | `make_host_task_envelope()` via `derive_child` |
| `dispatcher.py` | Frame/reply handlers, timeout sweep |
| `metrics.py` | `RouterMetrics`, health envelope builder |
| `main.py` | `FrameRouterService`, FastAPI lifespan |

## Code review fixes (post-review)

| Issue | Fix |
|-------|-----|
| Race: concurrent frames could bypass `max_inflight_*` between `decide()` and `mark_dispatched()` | `asyncio.Lock` around decide → publish → mark |
| False timeout metric when reply cleared pending before sweep | Only increment `host_timeouts_total` when `clear_pending` succeeds |
| Reply vs timeout sweep state races | Same lock for reply clear and timeout sweep |
| `sweep_timeouts` sync while handlers are async | Made `sweep_timeouts` async; `main` awaits it |
| Missing dispatcher contract assertions | Tests assert `reply_to` and `correlation_id` on published task |

## Requirements checklist

- [x] Subscribes `orion:vision:frames`, validates `VisionFramePointerPayload`
- [x] Policy sampling/backpressure via `config/vision_frame_router.yaml` + env settings
- [x] Publishes `VisionTaskRequestPayload` to `orion:exec:request:VisionHostService`
- [x] `reply_to = orion:vision:reply:<correlation_id>`
- [x] `derive_child` preserves causality chain + correlation id
- [x] Pending cleared on reply or timeout
- [x] Health on `orion:system:health`
- [x] No cv2/YOLO/capture imports (`test_no_contamination.py`)

## Verification

```bash
cd .worktrees/orion-vision-frame-router
PYTHONPATH=.:services/orion-vision-frame-router pytest services/orion-vision-frame-router/tests -q
# 17 passed
```

## Test plan

- [x] `PYTHONPATH=.:services/orion-vision-frame-router pytest services/orion-vision-frame-router/tests -q` (17 passed)
- [ ] `cd services/orion-vision-frame-router && docker compose build && docker compose up -d`
- [ ] Retina publishing frames → router dispatches sampled tasks to host intake
- [ ] `redis-cli PSUBSCRIBE 'orion:vision:reply:*'` receives `vision.task.result` replies
- [ ] `orion:system:health` includes router `details` (inflight, skip counts, policy path)
- [ ] `DRY_RUN=true` increments metrics/inflight without host publish

## Risk

- **Backpressure:** Global/camera inflight limits drop frames under load; tune `max_inflight_*` and `every_n_frames` per camera in YAML.
- **Shared frame volume:** Router and host must share read access to `image_path` paths (compose mounts `/mnt/telemetry/vision/frames`).
- **Pattern subscribe:** Reply consumer uses `PSUBSCRIBE` on `orion:vision:reply:*`; Redis catalog enforcement must allow pattern channel.
- **Single-process state:** Pending/inflight is in-memory; restart clears tracking (host may still complete in-flight tasks).

## Relationship to Retina / Host

Retina publishes `orion:vision:frames`; this router is the **policy gate** before GPU host work. Host remains the inference RPC surface (`VisionTaskRequestPayload`). Does not change host behavior.
