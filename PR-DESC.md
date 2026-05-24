# PR: feat(vision-retina): canonical frame intake organ

**Branch:** `feat/orion-vision-retina-canonical-v1`  
**Base:** `main`  
**Worktree:** `.worktrees/orion-vision-retina-canonical` (main checkout `feat/repair-pressure-v1` untouched)

**Create PR:** https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/orion-vision-retina-canonical-v1?expand=1

> `gh pr create` failed in this environment (`gh auth login` required). Branch is pushed; open PR via link above or run `gh auth login` then `gh pr create` from the worktree.

---

## Summary

Upgrade `orion-vision-retina` from a mock folder pointer stub into the **canonical visual intake organ** for Orion. Retina samples frames from `folder` / `rtsp` / `webcam` / `mock` sources, writes JPEG artifacts to shared storage, publishes `VisionFramePointerPayload` on `orion:vision:frames`, and emits `SystemHealthV1` on `orion:system:health`.

**Retina is the eye, not the cortex** — no YOLO, face detection, motion/presence, security watcher, substrate emitters, or `detector_worker` imports.

## Architecture

```text
source adapter → capture_once → frame_store (JPEG) → VisionFramePointerPayload
                              → BaseEnvelope (kind=vision.frame.pointer)
                              → orion:vision:frames
                              → orion:system:health (periodic)
```

## Commits (10)

| Commit | Description |
|--------|-------------|
| `52db49f2` | opencv + numpy deps |
| `6544135b` | expanded settings + `.env_example` |
| `6f217437` | frame_store save/cleanup |
| `1e8022a8` | sources (folder/mock/rtsp/webcam) |
| `22f003e5` | envelopes helper |
| `c206f01d` | health helper |
| `a384c3b1` | RetinaService + capture_once tests |
| `b180ad46` | no-detector regression guard |
| `0d52a582` | docker-compose + bus catalog |
| `a0264c97` | implementation plan doc |

## Verification evidence

```bash
cd .worktrees/orion-vision-retina-canonical
PYTHONPATH=. ../venv/bin/python -m pytest tests/test_vision_retina_*.py -v
# 15 passed in ~1.8s
```

## Code review fixes applied

- Removed repo-root `tests/conftest.py` (avoided `app` package collision with other services)
- Per-test `sys.path` bootstrap for retina imports only
- Health: `last_error` cleared on success; degraded when reads fail persistently (`_source_ok()`)
- Settings: `RETINA_SOURCE` wins over `RETINA_SOURCE_PATH` when both set

## Local sync (operator)

`.env` and `.env_example` copied to main checkout `services/orion-vision-retina/` (gitignored `.env` only).

## Test plan

- [x] `pytest tests/test_vision_retina_*.py` (15 passed)
- [ ] `cd services/orion-vision-retina && docker compose up -d`
- [ ] JPG in `/mnt/telemetry/vision/intake` → frames in `/mnt/telemetry/vision/frames`
- [ ] `redis-cli SUBSCRIBE orion:vision:frames` → `vision.frame.pointer` envelopes
- [ ] `orion:system:health` includes retina `details`

## Non-goals (later PRs)

Substrate grammar emitters, vision council/scribe, detectors, GraphDB/SQL/vector writes.
