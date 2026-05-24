# PR: Orion Vision Retina — canonical frame intake organ

**Branch:** `feat/orion-vision-retina-canonical-v1`  
**Base:** `main`  
**Worktree:** `/mnt/scripts/Orion-Sapienform/.worktrees/orion-vision-retina-canonical`

## Summary

Upgrades `orion-vision-retina` from a mock folder pointer stub into the **canonical visual intake organ**: sample frames from `folder` / `rtsp` / `webcam` / `mock`, persist JPEGs, publish `VisionFramePointerPayload` on `orion:vision:frames`, emit `SystemHealthV1` on `orion:system:health`.

**Retina is the eye, not the cortex** — no YOLO, detectors, substrate emitters, or `detector_worker` coupling.

## Architecture

```text
FrameSource.read()
  → capture_once()
  → frame_store.save_frame (JPEG + retention)
  → VisionFramePointerPayload
  → BaseEnvelope (kind=vision.frame.pointer, schema_id=orion.envelope)
  → orion:vision:frames

_health_loop → SystemHealthV1 → orion:system:health
```

| Module | Role |
|--------|------|
| `settings.py` | Full config; `RETINA_SOURCE_PATH` alias |
| `sources.py` | folder / mock / rtsp / webcam adapters |
| `frame_store.py` | Save + retention cleanup |
| `envelopes.py` | Canonical frame pointer envelopes |
| `health.py` | System health + retina metrics in `details` |
| `main.py` | `RetinaService.capture_once()` (unit-testable) |

## Code review fixes (post-review)

| Issue | Fix |
|-------|-----|
| Branch stale vs main (would drop substrate MVP) | Rebased onto `origin/main` |
| `fps_observed` always 0 | Rolling estimate from publish intervals |
| Dockerfile missing `orion` | Repo-root build context + `COPY orion` |
| Compose env gaps | Full parity with `.env_example` |
| No health tests | `tests/test_vision_retina_health.py` |
| Folder list static after start | `_refresh_files()` on empty read |
| Global `tests/conftest.py` pollution | Removed; per-test path bootstrap |
| Health `last_error` / `source_ok` | Cleared on success; degraded after failed sample |

## Verification

```bash
cd .worktrees/orion-vision-retina-canonical
PYTHONPATH=. ../venv/bin/python -m pytest tests/test_vision_retina_*.py -v
# 19 passed
```

## Test plan

- [x] `pytest tests/test_vision_retina_*.py` (19 passed)
- [ ] `cd services/orion-vision-retina && docker compose build && docker compose up -d`
- [ ] JPG in `/mnt/telemetry/vision/intake` → frames under `/mnt/telemetry/vision/frames`
- [ ] `redis-cli SUBSCRIBE orion:vision:frames` → `vision.frame.pointer`
- [ ] `orion:system:health` includes retina `details` (`fps_observed`, `source_ok`)

## Create PR

https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/orion-vision-retina-canonical-v1?expand=1

```bash
gh pr create --title "feat(vision-retina): canonical frame intake organ" \
  --body-file docs/superpowers/pr-reports/2026-05-23-orion-vision-retina-canonical-pr.md
```

## Relationship to `orion-vision-host`

Retina does **not** run on or inside vision-host. Host is GPU task RPC (`VisionTaskRequestPayload`, `retina_fast` profile); retina is continuous capture publishing `orion:vision:frames`. Shared: Redis bus, `image_path` on disk, `VisionFramePointerPayload` schema. Host does not auto-subscribe to frame pointers yet — see `services/orion-vision-retina/README.md`.

## Non-goals (follow-up)

Substrate grammar emitters, vision council/scribe, detectors, GraphDB/SQL/vector writes, host←frames auto-bridge.
