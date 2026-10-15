## Summary

- The walkway camera is mounted (Reolink, 192.168.1.87, edge reads the 640x480 substream `h264Preview_01_sub`).
- Replaces the placeholder zone shapes in `config/vision_zones.yaml` with ones traced from a live 2560x1920 frame:
  - **patio:** the porch slab, its plant bed and the furniture, lower right. Never embedded and never thumbnailed.
  - **mailbox:** the community mailbox cluster, left-middle.
  - **walkway:** everything public below the sky line.
- Tests that hardcoded the placeholder coordinates now use boxes on the real shapes.

## Outcome moved

Before this, the patio rule protected the wrong part of the picture. The placeholder put the "patio" on the lower-left lawn, while the real porch is lower right. Starting the camera on the placeholder would have embedded crops of people on the porch.

## Current architecture

The walkway pipeline is merged and deployed (PR #2287). The zones file is baked into the vision-host, vision-council and sql-writer images, and the walkway edge instance (`vision-edge-walkway` compose profile) is not started.

## Architecture touched

Config only: `config/vision_zones.yaml`. No code changes.

## Files changed

- `config/vision_zones.yaml`: real polygons traced from a frame.
- `orion/vision/tests/test_zones.py`: boxes on the real patio and walkway.
- `services/orion-vision-council/tests/test_unresolved.py`: same.

## Schema / bus / API changes

None.

## Env/config changes

- No `.env_example` change.
- Local `services/orion-vision-edge/.env`: `WALKWAY_RTSP_URL` set to the camera's substream. Same credentials as cam0, verified with ffprobe. Never committed; never-sync key.

## Tests run

```text
orion/vision/tests: 25 passed
services/orion-vision-council/tests: 110 passed
orion-vision-host test_crop_embeddings_patio + test_crop_thumbnails (inside the existing image): 22 passed
services/orion-sql-writer/tests: no new failures (failure list identical to origin/main)
```

## Evals run

```text
None. This is a config change. The shapes were checked visually by drawing them over the live frame.
```

## Docker/build/smoke checks

```text
ffprobe on rtsp://<cam0 creds>@192.168.1.87:554/h264Preview_01_sub -> h264 640x480
```

## Review findings fixed

- Finding: the zone overlay drawn on the live frame was the review.
  - Fix: none needed. The patio covers the slab, plant bed and furniture; the mailbox box covers the mailbox cluster plus the ground in front of it.
  - Evidence: the overlay image was checked in-session.

## Restart required

Rebuild the three images that bake the zones file, then start the walkway edge. All of this runs from the primary checkout on main after merge:

```bash
scripts/safe_docker_build.sh orion-vision-host up -d --build
scripts/safe_docker_build.sh orion-vision-council up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-vision-edge --profile walkway up -d --build vision-edge-walkway
```

## Risks / concerns

- **Severity: medium. The Halloween skeletons on the porch may read as "people on the patio".**
  - Two skeleton props sit at the patio table. The detector may count them as `person`, so the patio presence count, and the street line stance reads, could say people are on the patio all night.
  - They are never embedded or named, because the patio rule holds.
  - Mitigation: watch the patio presence row after go-live.
- **Severity: low. The shapes were traced from a night frame.** Recheck the mailbox box against a daytime frame.

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
