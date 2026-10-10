## Summary

- **Every face check that finds a face now keeps its frame,** plus a JSON note of what the check concluded: each face's similarity, match band and detection confidence, plus the thresholds used.
- **Where they're stored:** `/mnt/telemetry/orion-vision-host/face_frames/<camera>/<UTC date>/<time>_<band>_<similarity>.jpg`.
- **Why:** the camera frame buffer (`/mnt/telemetry/vision/frames`) rolls over in about 65 seconds. Juniper's 10-10 04:02 visit to cam0 scored 0.21 and 0.27 against her enrolled face, with the detector 99.7–100% sure it saw a face, and those frames were already gone when we looked.
- **What they're for:**
  - re-enrolling her face from cam0's own angle;
  - auditing sightings (#2558/#2572), and any stranger matched as her.
- **Limits:**
  - at most one frame per camera every 5 s, and only when a face was detected;
  - frames are downscaled to at most 1280 px;
  - pruned after 14 days on a background thread, never on the detection path.
- **Implementation:** reuses the existing `ThumbRateLimiter` from the crop-thumbnail store rather than a new limiter.

## Files changed

- `services/orion-vision-host/app/face_frames.py` (new): `FaceFrameStore` (save + sidecar, per-camera rate limit, age-based pruning) and `best_candidate`.
- `services/orion-vision-host/app/runner.py`: `_run_identity_face` saves the frame when it found faces. A save failure only adds a warning, never fails the check.
- `services/orion-vision-host/app/settings.py`, `.env_example`, `docker-compose.yml`: `VISION_FACE_FRAMES_DIR`, `VISION_FACE_FRAMES_RETENTION_DAYS=14`, `VISION_FACE_FRAMES_MIN_INTERVAL_SEC=5`.
- Tests: `tests/test_face_frames.py` (new, 4), `tests/test_run_identity_face.py` (+1).

## Env/config changes

- Added: the three `VISION_FACE_FRAMES_*` keys, synced to the local `.env`.
- The folder lives under the existing `/mnt/telemetry/orion-vision-host` mount, so no compose volume change is needed.

## Tests run

```text
vision-host tests (in the orion-vision-host image, torch available)   236 passed, 4 failed, 1 skipped
main checkout, same image                                             231 passed, 4 failed (same 4: heartbeat chassis, broadcast suppression)
```

## Schema / bus / API changes

None. Files only.

## Risks / concerns

- **Medium: frames of anyone with a detected face are kept.**
  - Concern: that includes other household members and visitors, for 14 days. The identity gallery's rule ("non-matches are never stored") is about face *embeddings* growing the gallery. This stores frames for review, not embeddings, and nothing reads them automatically.
  - Mitigation: Juniper asked for this directly (2026-10-10). Set `VISION_FACE_FRAMES_DIR=` (empty) to turn it off.
- **Low: disk use.**
  - Concern: each frame is roughly 100–200 KB. Continuous face presence is at most about 720 frames an hour, so the bound depends on how long people sit in view.
  - Mitigation: realistic use is far lower, and 14-day pruning bounds it.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-vision-host/.env -f services/orion-vision-host/docker-compose.yml up -d --build
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)
