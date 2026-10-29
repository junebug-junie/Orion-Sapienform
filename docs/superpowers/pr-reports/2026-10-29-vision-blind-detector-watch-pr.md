## Summary

- **An alarm for a blind detector.** The ladder watch (host cron, every 10 min, Hub card) has a new check per 24/7 camera: `detection_rate:walkway` and `detection_rate:cam0`. It goes red when the detector keeps running but the share of windows that see anything falls under a quarter of what the same hours saw on the previous 7 days. Night reads calm because the baseline is also quiet at night. No data reads "cannot check", never calm.
- **One frame per camera per hour, kept 14 days.** orion-sql-writer copies the newest frame of each UTC hour (walkway, cam0, and the doorbell in hours with a Ring event) into `/mnt/telemetry/vision/hourly_frames/<day>/`, with one index line per frame: brightness, contrast, and what the detector saw in the 10 minutes before. It prunes its own day folders.
- **Cut-offs per camera and per label.** The detector's box cut-off can now be set per stream and per label (`box_threshold_by_stream` in `config/vision_profiles.yaml`). **No value changed**: walkway person and vehicle are written out at today's 0.35, cam0 is not listed. The evidence for moving them is not there yet (below).

## Outcome moved

- Before: on 2026-10-22..23 the walkway detector was half-blind for about 30 hours and nothing alarmed. The camera-input check only asks whether frames arrive, and they did.
- After, replayed over stored data (one evaluation per 10-minute tick, `scripts/replay_detection_rate_watch.py --start 2026-10-16 --end 2026-10-29`):

| walkway day (UTC) | red ticks | calm ticks | cannot-check ticks | red UTC hours |
|---|---|---|---|---|
| 10-16, 10-17 | 0 | 0 | 144 | - (camera too new: under 3 baseline days) |
| 10-18 | 0 | 109 | 35 | - |
| 10-19 .. 10-21 | 0 | 144 each | 0 | - |
| **10-22** | **60** | 84 | 0 | 14-23 |
| **10-23** | **56** | 88 | 0 | 0-3 (evening of 10-22 local), 14-19 |
| 10-24 | 12 | 132 | 0 | 2-3, 14-15 |
| 10-25 | 0 | 144 | 0 | - |
| 10-26 (DHCP outage) | 0 | 65 | 79 | - (outage reads cannot-check, not calm) |
| 10-27 | 0 | 108 | 36 | - (rest of the outage) |
| 10-28 | 0 | 144 | 0 | - |

  cam0: 0 red, 0 cannot-check, 144 calm every day 10-16..10-28. cam0's room is never empty for the detector (share 1.00), so on cam0 this only fires on total blindness.
- 10-24's 12 red ticks: two short spells (02-03 UTC = evening of 10-23 local; 14-15 UTC = the first daylight hours). Both are within a day of the incident and may be its tail (the 10-24 dawn hours saw 0.05-0.17 of windows with something vs a usual ~0.6). I cannot tell from stored data, because no frames from then were kept. That is part 1's reason to exist.
- First red on 10-22: 14:00 UTC (the 3-hour span needs about an hour of blind daylight to cross the line).
- Sensitivity (walkway, 10-22..10-28): ratio 0.20 / 0.25 / 0.33 gives 10 / 12 / 15 red ticks on 10-24 and 0 on 10-25..10-28 in every case. A 1-hour span adds 4 red ticks on 10-25; 3 hours is quieter.

## Current architecture

- `orion/substrate_ladder_liveness.py` + `scripts/check_substrate_ladder_liveness.py`: freshness rungs, including `camera_input:cam0|walkway` (newest embedding per stream). Host cron runs `make substrate-ladder-watch` every 10 min from the primary checkout and cards new red keys through orion-notify.
- `services/orion-vision-host/app/runner.py` `_run_detect_grounding_dino`: one `box_th` from `score_threshold` (0.35) for every stream and label.
- Shared frames dir `/mnt/telemetry/vision/frames`: each writer prunes its own files at ~5 min (`FRAME_RETENTION_SECONDS=300`). orion-sql-writer already reads it (read-only) for story pictures.

## Architecture touched

- Ladder watch: new module `orion/vision_detection_rate_watch.py` (stdlib, pure), `LadderReport.detection_rates`, `read_detection_rates` (one bounded query per camera, 8 index range scans on `vision_scene_inventory(stream_id, observed_at)`; ~0.1 s cam0, ~1 s walkway live).
- orion-vision-host: new `app/box_thresholds.py`; runner edits limited to choosing and applying the cut-off (NMS, tile merge and duplicate suppression untouched).
- orion-sql-writer: new loop `app/vision_hourly_frames.py`, new rw mount `/mnt/telemetry/vision/hourly_frames`.
- No bus, schema, or migration changes.

## Metric gate (CLAUDE.md 0A): `detection_rate:<stream>`

1. **Provenance.** `vision_scene_inventory` rows are written by orion-sql-writer from `vision.scene.inventory.v1`, published by orion-vision-window `_publish_scene_inventory` (`services/orion-vision-window/app/main.py`) on every window flush. `counts` = `summary.object_counts`, built from the detector replies in that window; `frame_count` = number of replies. So a row exists only when the detector ran, and `counts <> '{}'` means it kept at least one box.
2. **Independence.** Not independent of the detector, by design: it is the detector's own output rate. It is independent of `camera_input:<stream>` (that one reads embedding arrival in `substrate_perception_embedding_baseline`; frames arriving vs the detector finding something). During the incident camera_input was green and this was red; during the 10-26 outage camera_input was red and this was cannot-check.
3. **Theory anchor.** The scene a fixed camera sees follows the day (light, traffic, parked cars). For the same clock span, the share of windows with any detection is stable day to day (walkway 16-22 UTC: 0.5-1.0 on normal days). A detector whose scores slide under its cut-off loses boxes everywhere at once, so the share collapses while the denominator (detector runs) does not.
4. **Live data.** Not degenerate: walkway share ranges 0.00 (night) to 1.00 (afternoon) and was 0.01-0.22 on the blind afternoon of 10-22. Rest state: a normal night reads calm because its own baseline is under the 0.2 floor (10-28 08:00Z: share 0.00 vs baseline 0.00 -> ok). Absence: 0 windows reads cannot-check (10-26 15:00Z). Weak spots: (a) the baseline is the median of 7 days, so a blindness that lasts 4+ days becomes the new normal and the alarm stops; the onset still cards. (b) The span is UTC clock time, so local traffic shifts by an hour at a DST change; sunlight does not. (c) cam0 can only catch total blindness.
5. **Existing mechanism.** Reuses the ladder watch, its dedupe, Hub card path and exit codes; reuses `vision_scene_inventory` and its index. Nothing new is stored.
6. **Reversibility.** Pure module + list entry. Removing it is deleting the module and four lines in the report; no schema, no table, no env key.

## Threshold decisions and evidence (part 3)

Nothing moved. The mechanism ships with today's values written out.

- **Walkway person: kept at 0.35.** Juniper's Hub verdicts (`vision_motion_story_verdict`, read 2026-10-29): 67 walkway person stories "right", 2 "wrong_other". Detection scores in "right" stories: p05 0.362, p10 0.374, p25 0.407, p50 0.446; 43 of 341 detections are under 0.38. The 2 wrong stories' best boxes are 0.426 and 0.468. So raising the cut-off would cost real people and still keep both wrong stories. Lowering it cannot be judged: every stored detection is >= 0.35 by construction, so nothing says what lies under it.
- **Walkway vehicle: kept at 0.35.** Two parked-car spots found from stored crops (boxes seen on 3+ days in 12+ daytime hours; 15-22 UTC, 10-22..10-28; crops before 10-22 are past their 7-day retention):

| day | spot A seen in % of detector frames | spot B % | median score on spots |
|---|---|---|---|
| 10-22 | 0 % | 3 % | 0.358 |
| 10-23 | 32 % | 8 % | 0.360 |
| 10-24 | 79 % | 25 % | 0.375 |
| 10-25 | 86 % | 48 % | 0.423 |
| 10-27 | 69 % | 12 % | 0.382 |
| 10-28 | 96 % | 59 % | 0.403 |

  On the blind days the visible scores pile up just over the cut-off (median 0.36), which says most of the distribution moved under 0.35. Whether the cars were parked there on 10-22 is UNVERIFIED (no frames). False-positive cost of a lower cut-off: on good days, 7,640 of 8,102 vehicle boxes scoring 0.35-0.37 (94 %) sit on the two parked spots, vs 9,595 of 9,774 (98 %) at >= 0.42. So the lowest stored band is still mostly real cars, but that says nothing about 0.30-0.35, which is never stored. The 10-23 manual probe (two real vehicles at 0.33 / 0.32) is one frame. The cam0 sweep in `vision_profiles.yaml` (0.25 -> 3.8 phantom desks per frame) shows the low tail can be mostly phantoms. Not enough to move it.
- **Next step that would settle it:** re-score the hourly samples from part 1 at a low cut-off (e.g. 0.20), on a sunny and a grey day, and count real vs phantom vehicles in the 0.30-0.35 band by eye. Then the walkway vehicle line is a one-line edit.
- **cam0: not touched** (another agent is fixing cam0 phantom people).

## Files changed

- `orion/vision_detection_rate_watch.py`: new; the blind-detector rule and its SQL.
- `orion/substrate_ladder_liveness.py`: `LadderReport.detection_rates`, red/green keys, warning severity, card text, JSON.
- `scripts/check_substrate_ladder_liveness.py`: `read_detection_rates`, human output.
- `scripts/replay_detection_rate_watch.py`: new; read-only replay over stored history (eval).
- `tests/scripts/test_vision_detection_rate_watch.py`: new; real fixtures for the incident, a normal night, the outage, a normal day.
- `services/orion-vision-host/app/box_thresholds.py`: new; resolve and apply per-stream, per-label cut-offs.
- `services/orion-vision-host/app/runner.py`: threshold selection only.
- `services/orion-vision-host/tests/test_box_thresholds.py`: new.
- `services/orion-vision-host/README.md`: new section.
- `config/vision_profiles.yaml`: `box_threshold_by_stream` with today's walkway values and the evidence.
- `services/orion-sql-writer/app/vision_hourly_frames.py`: new; sampler, index, prune, loop.
- `services/orion-sql-writer/app/main.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: wiring, keys, rw mount, docs.
- `services/orion-sql-writer/tests/test_vision_hourly_frames.py`, `tests/test_vision_hourly_frames_pg.py`: new.
- `scripts/sync_local_env_from_example.py`: `VISION_HOURLY_FRAMES_` prefix (the sync skipped the new keys without it).
- `.github/workflows/orion-static-gates.yml`, `.github/workflows/orion-sql-writer-tests.yml`: run the new tests.

## Schema / bus / API changes

- Added: none (bus/schema). Ladder `--json` output gains a `detection_rates` list.
- Removed: none.
- Renamed: none.
- Behavior changed: vision-host detect artifact JSON gains `box_threshold_by_label`. The ladder card can now carry a `detection_rate:<stream>` key (severity warning when it is the only finding).
- Compatibility notes: with the shipped config the detector's output is identical to before (every cut-off 0.35; test `test_runner_shipped_profile_behaves_like_single_threshold`).

## Env/config changes

- Added keys (orion-sql-writer): `VISION_HOURLY_FRAMES_ENABLED=true`, `VISION_HOURLY_FRAMES_INTERVAL_SEC=60`, `VISION_HOURLY_FRAMES_DIR=/mnt/telemetry/vision/hourly_frames`, `VISION_HOURLY_FRAMES_RETENTION_DAYS=14`, `VISION_HOURLY_FRAMES_STREAMS=walkway,cam0=frame,doorbell`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (orion-sql-writer).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, all 5 keys landed in `services/orion-sql-writer/.env` (checked by grep).
- skipped keys requiring operator action: none. Pre-existing, not from this PR: `VISION_HANDOFF_MIN_MET` is missing from the live sql-writer `.env`.

## Tests run

```text
tests/scripts/test_vision_detection_rate_watch.py + test_substrate_ladder_liveness.py: 80 passed
orion-vision-host full suite in the service image (PYTHONPATH=repo): 382 passed, 1 skipped
orion-sql-writer test_vision_hourly_frames.py + test_vision_story_strip.py + test_vision_motion.py: 64 passed, 18 skipped (PG)
orion-sql-writer test_vision_hourly_frames_pg.py on a throwaway postgres:16 with the CI migrations: 2 passed
Static gates: env template parity PASS; env key single source OK; compose relative mounts PASS; metric lineage PASS;
definition drift PASS; sync/env/settings/journald/allowlist/personal-terms pytest: 73 passed; ci_service_image_sources + settings defaults: 27 passed
```

## Evals run

```text
scripts/replay_detection_rate_watch.py --start 2026-10-16 --end 2026-10-29 (live, read-only): table above.
Sensitivity sweep --ratio 0.2/0.33, --span-hours 1/2: in "Outcome moved".
Live ladder run (--skip-docker --skip-migrations): detection_rate:cam0 ok (1611/1611, baseline 1.00),
detection_rate:walkway ok (1439/1515, baseline 0.79), GREEN.
Hourly sampler live dry run into a scratch dir with real frames + live DB: saved walkway (2560x1920, 719 KB,
luma 0.32, 89 windows / 1 with a detection) and cam0 (640x480); doorbell no_frame (no Ring event then).
```

## Docker/build/smoke checks

```text
orion-vision-host tests ran inside the existing orion-vision-host-vision-host:latest image with the worktree mounted.
No image was built and nothing was deployed (task rule: do not deploy).
```

## Review findings fixed

(filled in after review)

## Restart required

```bash
# Ladder watch: nothing to restart; host cron picks it up after `git pull` on main.
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-vision-host up -d --build
```

## Risks / concerns

- Severity: low. Concern: a blindness that lasts 4+ days becomes its own baseline and the alarm stops. Mitigation: the onset still cards; the card stays delivered until the key reads ok.
- Severity: low. Concern: the hourly sample is the newest frame of the hour, so it reflects the first minute(s) of the hour, not the whole hour. Mitigation: the index records the exact frame time and the 10 min detector summary.
- Severity: low. Concern: vision-host config is baked into the image; a later cut-off change needs a rebuild. Mitigation: documented in the README.
- UNVERIFIED: the weather cause of 10-22..23; whether the parked cars were present on 10-22; the doorbell sample (no Ring event during the dry run); the live loop in the container (not deployed).

## PR link

(filled in after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
