## Summary

- The room camera (cam0) counted the two people in a framed photo on the desk as real people. With Juniper alone in the room it read 3 people; with nobody there it read 2.
- New per-camera size floor in orion-vision-host: on cam0, a `person` box whose long side is under 10% of the frame's short side (48 px on the 640x480 frame) is dropped before NMS. Boxes that touch the frame edge are exempt (cut off, size unknown).
- Config only for cam0 (`min_box_long_side_frac_by_stream` in `config/vision_profiles.yaml`). Walkway, doorbell and every other stream are unchanged.
- No threshold change, no NMS change, no schema or env change.

## Outcome moved

Root cause, with evidence:

- Real cam0 frame, 10-22 19:44 MDT (Juniper packing a server into a box): the detector returns Juniper (0.73) plus two boxes 12x23 px and 10x26 px on the landscape photo on the desk. The boxes sit exactly on the two pictured people. Reproduced by running `app.gdino.detect_images` on archived frames.
- All 2.2M stored detection artifacts (2026-08-12..10-28) were scanned. cam0 person boxes not on the frame edge, by long side: 250,060 at 8-47 px, 5 at 48-63 px, then the real people from 64 px. The small boxes sit at one spot per period and move only when the photo moved on the desk (x~360 in August, x~240-250 Sep..10-21, x~254-273 since 10-22).
- The identity face path also matched faces at the same spot (10-28: 109 face checks at the photo position), because the photo "person" kept triggering cam0's triggered tier.

Why it changed on 10-17 ~15:00 MDT: before that, usually only one pictured person scored over 0.35. From 10-17 afternoon both did (2-box frames: 10-16 70, 10-17 2,286). After 10-22 the photo stood a little to the right and both figures fired in most daylight frames. No code change caused it.

Why it dropped on 10-27: no event. The second figure scores right at the 0.35 cut-off, and daylight moves it above or below hour by hour (10-26 12:00 MDT: 17 two-box frames, 13:00: 741). Frames with any photo box stayed the same (10-26 14,792, 10-27 12,434).

Before/after (metric gate, CLAUDE.md 0A):

1. Provenance: `vision_scene_inventory.counts->>'person'` = per-frame max of `person` objects (`services/orion-vision-window/app/projection.py` `summarize_items`), objects from `runner._run_detect_grounding_dino`. The fix acts at the producer.
2. Independence: not a new metric. It removes a known false source from an existing count.
3. Theory: a whole person in this room cannot be under 48 px tall or wide on the 640x480 frame (doors are ~230 px). The live distribution has an empty valley between the photo (max 47 px) and people (64 px up).
4. Live data replay:
   - Real frames through the real detector (CPU, `detect_images`, 0.35): 31 frames 10-22 19:40-20:20 MDT, before 1/2/3 people = 5/7/19 frames, after = 1 person in all 31. 9 empty-room frames 10-28 09:00-14:00 MDT: before 1 or 2 people in all 9, after 0 in all 9.
   - Window estimate, 10-17..10-28 (stored artifacts mapped onto `vision_scene_inventory` windows; replay before = 14,562 vs DB 14,364, so the mapping is close): windows with 2+ people 14,364 -> ~61; windows with 1+ people 38,810 -> ~2,500.
   - Per day 2+ windows: 10-23 4,197 -> ~1; 10-27 455 -> ~1; 10-28 629 -> ~3.
5. Existing mechanism: zones (`config/vision_zones.yaml`) and the stationary-fixture flag work on walkway crop individuals, not on cam0 detections. A fixed exclusion zone would break each time the photo moves (it has moved at least 4 times since August). NMS cannot help: the two photo boxes are side by side (IoU ~0.25), not duplicates.
6. Reversibility: delete the `cam0` entry in `min_box_long_side_frac_by_stream` and rebuild. No schema, no stored data.

Nested-box duplicates (whole body + upper body of one person) were measured too: a containment rule would remove only 123 more 2+ frames out of 73,338 over 2.5 months. Not shipped; the cause is the photo.

## Current architecture

orion-vision-host runs GroundingDINO on each routed cam0 frame, applies per-label NMS (IoU 0.6) and a score-ordered cap, and publishes objects. orion-vision-window takes the per-frame max of each label into `object_counts`, which reaches `vision_scene_inventory`, presence and the vision council prompt.

## Architecture touched

- orion-vision-host detection post-processing only (`drop_undersized` before `nms`).
- `config/vision_profiles.yaml` (baked into the vision-host image).

## Files changed

- `services/orion-vision-host/app/detections.py`: new `drop_undersized` (pure, no torch).
- `services/orion-vision-host/app/runner.py`: apply the stream's size floor before NMS; on-disk artifact JSON records `undersized_dropped`.
- `config/vision_profiles.yaml`: `min_box_long_side_frac_by_stream: {cam0: {person: 0.1}}` with the evidence.
- `services/orion-vision-host/tests/test_person_size_floor.py`: regression tests with the real live box coordinates.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: cam0 no longer emits `person` boxes under 48 px (long side) that do not touch the frame edge.
- Compatibility notes: the debug artifact JSON on disk gains `undersized_dropped`; it is not on the bus.

## Env/config changes

- Added keys: none (profile YAML param only)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no env template change)
- skipped keys requiring operator action: none

## Tests run

```text
docker run (orion-vision-host-vision-host image, --network none, worktree mounted)
  python3 -m pytest services/orion-vision-host/tests/test_person_size_floor.py tests/test_detections.py tests/test_tiled_detection.py
  45 passed
  python3 -m pytest services/orion-vision-host/tests
  374 passed, 1 skipped, 5 failed -- all 5 try to reach huggingface.co (--network none); unrelated to this change
pytest tests/test_no_personal_terms.py tests/test_no_private_eval_artifacts.py -> 10 passed
python scripts/check_metric_lineage.py --gate -> PASS
git diff --check -> clean
```

## Evals run

```text
Replay on real frames through the real detector (see Outcome moved, item 4).
No standing eval harness for cam0 person counts; follow-up: turn the 40-frame replay into
services/orion-vision-host/evals/cam0_person_count_eval.py once frames can be kept.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-vision-host build -> not run from the worktree (no worktree .env;
and a worktree build would retag the live image name). Config parse + runner wiring verified in the
live image by the tests above. Live effect after deploy: UNVERIFIED.
```

## Review findings fixed

Code review ran in a subagent against `git diff origin/main...HEAD`. No blockers. It confirmed that `stream_id` reaches the runner on every path, that box and frame coordinates match (no resize), and that the runner test fails without the fix (photo boxes IoU ~0.25 < 0.6).

- Finding: labels were matched exactly; GroundingDINO labels are free text (`Person`, ` person`).
  - Fix: both sides are matched after `strip().lower()`. Merged phrases (`person phone`) are documented as not matched.
  - Evidence: `test_label_match_ignores_case_and_spaces`.
- Finding: the evidence did not cover non-edge boxes with a 32-47 px long side, where a partly hidden real person could land.
  - Fix: counted them from the live data. There are 10 in 2.5 months: 5 at the photo spot (10-08, alone), 3 at x~112-152 next to the two photo boxes (09-05), 1 next to a 0.60 real person (09-03), and 1 unexplained (10-25, 472,117-500,160). None is a lone real person, so the floor stays at 0.1. The docstring and the YAML now carry these numbers.
  - Evidence: index scan over the stored detection artifacts.
- Finding: a malformed YAML value would raise `AttributeError` on every cam0 detect task.
  - Fix: a non-mapping value is treated as "no floor" in both the runner and `drop_undersized`.
  - Evidence: `test_malformed_rule_is_a_passthrough_not_a_crash`.
- Finding: the debug log line did not report the drop count.
  - Fix: added `undersized_dropped=` to `[DETECT]`.
- Finding (merge note): the concurrent per-camera score cut-off work edits the same lines. Both filters belong before `nms()`. Whoever merges second keeps both filters and the `undersized_dropped` key.
- Finding: the tests need torch, because conftest imports the runner. They ran in the vision-host image, as listed under Tests run.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-vision-host up -d --build
```

## Risks / concerns

- Severity: low
- Concern: a real person seen only as a small head, mid-frame (for example peeking around a door), under 48 px is dropped. In 2.5 months of live data there were 5 such mid-frame boxes between 48 and 63 px and none between 48 px and the photo's size that looked like people.
- Mitigation: boxes on the frame edge are always kept; the floor is per camera and per label.

- Severity: low
- Concern: the photo still drives identity face checks if something else triggers the cam0 triggered tier (the face path is separate). With the photo no longer a `person`, the triggered tier stops firing on an empty room, so this should mostly stop. UNVERIFIED until deployed.
- Mitigation: watch `face_frames/cam0` for faces at x~1020-1090, y~550-580 (2560 px still) after deploy.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2953
