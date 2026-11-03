## Summary

- **The walkway detector now sees parked cars on rainy days.** On rainy and grey days the detector's confidence in the parked cars drops to 0.27-0.35. The cut-off was 0.35, so most of those cars fell just under it and the camera read "empty street". On 2026-11-01 it saw something in 0 of 90 windows in the 23Z hour while a silver car stood mid-street.
- **Only the walkway `vehicle` cut-off moves: 0.35 -> 0.25.** The walkway `person` cut-off stays 0.35. Every other label (dog, bicycle, stroller, "mail truck") and every other camera stays 0.35.
- **The evidence is a labelled re-score, now a repeatable eval.** 142 hourly walkway frames (2026-10-29..11-03, day and night, sun, grey and rain) were re-scored at a 0.20 cut-off with the live detector code. 98 frames were labelled by eye: every box is real, false or a duplicate, plus the objects the detector missed entirely. `services/orion-vision-host/evals/walkway_cutoff_calibration_eval.py` replays the live pipeline at any cut-off from the stored boxes, with no model and no frames.
- **The 2026-11-03 "empty street" was not empty.** A dump trailer stood at the far corner all day. It scored 0.25-0.29 on the whole frame most of the day, so the blind-detector alarm's red that morning was real partial blindness and not a false alarm.

## Outcome moved

Real vehicles seen, whole-frame pass (most live frames are whole-frame only), 98 labelled frames:

| condition | at 0.35 (before) | at 0.25 (after) |
|---|---|---|
| rain (15 frames) | 4 of 28 (14 %) | 28 of 28 (100 %) |
| sunny (35 frames) | 15 of 54 (28 %) | 44 of 54 (81 %) |
| overcast (14 frames) | 18 of 22 (82 %) | 22 of 22 (100 %) |
| false vehicle boxes | 0 | 0 |

Hourly-frame proxy for the blind-detector alarm (does the hour's frame keep any box; daylight hours 14Z-00Z):

| day (UTC) | weather | before (0.35) | after (shipped) |
|---|---|---|---|
| 10-29 | sun, then grey | 5/10 | 10/10 |
| 10-30 | sun, grey, rain late | 8/11 | 11/11 |
| 10-31 | sun | 11/11 | 11/11 |
| 11-01 | rain all day | 3/11 | 11/11 |
| 11-02 | grey, then sun | 1/11 | 10/11 |
| 11-03 | sun, low sun glare | 1/10 | 7/10 (9/10 with tiles) |

## Current architecture

- `config/vision_profiles.yaml` `retina_detect_open_vocab.params.box_threshold_by_stream.walkway` held `person: 0.35, vehicle: 0.35`. These were the old single-threshold values, kept by PR #2952 "until the hourly frame samples are re-scored at a low cut-off".
- `services/orion-vision-host/app/box_thresholds.py` picks the cut-offs per request `stream_id`. `app/runner.py` `_run_detect_grounding_dino` runs GroundingDINO once at the lowest cut-off that applies. It then keeps each box only when the box reaches its own label's cut-off, on the whole frame and on each tile.
- Tiled detection is on for walkway (`config/vision_frame_router.yaml` `streams.walkway.tiled_detect.enabled: true`, at most 1 frame per 20 s). It adds 2 band tiles (y 15-60 %) to the whole-frame pass.
- Hourly frames: orion-sql-writer keeps 1 frame per camera per hour under `/mnt/telemetry/vision/hourly_frames/<day>/` with `index.jsonl` (PR #2952).

## Architecture touched

- Config only at runtime: one value in `config/vision_profiles.yaml`. This file is copied into the orion-vision-host image (`Dockerfile` `COPY config/vision_profiles.yaml`), so a rebuild picks it up. `scripts/rebuild_affected_services.py --plan --base origin/main` returns `orion-vision-host	build`.
- New eval and fixtures under `services/orion-vision-host/evals/`. No bus, schema, migration or env changes.

## Files changed

- `config/vision_profiles.yaml`: walkway `vehicle` 0.35 -> 0.25. The comment holds the evidence and points to this report and the eval.
- `services/orion-vision-host/evals/walkway_cutoff_calibration_eval.py`: new. `score` stores raw boxes at a low floor with the host's own `app.gdino.detect_images` and the shipped tile layout. `pool` writes numbered JPEGs, contact sheets and a labels template. `report` gives recall / false / duplicate per label and condition at each cut-off. `hours` gives the per-day blind-detector proxy.
- `services/orion-vision-host/evals/fixtures/walkway_cutoff_scored.jsonl`: raw boxes for 142 frames (boxes only, no pixels).
- `services/orion-vision-host/evals/fixtures/walkway_cutoff_labels.csv`: 111 labelled boxes (105 real, 1 false, 1 duplicate, 4 people among them).
- `services/orion-vision-host/evals/fixtures/walkway_cutoff_frames.csv`: 98 frames with condition, objects missed even at 0.20, and a note.
- `services/orion-vision-host/tests/test_walkway_cutoff_calibration_eval.py`: new. It pins the replay order, the bookkeeping, and the shipped result on the committed set: rain recall >= 95 %, rain recall at 0.35 < 50 %, and 0 false vehicle or person boxes.
- `services/orion-vision-host/tests/test_box_thresholds.py`: the shipped-config tests now pin walkway vehicle at 0.25 and everything else at 0.35.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: walkway `vehicle` boxes scoring 0.25-0.35 are now kept. The detector floor for walkway requests is now 0.25.
- Compatibility notes: none

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no env template changed)
- skipped keys requiring operator action: none

## Calibration method

1. **Re-score.** All 142 walkway hourly frames were re-scored. The run used CPU inference in a throwaway container from the live `orion-vision-host-vision-host` image, with 32 cores, the model cache mounted read-only, and about 18 s per frame (whole frame plus 2 tiles). GPU was deliberately not used. athena's T10 runs the live host with ~15.3 of 16 GB used, so a second GroundingDINO copy there risks OOM. The P4 is a live lane too. The run used the floor 0.20 and the live `text_threshold` (0.25), prompts and tile layout.
2. **Label.** Every box at the floor (whole and tiled passes, merged) was labelled from contact sheets of crops and from annotated frames: 98 frames (35 sunny, 14 overcast, 15 rain, 34 night). Objects were tracked by name across hours: `trailer_far`, `silver_car_mid`, `white_car_far_right`, and so on. Objects missed even at 0.20 were recorded per frame. Night frames are IR or near-dark. In 33 of 34 no object could be judged, so they test false boxes, not recall. The exception (11-03 01Z) has 2 parked cars in IR range.
3. **Replay.** The replay runs in the runner's order: per-label cut-off on each pass, tile merge, size floor, NMS, cap. Each kept box is matched to its labelled box by IoU or containment.

Labelled sample counts: 98 frames, 111 boxes: vehicle 105 real / 1 false / 1 duplicate, person 4 real. There are 106 distinct vehicle-object sightings, of which 1 was missed at every cut-off (sun straight into the lens on 11-03 15Z), and 4 person sightings.

## Recall / false-positive table

Recall = sightings with at least one kept real box / all sightings (including ones missed at 0.20). Each row applies one cut-off to every label.

### Whole frame (most live frames)

| cut-off | label | sunny | overcast | rain | night | all | false boxes |
|---|---|---|---|---|---|---|---|
| 0.25 | vehicle | 44/54 (81%) | 22/22 (100%) | 28/28 (100%) | 2/2 (100%) | 96/106 (91%) | 0 |
| 0.25 | person | 0/1 (0%) | 3/3 (100%) | - | - | 3/4 (75%) | 0 |
| 0.28 | vehicle | 36/54 (67%) | 22/22 (100%) | 27/28 (96%) | 2/2 (100%) | 87/106 (82%) | 0 |
| 0.28 | person | 0/1 (0%) | 3/3 (100%) | - | - | 3/4 (75%) | 0 |
| 0.30 | vehicle | 33/54 (61%) | 21/22 (95%) | 20/28 (71%) | 2/2 (100%) | 76/106 (72%) | 0 |
| 0.30 | person | 0/1 (0%) | 3/3 (100%) | - | - | 3/4 (75%) | 0 |
| 0.32 | vehicle | 29/54 (54%) | 21/22 (95%) | 14/28 (50%) | 1/2 (50%) | 65/106 (61%) | 0 |
| 0.32 | person | 0/1 (0%) | 3/3 (100%) | - | - | 3/4 (75%) | 0 |
| 0.35 | vehicle | 15/54 (28%) | 18/22 (82%) | 4/28 (14%) | 0/2 (0%) | 37/106 (35%) | 0 |
| 0.35 | person | 0/1 (0%) | 1/3 (33%) | - | - | 1/4 (25%) | 0 |

### Whole frame + tiles (live: about 1 frame per 20 s)

| cut-off | label | sunny | overcast | rain | night | all | false boxes |
|---|---|---|---|---|---|---|---|
| 0.25 | vehicle | 53/54 (98%) | 22/22 (100%) | 28/28 (100%) | 2/2 (100%) | 105/106 (99%) | 1 |
| 0.25 | person | 1/1 (100%) | 3/3 (100%) | - | - | 4/4 (100%) | 0 |
| 0.28 | vehicle | 50/54 (93%) | 22/22 (100%) | 27/28 (96%) | 2/2 (100%) | 101/106 (95%) | 1 |
| 0.28 | person | 1/1 (100%) | 3/3 (100%) | - | - | 4/4 (100%) | 0 |
| 0.30 | vehicle | 46/54 (85%) | 22/22 (100%) | 20/28 (71%) | 2/2 (100%) | 90/106 (85%) | 1 |
| 0.30 | person | 0/1 (0%) | 3/3 (100%) | - | - | 3/4 (75%) | 0 |
| 0.32 | vehicle | 40/54 (74%) | 22/22 (100%) | 14/28 (50%) | 2/2 (100%) | 78/106 (74%) | 1 |
| 0.32 | person | 0/1 (0%) | 3/3 (100%) | - | - | 3/4 (75%) | 0 |
| 0.35 | vehicle | 21/54 (39%) | 18/22 (82%) | 4/28 (14%) | 1/2 (50%) | 44/106 (42%) | 0 |
| 0.35 | person | 0/1 (0%) | 2/3 (67%) | - | - | 2/4 (50%) | 0 |

The one false box is a tile-only box at 0.33 on 11-03 01Z (IR night): a small bright object on a porch behind the fence. It is above 0.32, so it does not decide between 0.25 and 0.32.

Why not lower than 0.25: every one of the 43 raw boxes under 0.25 came back with an empty label (41 of them in labelled frames; `report` prints this count). So the eval cannot measure below 0.25, and it now refuses to: the lowest row is 0.25. The live `text_threshold` is 0.25, and a box needs a prompt word above it to get a name. So 0.25 is the lowest vehicle cut-off that changes anything while `text_threshold` stays 0.25.

### Choice

- **vehicle 0.25.** This is the only value that sees every rain-day sighting (28/28) and the most sunny-glare sightings (44/54 whole frame). There were 0 false vehicle boxes on the whole frame at any value down to 0.20. 0.28 would leave 1 rain sighting and 8 more sunny sightings unseen, for no measured precision gain.
- **person 0.35, kept.** This sample has 4 people, too few to move a cut-off that feeds motion stories, person learning and the street line. 0 false person boxes were seen at any value, but 0 of 4 says little. The low-scoring real person was a mail carrier half-hidden behind the mailbox door (0.29). The Hub verdict evidence from PR #2952 still holds (p05 of right detections 0.362).
- **Tiling.** Tiling is on for walkway and helps far and small vehicles: whole-frame plus tiles at 0.25 sees 53/54 sunny sightings vs 44/54 whole only. The white car seen through the neighbour's porch and the cars half-hidden behind the porch post are found mostly by the tiles. No tiling change is made. The tiles already run on about 1 frame in 20 s, which is enough for a parked car.

## Blind-detector alarm (`detection_rate:walkway`)

- Live alarm replay at the OLD cut-off, `scripts/replay_detection_rate_watch.py --start 2026-10-29 --end 2026-11-04`: red ticks 10-30 1 (22Z), 11-01 22 (14-18Z), 11-02 35 (0-3, 17, 18, 22, 23Z), 11-03 24 (0, 16-19Z).
- **The live alarm cannot be replayed at the new cut-off from stored data.** Its windows come from `vision_scene_inventory`, which only holds boxes that passed 0.35. The hourly proxy above is the closest stand-in. At the shipped cut-off every day 10-29..11-03 keeps a box in 7-11 of 10-11 daylight frames, against 1-3 on the blind days at 0.35. On that proxy no day would cross the red line (share under a quarter of the usual).
- **11-03 was not an empty street.** The trailer was in every daylight frame. The red hours 16-19Z match frames where it scored 0.25-0.29 on the whole frame (0.29-0.37 with tiles): real partial blindness. The only frame with nothing found even at 0.20 is 11-03 15Z (low sun straight into the lens), a genuinely blind hour. So the "empty-street false red" does not remain on this evidence. The score-margin signal (median stored score vs cut-off from `index.jsonl` `detections.crop_scores`) is not needed now, and nothing is built.
- First-week effect: the alarm's baseline is the median of the 7 days before, measured at the old cut-off, so it is lower than what the new cut-off produces. For about a week the alarm can only read ok or quiet because of this change. A real blindness must still fall under a quarter of the (lower) old baseline to go red. So the alarm is slightly less sensitive until 7 new-cut-off days are in the baseline.

## Downstream volume estimate

Kept vehicle boxes over all hourly frames, whole pass, old -> new: good days 10-29..10-31 31 -> 50 (x1.6); blind days 11-01..11-03 6 -> 46 (x7.7). Live walkway vehicle crop rows per day (`vision_crop_observation`): 10-28 25,895; 10-31 20,781; 11-01 4,949; 11-02 3,117; 11-03 2,258.

- **sql-writer crop rows:** expect about x1.6 a good day: ~33-41k vehicle rows/day. At ~5 KB per row (measured average, embedding included) that is ~165-210 MB/day, vs ~105-130 MB on good days now. Blind days go back to good-day volume or more.
- **Crop embeddings / GPU time:** vehicle crops get SigLIP only. OSNet and faces are person-only (`OSNET_LABELS`, `crop_face_labels`). The SigLIP pass runs once per frame that has any box, batched. On good days nearly every daylight frame already has one (10-31: 11/11 hourly frames), so the change adds 1 crop to a batch, not a new pass. On blind days it brings back the passes a normal day already runs. The per-frame cap (`crop_embedding_max_per_frame: 8`) is not reached (at most 3 vehicles per frame in the sample). Measured GPU time after deploy: UNVERIFIED.
- **Motion stories:** vehicle passages are never published (`out_of_scope`, `prop`, `parked`, `glimpse`). Expect vehicle story rows to grow with the boxes (good days now 400-560 `prop` + ~120 `glimpse`/day). Person stories do not change, because the person cut-off does not change. Small risk: a moving car seen more often can show up as a "together" companion of a walking person.
- **Vehicle rhythm (`vision_rhythm.py`, vehicle arrivals):** rain-day flicker (a parked car dropping in and out under 0.35) should make fewer false "arrivals". UNVERIFIED.

## Metric gate (CLAUDE.md 0A)

This patch moves a threshold and adds an eval. It wires no new metric into a model. The gate is applied to the eval's two measures.

1. **Provenance.** Recall and false boxes come from `evaluate()` over `replay()`. `replay()` calls the host's own `apply_box_thresholds`, `merge_tiled_detections`, `drop_undersized`, `nms` and `cap_by_score` on raw boxes from `app.gdino.detect_images` (the function `runner.py` calls). The labels are human verdicts in `walkway_cutoff_labels.csv`. The hourly proxy uses the same replay on frames written by orion-sql-writer `app/vision_hourly_frames.py`.
2. **Independence.** The hourly proxy is not independent of `detection_rate:walkway`: it is the same detector output, sampled once per hour instead of per window. It is used only as a stand-in for a replay that stored data cannot support. Recall is independent of the alarm, because it rests on human labels.
3. **Theory anchor.** A detector score is a calibrated-ish confidence. A fixed scene (parked cars) under changing light and wet roads shifts the score distribution as a whole (whole frame, silver car: 11-01 0.31-0.35 all day; 10-31 0.36-0.43). Recall against labelled objects at each cut-off measures exactly the miss rate that shift causes.
4. **Live data.** Not degenerate: per-condition recall at 0.35 ranges 14-82 %. At rest (night, empty street beyond IR range) the measure correctly reads 0 objects and 0 false boxes. It does not read a fake "seen". The one IR night with parked cars in range (11-03 01Z) shows both cars found.
5. **Existing mechanism.** This extends PR #2952's per-label cut-offs and hourly frames and the tiled eval's harness pattern (`walkway_tiled_detection_eval.py`). Nothing parallel is built.
6. **Reversibility.** It is one YAML value. Reverting it to 0.35 and rebuilding orion-vision-host restores the old behaviour. The eval and fixtures are standalone.

## Tests run

```text
docker run ... orion-vision-host-vision-host -m pytest tests -q   (worktree)
  398 passed, 1 skipped, 5 failed (after review fixes)
  5 failures are pre-existing: same 5 fail on clean main in the same image
  (test_heartbeat_chassis x2, test_identity_face_broadcast_suppression x2,
   test_model_manager_vlm_family x1)
docker run ... -m pytest tests/test_walkway_cutoff_calibration_eval.py tests/test_box_thresholds.py -q
  22 passed (after review fixes; parity test fails when drop_undersized is removed from replay)
```

## Evals run

```text
PYTHONPATH=services/orion-vision-host .venv/bin/python \
  services/orion-vision-host/evals/walkway_cutoff_calibration_eval.py report --families person vehicle
  labelled frames: 98 (night=34, overcast=14, rain=15, sunny=35)
  labelled pool boxes: person {'real': 4}; vehicle {'real': 105, 'false': 1, 'dup': 1}
  (tables above)
PYTHONPATH=services/orion-vision-host .venv/bin/python \
  services/orion-vision-host/evals/walkway_cutoff_calibration_eval.py hours --cutoff 0.35 [--tiled]
  (day table above)
evals/walkway_cutoff_calibration_eval.py score /frames ... (CPU, live image, 142 frames, ~18 s/frame)
```

## Docker/build/smoke checks

```text
No image build or deploy was run (task: do not deploy/restart).
scripts/rebuild_affected_services.py --plan --base origin/main -> orion-vision-host  build
```

## Review findings fixed

Review ran in a subagent (orion-repo-agent). It found no blocker. It confirmed that `replay()` matches `runner.py` in order and semantics, and that hourly frames are byte copies of live frames.

- Finding: the 0.20 scoring floor is not measurable. Boxes under `text_threshold` 0.25 have empty labels, so the 0.20 and 0.25 rows were identical by construction, and the shipped value sits on the lowest measurable point.
  - Fix: `CUTOFFS` now starts at 0.25. `replay()` refuses a cut-off under `max(floor, text_threshold)` (only `pool` may go lower). `report` prints the empty-label count. The yaml comment and this report now state the limit and its consequence: a further drop in scores needs `text_threshold` to move too.
  - Evidence: `test_report_refuses_cutoff_below_text_threshold`; `report` prints `raw boxes with an empty label ...: 41`.
- Finding: there was no parity test between `replay()` and the real runner.
  - Fix: `test_replay_matches_real_runner` feeds the same raw boxes (whole frame plus 2 tiles, a seam pair, a tile fragment, an undersized person, a low person, a mail truck) through `_run_detect_grounding_dino` with `detect_tiles` and through `replay()`, and asserts the same kept boxes.
  - Evidence: passes. With `drop_undersized` removed from `replay()` as a mutation check, it fails.
- Finding: a second kept box on an already-seen object counted as real (tiled `real 109` against `objects 106`).
  - Fix: it now counts as `dup`.
  - Evidence: tiled 0.25 now reads real 105, dup 5. Recall is unchanged.
- Finding: `frames_with` left out frames whose only vehicle was missed at the floor.
  - Fix: it now counts missed objects.
  - Evidence: sunny frames with a vehicle now read 35, not 34 (11-03 15Z).
- Finding (latent, nit): `build_pool` keeps the higher-scoring geometry, so a partly overlapping whole-frame box could read "unlabelled".
  - Fix: documented in code. The count is 0 on the committed set, and the `unlabelled` column is printed, so a future set shows it.
- Finding: the blind-detector watch counts any label, so steady parked cars keep the walkway share near 1.0, and a blindness that hits only people will not card.
  - Fix: not in this PR. Recorded in Risks with a follow-up.
- Finding: walkway `caption_dispatch.min_score` is 0.25 and includes `vehicle`, so parked cars now pass the caption gate on triggered frames.
  - Fix: none. The same-scene gate (IoU 0.5, 300 s) caps it. Recorded as UNVERIFIED in Risks.

## Restart required

```bash
scripts/safe_docker_build.sh orion-vision-host up -d --build
```

Run from the clean primary `main` checkout after merge. The post-pull rebuild (`scripts/rebuild_services_from_git_diff.sh`) plans this same rebuild. `config/vision_profiles.yaml` is baked into the image, so a plain restart does not pick it up.

## Risks / concerns

- Severity: medium. Concern: the hourly sample is 1 frame per hour (98 labelled). Live runs ~240 detector frames per hour, so a rare false vehicle box pattern (headlight glare, a passing shadow) could exist that the sample never saw. Mitigation: after deploy, check `vision_crop_observation` walkway vehicle boxes scoring 0.25-0.35 for a day, for example with a crop-thumb contact sheet. Revert is one value.
- Severity: low. Concern: the alarm is slightly less sensitive for about 7 days (baseline from the old cut-off). Mitigation: none needed. It self-heals when the baseline window rolls over.
- Severity: low. Concern: walkway person recall is still low on hard frames (the mail carrier at 0.29). Mitigation: no change without more person evidence. Re-run this eval when more people appear in hourly frames, or label low-score person crops from the Hub.
- Severity: medium. Concern: `detection_rate:walkway` counts windows with any detection. With parked cars now seen in 81-100 % of daylight frames, the walkway share sits near 1.0, so a blindness that hits only people (or only moving things) will not card. Mitigation: follow-up to give the watch a per-label share or count only non-stationary detections. Not built here.
- Severity: low. Concern: walkway caption dispatch (`min_score 0.25`, `vehicle` in its labels) can now fire on parked cars in triggered frames. Mitigation: the same-scene gate (IoU 0.5, TTL 300 s) caps it at about 1 call per 5 min per unchanged scene. Check dispatch counts after deploy (UNVERIFIED).
- Severity: low. Concern: the shipped vehicle value sits on the lowest point the data can measure (`text_threshold` 0.25). If scores fall further, lowering `vehicle` alone does nothing. Mitigation: move `text_threshold` with it, and re-score with this eval first.
- UNVERIFIED: live share of windows with a detection after deploy, GPU time of crop embeddings after deploy, and the rhythm effect.

## PR link

(added after `gh pr create`)
