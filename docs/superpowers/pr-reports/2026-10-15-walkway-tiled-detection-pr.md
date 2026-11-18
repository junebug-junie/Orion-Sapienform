# Walkway tiled full-resolution detection

## Summary

- The walkway camera now gets a second, closer look on some frames. Once every 20 seconds, the frame router marks one walkway frame "tiled". The athena vision host then runs the person/dog detector three times on that frame: on the whole frame, as before, and on two overlapping full-resolution crops of the band holding the street and both sidewalks.
- Boxes from the crops are moved back into whole-frame pixels and merged. A box clipped by a crop edge is dropped when a whole box continues past that edge. Two halves of one object split by the crop seam are joined. Duplicates are removed per label, and the existing 60-box cap still applies. Everything downstream (zones, crop embeddings, faces, thumbnails, crop observations) sees ordinary full-frame boxes.
- cam0 and carbon never get tiles. The host also refuses to tile any frame whose short side is under 1200 px.
- New eval, `services/orion-vision-host/evals/walkway_tiled_detection_eval.py`, runs on a directory of real frames. It also flags RTSP packet-loss smear, so tiling is not judged on garbage pixels.

## Outcome moved

The detector shrinks every image to 800 px on the short side. On the 2560x1920 walkway frame, a far-sidewalk person (~50 px tall) reaches the model at ~21 px. In the band tiles the same person reaches it at ~46 px.

These counts come from real frames collected 2026-10-15, 16:00-16:58 UTC. Each tiled detection was checked against the whole-frame pass. Every "real" item below was confirmed by looking at the image.

| Frame set | Frames | Whole-frame pass | Tiled (shipped) | Extra, real | Extra, false | Lost |
|---|---|---|---|---|---|---|
| Timed sample (one frame per 20 s) | 173 | person 117, dog 5 | person 151, dog 7, stroller 2 | mail carrier behind the mailbox x3, far walker x2, walked dog x1, stroller x2, walked dog labelled "bicycle" x1 | second seated patio skeleton as "person" ~28 | 0 |
| Bursts around 30 live person/dog sightings (one frame per 6 s) | 171 | person 180, dog 9 | person 191, dog 15, stroller 4 | dogs x5 (4 eyeballed, all real, one half-hidden behind the tree), stroller x4 (same one), cyclists' bikes x2, mail carrier / far walkers ~7 | skeleton x3 | 0 |
| Juniper's own walk (known positive, 17:51-17:55 UTC) | 63 | her: 42 frames | her: 45 frames (adds 17:52:15, 17:52:23, 17:54:38) | 3 frames | 0 | 0 |

Still missed by both passes: very small leashed dogs on the far sidewalk.

The 2x2 whole-frame grid was also tried. On the 173 timed frames it added 71 "person" boxes, almost all of them the skeleton, plus porch-swing "bicycles". It found none of the far walkers the band found, and it costs more. It was not shipped.

## Current architecture

The router dispatches walkway frames (`retina_fast`, prompts `person, dog, bicycle, stroller, vehicle, package, mail truck`) to the athena `orion-vision-host`. One GroundingDINO pass runs on the whole frame, on `cuda:0` = physical GPU 0, the Tesla P4. That container has `CUDA_VISIBLE_DEVICES=0`, and the P4 sits at 100% utilisation.

Live per-task GPU time, matched to the router's dispatch log by timing: walkway ~5.4 s, cam0 ~2.3 s, identity ~0.4 s. Each runs back to back in one ~8 s cycle. The walkway sent ~600-720 frames/hour, mostly on the triggered tier in daylight, because the patio skeleton reads as `person`.

## Architecture touched

- `orion-vision-host`: the detect path factored into `app/gdino.py` (one forward pass, any number of images), new `app/tiling.py` (tile layout and merge), and `_run_detect_grounding_dino` (tiling, `[DETECT_TILES]` log line, `detect_tiles` stats block in the result).
- `orion-vision-frame-router`: `FrameDispatchPolicy.apply_tiled_detect`, `CameraState.last_tiled_detect_ts`, and `tiled=` added to the dispatch log line.
- Config: `config/vision_profiles.yaml` (`retina_detect_open_vocab` `tile_*` params) and `config/vision_frame_router.yaml` (the walkway `tiled_detect` block).

## Files changed

- `services/orion-vision-host/app/tiling.py`: new. Layout, tile-to-frame mapping, cut flags, merge.
- `services/orion-vision-host/app/gdino.py`: new. Shared GroundingDINO forward pass. Pads so there is one list per image.
- `services/orion-vision-host/app/runner.py`: tiling in the detect path, time budget, stats.
- `services/orion-vision-host/tests/test_tiled_detection.py`: new, 21 tests.
- `services/orion-vision-host/evals/walkway_tiled_detection_eval.py`: new real-frame eval, with smear flagging.
- `services/orion-vision-host/README.md`: "Tiled detection" section.
- `services/orion-vision-frame-router/app/{policy,state,dispatcher}.py`: the tiled gate.
- `services/orion-vision-frame-router/tests/test_walkway_tiled_detect.py`: new, 5 tests.
- `services/orion-vision-frame-router/README.md`: walkway tiled detection, and the intermittent-detection note.
- `config/vision_profiles.yaml`, `config/vision_frame_router.yaml`: knobs.

## Schema / bus / API changes

- Added: `request.detect_tiles` (bool, or a dict of `tile_*` overrides) inside the free-form `VisionTaskRequestPayload.request`. A `detect_tiles` stats block in the detect artifact (`VisionArtifactOutputs` allows extra keys).
- Removed / Renamed: none.
- Behavior changed: walkway detections on tiled frames include objects the whole-frame pass misses.
- Compatibility notes: no schema, registry or channel change. `VisionObject` is unchanged, and merged boxes carry no extra keys.

## Env/config changes

- Added / removed / renamed env keys: none.
- `.env_example` updated: no. Local `.env` sync: not needed.
- Config knobs (YAML, not env): `tile_grid [2,1]`, `tile_overlap 0.2`, `tile_region [0,0.15,1,0.6]`, `tile_include_full_frame true`, `tile_batch false`, `tile_min_short_side_px 1200`, `tile_contain_frac 0.7`, `tile_seam_merge_frac 0.25`, `tile_time_budget_s 10`. Router: `streams.walkway.tiled_detect {enabled: true, tiers: [baseline, triggered], min_seconds_between: 20}`.

## Tests run

```text
orion-vision-host (in the orion-vision-host-vision-host image, read-only mount):
  python3 -m pytest -q tests                         -> 291 passed, 1 skipped
orion-vision-frame-router (repo .venv, PYTHONPATH=worktree):
  pytest -q services/orion-vision-frame-router/tests -> 104 passed
Static gates (subset run locally): test_no_personal_terms, test_no_private_eval_artifacts,
  test_ci_service_image_sources, check_metric_lineage --gate, check_definition_drift --gate,
  check_service_hostname_refs, check_compose_no_relative_mounts, check_scripts_dir_no_stdlib_shadow -> all pass
CI on the PR: Static repo gates pass, hub-schedule-browser-smoke pass
```

## Evals run

```text
evals/walkway_tiled_detection_eval.py, CPU, throwaway orion-vision-host containers (read-only model cache).
Frames copied to a scratch dir outside the repo and deleted afterwards; never committed.
  timed sample 177 frames (173 readable)  --layout shipped --layout 2x2:2,2,0.2
  live-sighting bursts 171 frames         --layout shipped
  Juniper walk 114 frames (102 smeared)   --layout shipped --load-truncated
  re-run of 66 frames after the band-edge fix: duplicate person boxes 9 -> 0 on the walk frames
Smear: 0 of 173 timed frames and 0 of 171 burst frames (all before ~17:00 UTC). 102 of 114
walk frames are smeared from 5-65% of the height down. Walk counts above only use
detections whose box centre is above the smear line, and the walk boxes counted for Juniper
were all above it.
```

## Docker/build/smoke checks

```text
CPU latency, GroundingDINO-base fp32, one frame, mean:
  one container, 24 cores:      whole 4.6 s | band 2 tiles + whole 16.3 s (3.5x) | 2x2 + whole 22.3 s (4.8x)
  4 containers at once, 16 cores each: whole 7.5 s | band + whole 26.9 s (3.6x) | 2x2 + whole 37.3 s (5.0x)
  batched tiles on CPU: 2x2 25.0 s vs 22.3 s sequential, band 16.5 s vs 16.3 s -> no gain, shipped sequential
Live P4 latency: UNVERIFIED. GPU measurement would have taken VRAM from the live host. The P4 has ~2.7 GB free,
and vision-host refuses work below its hard floor, so no GPU run was done.
Bound: one P4 GroundingDINO pass at this input size is <= ~2.3 s (cam0's whole task, which includes one
pass at the same model input size). So a tiled walkway task adds <= ~6 s, once per 20 s.
```

Cost decision. "Tile only the baseline tier" would tile almost nothing in daylight, because the skeleton keeps the walkway on the triggered tier. So tiling runs on both tiers, time-gated to one frame per 20 s. That is about one walkway task in three. A far walker stays in view for ~30 s or more, so most walkers get at least one tiled look.

Expected effect on the full P4: up to ~+30% demand. The walkway and cam0 cycle may stretch from ~8 s toward ~10 s. This is UNVERIFIED until the first live hour of `[DETECT_TILES]` lines. `tile_time_budget_s: 10` stops starting new tiles once the budget is spent. Router and host timeouts are 30 s, and walkway tasks take <= ~10 s end to end today. Router in-flight limits are unchanged: one per camera, three total.

## Review findings fixed

- Finding: an empty post-process result raised IndexError and failed the whole detect step (regression from the refactor).
  - Fix: `detect_images` pads to one list per image.
  - Evidence: `test_detect_images_none_boxes_and_empty_results_mean_nothing_found`.
- Finding: the seam union chained through the growing box, which could merge two neighbours.
  - Fix: each piece is compared with the original pieces only.
  - Evidence: `test_seam_union_does_not_chain_through_the_growing_box` (would merge on the old code).
- Finding: a small cut person was dropped whenever any bigger same-label box contained it.
  - Fix: the drop applies only if that box continues past the cut edge.
  - Evidence: `test_small_cut_person_inside_big_box_that_stops_at_the_seam_survives`.
- Finding: no bound on tile time.
  - Fix: `tile_time_budget_s`. The first tile always runs.
  - Evidence: `test_time_budget_skips_remaining_tiles`.
- Finding: tiles could be set on task types with no detect step.
  - Fix: `TILED_DETECT_TASK_TYPES` guard.
  - Evidence: `test_task_type_without_detect_gets_no_tiles`.
- Finding: `gdino.py` had no direct test.
  - Fix: tests with a fake processor and model (batch order, None boxes, labels).
- Finding: the review suggested treating `tile_region` edges as frame edges.
  - Fix: applied, then reverted. Juniper's walk frames showed it double-counting a person standing close (whole box plus a copy clipped at the band edge, 9 times in 114 frames). Region edges are seams again; the crossing rule removes the clipped copy.
  - Evidence: `test_band_edge_is_a_seam_so_a_near_person_is_not_double_counted`, `test_band_edge_clipped_person_survives_when_whole_pass_missed_them`, re-run 9 -> 0.
- Finding: intermittent far detections, because only one frame per 20 s is tiled.
  - Fix: documented in the router README and the config comment. A tile-only person appears on tiled frames only, and the 8 s trigger TTL lapses between them.
- Not changed (nit): the tiled clock is stamped before `bus.publish`. A failed publish uses up one 20 s slot. Harmless.

## Restart required

```bash
# from the clean primary main checkout after merge (vision-host bakes config/vision_profiles.yaml into the image):
scripts/safe_docker_build.sh orion-vision-host up -d --build
# router code is bind-mounted; its policy file is a single-file mount -> recreate:
scripts/safe_docker_build.sh orion-vision-frame-router up -d --force-recreate
```

## Risks / concerns

- Severity: medium.
  - Concern: P4 load. Live per-pass GPU time is UNVERIFIED, and the P4 is already full, so cam0 face checks and walkway cadence may slow.
  - Mitigation: watch `[DETECT_TILES] ... tiles_s=` and `vision_task_complete` `inference_s` for an hour. Raise `min_seconds_between`, or set `tiled_detect.enabled: false` (router recreate only, no rebuild).
- Severity: low.
  - Concern: the second patio skeleton becomes a constant extra `person` on tiled frames. It is the same kind of false positive the whole-frame pass already makes for the first skeleton. It is seasonal, and the window's stationary-prop caption gate already drops it.
  - Mitigation: none in this patch.
- Severity: low.
  - Concern: GroundingDINO's garbled multi-word labels (`vehicle mail truck`, `bicycle vehicle`) appear a little more often with tiles (1-2 per 170 frames). Pre-existing.
- Severity: info.
  - Concern: the RTSP smear (UDP packet loss, bottom ~40% of many frames after ~17:00 UTC) is being fixed separately in `fix/retina-rtsp-stall-watchdog`. Tiling covers the 15-60% band, so a smeared frame still gets its upper band tiled.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2691

🤖 Generated with [Claude Code](https://claude.com/claude-code)
