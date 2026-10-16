# fix(vision): walkway captions never landed -- Qwen said "nothing notable" to every frame

## Summary

- The walkway camera's caption lane (Qwen2-VL-2B on the T10) answered "nothing notable" to every task: 24 of 24 on 2026-10-16, 44 of 50 on 2026-10-15. Zero captions reached the window.
- Cause, shown live: the prompt asked an existence question ("Are there any people, animals or vehicles ...? If none, reply only: nothing notable"). Qwen2-VL-2B answers "nothing notable" to that even on a clear 448 px crop of a parked car, and describes the same crop correctly when asked to describe it. The prompt was chosen on empty night frames only, so the eval never saw a frame with something in it.
- Second cause: the model reads the whole 2560x1920 frame shrunk to about 1 MP, so a person or car down the street is a few pixels.
- Fix: the router now sends the detector's boxes with each caption task, the host captions a padded crop around them, and the prompt asks for a description with no "nothing here" escape answer.
- So this cannot hide again: a rejected caption logs the model's raw reply, and the router keeps a "no caption in a row" streak and logs a warning every 10.

## Outcome moved

Walkway caption yield: 0% before. On a live clean frame (2026-10-16 ~19:50 UTC), with the old prompt the live lane said `nothing notable`. With the new crop and prompt it said: "A car is driving down a street in a residential area." (The car is parked. The motion part is wrong, see Risks.) Live yield after deploy: UNVERIFIED until the router and the Qwen lane are redeployed.

## Current architecture

vision-frame-router `caption_dispatch` (config/vision_frame_router.yaml `streams.walkway`) sends a `caption_frame` task to `orion:exec:request:VisionHostService:circe-vl` after a triggered detect reply finds a gate label. orion-vision-host (Qwen lane) runs `_run_caption_frame`. A reply that exactly matches `caption_null_answers` becomes `rejected_reason="null_answer"`. orion-vision-window's caption gate counts it as `host:null_answer`.

## Architecture touched

- vision-frame-router: caption task request gains `caption_crop_boxes` and `caption_crop_pad`. There is a new health detail, `caption_no_caption_streak`, and a new warning log line, `caption_lane_silent`.
- orion-vision-host: new `app/caption_crop.py`. `_run_caption_frame` crops before generation, returns `crop_box`, and logs the raw reply when it rejects a caption.
- config: walkway `caption_dispatch` sets `crop_to_detections: true` and `crop_pad: 0.5`, uses the new prompt, and has no `null_answers`.

## Files changed

- `config/vision_frame_router.yaml`: crop on, describe prompt, null answer removed, and the incident written into the comment.
- `services/orion-vision-frame-router/app/policy.py`: sends the crop boxes.
- `services/orion-vision-frame-router/app/metrics.py`, `app/dispatcher.py`: the no-caption streak and its warning.
- `services/orion-vision-frame-router/tests/test_walkway_caption_dispatch.py`: real-config assertions, crop-box request, streak.
- `services/orion-vision-frame-router/README.md`, `services/orion-vision-host/README.md`: docs.
- `services/orion-vision-host/app/caption_crop.py`: the crop helper (new).
- `services/orion-vision-host/app/runner.py`: crop wiring, `crop_box`, raw-reply log.
- `services/orion-vision-host/tests/test_caption_crop.py` (new), `tests/test_run_caption_frame_walkway.py`: regression tests.
- `services/orion-vision-host/evals/walkway_caption_eval.py`: `--crop-box`. `null_answers` is now optional.

## Schema / bus / API changes

- Added: optional `request.caption_crop_boxes` (list of `[x1,y1,x2,y2]` frame pixels) and `request.caption_crop_pad` (float) on `caption_frame`. `VisionTaskRequestPayload.request` is `Dict[str, Any]`, so there is no schema change.
- Added: `crop_box` key in the host's caption_frame result dict. Not copied into `VisionArtifactPayload`.
- Added: `caption_no_caption_streak` in the router's `system.health.v1` details.
- Removed / renamed: none.
- Behavior changed: walkway captions describe a crop. A walkway task no longer sends `caption_null_answers`.
- Compatibility: a host without this patch ignores the crop fields and captions the whole frame. Every other `caption_frame` caller (cam0 BLIP, orion-thought re-observation) sends no boxes, so it is unchanged.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed.
- skipped keys: none.

## Tests run

```text
services/orion-vision-frame-router: pytest tests -q  -> 108 passed
services/orion-vision-host (in the vision-host image, pytest pip-installed in a --rm container): pytest tests -q -> 364 passed, 2 skipped
Regression: with the old config, test_caption_task_carries_the_detector_boxes_for_a_crop failed (no caption_crop_boxes).
```

## Evals run

```text
evals/walkway_caption_eval.py --mode qwen (CPU, real Qwen2-VL-2B weights, 2 live clean walkway frames, 2026-10-16):
  --crop-box 1280,454,1408,518 -> "A car is driving down a street in a residential area, passing by a house with a white fence and a tree."  crop_box [1120,262,1568,710]
  no crop                      -> long whole-scene description (timestamp overlay, porch, street)

Live lane, same frame family, via POST :6602/v1/vision/task (read-only inference, no stream_id, so no broadcast):
  old prompt, full frame        -> "nothing notable"
  old prompt, 800x400 car crop  -> "nothing notable"
  "Describe this image..." crop -> "A car is parked on a street in a residential neighborhood ..."
  "What color is the car?" crop -> "The car in the image is gray."   (so the model sees the image)
  new prompt, chair crop (the 'skeleton' false-positive box) -> "A wooden chair is placed on a grassy lawn ..."
```

No positive frame with a walking person came through during the session (the walkway was quiet 19:33-20:05 UTC). A person-on-the-walkway caption is still UNVERIFIED.

## Docker/build/smoke checks

```text
No deploy or restart from this session (task rule). Tests ran in the existing orion-vision-host-qwen image.
```

## Review findings fixed

Review ran in a subagent (orion-repo-agent). It found no blockers. It confirmed that the detector boxes are full-frame pixels of the same image the host loads. Gates it ran: `check_metric_lineage.py --gate` PASS and `check_metric_dead_wiring.py` clean.

- Finding: the crop was the union of every box. A person at one edge plus a prop or parked car at the other made the crop nearly frame-wide, and the description drifted to the prop.
  - Fix: the crop is now anchored on the highest-score box. Other boxes count only if they touch its padded region.
  - Evidence: `test_far_apart_boxes_crop_to_the_top_box_only`, `test_nearby_boxes_are_cropped_together`.
- Finding: the streak is one counter for the whole router, but the warning printed `stream_id`, which read like a per-stream count. Only the arithmetic was tested.
  - Fix: the warning now says `(all captioned streams) last_stream_id=`.
  - Evidence: a new dispatcher-level test sends 12 null replies and checks that exactly one `caption_lane_silent streak=10` line is logged.
- Finding: the eval used one fixed box for every frame.
  - Fix: a per-frame `<frame>.boxes.json` sidecar, and `--crop-box` is checked for exactly 4 numbers.
  - Evidence: the eval ran on CPU (see Evals).
- Finding: the raw-reply log would also fire on cam0 BLIP rejections.
  - Fix: it now logs only for caller-prompted tasks.
  - Evidence: host suite 364 passed.
- Finding (nits): the constant sat inside the import block, and the docs did not say that the pad applies on all sides or that `crop_box: null` means a whole-frame fallback.
  - Fix: all done.
- Finding NOT fixed (accepted, listed in Risks): with no null answer, a detector false positive becomes a confident caption of whatever is in the crop. The window gate trusts the same detector, so it is not an independent check.

## Restart required

Run from the clean primary `main` checkout on athena after merge. The router rebuilds post-pull. The Qwen lane does not: its compose file is outside post-pull coverage and outside `safe_docker_build.sh`.

```bash
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-vision-frame-router up -d --build
cd /mnt/scripts/Orion-Sapienform && docker compose --env-file .env --env-file services/orion-vision-host/.env -f services/orion-vision-host/docker-compose.circe-qwen.yml up -d --build
```

The second command also picks up #2697's corrected `CUDA_VISIBLE_DEVICES` for the T10. Before running it, confirm the index with `nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv` (T10 = 00000000:D8:00.0).

## Risks / concerns

- Severity: medium. Concern: Qwen2-VL-2B adds motion it cannot see. A parked car comes back as "driving". Mitigation: orion-vision-window's stationary gate already drops captions whose detections are all fixtures. Follow-up: tune the prompt on real moving-subject frames.
- Severity: medium. Concern: there is no "nobody here" answer any more. A detector false positive (a 0.25-score `vehicle` on a shadow, the patio chair or skeleton read as `person`) now gets a confident caption of whatever is in the crop. Live: the chair box got "A wooden chair is placed on a grassy lawn ...". orion-vision-window's gate trusts the same detector, so it is not an independent check. It only drops boxes that are stationary for long enough. Mitigation now: the stationary gate. Follow-up: collect real true-positive and false-positive walkway frames with their boxes (the eval sidecar), then pick either a higher caption `min_score` or a VQA second opinion on the crop, based on that data.
- Severity: low. Concern: the Qwen lane container is still outside post-pull rebuild coverage. Adding it needs `safe_docker_build.sh` to target a second compose file, which touches the deploy safety wrapper. It is left as the existing open follow-up, not done here.

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
