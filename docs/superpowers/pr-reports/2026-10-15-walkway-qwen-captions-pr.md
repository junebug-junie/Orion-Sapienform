## Summary

- Walkway captions now come from the idle Qwen2-VL-2B lane on athena's T10, not from BLIP-base. BLIP is switched off for the walkway (`want_caption: false`); cam0 and carbon are unchanged.
- The frame router asks Qwen for a caption only after the detector found a person, dog, bicycle, stroller, vehicle, package or mail truck in that frame, with an instruction prompt that tells it to describe only those, or answer "nothing notable". That answer becomes no caption.
- A new gate in orion-vision-window lets a walkway caption into the window summary (and so into the council narrative, `vision_events`, the chat street line) only if the detections of its own frame include something that is not a stationary prop, and only if it arrived within 20 s of the frame.
- Every rejection is kept as a rejected caption (empty text, confidence 0.0, `gate:<reason>`), counted (`summary.caption_gate`, window metrics, router heartbeat) and logged. Nothing is published as an empty caption at confidence 1.0.
- Small eval replays real walkway frames through the host's own caption code, BLIP vs Qwen with the shipped prompt.

## Outcome moved

The walkway street line stops repeating BLIP's hallucination. Last 24 h before this patch: 30 of 69 walkway `vision_events` narratives were the "camera on a bench" family ("Camera is placed on a bench in the yard.", "A camera is positioned on the ground near a bench."). Under this patch a frame with no detections never reaches a captioner, and a caption without agreeing detections is dropped at the window.

## Current architecture

Walkway frames -> frame router -> athena vision host `retina_fast` (GroundingDINO + SigLIP + BLIP caption on the triggered tier) -> `orion:vision:artifacts` -> orion-vision-window (30 s windows, `summary.captions`) -> orion-vision-council (narrative; falls back to `captions[0]` for `scene_summary`) -> `vision_events` -> chat situation brief / reveries / curiosity. The Qwen2-VL lane (`orion-athena-vision-host-qwen`, channel `orion:exec:request:VisionHostService:circe-vl`) served only orion-thought's dream re-observation (37 calls in 48 h, ~3 s each on the T10), had no frames mount and its artifact broadcast off.

## Architecture touched

- orion-vision-frame-router: new caption lane (`PendingTask.lane="caption"`), chained after the primary detect reply; own in-flight cap, per-camera clock, per-task timeout; host in-flight cap now counts host-lane tasks only.
- orion-vision-host: caller prompt on instruction-tuned VLMs, null-answer mapping, stream-scoped artifact broadcast; qwen compose mounts frames read-only and queues up to 2.
- orion-vision-window: caption gate before the summary is built.
- Contract: frame router added as a producer on `orion:exec:request:VisionHostService:circe-vl`; qwen lane now a walkway-scoped producer on `orion:vision:artifacts`.

## Files changed

- `config/vision_frame_router.yaml`: walkway `want_caption: false`; `caption_dispatch` block (labels, min_score, 8 s rate, 1 in flight, 15 s timeout, prompt, null answer).
- `services/orion-vision-frame-router/app/{dispatcher,policy,state,metrics,settings,grammar_emit}.py`: caption dispatch, lane accounting, outcome counters, `CHANNEL_CAPTION_INTAKE`, caption yield counts secondary replies.
- `services/orion-vision-frame-router/{.env_example,README.md}`, tests `test_walkway_caption_dispatch.py`, `test_grammar_emit.py`, `test_walkway_policy_and_expectation.py`.
- `services/orion-vision-host/app/{vlm_family,runner,caption_sanitize,main,settings}.py`: `resolve_caption_prompt`, `matches_null_answer`, prompt-aware echo check, `artifact_broadcast_allowed`, `VISION_ARTIFACT_BROADCAST_STREAMS`.
- `services/orion-vision-host/{.env_example,docker-compose.circe-qwen.yml,README.md}`, tests `test_walkway_caption_prompt.py`, `test_artifact_broadcast_streams.py`, `test_circe_qwen_lane_compose.py`; eval `evals/walkway_caption_eval.py`.
- `services/orion-vision-window/app/{caption_gate,fixtures,main,settings}.py`, `.env_example`, `docker-compose.yml`, `README.md`, test `test_caption_gate.py`.
- `orion/bus/channels.yaml`, `orion/schemas/vision.py` (comment only: `null_answer` reason), `config/metrics/metric_definitions.lock.json` (routing change on the circe-vl channel).

## Schema / bus / API changes

- Added: frame router publishes `caption_frame` `VisionTaskRequestPayload` on `orion:exec:request:VisionHostService:circe-vl`. New request keys `caption_prompt`, `caption_null_answers`; new meta keys `caption_secondary_dispatch`, `source_task_correlation_id`, `source_objects` (all inside existing free-form dicts).
- Added: qwen lane broadcasts `VisionArtifactPayload` on `orion:vision:artifacts` for `stream_id=walkway` only.
- Added: `summary.caption_gate` (`accepted`, `rejected` by reason) on walkway window payloads (free-form summary dict).
- Removed / renamed: none.
- Behavior changed: walkway captions are Qwen-only and gated; `VisionCaption.rejected_reason` gains `null_answer` and `gate:*` values.
- Compatibility: additive. Council reads only `summary.captions`, which simply no longer carries rejected text.

## Env/config changes

- Added keys: `CHANNEL_CAPTION_INTAKE` (frame router); `WINDOW_CAPTION_GATE_STREAMS`, `WINDOW_CAPTION_GATE_LABELS`, `WINDOW_CAPTION_GATE_MIN_SCORE`, `WINDOW_CAPTION_MAX_AGE_SEC` (window); `VISION_ARTIFACT_BROADCAST_STREAMS`, `CIRCE_QWEN_VISION_ARTIFACT_BROADCAST_STREAMS` (vision host).
- Removed / renamed keys: none.
- `.env_example` updated: yes, all three services.
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-vision-frame-router orion-vision-window orion-vision-host`: yes, 7 keys added.
- skipped keys requiring operator action: none.

## Tests run

EVAL_AND_TESTS_PLACEHOLDER

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

Deploy from the clean primary `main` checkout after merge. The router's policy file is a single-file bind mount, so the router must be recreated (not just restarted) to see the new YAML.

```bash
scripts/safe_docker_build.sh orion-vision-frame-router up -d --build --force-recreate
scripts/safe_docker_build.sh orion-vision-window up -d --build
docker compose --env-file .env --env-file services/orion-vision-host/.env -f services/orion-vision-host/docker-compose.circe-qwen.yml up -d --build
```

The main athena `orion-vision-host` needs no restart for behavior (its code change is inert while `VISION_ARTIFACT_BROADCAST_ENABLED=true`). Order: qwen lane first (so it can read frames and broadcast walkway), then window (gate), then router (starts sending caption tasks).

## Risks / concerns

RISKS_PLACEHOLDER

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
