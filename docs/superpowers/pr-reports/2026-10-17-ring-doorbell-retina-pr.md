## Summary

- New retina source `RETINA_SOURCE_TYPE=ring`. Juniper's Ring doorbell is battery-powered with no Ring Protect plan, so it has no RTSP, no recordings, and no snapshot while it records. On each Ring motion/ding push event, the source opens one short WebRTC live view (15 s). The capture loop samples it at 2 frames/s. **Never continuous video.**
- New `orion-vision-retina-doorbell` instance (profile `doorbell`, stream/camera id `doorbell`, athena, port 8029). One-time login: `python -m app.ring_login`.
- The doorbell gets the walkway's downstream treatment: router policy (detection, frame and crop embeddings, crop faces, Qwen captions on the T10 lane), a zone, caption broadcast and gate, motion stories, situation / reverie / curiosity street lines, the Hub frame cache plus a "Doorbell (events)" live view, and a camera description.
- Situation context now shows every camera that has something to say, doorbell first. Before, only the first camera spoke, so a busy walkway hid a visitor at the door.
- Changes for a sparse stream: no tiling, `every_n_frames: 1`, no head-count presence, and no substrate surprise node (the surprise is scored and logged only). A quiet doorbell reads "waiting", not "broken".

## Outcome moved

Orion can now see who comes to the front door, through the same detection, identity, caption and narrative pipeline as the walkway. Before this, the doorbell was invisible to Orion.

## Current architecture

`orion-vision-retina` instances (RTSP / webcam / folder) publish `vision.frame.pointer` on `orion:vision:frames`. The frame router, vision host, Qwen caption lane, vision window, sql-writer reducers, substrate and situation context then process the frames per `stream_id`. Several downstream consumers carried `walkway`-only stream lists. `_build_street_fields` returned only the first stream that had lines.

## Architecture touched

- `orion-vision-retina`: the new source, the login CLI, health fields, and a compose service.
- Config: `vision_frame_router.yaml` `streams.doorbell`, and `vision_zones.yaml` `streams.doorbell`.
- Stream lists in orion-vision-window, orion-vision-host (circe-qwen compose), orion-sql-writer, orion-thought, orion-hub, and orion-cortex-exec.
- `orion-substrate-runtime`: new log-only stream set.
- `orion/situational`: `context.py` (multi-camera street lines) and `camera_eyes.py` (doorbell description, `event_driven` rendering).
- Hub live-view JS and template.

## Files changed

- `services/orion-vision-retina/app/ring_source.py`: the event source. It covers push events, live-view bursts, a supervisor for the push link, and credential files.
- `services/orion-vision-retina/app/ring_login.py`: one-time interactive Ring login.
- `services/orion-vision-retina/app/{sources,main,settings,health}.py`: factory branch, idle-is-not-failure, `RETINA_RING_*` settings, and `ring_*` health details.
- `services/orion-vision-retina/{docker-compose.yml,.env_example,requirements.txt,README.md}`: the doorbell instance, `DOORBELL_*` keys, `ring_doorbell` / `aiortc` / `av` / `firebase-messaging` pins, and docs.
- `config/vision_frame_router.yaml`, `config/vision_zones.yaml`: doorbell policy and zone.
- `orion/situational/context.py`: every speaking street camera gets its own part. A failed read no longer stops the loop or gets reported as ok.
- `orion/situational/camera_eyes.py`: doorbell description, and `event_driven` wording.
- `services/orion-substrate-runtime/app/{worker,settings}.py`, compose, `.env_example`: `SUBSTRATE_PERCEPTION_LOG_ONLY_STREAM_IDS=doorbell`.
- Stream-list defaults: orion-vision-window, orion-vision-host, orion-sql-writer, orion-thought, orion-hub, orion-cortex-exec (settings, `.env_example`, compose).
- `services/orion-hub/static/js/vision-live-frame.js`, `templates/index.html`: Doorbell live view.
- Tests: `services/orion-vision-retina/tests/test_ring_source.py` (new, 21 tests), plus updated defaults and new cases in the situational, substrate, hub, frame-router and vision-host tests.

## Schema / bus / API changes

- Added: none. `VisionFramePointerPayload` is unchanged. No new channel. `CameraSenseV1` is unchanged (the event-camera wording lives in the rendering layer).
- Behavior changed: `PerceptionContextV1.street_summary` can now hold several framed camera parts. `street_stream_id` is the first camera that spoke.
- Compatibility notes: none needed. Nothing new crosses the bus.

## Env/config changes

- Added keys:
  - orion-vision-retina: `DOORBELL_RING_DEVICE`, `DOORBELL_RETINA_FPS`, `DOORBELL_RETINA_JPEG_QUALITY`, `DOORBELL_RING_EVENT_KINDS`, `DOORBELL_RING_BURST_SECONDS`, `DOORBELL_RING_COOLDOWN_SECONDS`, `DOORBELL_RETINA_HTTP_PORT`.
  - Retina settings: `RETINA_RING_STATE_DIR`, `RETINA_RING_EVENT_KINDS`, `RETINA_RING_BURST_SECONDS`, `RETINA_RING_COOLDOWN_SECONDS`.
  - orion-substrate-runtime: `SUBSTRATE_PERCEPTION_LOG_ONLY_STREAM_IDS`.
- Changed defaults (doorbell added):
  - `WINDOW_CAPTION_GATE_STREAMS`, `CIRCE_QWEN_VISION_ARTIFACT_BROADCAST_STREAMS`, `VISION_MOTION_STORY_STREAMS`, `ORION_REVERIE_STREET_STREAM_IDS`, `HUB_CURIOSITY_STREET_STREAM_IDS`.
  - `ORION_SITUATION_STREET_STREAM_IDS=doorbell,walkway`.
  - `VISION_FRAME_CACHE_STREAM_IDS=carbon,walkway,doorbell`.
  - `ORION_CAMERA_DESCRIPTIONS` (+doorbell, `event_driven`).
- Removed / renamed keys: none.
- `.env_example` updated: yes.
- Local `.env` synced:
  - New keys were added with `python scripts/sync_local_env_from_example.py --all-keys <services>`.
  - The changed stream lists and `ORION_CAMERA_DESCRIPTIONS` were edited in place. Each local value equalled the old template exactly, so each was a stale default, not an override.
  - A final sync run reports no divergence on these keys.
- Skipped keys requiring operator action: `DOORBELL_RING_DEVICE` is left empty, which means the first camera on the Ring account. Set it to the doorbell's Ring app name if the account has more than one camera.

## Tests run

```text
retina tests (CI recipe, requirements installed):  101 passed
orion/situational + orion/vision:                   256 passed
orion-vision-window: 168 passed | orion-vision-frame-router: 108 passed | orion-dream: 270 passed
substrate walkway split + receipt: 16 passed | hub vision_live_frame: 37 passed
hub curiosity street: 4 passed | thought reverie street: 15 passed | metric_rest_range: 25 passed
endogenous_curiosity_walkway_fence: 3 passed | circe_qwen_lane_compose: 6 passed
node --test vision-live-frame.test.js: 11 pass
scripts/check_env_template_parity.py: PASS (94 services)
```

The whole orion-thought and orion-sql-writer suites each have one failure that this PR does not touch: `test_settings_mind_enrichment` and `test_biometrics_induction_timestamp_format`, both environment-sensitive.

## Evals run

```text
No eval harness covers the Ring source. Real-aiortc loopback in the built image
(the actual _burst code against an aiortc peer that sends video):
frames read 8 [(640, 480), (640, 480)] failed 0 err None closed 1
```

Follow-up: once real doorbell frames exist, add a caption and detection eval on them, the way `walkway_caption_eval.py` covers the walkway.

## Docker/build/smoke checks

```text
docker build -f services/orion-vision-retina/Dockerfile .   -> ok (python 3.10)
in-image: aiortc 1.15.0, av 17.1.0, numpy 1.26.4 import; offer has 2 m-lines;
Ring session-id parse ok; create_frame_source('ring') -> RingEventFrameSource
```

Live Ring path: **UNVERIFIED**. It needs Juniper's Ring login and a real event.

## Review findings fixed

- Finding H1: if Ring didn't answer, the live-view session was never closed. The camera could keep streaming to nobody.
  - Fix: switched to `generate_async_webrtc_stream`, which registers the session before it talks to Ring. The burst code applies its own 10 s answer timeout and closes the session in `finally` on every path. Trickled ICE candidates are handled.
  - Evidence: `test_no_answer_is_named_failed_and_still_closed`, `test_ring_errors_are_failed_and_still_closed[raise|error]`, and the loopback.
- Finding H2: a dead push link looked healthy forever.
  - Fix: a `_supervise` loop checks the FCM receiver every 60 s, reports `ring push listener stopped`, and rebuilds the link. It also reconnects fully every 12 h.
  - Evidence: `test_supervisor_reconnects_a_dead_push_listener`, `test_listener_alive_trusts_the_receiver_not_the_started_flag`.
- Finding H3: the 520-character cut in chat_stance could drop the doorbell line.
  - Fix: the default order is now `doorbell,walkway`.
  - Evidence: `test_doorbell_line_survives_a_long_walkway_line_cut_at_520`.
- Finding M1: the doorbell overwrote the walkway surprise node.
  - Fix: `SUBSTRATE_PERCEPTION_LOG_ONLY_STREAM_IDS=doorbell`. The doorbell is scored, logged and baselined, and feeds no node.
  - Evidence: `test_doorbell_is_scored_and_baselined_but_feeds_no_node`.
- Finding M2: the brief would call a quiet doorbell broken.
  - Fix: a description flag `event_driven`. The line now reads "sends frames only when something happens there; last time 3 h ago".
  - Evidence: `test_quiet_doorbell_reads_waiting_not_broken`.
- Finding M3: Ring's 30 s keep-alive could cut a long burst, and that was counted as a failure.
  - Fix: `keep_alive_timeout = burst + 30`. A stream that ends after frames arrived is not a failure.
  - Evidence: `test_stream_ending_early_after_frames_is_not_a_failure`.
- Finding M4: `stop()` didn't wait for the burst cleanup.
  - Fix: it now gathers the cancelled tasks before closing auth.
  - Evidence: `test_stop_mid_burst_waits_for_cleanup`.
- Finding M5: no tests covered the burst path.
  - Fix: 8 fake-device tests, plus the real loopback.
- Finding L1: a street-read error stopped the loop and was then overwritten with "ok".
  - Fix: `continue`, and the error status stays visible.
  - Evidence: `test_one_failed_street_read_does_not_hide_the_other`.
- Finding L2: other retina containers could read the Ring token.
  - Fix: a dedicated `/mnt/telemetry/vision/ring_state` mount, used only by the doorbell container.
  - Evidence: a compose test.
- Finding L3: temp-file race on credential writes.
  - Fix: `mkstemp` + `fchmod 0600`.
- Finding L4: the login text and the docs disagreed on restart.
  - Fix: aligned (no restart needed).
- Finding L5: old-format ding pushes are silently mislabelled by the library.
  - Fix: ignored kinds are now logged at INFO.
- Finding L6: `firebase-messaging` was not pinned.
  - Fix: pinned at 0.4.5.

## Metric gate (substrate input)

No new metric. The doorbell's frame-embedding surprise uses the existing `perception_prediction_error()` with its own per-stream baseline. It is logged only.

- Independence: the doorbell is a separate sensor, but its sparse bursts give a baseline only after events. That biases it toward "something is there".
- Live sanity check: impossible until real bursts exist.

So it deliberately feeds no node. A doorbell node needs the full gate on real data first.

## Restart required

First deploy the merged main from the primary checkout (athena). Then:

```bash
scripts/safe_docker_build.sh orion-vision-retina --profile doorbell up -d --build orion-vision-retina-doorbell
docker exec -it orion-vision-retina-doorbell python -m app.ring_login
```

The services whose stream lists or code changed pick them up on the post-pull rebuild: orion-vision-window, the orion-vision-host circe-qwen lane, orion-sql-writer, orion-substrate-runtime, orion-thought, orion-hub, orion-cortex-exec, and orion-vision-frame-router (config).

## Risks / concerns

- Severity: medium.
  - Concern: the live path is unverified. Ring may answer an `aiortc` offer differently than the Home Assistant / go2rtc clients this library is used with.
  - Mitigation: a failed burst is visible (`ring_bursts_failed`, health degraded, a named `last_error`). The loopback proves the local half.
- Severity: medium.
  - Concern: battery. Each event wakes the camera for about 15 s of live view.
  - Mitigation: one burst at a time and a 30 s cooldown. Tune `DOORBELL_RING_BURST_SECONDS` and `DOORBELL_RING_COOLDOWN_SECONDS`, or set `DOORBELL_RING_EVENT_KINDS=ding` for rings only.
- Severity: medium.
  - Concern: privacy. Crop faces of door visitors are embedded, the same as the walkway.
  - Mitigation: Juniper asked for the walkway's full treatment. The walkway face policy (2026-10-15) applies.
- Severity: low.
  - Concern: the zone is the whole frame until a real doorbell frame is traced. Rhythm forecasts and grading are not wired for the doorbell, because a burst stream never reaches the 0.8 coverage that grading needs.
  - Mitigation: follow-ups after the first real frames.

## PR link

<link>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
