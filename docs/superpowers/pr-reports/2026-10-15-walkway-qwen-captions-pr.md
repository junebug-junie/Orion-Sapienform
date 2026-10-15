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

```text
PYTHONPATH=.:services/orion-vision-frame-router pytest services/orion-vision-frame-router/tests -q   -> 99 passed
PYTHONPATH=.:services/orion-vision-window pytest services/orion-vision-window/tests -q               -> 159 passed
vision-host tests in the orion-vision-host image (read-only mount, + pytest-asyncio)                   -> 270 passed, 1 skipped
  (without pytest-asyncio, as the image ships: 4 async tests fail identically on origin/main -- baseline, not this branch)
scripts/check_metric_lineage.py --gate / --prompt-semantics                                           -> PASS / PASS
scripts/check_definition_drift.py --gate (after --update for the real circe-vl routing change)         -> PASS
scripts/check_env_template_parity.py                                                                  -> PASS
ORION_BUS_URL=redis://100.92.216.81:6379/0 scripts/check_single_consumer_channels.py (read-only)       -> OK
check_bus_reply_channels, check_inner_state_registry, check_service_hostname_refs, compose mount gates,
check_system_health_producers, check_async_routes_not_blocking, check_circe_worker_refs                -> all OK
pytest tests/test_no_personal_terms.py tests/test_check_settings_defaults.py tests/test_compose_logging_journald.py
  tests/test_grammar_event_producer_catalog.py tests/test_no_private_eval_artifacts.py
  tests/test_metric_prompt_semantics.py tests/test_metric_rest_range.py                               -> passed
git diff --check origin/main...HEAD                                                                   -> clean
```

New tests include tonight's real case: no detections + "A camera is placed on a wooden bench in a garden" -> dropped (`test_tonights_blip_caption_on_an_empty_dark_frame_is_dropped`, and end to end through the window flush in `test_flush_keeps_rejected_caption_out_of_summary_and_counts_it`).

## Evals run

`services/orion-vision-host/evals/walkway_caption_eval.py` replays real walkway frames through the host's own `_run_caption_frame`. 8 frames copied from `/mnt/telemetry/vision/frames/walkway_*` at 2026-10-15 01:04 MDT (night; empty walkway, two Halloween skeletons at the patio table -- the same scene the live 30/69 hallucinations came from). Frames not committed. Run on CPU in one-off containers from the live images, read-only mounts: the T10 had ~2.8 GB free with the live Qwen loaded, not enough for a second copy, and the running Qwen container cannot take a custom prompt until this code is deployed.

| Captioner | Prompt | Result on 8 night frames | Becomes a caption |
|---|---|---|---|
| BLIP-base (today) | none | "a camera is on the side of a building", "a camera is on a wooden bench with a basket on the ground", "a camera is sitting on a bench", "a camera sitting on top of a bench", "a camera is shown with a small camera", "a camera is on the ground in the dark", "a camera is positioned on a table outside", "a camera is sitting on a bench next to a bench" | 8/8, all at confidence 1.0 |
| Qwen2-VL-2B | first draft: "Describe only people, animals and vehicles ... If there are none, answer exactly: nothing notable." | "There are no people, animals, or vehicles visible on the walkway or street." (8/8) | 8/8 -- a caption that says nothing |
| Qwen2-VL-2B | trial: same + "Statues, decorations and pictures do not count ... your whole answer must be: nothing notable" | "There are no people, animals, or vehicles visible in the image." (2/2) | 2/2 |
| Qwen2-VL-2B | trial: "Reply with exactly "nothing notable" unless you can clearly see a real person ..." | "nothing notable" (2/2) | 0/2 |
| Qwen2-VL-2B | **shipped**: "Are there any people, animals or vehicles in this image? If there are none, reply with only: nothing notable. If there are, describe them and what they are doing in one short sentence." | "nothing notable" (8/8) | 0/8 (no caption, `null_answer`) |

Qwen never mistook the skeletons for people; BLIP never noticed them and invented a camera every time. CPU latency ~30 s/frame is a CPU artifact; the live lane logs 2.2-3.8 s per caption on the T10.

**Not covered (UNVERIFIED):** a positive frame (a real passer-by, dog or car). It was night, the walkway was empty, and no bundled sample photo exists on the host. Whether the shipped prompt still describes a real person in one sentence, rather than answering "nothing notable", is unmeasured until daytime. Re-run the eval on daytime frames before relying on walkway captions.

## Review findings fixed

Independent review by a fresh subagent (read-only). No MUST findings; it traced no path for a BLIP or stale caption to reach the walkway summary. SHOULD/NIT findings, all fixed:

- Finding: the patio skeleton reads as `person` (0.37) all night, so the router would send Qwen a caption task every 8 s (~450/hour) on a lane shared with dream re-observation.
  - Fix: router `same_scene` skip -- no new caption while every gate box matches the last captioned frame's (same label, IoU >= 0.5) within 300 s. README documents the gate's 30-minute cold start after a window restart.
  - Evidence: `test_same_scene_is_not_recaptioned_until_it_changes_or_ages_out`.
- Finding: caption-lane replies were added to the vision organ's `replies_ok`, failures/timeouts were not counted as requested, so a down lane read as "never asked" and healthy replies diluted athena-host failure pressure.
  - Fix: `record_caption_lane` counts `caption_requested` at dispatch and `captions` on a real caption; `replies_ok`/`failed` stay the athena host's. Lane failures go to router metrics.
  - Evidence: `test_caption_lane_yield_counts_at_dispatch_and_leaves_host_counts_alone`.
- Finding: if the qwen lane ever ran a BLIP model, the walkway prompt would be dropped with a warning and BLIP's own caption published at 1.0.
  - Fix: a caller prompt on a non-instruction model returns no caption (`caller_prompt_unsupported`) without generating.
  - Evidence: `test_blip_on_the_lane_returns_no_caption_not_its_own`.
- Finding: caption-only artifacts counted toward `item_count`/scene-inventory `frame_count`, and a late caption alone in a window would vote "saw nobody" into belief/presence.
  - Fix: `is_caption_only` excluded from `item_count`; a window with no detection pass skips belief/presence.
  - Evidence: `test_caption_only_window_is_not_a_look_at_the_scene`.
- Finding (NIT): missing `frame_ts` skipped the staleness check. Fix: rejects as `no_frame_ts`. Evidence: `test_caption_without_frame_ts_is_dropped`.
- Finding (NIT): router timeout 15 s vs window max age 20 s disagreed. Fix: both 20 s; a reply after the timeout is ignored. Evidence: `test_caption_lane_timeout_uses_its_own_timeout_and_counter`.
- Finding (NIT): mixed-case stream id disabled prop tracking. Fix: normalised in `CaptionGate.apply`. Evidence: `test_mixed_case_stream_id_still_checks_props`.
- Finding (NIT): misleading compose comment, runner-level test gap. Fix: comment corrected; `tests/test_run_caption_frame_walkway.py` covers prompt, null answer, BLIP refusal, cam0 unchanged.

Separately, the eval itself found the first-draft prompt failed (Qwen said "There are no people..." instead of the null answer, 8/8); the prompt was replaced by one that measured 8/8 correct, without adding phrase lists.

## Restart required

Deploy from the clean primary `main` checkout after merge. The router's policy file is a single-file bind mount, so the router must be recreated (not just restarted) to see the new YAML.

```bash
scripts/safe_docker_build.sh orion-vision-frame-router up -d --build --force-recreate
scripts/safe_docker_build.sh orion-vision-window up -d --build
docker compose --env-file .env --env-file services/orion-vision-host/.env -f services/orion-vision-host/docker-compose.circe-qwen.yml up -d --build
```

The main athena `orion-vision-host` needs no restart for behavior (its code change is inert while `VISION_ARTIFACT_BROADCAST_ENABLED=true`). Order: qwen lane first (so it can read frames and broadcast walkway), then window (gate), then router (starts sending caption tasks).

## Risks / concerns

- Severity: medium. Concern: positive frames are unmeasured (see Evals). The question-first prompt could bias Qwen toward "nothing notable" when someone is there, which would cost captions, never add junk. Mitigation: re-run `evals/walkway_caption_eval.py` on daytime walkway frames; watch `caption_no_caption_counts.null_answer` vs `caption_captioned_total` in the router heartbeat after deploy.
- Severity: low. Concern: prop memory is in-memory; for 30 min after a window restart the skeleton is "moving", so a non-null Qwen caption of it would pass. Mitigation: Qwen answered "nothing notable" for it 8/8; `same_scene` limits asks to one per 5 min.
- Severity: low. Concern: the T10 lane is shared with orion-thought's dream re-observation. Mitigation: router sends at most 1 in flight, 1 per 8 s while something new is on the walkway; host now queues 2 instead of refusing; orion-thought's caption timeout is 60 s.
- Severity: low. Concern: a passer-by who appears only on the baseline tier (vehicle, no person/dog trigger) gets no caption. Mitigation: deliberate (Juniper's "triggered frames"); `caption_dispatch.tiers` can add `baseline`.
- Severity: low. Concern: live path is UNVERIFIED -- not deployed, per instructions. After deploy, check: router heartbeat `caption_dispatched_total` moving, qwen logs `vision_task_complete task_type=caption_frame`, window `/api/vision-window/metrics` `vision_window_caption_gate_*`, and `vision_events` walkway narratives no longer containing "camera ... bench".

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2690

🤖 Generated with [Claude Code](https://claude.com/claude-code)
