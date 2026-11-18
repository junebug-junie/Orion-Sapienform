# PR: the office camera goes through its own retina and gets what the walkway gets

## Summary

- On 2026-11-04 Juniper was in the office for 12 minutes (16:09-16:21 MDT). Orion saw "a person", but did not know it was Juniper, gave the room no attention, and only listed furniture. The office camera had none of the walkway's person learning.
- The office camera (stream id `cam0`) is now captured by a new retina instance, `orion-vision-retina-office`, from its 2560x1920 main stream at 2 fps. It is no longer captured by `orion-vision-edge` from the 640x480 substream.
- cam0 now gets walkway parity: person and dog crops with body and face vectors, individuals and naming asks, activity captions from the Qwen lane ("describe what the person is doing"), hourly rollups, and a "who is around" part in Orion's situation brief.
- Orion and the Hub call it the **office camera** ("your office camera", "On the office camera just now: ...", Hub "Office (live)"). The stream id stays `cam0`, so presence, history, attributions and config keep working.
- `orion-vision-edge` is **kept, not used**. It is in `exclude_services.txt`, so the rebuild and bring-up sweeps never start it. A running edge would publish a second cam0 frame stream. The router no longer asks it for a high-res still.
- A person who sits still at the desk is never filed as a prop. The walkway's stationary rule (the patio skeleton) now runs only on the streams that have props: vision-window's caption gate follows `WINDOW_STATIONARY_STREAMS`, and sql-writer gains `VISION_STATIONARY_STREAMS=walkway,doorbell`.
- Fixed in passing: the host's "look now" took the newest file in the shared frames folder from **any** camera. After this change the walkway and office both write 2 frames/s, so "look at the office" would have answered from the walkway about half the time. It now picks the newest frame of one camera (default cam0). vision-council's foveal probe had the same bug and gets the same fix.

## Outcome moved

- Before: cam0 has 0 person crops, 0 face vectors and 0 individuals, ever. Its captions are BLIP furniture lists. The face check last matched Juniper on 10-27.
- After deploy (expected, UNVERIFIED until live): every person in the office becomes crops and an individual on cam0. Juniper names those clusters through the existing asks. After that, the brief says "On the office camera in the last 15 minutes: Juniper", and an open ask becomes an event want that competes for attention (#3122).
- Live check: `services/orion-vision-retina/evals/run_office_parity_live_eval.py` (pre-deploy baseline: every step 0, FAIL).

## Current architecture

- `orion-vision-edge` captured cam0 (RTSP substream 640x480, 15 fps), published `orion:vision:frames` pointers named `frame_*`, and served `POST /still` (a main-stream still for each face check).
- The router (`config/vision_frame_router.yaml` `streams.cam0`) sent `retina_fast` tasks with a host BLIP caption, plus an `identity_face` check on the still. No crops, no faces.
- `config/vision_zones.yaml` had no cam0 zone, so the host's fail-closed rule could never embed a cam0 crop.
- The walkway (and the doorbell since 10-17) had the full person-learning path. It is camera-agnostic and switched on per stream by config lists.

## Architecture touched

Config and per-stream lists across vision-retina, vision-frame-router (config only), vision-host, vision-window, sql-writer, hub, cortex-exec, thought, equilibrium; `orion/situational`. No bus or schema contract changes.

## Files changed

- `services/orion-vision-retina/docker-compose.yml`, `.env_example`, `README.md`: `orion-vision-retina-office` (profile `office`), `OFFICE_*` keys, first-start command.
- `services/orion-vision-retina/tests/test_office_instance_compose.py`: instance identity, profile, template, router/zone parity, edge kept but excluded.
- `services/orion-vision-retina/evals/run_office_parity_live_eval.py`: live parity check.
- `config/vision_frame_router.yaml`: cam0 baseline + triggered tiers with crops/faces, caption_dispatch with an activity prompt, identity check without `still_url`.
- `config/vision_zones.yaml`: whole-frame `office` zone, `embed: true`.
- `config/vision_profiles.yaml`: note that the cam0 size floor (a fraction) still holds at 2560x1920.
- `services/orion-vision-window/*`: `WINDOW_CAPTION_GATE_STREAMS` gains cam0.
- `services/orion-vision-host/.env_example`, `docker-compose.circe-qwen.yml`: qwen lane broadcasts cam0 captions. `app/runner.py`: one-camera "look now". Tests updated and added.
- `services/orion-sql-writer/*`: `VISION_ROLLUP_STREAMS` gains cam0; hourly frames read cam0 with retina naming.
- `services/orion-hub/*`, `services/orion-cortex-exec/*`, `orion/situational/context.py`: Street line order `doorbell,walkway,cam0` (office last, so its near-constant part never pushes walkway lines past the 900-character cut); Hub frame cache + "Office (live)" view; camera description "your office camera".
- `services/orion-thought/.env_example`, `orion/situational/camera_eyes.py`: camera description.
- `orion/situational/perception_reader.py`: `STREET_CAMERA_WORDS` (cam0 -> "office camera").
- `services/orion-vision-window/app/caption_gate.py`, `app/main.py`: prop tracking only on `WINDOW_STATIONARY_STREAMS`.
- `services/orion-sql-writer/app/vision_individuals.py`, `vision_individuals_loop.py`, `settings.py`, `.env_example`: `VISION_STATIONARY_STREAMS`.
- `services/orion-vision-council/app/foveal_probe.py`: one-camera newest frame.
- `services/orion-equilibrium-service/*`: expects `vision-retina-office` instead of `vision-edge`.
- `mesh-utilities/common/exclude_services.txt`, `services/orion-vision-edge/README.md`: edge kept, not run.
- Router/host/window/situational tests: updated where they pinned the old cam0 behavior ("room cameras never ask for faces", BLIP for cam0, defaults).

## Schema / bus / API changes

- Added: none. Removed: none. Renamed: none.
- Behavior changed: cam0 frames come from the retina (`cam0_cam0_*` files) at 2560x1920. Host `use_latest_frame` honors `request.stream_id` (default cam0).
- Compatibility notes: stream id unchanged. `orion:vision:frames` producer set unchanged in the catalog (edge stays listed; it just does not run).

## Env/config changes

- Added keys: `VISION_STATIONARY_STREAMS` (orion-sql-writer, default `walkway,doorbell` = today's behavior), `OFFICE_RTSP_URL`, `OFFICE_RETINA_FPS`, `OFFICE_RETINA_JPEG_QUALITY`, `OFFICE_RETINA_HTTP_PORT` (orion-vision-retina).
- Changed values: `WINDOW_CAPTION_GATE_STREAMS`, `CIRCE_QWEN_VISION_ARTIFACT_BROADCAST_STREAMS`, `VISION_ROLLUP_STREAMS`, `VISION_HOURLY_FRAMES_STREAMS`, `ORION_SITUATION_STREET_STREAM_IDS` (hub, cortex-exec), `VISION_FRAME_CACHE_STREAM_IDS`, `ORION_CAMERA_DESCRIPTIONS` (hub, cortex-exec, thought), `EQUILIBRIUM_EXPECTED_SERVICES`.
- `.env_example` updated: yes.
- Local `.env` synced: yes, but by targeted key writes, not `sync_local_env_from_example.py`. Every changed key already had a value locally, so the script would report it "diverged" and keep the old value, and `--force` would also overwrite unrelated intentional overrides.
- Operator action: none. `OFFICE_RTSP_URL` (secret) was copied into the local retina `.env` from the edge's `REOLINK_HIRES_URL`.

## Tests run

```text
orion-vision-frame-router  pytest tests -q            124 passed
orion-vision-council       pytest tests -q            128 passed
orion-vision-retina        pytest tests (in image)    177 passed
orion-vision-host          pytest tests (in image)    405 passed, 1 skipped
orion-vision-window        pytest tests -q            172 passed (new office-desk test fails without the fix)
orion-sql-writer           pytest tests -q            1199 passed, 183 skipped
orion-equilibrium-service  pytest tests -q            257 passed
orion/situational + orion/curiosity   549 passed, 29 skipped
orion-hub                  pytest tests -q            same 8 failures as main, none new
orion-cortex-exec          (in image)                 same failures as main; 5 extra in the full run are order-dependent and pass alone on both trees
node --test vision-live-frame.test.js                 11 pass
```

## Evals run

```text
run_office_parity_live_eval.py --hours 2 (pre-deploy)  every step 0, and 819 fresh edge frames -> FAIL, as expected
```

## Docker/build/smoke checks

```text
docker compose --profile office config orion-vision-retina-office   renders: rtsp main stream, cam0, vision-retina-office, port 8030
gates: check_definition_drift --gate, check_metric_lineage --gate, check_env_key_single_source,
       check_service_hostname_refs, check_system_health_producers, check_env_template_parity,
       check_settings_defaults orion-hub --example-drift, git diff --check   all ok
```

## Review findings fixed

Review ran in a subagent.

- Finding (HIGH): a person seated for 30 min on cam0 would be marked a stationary prop in both the window caption gate and the sql-writer individuals reducer: captions rejected, dropped from "who is around", no asks.
  - Fix: prop tracking only on `WINDOW_STATIONARY_STREAMS` (window) and the new `VISION_STATIONARY_STREAMS` (sql-writer).
  - Evidence: `test_office_person_sitting_still_for_hours_is_not_a_prop` (fails without the fix), `test_office_person_at_a_desk_is_never_a_prop`.
- Finding (HIGH): the edge still runs live; the exclude list does not stop a running container. Two cam0 streams would publish.
  - Fix: restart step stops the edge first; the live eval fails while edge `frame_*` files are fresh.
  - Evidence: eval reports 819 fresh edge frames today.
- Finding (MEDIUM): vision-council `foveal_probe.resolve_latest_frame_path` had the same newest-file-of-any-camera bug.
  - Fix: same one-camera prefix rule, default cam0. Evidence: `test_resolve_latest_frame_path_is_the_office_cameras_not_the_newest_file`.
- Finding (MEDIUM): office second in the Street order could cut walkway lines at 900 chars.
  - Fix: order `doorbell,walkway,cam0`.
- Finding (MEDIUM): GPU load with Juniper in view for hours.
  - Fix: cam0 triggered tier at most one task per 5 s (was 2 s). UNVERIFIED on live GPU until deploy.
- Findings (LOW): profile name wording, a garbled comment, implicit `presence`, `__import__("re")`.
  - Fix: all corrected.
- Not changed: the router's generic still path stays (edge kept; cam0 just has no `still_url`).

## Restart required

After merge, the post-pull rebuild redeploys the changed services. Then, once, on athena from the primary main checkout:

```bash
docker stop orion-athena-vision-edge && scripts/safe_docker_build.sh orion-vision-retina --profile office up -d --build orion-vision-retina-office
```

Then, after someone has been in the office a while: `python services/orion-vision-retina/evals/run_office_parity_live_eval.py --hours 2`

## Risks / concerns

- Severity: medium. Concern: one person is split into several cam0 individuals (the walkway's known behavior), so Juniper may get several "who is this?" asks before the office line names her reliably. Mitigation: the shared daily ask budget caps it.
- Severity: low. Concern: GPU load. The baseline tier is capped at one task per 5 s, as before. Crops and faces add work only while someone is in view. Qwen captions skip an unchanged seated scene for 300 s.
- Severity: low. Concern: Hub "Eye-Ball-1" (the edge's MJPEG view) is dark once the edge is stopped. Use "Office (live)".
- Severity: low. Concern: an empty office now has no description. Captions are asked only when a person or dog is in view (BLIP used to list furniture). Intended.
- Not done, on purpose: motion stories for cam0 (street wording such as "toward the house" and a spot-check step that does not fit a room).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3125
