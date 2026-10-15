## Summary

You can now watch the walkway camera in Hub. It is the new **Walkway (live)** choice in the camera dropdown of the Vision panel ("Orion's Vision"), next to Carbon. The picture refreshes about every 1.5 seconds, and a corner badge says whether it is actually live.

- **Hub now remembers walkway frames.** Hub keeps a short list of each camera's newest frame, filled from frame announcements on the bus. Walkway announcements used to be thrown away because they don't carry a percept-store hash, so Hub had nothing to show. They now keep the frame's file location on athena instead (`scripts/vision_frame_cache.py`).
- **New routes serve the newest frame for any camera on Hub's list.** These are `GET /api/vision/frames/latest` and `GET /api/vision/frames/latest/image` (`scripts/api_routes.py` + `scripts/vision_live_frame.py`).
  - For walkway, Hub reads the file from the frames folder, which is mounted into Hub read-only. It only reads files sitting directly in that folder, refuses symlinks, and caps the file size.
  - For carbon, Hub fetches the frame from percept-store and checks it against its hash.
  - Either way the picture is shrunk to 1280px wide before it reaches the browser, without blocking the server: 788KB becomes 174KB, in about 40ms on a real walkway frame.
- **A frozen picture never shows as live.** The badge says `LIVE · Ns ago` only when Hub has had a new frame in the last 10 seconds *and* the picture on screen actually loaded recently. Otherwise it says `STALE …` or `IMAGE ERROR` and the picture is dimmed. A missed poll keeps the last picture on screen, marked stale (`static/js/vision-live-frame.js`).
- **The old carbon addresses still work.** `/api/vision/carbon/latest-frame{,/image}` now point at the same code as the new routes.

## Outcome moved

- **Before:** walkway was not in the dropdown at all. Hub dropped every walkway frame announcement because it had no hash.
- **After:** Juniper picks **Walkway (live)** and sees a walkway frame at most a couple of seconds old, with an honest LIVE/STALE badge. Each poll sends about 170KB instead of the full 750KB+ original.

## Current architecture

- **Carbon (live):** polled `/api/vision/carbon/latest-frame` every 5s. It relied on a bus-fed cache of frame pointers, allowlist `carbon` only, holding hashes only, with bytes proxied from percept-store.
- **cam0 ("Eye-Ball-1"):** an MJPEG stream (motion-JPEG, a continuous image stream) straight from the vision edge, `${VISION_EDGE_BASE}/stream.mjpg`. Untouched.
- **Walkway (`orion-vision-retina-walkway` on athena):** publishes `vision.frame.pointer` on `orion:vision:frames`. Each pointer carries `image_path=/mnt/telemetry/vision/frames/walkway_walkway_<ts>_<id>.jpg` and `sha256=null`, at 2560x1920, about 2 per second. This was confirmed live by subscribing to the channel: 8 walkway and 55 cam0 pointers in 4 seconds.

## Architecture touched

- orion-hub only, plus one CI workflow step.
- No bus or schema changes: Hub reads the existing `VisionFramePointerPayload` `image_path` field, which it already received but discarded.
- New read-only bind mount of `/mnt/telemetry/vision/frames` into Hub.

## Files changed

- `services/orion-hub/scripts/vision_frame_cache.py`: keeps pointers that have only an `image_path`, and stores `image_path`.
- `services/orion-hub/scripts/vision_live_frame.py` (new): confines the path to the frames folder, does the safe read, does the shrinking, builds the browser-facing metadata (with no host paths), keeps a small per-camera cache of the shrunk picture, and cleans the frame-id header.
- `services/orion-hub/scripts/api_routes.py`: the new generic routes, with the carbon routes as aliases. The percept-store fetch is pulled out into a helper.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: three new keys, the default allowlist changed to `carbon,walkway`, and the read-only frames mount.
- `services/orion-hub/static/js/vision-live-frame.js` (new), plus `.test.js`: the pure LIVE/STALE logic, with node tests.
- `services/orion-hub/static/js/app.js`: a generic `renderLiveFrame`/`paintLiveFrame` used for both carbon and walkway.
- `services/orion-hub/templates/index.html`: the `Walkway (live)` option and the script tag (loaded before app.js).
- `services/orion-hub/tests/test_vision_live_frame.py` (new), `test_vision_live_frame_browser_smoke.py` (new), `test_vision_frame_cache.py`, `test_vision_affect_capture_api.py`.
- `services/orion-hub/README.md`: new section, "Vision panel: live camera views".
- `.github/workflows/schedule-browser-smoke.yml`: runs the new Chromium smoke test and installs Pillow.

## Schema / bus / API changes

- **Added:** `GET /api/vision/frames/latest?stream_id=` and `GET /api/vision/frames/latest/image?stream_id=`.
- **Removed:** none.
- **Renamed:** none.
- **Behavior changed:**
  - Carbon routes now return `stream_id`, `frame_id`, `age_sec`, `stale` and `source`.
  - A stream not on the allowlist now returns `stream_not_cached`.
  - Carbon images now pass through the shrinking step. Anything wider than 1280px is re-encoded at quality 80. Bytes that will not decode return 404 instead of being sent as-is.
- **Compatibility notes:** the old carbon addresses keep working. No bus contract change.

## Env/config changes

- **Added keys:** `HUB_VISION_FRAMES_DIR=/mnt/telemetry/vision/frames`, `HUB_VISION_LIVE_MAX_WIDTH=1280`, `HUB_VISION_LIVE_STALE_SEC=10`.
- **Changed default:** `VISION_FRAME_CACHE_STREAM_IDS` from `carbon` to `carbon,walkway`.
- **`.env_example` updated:** yes.
- **Local `.env` synced:** yes, with `python scripts/sync_local_env_from_example.py --all-keys orion-hub`. That added the three new keys.
- **Skipped keys requiring operator action:** none. The sync script reported `VISION_FRAME_CACHE_STREAM_IDS` as diverged (local value `carbon`) and left it alone, because it never overwrites without `--force`. That line was updated by hand to `carbon,walkway` in the primary checkout's `services/orion-hub/.env`. Without that, walkway would never show up.

## Tests run

```text
pytest tests/test_vision_live_frame.py tests/test_vision_frame_cache.py tests/test_vision_affect_capture_api.py  -> 67 passed
node --test static/js/vision-live-frame.test.js                                                               -> 11 passed
node --check static/js/app.js static/js/vision-live-frame.js                                                  -> ok
pytest tests/test_vision_live_frame_browser_smoke.py (Chromium, real index.html + app.js)                     -> 1 passed
Full Hub suite (tests/, excluding e2e), branch vs origin/main baseline:
  branch:   35 failed, 3683 passed, 105 skipped
  baseline: 35 failed, 3646 passed, 104 skipped
  The failure sets are identical: 0 new failures, 0 fixed. All 35 failures already exist on main.
scripts/check_async_routes_not_blocking.py           -> no blocking calls inside async routes
scripts/check_settings_defaults.py orion-hub --example-drift -> OK
scripts/check_env_template_parity.py                 -> PASS
check_service_hostname_refs / compose_no_relative_mounts / compose_no_host_claude_json_mount / scripts_dir_no_stdlib_shadow -> PASS
tests/test_no_personal_terms.py, test_compose_logging_journald.py, test_gitignore_no_bare_catchall.py -> 12 passed
git diff --check -> clean
```

## Evals run

```text
None. orion-hub has no eval harness for UI views, and this is a viewer with no cognition output.
Its behaviour is covered by the Chromium smoke and the live-frame smoke below.
```

## Docker/build/smoke checks

```text
docker compose ... -f services/orion-hub/docker-compose.yml config
  -> /mnt/telemetry/vision/frames bind, read_only: true; VISION_FRAME_CACHE_STREAM_IDS=carbon,walkway; 3 new keys present
Live-frame smoke (module code against real files on athena, not the deployed container):
  walkway_walkway_1792047118721_5df99c63.jpg: 788055 B, read 1.3 ms, shrink 39.4 ms -> 173581 B at 1280x960
  path traversal (../../../etc/hostname) and /etc/passwd -> refused (None)
Frames dir permissions: 755, files 644. The Hub container runs as root, so it can read them.
End-to-end in the deployed Hub: UNVERIFIED (not deployed; needs the recreate below).
```

## Review findings fixed

A fresh review subagent found no MUST-level issues. All of its SHOULDs were fixed, plus the relevant NITs:

- **Finding:** the LIVE badge depended only on whether the bus had announced a new frame, so it could sit over a blank or frozen picture whenever the image route kept failing (missing mount, file that won't decode).
  - **Fix:** LIVE now also requires that the picture on screen loaded recently, allowing two poll intervals of slack. An image that fails to load shows `IMAGE ERROR`/`STALE`.
  - **Evidence:** node tests `fresh backend but the image never loaded is not LIVE` and `...picture stopped updating is STALE`.
- **Finding:** image fetches could overlap, and a late, older response could swap an older frame back onto the screen.
  - **Fix:** one image load at a time, plus a sequence number so late responses are ignored.
  - **Evidence:** `seq !== liveImageSeq` guard, pinned in `test_page_badge_logic_lives_in_the_tested_module`.
- **Finding:** switching views reset the "request in flight" flag while the old view's request was still pending, so requests could overlap.
  - **Fix:** the in-flight markers now record which view owns them and are only cleared by that view.
- **Finding:** when a pointer carries both a local path and a hash, Hub served the local file unchecked, skipping the hash check carbon relies on.
  - **Fix:** the local bytes must match the hash; otherwise Hub falls back to percept-store.
  - **Evidence:** `test_local_copy_that_does_not_match_sha256_falls_through_to_the_store` and `..._matches_..._is_served_without_the_store`.
- **Finding:** a single failed poll blanked the picture, which would flicker at a 1.5s poll rate.
  - **Fix:** the last picture is kept, badged `STALE · connection lost`.
  - **Evidence:** node test `a failed poll keeps the picture but badges it STALE`.
- **NIT:** `X-Orion-Frame-Id` echoed a filename from the bus unchecked. Now only plain JPEG names or hashes are echoed (parametrized test).
- **NIT:** `or DEFAULT` silently replaced a configured 0. Now uses explicit None checks (test).
- **NITs:** the carbon behaviour change and the hard-link caveat are documented in the module docstring and in this report. `f=` is documented as a cache-buster only.

## Restart required

The compose mount and env changed, so Hub needs a **recreate**, not just a restart. Run from the clean primary `main` checkout after merge:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-hub up -d --build --force-recreate
```

## Risks / concerns

- **Low:** the walkway retina deletes frames after about 60s. If a poll lands just after a frame is deleted, that image request returns 404. The panel keeps the last picture and the next tick catches up. If it keeps failing, the badge goes STALE.
- **Low:** each open tab pulls about 170KB every 1.5s while Walkway is selected, roughly 115KB/s. Hub decodes each frame once, however many tabs are open. Nothing is fetched when another view is selected.
- **Low:** this only works where Hub and the walkway retina share a filesystem (athena). On any other host the frames folder is empty, and the panel says "No frame"/"IMAGE ERROR" rather than pretending.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
