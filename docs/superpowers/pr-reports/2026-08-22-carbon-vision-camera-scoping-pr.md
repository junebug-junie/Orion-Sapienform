## Summary

Two direct asks from Juniper, addressed in one PR:

1. **"I want this to only run on my carbon webcam... not on the office webcam."** `orion:exec:request:RetinaClipCaptureService` had no built-in per-instance routing — any retina instance subscribed to it with `RETINA_CLIP_ENABLED=true` would respond to any request. `RetinaClipCaptureRequestPayload.target_stream_id` (required, no default) closes that: every retina instance — both the bus RPC path and the pre-existing HTTP route — checks it against its own `RETINA_STREAM_ID` before anything else, refusing (`error_code="wrong_camera"`) on a mismatch. Confirmed live via investigation that no second retina deployment actually exists today (the office camera, "Eye-Ball-1"/`cam0` in Hub's Vision panel, is a completely separate service, `orion-vision-edge`, sharing zero code path with this one) — but this service's own docs already anticipate a future room-camera deployment, so this closes a real, if currently latent, gap.

2. **"Port that into hub's camera drop down... one for streaming carbon, and one for the most recent affect snapshot."** New `services/orion-hub/scripts/vision_frame_cache.py` — a bus-subscriber cache remembering the latest frame pointer per stream_id (confirmed live via investigation that no persisted "latest frame for stream X" lookup exists anywhere else in this repo). Two new Vision panel dropdown options: "Carbon (live)" (polls the latest captured frame — carbon has no continuous video stream, only a ~5s still-frame loop) and "Carbon (affect snapshot)" (shows the last successful AffectGPT reading, from either "Check now" or the ambient toggle).

Code review (8 finder angles) found 10 confirmed issues in the first pass — all fixed in a follow-up commit on this same branch (see "Review findings fixed" below).

## Outcome moved

Before this PR, the affect-capture bus channel would respond to a request from any retina instance that happened to have `RETINA_CLIP_ENABLED=true`, with nothing distinguishing "carbon" from any other camera. After this PR, that's a structural check, not an operational convention to remember. And the Vision panel — previously showing nothing for carbon at all — now shows both a live-ish view and the AI's actual read of Juniper's face/voice, sourced from real backend state rather than a guess.

## Current architecture

Builds directly on PR #1838 (the initial capture pipeline) and PR #1840 (the ambient toggle). Before this PR: `RetinaClipCaptureRequestPayload` had no fields at all; Hub's Vision panel dropdown had only "Eye-Ball-1" (office/`orion-vision-edge`) and "Simulated Feed"; no route or cache anywhere could answer "what's the latest frame for a given camera."

## Architecture touched

- `orion/schemas/vision.py` — `RetinaClipCaptureRequestPayload.target_stream_id` (required).
- `services/orion-vision-retina/app/main.py` — `_handle_clip_request` (bus) and `capture_clip_endpoint` (HTTP) both check `target_stream_id` first, before `RETINA_CLIP_ENABLED` or anything else.
- `services/orion-juniper-affective-state` — new `AFFECT_TARGET_STREAM_ID` setting (default `"carbon"`), sent on every retina RPC.
- `services/orion-hub/scripts/vision_frame_cache.py` (new) — bus-subscriber latest-frame cache, mirroring `biometrics_cache.py`'s pattern.
- `services/orion-hub/scripts/api_routes.py` — `GET /api/vision/carbon/latest-frame` (metadata) and `.../image` (proxies JPEG bytes from percept-store, server-side, with sha256 verification).
- `services/orion-hub/scripts/vision_affect_ambient.py` — `AffectAmbientState` gains `last_raw_response`/`last_video_sha256`, populated by both the manual and ambient capture paths.
- `services/orion-hub/templates/index.html` / `static/js/app.js` — two new dropdown options, render-generation-guarded async view rendering, Pop Out disabled for carbon views.

## Files changed

- `orion/schemas/vision.py` — `target_stream_id` field
- `services/orion-vision-retina/app/main.py` — camera-identity check on both entry points
- `services/orion-vision-retina/README.md` — documents the check, its real scope (not authentication), and deploy-order safety
- `services/orion-juniper-affective-state/app/settings.py`, `app/main.py`, `.env_example`, `docker-compose.yml` — `AFFECT_TARGET_STREAM_ID`
- `services/orion-hub/scripts/vision_frame_cache.py` (new) — the latest-frame cache
- `services/orion-hub/scripts/vision_affect_ambient.py` — `last_raw_response`/`last_video_sha256`
- `services/orion-hub/scripts/api_routes.py` — new carbon-frame routes (sha256-verified, aiohttp-based, `stream_id` param not hardcoded)
- `services/orion-hub/scripts/main.py` — cache startup/shutdown wiring
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml` — `VISION_FRAME_CACHE_*`, `PERCEPT_STORE_*` settings
- `services/orion-hub/templates/index.html`, `static/js/app.js` — dropdown options, race-safe rendering, Pop Out handling
- `docs/operations/carbon-webcam.md` — updated curl examples with the new required query param
- Tests: `tests/test_vision_retina_clip_rpc.py` (+7), `services/orion-juniper-affective-state/tests/test_capture_and_assess.py` (+2), `services/orion-hub/tests/test_vision_frame_cache.py` (new, 7), `test_vision_affect_ambient.py` (+3), `test_vision_affect_capture_api.py` (+13 net)

## Schema / bus / API changes

- **Added:** `RetinaClipCaptureRequestPayload.target_stream_id` (required — this is a breaking change to that payload's shape, see Compatibility notes); `GET /api/vision/carbon/latest-frame`, `GET /api/vision/carbon/latest-frame/image` (both take an optional `stream_id` query param, default `"carbon"`); `?target_stream_id=` required query param on `POST /capture/clip`.
- **Removed:** nothing.
- **Renamed:** nothing.
- **Behavior changed:** both retina capture entry points now reject a request whose `target_stream_id` doesn't match `RETINA_STREAM_ID`, even if the capture would otherwise have succeeded.
- **Compatibility notes:** `target_stream_id` being required with no default means an old caller (pre-2026-08-22 orchestrator, or a bare curl without the new query param) gets a clean `invalid_request`/400 rather than a silent wrong-camera capture — fails closed in both deploy-order directions, no coordinated rollout required.

## Env/config changes

- **Added keys:** `orion-juniper-affective-state`: `AFFECT_TARGET_STREAM_ID` (default `carbon`). `orion-hub`: `VISION_FRAME_CACHE_ENABLED`, `VISION_FRAME_CACHE_STREAM_IDS`, `VISION_FRAME_CHANNEL`, `PERCEPT_STORE_BASE_URL`, `PERCEPT_STORE_TIMEOUT_SEC`, `PERCEPT_STORE_TOKEN`.
- **Removed keys:** none.
- **Renamed keys:** none.
- **`.env_example` updated:** yes, both services.
- **local `.env` synced with `python scripts/sync_local_env_from_example.py`:** yes — `--all-keys` on the primary checkout for `orion-vision-retina`, `orion-juniper-affective-state`, `orion-hub`.
- **skipped keys requiring operator action:** none.

## Tests run

```text
PYTHONPATH=. venv/bin/python -m pytest tests/test_vision_retina_*.py -q
  53 passed

cd services/orion-juniper-affective-state && PYTHONPATH=.:<repo root> venv/bin/python -m pytest tests/ -q
  22 passed

cd services/orion-hub && PYTHONPATH=.:<repo root> venv/bin/python -m pytest tests/test_vision_affect_ambient.py tests/test_vision_affect_capture_api.py tests/test_vision_frame_cache.py -q
  43 passed
```

118 total, all passing. `services/orion-hub/scripts/main.py` verified to import cleanly end-to-end (confirms the new cache's task registration doesn't break app startup).

## Evals

No eval harness exists for the touched services beyond the unit-test layer above. The full "toggle a real retina instance targeting the wrong camera and confirm it refuses" path has not been exercised live (would need a second physical retina deployment, which doesn't exist yet) — the unit tests exercise the logic with mocked settings instead.

## Docker/build/smoke checks

Not run — no Docker available in this environment for a live retina/orchestrator/Hub triple. `scripts/check_service_env_compose_parity.py` run instead (deterministic) — all three touched services report OK.

## Review findings fixed

Code review (8 finder angles, all candidates independently verified) found 10 confirmed issues in the first-pass diff — all fixed in a follow-up commit on this branch:

- **HTTP `/capture/clip` was missing the camera-identity check entirely** (highest severity — the bus RPC path got it, this pre-existing route did not, in the same commit): fixed with a required `?target_stream_id=` query param, checked identically.
  - Evidence: `test_http_route_rejects_mismatched_target_stream_id`, `test_http_route_rejects_a_missing_target_stream_id`, `test_http_route_proceeds_when_target_stream_id_matches`.
- **JS stale-callback race**: switching the dropdown mid-flight let an old `renderCarbonLiveFrame`/`renderCarbonAffectSnapshot` callback overwrite whatever the user had switched to.
  - Fix: a render-generation counter, checked before every DOM mutation in both async renderers.
- **Pop Out silently did nothing (and desynced `visionIsFloating`) while a carbon view was selected**: fixed by disabling the button and forcing `visionIsFloating = false` for those two dropdown values.
- **Image-proxy route trusted percept-store's response uncritically** (unlike the orchestrator's own percept fetch, which has verified this since it was built): fixed with the same sha256-recompute-and-compare check.
  - Evidence: `test_latest_frame_image_502s_when_fetched_bytes_dont_match_the_sha256`.
- **Image-proxy route used `asyncio.to_thread(requests.get, ...)`** instead of this file's own established `aiohttp.ClientSession` convention, tying up a shared thread-pool slot (also used by the ambient loop's up-to-~195s blocking calls) for the whole fetch: switched to `aiohttp`.
- **README overclaimed** the camera-identity check as "a real structural guarantee": corrected to precisely state what it does and doesn't protect against (accidental misconfiguration, not deliberate spoofing on an unauthenticated channel — same disclosed risk as the pre-existing "Known, accepted risk" section).
- **No documented deploy-order safety**: added a note explaining both mismatch directions fail closed (`invalid_request`), no coordinated rollout needed.
- **Hardcoded `"carbon"` in the new Hub routes**, even though `VisionFrameCache` already supports an arbitrary allowlist: routes now take `stream_id` as a real query param (default `"carbon"`).
  - Evidence: `test_latest_frame_routes_accept_a_stream_id_param_not_hardcoded_carbon`.
- **Unconditional full-image re-fetch every 5s poll tick**, even when the frame hadn't changed: `renderCarbonLiveFrame` now checks the lightweight metadata route first and only fetches image bytes when the sha256 actually differs from what's already shown.
- **Missing test coverage for the pre-existing `not_configured` branch** (RETINA_PERCEPT_STORE_URL unset) — had zero direct coverage before or after the try/except/else refactor that added the camera check ahead of it.
  - Evidence: `test_handle_clip_request_reports_not_configured_without_capturing`.
- Two of the fixed tests (`test_latest_frame_image_proxies_bytes_from_percept_store`, `...transport_failure`) were themselves silently making real (failing) network calls instead of being properly mocked, once the route switched to `aiohttp` — caught while fixing the `aiohttp` migration, rewritten with a proper fake `ClientSession`.

## Restart required

```bash
# carbon (orion-vision-retina) -- picks up the new required target_stream_id
# check on both the bus RPC and HTTP paths:
systemctl --user restart orion-retina
# or: docker compose -f services/orion-vision-retina/docker-compose.yml restart

# circe (orion-juniper-affective-state) -- picks up AFFECT_TARGET_STREAM_ID:
docker compose --env-file .env --env-file services/orion-juniper-affective-state/.env \
  -f services/orion-juniper-affective-state/docker-compose.yml up -d --build

# athena (orion-hub) -- picks up the new frame cache + dropdown routes:
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build hub-app
```

**Deploy order note:** retina and the orchestrator can be restarted in either order safely (see Compatibility notes above) -- a version mismatch during the rollout window just produces clean `invalid_request` rejections, never a wrong-camera capture.

## Risks / concerns

- **Severity: Low.** The camera-identity check is value-equality on an unauthenticated bus channel/HTTP route, not real authentication — disclosed explicitly in the README now, not a new gap, but worth remembering if this ever gets a formal threat-model pass.
- **Severity: Low.** The "Carbon (live)" and "Carbon (affect snapshot)" dropdown views are unit-tested against mocks only — the full live path (real retina publishing real frames, real Hub subscribing and caching them, real browser polling) has not been exercised end-to-end in this session.
- **Severity: Low.** `VisionFrameCache` is scoped to an explicit stream_id allowlist (default just `"carbon"`) by design, not every stream in the mesh — if an operator adds a second camera to `VISION_FRAME_CACHE_STREAM_IDS` later, the Hub routes can already serve it (via the `stream_id` param), but no dropdown UI exists for it yet — that's real, disclosed follow-up work, not built here.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1841
