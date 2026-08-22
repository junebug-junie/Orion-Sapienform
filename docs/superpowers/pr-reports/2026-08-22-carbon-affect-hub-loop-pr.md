## Summary

- Closed the loop from Hub's "Orion's Vision" panel to a live AffectGPT read: **Hub → (bus RPC) → carbon's webcam+mic → percept-store → AffectGPT worker → published event → back to Hub**, with no piece left as a stub or an empty shell.
- New bus channel `orion:exec:request:RetinaClipCaptureService` — the bus-reachable twin of retina's existing `POST /capture/clip`, since carbon accepts no inbound HTTP at all (only outbound to the bus and percept-store).
- `orion-juniper-affective-state.capture_and_assess()` — the real cross-host bridge: RPCs retina for a live clip, fetches both blobs from percept-store with hash verification on the received bytes, writes them into the same shared volume the worker already mounts read-only, then runs the existing worker round trip.
- Hub: `POST /api/vision/affect-capture` proxies to that; a new "Affect check" button in the Vision panel shows the model's `raw_response` inline.
- Retired the "UNVERIFIED against real hardware" disclaimers on the clip-capture code (this session ran it live on carbon and confirmed it byte-for-byte) and replaced them with the actual evidence, not just the claim.

## Outcome moved

Before this PR, `orion-vision-retina`'s `POST /capture/clip` returned sha256 refs nobody downstream could consume, and Hub's Vision panel was a stub ("Vision service stub."). After this PR, one button click in Hub produces a real AffectGPT assessment of Juniper's actual face/voice from carbon's webcam, end to end, with every failure mode (capture failed, fetch failed, worker busy, timeout) surfacing as a real published `orion:affectgpt:assessment` event rather than a silent drop.

## Current architecture

- `orion-vision-retina` (carbon): continuous low-fps presence capture (unchanged) + on-demand `POST /capture/clip` (built earlier this session, PR #1834/#1835/#1836) — HTTP-only, unreachable from anywhere but carbon's own localhost.
- `orion-juniper-affective-state` (circe): `POST /v1/juniper/affect/trigger` — manual trigger requiring an already-written video/audio path pair on the worker's own filesystem. No live capture source wired to it.
- `orion-affectgpt-worker` (circe GPU2): requires local `video_path`/`audio_path`, unaware of percept-store.
- Hub: "Orion's Vision" panel (`templates/index.html`) was a pure stub with a source-select dropdown (gopro-1/simulated) and no backend behind it for anything affect-related.

## Architecture touched

- `orion/schemas/vision.py` + `orion/schemas/registry.py` + `orion/bus/channels.yaml`: new `RetinaClipCaptureRequestPayload`/`RetinaClipCaptureResultPayload` schemas and the `orion:exec:request:RetinaClipCaptureService` / `orion:retina:clip:reply:*` channel pair.
- `orion-vision-retina`: `RetinaService.capture_and_upload_clip()` extracted as the one shared capture implementation for both the HTTP route and a new bus RPC consumer (`_clip_consume_loop` / `_handle_clip_request`) — no duplicated capture logic. Added a cooldown gate (`RETINA_CLIP_MIN_INTERVAL_SEC`, `ClipCaptureCooldownError`).
- `orion-juniper-affective-state`: new `capture_and_assess()` + `_capture_clip_via_retina()` + `_fetch_percept()` + `POST /v1/juniper/affect/capture_and_assess`. New `AFFECTGPT_SCRATCH_DIR` setting so the fetched clip lands in the volume the worker container can actually see.
- `orion-affectgpt-worker`: docker-compose.yml's scratch mount is now `AFFECTGPT_SCRATCH_DIR`-parameterized (was a bare literal) to stay in lockstep with the orchestrator's own setting.
- Hub: new `JUNIPER_AFFECTIVE_STATE_BASE_URL`/`_TIMEOUT_SEC` settings, `POST /api/vision/affect-capture` route, "Affect check" button + result panel in the Vision section.

## Files changed

- `orion/schemas/vision.py` — `RetinaClipCaptureRequestPayload`/`RetinaClipCaptureResultPayload`
- `orion/schemas/registry.py`, `orion/bus/channels.yaml` — schema/channel registration
- `config/metrics/metric_definitions.lock.json` — `--update`d for the two new channel definitions (0 concerning drift)
- `services/orion-vision-retina/app/main.py` — shared `capture_and_upload_clip()`, bus RPC consumer, cooldown gate, catch-all exception symmetry between HTTP and bus paths
- `services/orion-vision-retina/app/clip_capture.py` — `ClipCaptureCooldownError`; retired stale UNVERIFIED disclaimer
- `services/orion-vision-retina/app/settings.py`, `.env_example`, `docker-compose.yml` — new channel/cooldown settings
- `services/orion-vision-retina/README.md` — bus RPC twin documented, live-verification evidence, explicit disclosed risk section
- `services/orion-juniper-affective-state/app/main.py` — `capture_and_assess()`, `_capture_clip_via_retina()`, `_fetch_percept()` (hash-verified), `PerceptFetchError`
- `services/orion-juniper-affective-state/app/settings.py`, `.env_example`, `docker-compose.yml` — new bridge/scratch/token settings
- `services/orion-juniper-affective-state/README.md` — the new capture path documented as the answer to the "cross-host capture" gap it used to flag as future work
- `services/orion-affectgpt-worker/.env_example`, `docker-compose.yml` — `AFFECTGPT_SCRATCH_DIR` parity fix
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml` — `JUNIPER_AFFECTIVE_STATE_*` settings
- `services/orion-hub/scripts/api_routes.py` — `POST /api/vision/affect-capture`
- `services/orion-hub/templates/index.html`, `static/js/app.js` — "Affect check" button + result panel
- `docs/operations/carbon-webcam.md` — bus RPC path documented, live-verification evidence block
- Tests: `tests/test_vision_retina_clip_rpc.py`, `tests/test_vision_retina_clip_cooldown.py`, `services/orion-juniper-affective-state/tests/test_capture_and_assess.py`, `services/orion-hub/tests/test_vision_affect_capture_api.py` (all new)

## Schema / bus / API changes

- **Added:** `RetinaClipCaptureRequestPayload`, `RetinaClipCaptureResultPayload` schemas; `orion:exec:request:RetinaClipCaptureService` (request) and `orion:retina:clip:reply:*` (result) channels; `POST /v1/juniper/affect/capture_and_assess` on the orchestrator; `POST /api/vision/affect-capture` on Hub.
- **Removed:** nothing.
- **Renamed:** nothing.
- **Behavior changed:** `orion-vision-retina`'s `POST /capture/clip` now shares its implementation with the bus RPC path and is subject to a 30s (default) cooldown between captures (`429`/`error_code=cooldown` if violated) — was previously unthrottled beyond the single-capture-in-flight lock.
- **Compatibility notes:** all additive; no existing request/response shape changed.

## Env/config changes

- **Added keys:** `orion-vision-retina`: `CHANNEL_RETINA_CLIP_INTAKE`, `CHANNEL_RETINA_CLIP_REPLY_PREFIX`, `RETINA_CLIP_MIN_INTERVAL_SEC`. `orion-juniper-affective-state`: `CHANNEL_RETINA_CLIP_INTAKE`, `CHANNEL_RETINA_CLIP_REPLY_PREFIX`, `RETINA_CLIP_RPC_TIMEOUT_S`, `PERCEPT_STORE_BASE_URL`, `PERCEPT_STORE_TIMEOUT_SEC`, `PERCEPT_STORE_TOKEN`, `AFFECTGPT_SCRATCH_DIR`. `orion-affectgpt-worker`: `AFFECTGPT_SCRATCH_DIR` (compose-only, no Settings() field). `orion-hub`: `JUNIPER_AFFECTIVE_STATE_BASE_URL`, `JUNIPER_AFFECTIVE_STATE_TIMEOUT_SEC`.
- **Removed keys:** none.
- **Renamed keys:** none.
- **`.env_example` updated:** yes, all four services.
- **local `.env` synced with `python scripts/sync_local_env_from_example.py`:** yes — `--all-keys` run against all four services on the primary checkout (`/mnt/scripts/Orion-Sapienform`), plus one hand-edit (`JUNIPER_AFFECTIVE_STATE_TIMEOUT_SEC` 120→240, a genuine upstream-default change caught by the sync script's own divergence check).
- **skipped keys requiring operator action:** none from the sync script's own `NEVER_SYNC_KEYS`. Operator action still needed: `RETINA_CLIP_ENABLED=true` + `RETINA_CLIP_TOKEN` must be set on carbon's own `.env` for the bus RPC path to actually respond (same prerequisite as the existing HTTP route, unchanged by this PR).

## Tests run

```text
PYTHONPATH=. venv/bin/python -m pytest tests/test_vision_retina_*.py -q
  45 passed

cd services/orion-juniper-affective-state && PYTHONPATH=.:<repo root> venv/bin/python -m pytest tests/ -q
  13 passed

cd services/orion-hub && PYTHONPATH=.:<repo root> venv/bin/python -m pytest tests/test_vision_affect_capture_api.py -q
  4 passed
```

62 tests total, all passing. Bus-free / hardware-free by design (mocked bus, mocked percept-store, fake ffmpeg) — see each test file's own docstring for exactly what it does and doesn't exercise.

## Evals

No eval harness exists for `orion-vision-retina`, `orion-juniper-affective-state`, or `orion-hub`'s Vision panel specifically. The real end-to-end capture path (retina → percept-store) was live-verified manually on carbon this session (see `docs/operations/carbon-webcam.md`'s evidence block: real sha256s, ffprobe output, byte counts, chain-of-custody re-fetch+re-hash). The full Hub→retina→percept-store→worker loop this PR adds has **not** been run live end-to-end (would require a live circe GPU + live carbon capture in the same session) — flagged as a concrete follow-up, not claimed as verified.

## Docker/build/smoke checks

Not run — no Docker available in this environment for `orion-vision-retina` (carbon-only hardware), `orion-affectgpt-worker` (circe GPU-only), or a live `orion-juniper-affective-state`/`orion-hub` pair. `scripts/check_service_env_compose_parity.py` run instead for all four touched services (deterministic, no Docker needed) — all four report OK. Real container smoke is on Juniper for the carbon/circe hosts.

## Review findings fixed

All 10 findings from the code-review subagent (9 finder angles, 4 verification batches, all CONFIRMED) were addressed:

- **Bus RPC trigger has no auth beyond bus reachability** (real, disclosed): added `RETINA_CLIP_MIN_INTERVAL_SEC` cooldown (`ClipCaptureCooldownError`) to bound the worst-case trigger rate. This does **not** add real per-caller authentication — that's disclosed explicitly in `services/orion-vision-retina/README.md`'s new "Known, accepted risk" section as legitimate follow-up work, not silently accepted.
  - Fix: `RetinaService.capture_and_upload_clip()` cooldown gate, shared by both HTTP and bus paths.
  - Evidence: `tests/test_vision_retina_clip_cooldown.py` (4 tests).
- **percept-store's default no-auth exposes the newly-live-recording payload**: pre-existing, documented, accepted risk (`PERCEPT_STORE_TOKEN` already exists and defaults empty "acceptable only on a closed tailnet") — this PR's capture path is what newly activates writing that content type, so flagged explicitly rather than left implicit.
  - Fix: disclosed in the same README section above; `PERCEPT_STORE_TOKEN` support added to the orchestrator (next finding) so the existing knob is actually usable end-to-end if an operator wants to close this.
- **Orchestrator's percept-store fetch has no auth header, would silently 401 if `PERCEPT_STORE_TOKEN` were ever enabled**:
  - Fix: `PERCEPT_STORE_TOKEN` setting added, sent as `X-Orion-Percept-Token`.
  - Evidence: `test_fetch_percept_sends_configured_token_header`, `test_fetch_percept_sends_no_token_header_when_unset`.
- **`asyncio.gather()` race in `capture_and_assess()`'s percept fetch**: a fast-failing fetch could return from inside the `TemporaryDirectory` block while the sibling fetch's real OS thread was still writing, racing `rmtree`.
  - Fix: `return_exceptions=True` — gather now waits for both threads to actually finish before either error handling or cleanup runs.
  - Evidence: `test_gather_waits_for_slow_sibling_fetch_before_cleanup` — real `time.sleep` in a real thread, asserts the slow side actually finished by the time the function returns (not just that no exception escaped).
- **No rate limit on the bus capture trigger**: same fix as the auth-boundary finding above (`RETINA_CLIP_MIN_INTERVAL_SEC`).
- **Bus RPC handler task not tracked/cancelled on shutdown**: acknowledged, not fixed in this PR — the fire-and-forget `asyncio.create_task` per request is consistent with `orion-affectgpt-worker`'s own pre-existing `_handle_envelope` pattern in this codebase; fixing it here alone (not the sibling pattern) would be inconsistent surgery. Flagged as a concern below, not silently dropped.
- **Hub's timeout (120s) less than the backend's own worst-case sequential sum (~195s)**:
  - Fix: bumped to 240s (`JUNIPER_AFFECTIVE_STATE_TIMEOUT_SEC`), with the arithmetic in the settings comment.
- **`AFFECTGPT_SCRATCH_DIR` parity gap between the orchestrator (parameterized) and the worker (hardcoded literal)**:
  - Fix: worker's `docker-compose.yml` now uses the same env var with the same default.
- **HTTP `/capture/clip` route missing the catch-all exception handler the bus path got**:
  - Fix: added matching `except Exception` returning the same `{"ok": false, ...}` shape.
- **"Live-verified"/"confirmed byte-for-byte" doc claims with no attached evidence artifact**:
  - Fix: replaced prose-only claims in `docs/operations/carbon-webcam.md` and `services/orion-vision-retina/README.md` with the actual measured values (sha256s, ffprobe output, byte counts) from the real capture this session ran on carbon.

## Restart required

```bash
# carbon (venv/systemd or Docker path, per docs/operations/carbon-webcam.md):
# no code change to the running capture loop's core behavior, but the new
# CHANNEL_RETINA_CLIP_INTAKE bus consumer only starts if RETINA_CLIP_ENABLED
# is already true -- restart to pick up the new .env keys either way.
systemctl --user restart orion-retina
# or: docker compose -f services/orion-vision-retina/docker-compose.yml restart

# circe (orion-juniper-affective-state, orion-affectgpt-worker):
docker compose --env-file .env --env-file services/orion-juniper-affective-state/.env \
  -f services/orion-juniper-affective-state/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-affectgpt-worker/.env \
  -f services/orion-affectgpt-worker/docker-compose.yml up -d --build

# athena (orion-hub):
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build hub-app
```

## Risks / concerns

- **Severity: Medium.** The bus RPC capture trigger (`orion:exec:request:RetinaClipCaptureService`) has no authentication beyond bus reachability — any bus-connected service can trigger a live webcam+mic recording of Juniper. This matches every other RPC channel's trust model in this codebase, but this is the first one whose action is "record a person," not "route a request." **Mitigation:** `RETINA_CLIP_MIN_INTERVAL_SEC` cooldown bounds the worst case to a known rate; explicitly disclosed as accepted-not-solved in `services/orion-vision-retina/README.md`. Real per-caller auth on this channel is legitimate, not-yet-done follow-up work.
- **Severity: Low.** `_handle_clip_request`'s per-request `asyncio.create_task` is untracked — `RetinaService.stop()` doesn't cancel/await in-flight handler tasks, only the consumer loop. A shutdown mid-capture could drop a reply the caller then just times out waiting for. Not fixed here (consistent with `orion-affectgpt-worker`'s own existing pattern); worth a follow-up pass across both services together rather than fixing one in isolation.
- **Severity: Low.** The full Hub→retina→percept-store→worker loop has not been exercised live end-to-end in this session (would need simultaneous live carbon + circe access). The retina↔percept-store half was; the orchestrator↔worker half was already live-verified in earlier PRs (#1831 era). The new glue (`capture_and_assess`) is only unit-tested with mocks.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1838
