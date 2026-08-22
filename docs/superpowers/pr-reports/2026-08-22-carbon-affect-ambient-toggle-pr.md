## Summary

- **Design correction.** PR #1838 shipped a one-shot "Affect check" button and called it "the toggle" — a silent substitution for what Juniper had actually asked for and approved before compaction ("a toggle that periodically grabs a clip... while on"). This PR is the real toggle. The one-shot button stays too, renamed "Check now" — it was correct on its own terms, just not a substitute.
- `services/orion-hub/scripts/vision_affect_ambient.py` (new): Hub-owned recurring capture loop — Juniper's explicit direction, overriding an earlier draft that put the loop on the orchestrator (circe). Always running once Hub starts, gated by an in-memory `enabled` flag the toggle route flips. Fails closed on Hub restart by construction (in-memory, no persistence layer). No retries on a failed tick, per Juniper's explicit instruction — just waits for the next scheduled one.
- 5-minute default interval (Juniper's choice, densest of three offered options), a decoupled 5s poll cadence so toggling off takes effect quickly rather than waiting up to a full interval.
- `JuniperMultimodalAffectV1` gains `trigger` (`manual`|`ambient`) and `correlation_id` — Juniper's explicit ask: "ensure the data model has good ability to be correlative with other components in the mesh." One id is now generated per attempt and threaded through the retina RPC, the worker RPC, and the published event (both the payload field and the standard envelope-level `correlation_id`).
- New `POST /api/vision/affect-ambient` (on/off) and `GET /api/vision/affect-ambient/status`, polled by the UI so it reflects real server-owned state (correctly shows "off" after a Hub restart, rather than guessing).

## Outcome moved

Before this PR, the only way to get an AffectGPT read was clicking a button once per attempt. After this PR, flipping the Vision panel's toggle on produces a real recurring signal every 5 minutes until flipped off — the actual capability Juniper asked for — with manual and ambient captures sharing one real mutual-exclusion mechanism so they can't silently collide.

## Current architecture

See PR #1838's own "Current architecture" section for the pre-existing pieces (retina's on-demand clip capture, the orchestrator's `capture_and_assess`, Hub's manual button). This PR's starting point was exactly what #1838 shipped: one-shot only, no recurring trigger anywhere.

## Architecture touched

- `orion/schemas/affectgpt.py` — `JuniperMultimodalAffectV1.trigger`/`.correlation_id`.
- `services/orion-juniper-affective-state/app/main.py` — `capture_and_assess()`/`_capture_clip_via_retina()`/`_call_worker()`/`trigger_assessment()`/`_wrap_event()`/`_publish_event()` all thread one `corr_id` through an attempt; new `CaptureAndAssessRequest` model + `_normalize_trigger()` helper for real validation.
- `services/orion-hub/scripts/vision_affect_ambient.py` (new) — the loop, the shared state, the shared capture-slot lock.
- `services/orion-hub/scripts/main.py` — startup/shutdown task wiring, mirroring the existing `_run_substrate_topic_foundry_scheduler` pattern.
- `services/orion-hub/scripts/api_routes.py` — new toggle/status routes; the existing manual route refactored to share `vision_affect_ambient`'s call site and capture slot.
- `services/orion-hub/templates/index.html` / `static/js/app.js` — toggle button + status line + read-only status polling (no capture-driving client timer — that pattern was explicitly rejected during design).

## Files changed

- `orion/schemas/affectgpt.py` — new `trigger`/`correlation_id` fields
- `services/orion-hub/scripts/vision_affect_ambient.py` — new module: state, shared lock, tick, loop
- `services/orion-hub/scripts/main.py` — task registration/cancellation
- `services/orion-hub/scripts/api_routes.py` — `/api/vision/affect-ambient` (toggle + status), manual route refactor
- `services/orion-hub/templates/index.html`, `static/js/app.js` — toggle UI, status line, polling
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml` — `AFFECT_AMBIENT_*` settings
- `services/orion-juniper-affective-state/app/main.py` — correlation_id/trigger threading, request validation
- `services/orion-juniper-affective-state/README.md` — retired stale "no ambient mode" claim
- `services/orion-juniper-affective-state/evals/test_trigger_eval.py` — consumer-side assertions on the new fields
- Tests: `services/orion-hub/tests/test_vision_affect_ambient.py` (new), `test_vision_affect_capture_api.py` (extended), `services/orion-juniper-affective-state/tests/test_capture_and_assess.py` + `test_trigger.py` (extended)

## Schema / bus / API changes

- **Added:** `JuniperMultimodalAffectV1.trigger` (`Literal["manual","ambient"]`, default `"manual"`), `.correlation_id` (`Optional[str]`); `POST /api/vision/affect-ambient`; `GET /api/vision/affect-ambient/status`; orchestrator's `capture_and_assess` now accepts `trigger` in its body.
- **Removed:** nothing.
- **Renamed:** nothing (the Vision panel's original button is relabeled "Check now" in the UI only — its route, `/api/vision/affect-capture`, is unchanged).
- **Behavior changed:** the manual "Check now" route and the ambient loop now share one exclusive capture slot — a collision returns an explicit 429 instead of an incidental "busy" from retina's own device lock.
- **Compatibility notes:** all additive; `/trigger` and `/capture_and_assess`'s existing request shapes are unchanged (trigger defaults to `"manual"` if omitted).

## Env/config changes

- **Added keys:** `orion-hub`: `AFFECT_AMBIENT_ENABLED` (default `true`, boot-time — whether the loop task is created at all, NOT a live kill switch), `AFFECT_AMBIENT_INTERVAL_SEC` (default `300`), `AFFECT_AMBIENT_POLL_SEC` (default `5`).
- **Removed keys:** none.
- **Renamed keys:** none.
- **`.env_example` updated:** yes (orion-hub).
- **local `.env` synced with `python scripts/sync_local_env_from_example.py`:** yes — `--all-keys orion-hub` on the primary checkout.
- **skipped keys requiring operator action:** none.

## Tests run

```text
cd services/orion-juniper-affective-state && PYTHONPATH=.:<repo root> venv/bin/python -m pytest tests/ -q
  20 passed (+5 over PR #1838's baseline)

cd services/orion-hub && PYTHONPATH=.:<repo root> venv/bin/python -m pytest tests/test_vision_affect_ambient.py tests/test_vision_affect_capture_api.py -q
  23 passed (+7)
```

43 total, all passing. `services/orion-hub/scripts/main.py` also verified to import cleanly end-to-end (confirms the new task registration doesn't break app startup).

## Evals

`services/orion-juniper-affective-state/evals/test_trigger_eval.py` updated with assertions on the new `trigger`/`correlation_id` fields (requires a live bus + live worker — not run in this environment, same as before this PR).

## Docker/build/smoke checks

Not run — no Docker available in this environment for a live Hub/orchestrator pair. `scripts/check_service_env_compose_parity.py` run instead (deterministic, no Docker needed) — `orion-hub` and `orion-juniper-affective-state` both report OK. Real smoke (toggle on, watch it actually recur every 5 minutes, toggle off, confirm it stops) is on Juniper.

## Review findings fixed

Code review (9 finder angles across the full diff, all findings verified) surfaced 9 confirmed issues (a 10th was already fixed mid-review), all addressed:

- **No interlock between manual "Check now" and the ambient loop** (highest severity — collisions only caught incidentally by retina's device lock, with a confusing generic "busy" and no record on the Hub side):
  - Fix: `vision_affect_ambient.try_begin_capture()`/`end_capture()`, a real `threading.Lock` (not just the GIL-atomic boolean this started as — the manual route runs in FastAPI's threadpool, a genuine different OS thread from the ambient loop's asyncio event loop, so this is a real cross-thread race). Manual route now gets an explicit 429 on collision.
  - Evidence: `test_try_begin_capture_excludes_a_concurrent_caller`, `test_check_now_returns_429_when_ambient_holds_the_capture_slot`, `test_ambient_tick_skips_without_retrying_when_manual_holds_the_slot`.
- **`call_capture_and_assess` degraded a malformed non-dict response to a bare `{}`**, silently dropping the error signal for both callers:
  - Fix: returns the same `{"result": {"ok": false, "error": "invalid_response"}}` shape real responses use.
  - Evidence: `test_call_capture_and_assess_returns_error_shape_on_non_dict_response`.
- **Envelope-level `correlation_id` was always an unrelated fresh `uuid4()`**, not the same id as the new payload field, defeating the join for any consumer using the standard mesh-wide convention:
  - Fix: `_publish_event()` now sets it explicitly from the event's own `correlation_id`.
  - Evidence: `test_trigger_generates_a_real_correlation_id_and_surfaces_it_on_the_envelope`.
- **`/trigger`'s plain path never surfaced its internally-generated `corr_id`** back to the published event (`correlation_id` was always `None` on that path):
  - Fix: `trigger_assessment()` now generates `corr_id` once at the top if not supplied, threading it to both `_call_worker` and `_wrap_event`.
  - Evidence: same test as above.
- **Manual captures never touched ambient state**, so the status UI couldn't show "capturing now" during a manual click:
  - Fix: both paths now update the same shared state via `try_begin_capture`/`end_capture`.
  - Evidence: `test_check_now_releases_the_slot_after_completing`.
- **`AFFECT_AMBIENT_ENABLED` was documented as "an operator kill switch"** but only gates task creation at startup — an already-running loop ignores it entirely:
  - Fix: corrected comments in `main.py`/`.env_example` to say what it actually is (boot-time switch) and point at the real live control (the runtime toggle).
- **JS toggle inferred intended state by parsing the button's rendered label text** (`"off"` substring) — a pure copy change would silently break which direction it flips:
  - Fix: tracks a real boolean (`affectAmbientEnabled`) set from the last status fetch.
- **Duplicated trigger-clamp logic in two places, untyped request body**:
  - Fix: single `_normalize_trigger()` helper; real `CaptureAndAssessRequest` pydantic model at the HTTP boundary (422 on a malformed trigger, matching `/trigger`'s own validation pattern).
  - Evidence: `test_capture_and_assess_request_rejects_unrecognized_trigger`, `test_normalize_trigger_only_accepts_the_literal_string_ambient`.
- **New schema fields shipped with a producer test but no consumer-side coverage** (CLAUDE.md §6 — contract changes need both):
  - Fix: added assertions to the live eval (`test_trigger_eval.py`) confirming `trigger`/`correlation_id` populate through a real bus round trip, not just mocked unit tests.

## Restart required

```bash
# athena (orion-hub) -- picks up the new AFFECT_AMBIENT_* settings and
# starts the (initially-off) ambient loop task:
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build hub-app

# circe (orion-juniper-affective-state) -- picks up the corr_id/trigger
# threading and request validation changes:
docker compose --env-file .env --env-file services/orion-juniper-affective-state/.env \
  -f services/orion-juniper-affective-state/docker-compose.yml up -d --build
```

## Risks / concerns

- **Severity: Low.** The shared capture-slot lock is a real fix for the concurrency hazard, but the two callers (manual route, ambient loop) still aren't tested together against a live retina/orchestrator — only against mocks. First live toggle-on session should watch for any collision behavior that doesn't match the unit tests' assumptions.
- **Severity: Low.** `AffectAmbientState`'s non-lock-protected fields (`tick_count`, `last_result_ok`, etc.) remain GIL-atomic single-attribute writes rather than fully wrapped in the lock, unlike `biometrics_cache.py`'s more defensive pattern for similarly cross-thread-shared state in this same service. Flagged by review as "safe today, fragile long-term" — not fixed here since the load-bearing mutual-exclusion concern (the actual capture slot) now has a real lock; revisit if this state grows more fields with multi-step atomic updates.
- **Severity: Low.** The full toggle-on-recurring-capture path has not been run live (would need a live circe + live carbon session watching multiple real ticks). The manual "Check now" path was live-verified in PR #1838; this PR's new loop mechanics are unit-tested with mocks only.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1840
