## Summary

- The world-pulse news digest already keeps two separate backend structures — the 12 regular source cards (`digest.items`) and the autonomy gap-fill "curiosity" articles (`digest.curiosity_followups`). This surfaces **both in one combined hub/chat digest**.
- Curiosity was previously invisible on the hub World Pulse panel (it only rode along inside `rendered_markdown`/`structured_payload` and the disabled email body). It now renders as an "Orion Went Looking (gaps our sources missed)" section directly under the digest cards.
- Backend data stays separate: gap-fill articles are **not** merged into the regular cards, and **no new email/delivery** is added — one digest, both sections.
- Hardened external article links (curiosity + worth-reading) to only accept `http(s)` schemes.

## Outcome moved

On the live hub surface the daily digest now shows the 12 curated cards *and* the curiosity gap-fill in a single view. Previously the "Orion went looking" articles never rendered in the panel.

## Current architecture

`orion-world-pulse` builds `DailyWorldPulseV1` with `items` (primary scrape → clustered cards) and a separate `curiosity_followups` field (autonomy web-fetch for missing sections). `render_hub_digest` produced a `HubWorldPulseMessageV1` carrying `cards` structurally but only exposed curiosity via `rendered_markdown`/`structured_payload`. The hub UI `renderWorldPulseDetails` rendered cards / worth-reading / worth-watching / rollups but never curiosity.

## Architecture touched

- Shared schema contract `HubWorldPulseMessageV1` (additive field).
- `orion-world-pulse` hub renderer (producer).
- `orion-hub` static JS panel renderer (consumer/UI).

## Files changed

- `orion/schemas/world_pulse.py`: add `curiosity_followups: list[CuriosityFollowupV1]` to `HubWorldPulseMessageV1` (additive, default empty).
- `services/orion-world-pulse/app/services/renderers.py`: `render_hub_digest` populates the new field from `digest.curiosity_followups`.
- `services/orion-hub/static/js/app.js`: `normalizeWorldPulsePayload` fallback carries `curiosity_followups`; `renderWorldPulseDetails` renders the "Orion Went Looking" section (section label, query, article list with salience + link); added `worldPulseSafeHttpUrl()` scheme guard applied to the curiosity and worth-reading anchors.
- `services/orion-world-pulse/tests/test_renderers_curiosity.py`: assert the hub message carries the structured field (populated + empty-safe).

## Schema / bus / API changes

- Added: `HubWorldPulseMessageV1.curiosity_followups` (list, default `[]`).
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing consumers.
- Compatibility: additive with default; backward-compatible under `extra="forbid"`. No bus channel changes; sql-writer persists the message as a `payload_json` blob so no column mapping breaks.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not required (no env template changed).
- skipped keys requiring operator action: none.

## Tests run

```text
.venv/bin/python -m pytest services/orion-world-pulse/tests -q
  -> 70 passed

.venv/bin/python -m pytest services/orion-hub/tests/test_world_pulse_proxy_routes.py -q
  -> 3 passed

.venv/bin/python -m pytest services/orion-world-pulse/tests/test_renderers_curiosity.py -q
  -> 4 passed

node --check services/orion-hub/static/js/app.js  -> JS_OK
HubWorldPulseMessageV1 round-trip with curiosity_followups -> roundtrip_ok
```

## Evals run

```text
No world-pulse eval harness for this render seam; covered by unit tests at
services/orion-world-pulse/tests/test_renderers_curiosity.py.
```

## Docker/build/smoke checks

```text
Not rebuilt in this session. Change is code-only (schema + producer + static JS).
Runtime verified against the live latest run (run 339f9acf): 12 cards + 1 curiosity
followup (5 GPU articles), confirming the digest data the panel now renders.
```

## Review findings fixed

- Finding (Minor): external article `href` not scheme-guarded (`app.js`), URLs come from untrusted search/RSS results.
  - Fix: added `worldPulseSafeHttpUrl()` and applied it to the new curiosity anchor and the sibling worth-reading anchor (only `http(s)` linked; others render as plain text).
  - Evidence: `node --check` OK; anchors still use `textContent` + `rel="noopener noreferrer"`.
- Non-blocking (no action): fallback curiosity carry is dead on the live path but kept for defensive consistency with the worth-reading/watching fallbacks.
- Non-blocking (no action): empty curiosity section is hidden, matching the plaintext renderer which omits the block when empty.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-world-pulse/.env -f services/orion-world-pulse/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

(Hard-refresh the hub page to bust the cached `app.js`.)

## Risks / concerns

- Severity: low
- Concern: static JS is cache-busted only by hard refresh; the panel already reads the full digest so the field is present without a schema change on the `/latest` path.
- Mitigation: restart commands above; the schema field is additive/backward-compatible.

## Follow-ups

- No JS test harness exists for `renderWorldPulseDetails`; the curiosity DOM render branch is unit-tested only at the Python schema/renderer seam (matches existing practice — sibling renderers are also untested). Worth a small hub JS test harness later.

## PR link

Branch pushed: `feat/world-pulse-combined-curiosity-digest`
Open PR: https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/world-pulse-combined-curiosity-digest
