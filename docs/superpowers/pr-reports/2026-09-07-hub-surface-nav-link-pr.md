# PR: Hub Surface nav link

## Summary

- The Hub Surface dashboard (`services/orion-hub/scripts/hub_surface_routes.py`, PR #2137) had a working page at `/hub-surface`, a template, a script, and tests — but no button anywhere in the Hub UI to reach it. Only way in was typing the URL.
- Added a "Hub Surface" tab to Hub's main nav bar (`index.html`), an embedded `<section data-panel="hub-surface">` with an iframe pointing at `/hub-surface` plus Refresh / Open-standalone buttons.
- Added `hub_surface_tab.js` to drive that panel's show/hide + hash lifecycle, following the same self-routed pattern `self_observability.js` uses (the page is a standalone document with its own `<html>`/JS, so app.js's generic `setActiveTab` router doesn't own it).
- Added `hub-surface` to this repo's own panel-reachability gate's exception list (`test_hub_panel_reachability.py`'s `PANELS_WITHOUT_APP_JS_TAB`) — that test exists precisely to catch an unreachable panel like this, and caught it against this diff during review.
- Added `test_hub_surface_nav.py` covering the new nav tab, panel, iframe, and script tag.

## Outcome moved

Before: Hub Surface only reachable by typing `/hub-surface` directly. After: one click from the main Hub nav, same as every other standalone panel (Substrate, Curiosity, Sentience Program, etc.).

## Current architecture

`services/orion-hub/templates/index.html` renders a single-page shell with a hash-routed nav (`hubPrimaryNav`, `a[data-hash-target]`) and one `<section data-panel="...">` per tab. Most panels are shown/hidden by a big `setActiveTab()` switch in `app.js`. Two panels (`self-observability`, and now `hub-surface`) are standalone full pages served at their own route and embedded via `<iframe>`; because they carry their own `<html>`/head/script, they route their own show/hide via a small dedicated JS file instead of `app.js`, and are named in `test_hub_panel_reachability.py`'s exception list so the generic reachability gate doesn't flag them as unreachable.

## Architecture touched

`services/orion-hub` only — template, one new static JS file, two test files. No contract, schema, bus, or env changes.

## Files changed

- `services/orion-hub/templates/index.html`: added the "Hub Surface" nav tab, the `#hub-surface` panel section (iframe + Refresh/Open-standalone), and the `hub_surface_tab.js` script include.
- `services/orion-hub/static/js/hub_surface_tab.js` (new): tab activation/deactivation, hash lifecycle, iframe refresh — mirrors `self_observability.js`.
- `services/orion-hub/tests/test_hub_surface_nav.py` (new): static-content coverage for the new wiring, mirroring `test_self_observability_ui_panel.py`.
- `services/orion-hub/tests/test_hub_panel_reachability.py`: added `hub-surface` to `PANELS_WITHOUT_APP_JS_TAB` with an inline reason (self-routed, same as `self-observability`).

## Schema / bus / API changes

None.

## Env/config changes

None.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/pytest \
  services/orion-hub/tests/test_hub_surface_nav.py \
  services/orion-hub/tests/test_self_observability_ui_panel.py \
  services/orion-hub/tests/test_hub_panel_reachability.py -q
12 passed in 0.18s
```

`test_hub_surface_routes.py` was attempted but fails to collect on both this worktree and the primary checkout (pre-existing, unrelated to this change): `Settings()` raises `ValidationError` for `CHANNEL_VOICE_TRANSCRIPT`/`CHANNEL_VOICE_LLM`/`CHANNEL_VOICE_TTS` (and, in a fresh worktree, also `CHANNEL_COLLAPSE_INTAKE`/`CHANNEL_COLLAPSE_TRIAGE`) — an env-parity gap in this environment's `.env`, not something this patch touches or introduces.

## Evals run

None — this is a static UI wiring fix with no behavior/quality dimension an eval would cover; the tests above are the appropriate gate.

## Docker/build/smoke checks

Not run. This is template/static-asset-only; no Python import, dependency, port, or boot-time behavior changed. Manual verification path: reload Hub, click the "Hub Surface" tab, confirm the dashboard renders in the embedded iframe and "Open standalone" opens the same `/hub-surface` route.

## Review findings fixed

- Finding: `hub-surface` wasn't added to `PANELS_WITHOUT_APP_JS_TAB`, so this repo's own reachability gate (`test_hub_panel_reachability.py::test_every_panel_can_be_activated`) failed against the diff.
  - Fix: added `hub-surface` to the exception set with an inline comment, matching the existing `self-observability` entry.
  - Evidence: `pytest services/orion-hub/tests/test_hub_panel_reachability.py -q` — 3 passed (previously would report `missing_active = {'hub-surface'}`).
- Finding: `hub_surface_tab.js` is a near-byte-identical copy of `self_observability.js` (same activation/deactivation/hash logic), duplicated rather than factored into a shared helper.
  - Fix: not applied. With only two instances of this pattern, extracting a shared module now would touch `self_observability.js` (a separately shipped, separately tested file) purely for a stylistic gain, and `test_self_observability_ui_panel.py` asserts the literal function bodies live in that file — refactoring it would require updating an unrelated, working test. Leaving the second copy in place until a third self-routed panel actually shows up (rule-of-three) is the smaller, lower-risk patch; a comment could be added at that point pointing to both files if this happens again.
  - Evidence: reviewed and accepted as a documented trade-off, not silently dropped.

## Restart required

```text
No restart required for the fix itself — Hub reads templates/static assets at request time
(hub_surface_routes.py serves hub_surface.html the same way). If Hub's process caches
index.html or static assets at boot in this deployment, restart with:
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build orion-hub
```

## Risks / concerns

- Severity: low
- Concern: `hub_surface_tab.js` duplicates `self_observability.js`'s tab-routing logic (see review finding above).
- Mitigation: accepted per rule-of-three; revisit if a third self-routed panel is added.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/hub-surface-nav-link
