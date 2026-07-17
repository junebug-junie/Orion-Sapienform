# Hub Drives Analytics — operator UI (PR3)

**Date:** 2026-07-17
**Branch:** `feat/hub-drives-analytics-ui`
**PR:** https://github.com/junebug-junie/Orion-Sapienform/pull/1127
**Plan:** `docs/superpowers/plans/2026-07-16-hub-drives-analytics.md` (Tasks 5-9)
**Spec:** `docs/superpowers/specs/2026-07-16-hub-drives-analytics-design.md`

## Summary

- Ships the Hub **Drives** operator tab: a standalone `/drives-analytics` page (7 cards — gauges, contributors, KPI strip, dual time series, divergence, goals, cross-links) with goal-aligned coloring, tooltips, and gentle 30s polling.
- Wires a session-preserving `#drives` tab into the Hub shell (`index.html` + `app.js`), mirroring the Causal Geometry iframe-isolation pattern exactly — toggling the tab never reloads the Hub chat session.
- Adds a `GET /drives-analytics` page route in `api_routes.py`.
- Documents the surface in `orion/autonomy/README.md`, `services/orion-hub/README.md`, and `services/orion-spark-concept-induction/README.md`.
- Fixes two issues found in orchestrator code review: a refresh race condition and duplicated align-color lookup logic.

PR 1 (persistence: `tick_attribution`/`tension_kinds` columns) and PR 2 (read-only API: subjects/snapshot/window/series/goal-alignment/divergence) were already merged and live on `main` (PRs #1123, #1124) before this work started.

## Outcome moved

Juniper can now see the live six-drive `DriveEngine` economy at a glance in the Hub — real contributor history, program-health KPIs, tick-rate + pressure time series, goal-alignment coloring, and a divergence/integrity check against `drive_state.v1` — without CLI archaeology (`measure_autonomy_gate.py`, `drive_state_divergence_audit.py`) or per-turn chat side panels.

## Orchestration approach

Two independent background agents ran in parallel in one worktree (`/mnt/scripts/Orion-Sapienform-hub-drives-analytics-ui`, branch `feat/hub-drives-analytics-ui`), scoped to disjoint file sets:

1. **Frontend agent** — built the standalone page (Tasks 5-7 of the plan): `templates/drives-analytics.html`, `static/js/drives-analytics.js`, the `GET /drives-analytics` route, and `tests/test_drives_analytics_page.py`. Ran its own internal code-review pass and fixed 2 findings before returning.
2. **README agent** — wrote the three README sections (Task 9), reading the real backend source (`drives_analytics.py`, `drives_analytics_queries.py`, `drive_audit.py`, concept-induction's `audit.py`/`drive_attribution.py`) so the docs describe what actually exists.

Both completed cleanly with no file collisions. The orchestrator then did the Hub shell tab wiring (Task 8: `index.html` + `app.js`) directly — a small, mechanical, exactly-specified change not worth a third agent dispatch — appending 6 shell-wiring tests to the frontend agent's test file.

## Current architecture

Before this PR: `drive_audits` persistence and the 6 read-only `/api/drives-analytics/*` endpoints existed, but no UI consumed them — the data was invisible to operators.

## Architecture touched

`services/orion-hub/` only (templates, static JS, one new route, tests, README). No bus/schema/registry changes. No backend/API changes (already merged in prior PRs).

## Files changed

- `services/orion-hub/templates/drives-analytics.html` (new): standalone page shell, 7 card containers, controls row, tooltip popover.
- `services/orion-hub/static/js/drives-analytics.js` (new, ~890 lines): fetch/render/polling bundle for all 7 cards, `TOOLTIP_COPY`, color-mode logic (`combined`/`align`/`funnel`), Live/Window contributors toggle, URL-query state sync.
- `services/orion-hub/scripts/api_routes.py`: added `GET /drives-analytics` page route (mirrors `causal_geometry_page` exactly).
- `services/orion-hub/templates/index.html`: `#drives` nav tab + iframe panel section.
- `services/orion-hub/static/js/app.js`: tab-switch/hash/refresh wiring for `#drives`, mirroring every existing tab's pattern.
- `services/orion-hub/tests/test_drives_analytics_page.py` (new): 21 tests — route registration, template/card ids, standalone-bundle isolation, tooltip structure, polling, color-mode/window-picker presence, no-mutation-calls, goals-card no-action-buttons, gate-verdict labeling, cross-link content, divergence banner, and 6 shell-wiring tests (session-preserving tab toggle, iframe-only refresh, no full-page navigation).
- `orion/autonomy/README.md`, `services/orion-hub/README.md`, `services/orion-spark-concept-induction/README.md`: operator-facing docs.

## Schema / bus / API changes

None. This PR only builds a UI on top of the already-merged read-only API.

## Env/config changes

None. No `.env_example` keys added/removed/renamed.

## Tests run

```
cd services/orion-hub
python -m pytest tests/test_drives_analytics_page.py tests/test_drives_analytics_api.py tests/test_drives_analytics_helpers.py tests/test_causal_geometry_page.py tests/test_causal_geometry_api.py -q
→ 75 passed

python -m pytest ../orion-sql-writer/tests/test_drive_audit_sql_shape.py -q
→ 16 passed

node --check static/js/drives-analytics.js && node --check static/js/app.js
→ syntax OK

git diff --check
→ clean
```

No eval harness exists for this UI-only slice beyond the pytest gate tests above.

## Docker/build/smoke checks

Not run — this patch only adds new frontend files (templates/static/route), no config/dependency/DB changes. A Hub container rebuild is needed to pick up the new static/template files before operators can use the tab (see Restart below).

## Review findings fixed

8-angle high-effort code review (parallel finder agents + orchestrator verification against the already-merged backend contract):

- **Finding (real, fixed):** `refreshAll()`/`fetchAll()` had no request-sequencing guard — an overlapping poll tick and a control-driven refresh (subject/window change) could resolve out of order, silently rendering stale-selection data while the URL/selects already showed the new selection.
  - Fix: added a monotonic `refreshToken`; any response superseded by a newer refresh is dropped instead of overwriting `payloads`.
  - Evidence: `services/orion-hub/static/js/drives-analytics.js`, `refreshAll()`.
- **Finding (real, fixed):** `gaugeColorClass` and `goalBorderClass` each independently re-implemented the same per-drive `color_align` lookup, with inconsistent Tailwind shade numbers (500 vs 700) — drift risk.
  - Fix: extracted `alignColorForKey()` as the single source of truth for both call sites; also dropped `gaugeColorClass`'s unused `kpis` destructured parameter.
  - Evidence: `services/orion-hub/static/js/drives-analytics.js`, `alignColorForKey`, `gaugeColorClass`, `goalBorderClass`.
- **4 findings verified as unreachable:** candidates about missing null-guards on `record_count`/`active_count`/timestamp fields were checked against the already-merged backend (`services/orion-hub/scripts/drives_analytics_queries.py`) — every field is guaranteed non-null/well-formed (`int(x or 0)` casts, `isoformat()` of real `datetime` objects) on every code path, so no fix was needed.
- **3 findings noted as follow-ups, not fixed:** `fetchSection()` duplication across standalone Hub pages, `DRIVE_KEYS`/`ALLOWED_HOURS`/verdict-string hardcoding in JS (no shared JS/backend contract mechanism exists yet), and refetching the static `/subjects` endpoint on every poll tick — all pre-existing repo patterns or low-impact at operator-dashboard/30s-cadence scale.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

sql-writer and its migration are unaffected by this PR (already deployed in PR1/PR2).

## Risks / concerns

- Severity: Low — Sparklines/tick-rate chart are hand-rolled SVG polylines, not a charting library; adequate but visually blunt.
- Severity: Low — `DRIVE_KEYS`/`ALLOWED_HOURS`/verdict strings are hardcoded in the client JS with no shared contract to the Python source of truth; a future backend change to these constants would silently desync the UI. Mitigation: same pattern as every other standalone Hub page; fixing it repo-wide is out of scope for this PR.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1127
