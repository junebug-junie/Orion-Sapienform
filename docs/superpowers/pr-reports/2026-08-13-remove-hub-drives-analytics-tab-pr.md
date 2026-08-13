## Summary

- Removes the Hub `Drives` tab entirely: standalone `/drives-analytics` page, the shell-embedded iframe panel, all 6 `/api/drives-analytics/*` endpoints, the two backend query/helper modules, the frontend JS/HTML, and the 3 dedicated test files.
- Drops the underlying `drive_audits` Postgres table (346,066 frozen rows, 261MB) this page read from — snapshotted first.
- Both actions follow up on this session's earlier finding: the `DriveEngine` producer this page visualized was already fully retired 2026-07-30 (PR #1486), and the page had been kept alive since then purely as a "kill the producer, not the reader" historical-forensics view. Juniper's direct call: kill the reader too, now that the history behind it is gone as well.

## Outcome moved

- Recovers 261MB of dead Postgres storage.
- Removes ~700 lines of dead frontend/backend surface (routes, queries, JS wiring, template) that could never again show anything but a frozen 2026-07-30 snapshot.
- Closes out a "removal debt" item explicitly flagged in the prior session as a disclosed-but-not-done follow-up.

## Current architecture

- `services/orion-hub`: FastAPI app (`scripts/api_routes.py`) serving a single-page shell (`templates/index.html`) with per-feature tabs, several implemented as session-preserving iframes over standalone pages (Causal Geometry, Concept Atlas, Drives Analytics, ...).
- Drives Analytics specifically: `GET /drives-analytics` (page), 6 `GET /api/drives-analytics/*` endpoints, backed by `scripts/drives_analytics.py` (KPI/verdict helpers) and `scripts/drives_analytics_queries.py` (asyncpg queries against Postgres `drive_audits`), rendered via `templates/drives-analytics.html` + `static/js/drives-analytics.js`, embedded in `index.html`'s `#drives` shell section and wired in `static/js/app.js`.

## Architecture touched

- `services/orion-hub/scripts/api_routes.py`: removed the page route + 6 API routes.
- `services/orion-hub/static/js/app.js`: removed all `drivesAnalytics*` DOM refs, tab-switch branches, and event listeners (10 distinct removal sites).
- `services/orion-hub/templates/index.html`: removed the nav tab button and the `#drives` panel section (nav link + iframe embed).
- Deleted outright: `scripts/drives_analytics.py`, `scripts/drives_analytics_queries.py`, `static/js/drives-analytics.js`, `templates/drives-analytics.html`, `tests/test_drives_analytics_api.py`, `tests/test_drives_analytics_helpers.py`, `tests/test_drives_analytics_page.py`.
- `services/orion-hub/README.md`, `orion/autonomy/README.md`: replaced the detailed operator docs with short "removed, here's why, here's the trail" pointers rather than deleting the sections outright.
- Postgres: `drive_audits` table dropped (out-of-band, via `psql`, not a schema migration — this table was never behind a versioned migration mechanism to begin with).

## Files changed

- `services/orion-hub/scripts/api_routes.py`: removed `drives_analytics_page()` and 6 `api_drives_analytics_*()` route handlers.
- `services/orion-hub/static/js/app.js`: removed 10 distinct `drivesAnalytics*`/`isDrivesAnalytics` reference sites (DOM lookups, tab-switch fallback, panel visibility toggle, tab button styling, hash router branch, click handlers, refresh handler).
- `services/orion-hub/templates/index.html`: removed the `Drives` nav tab button and the `#drives`/`data-panel="drives"` section.
- `services/orion-hub/scripts/drives_analytics.py`, `drives_analytics_queries.py`: deleted.
- `services/orion-hub/static/js/drives-analytics.js`: deleted.
- `services/orion-hub/templates/drives-analytics.html`: deleted.
- `services/orion-hub/tests/test_drives_analytics_api.py`, `test_drives_analytics_helpers.py`, `test_drives_analytics_page.py`: deleted.
- `services/orion-hub/README.md`: section "5.4 Drives Analytics panel" replaced with a short removal note + pointer to this PR report.
- `orion/autonomy/README.md`: section "Hub Drives Analytics" replaced with a short removal note.
- `services/orion-spark-concept-induction/README.md`: review finding — stale prose still claimed the Hub Drives tab "still renders it, relabeled (historical)"; updated to reflect the removal and fixed a dead anchor link (the `orion/autonomy/README.md` heading it linked to was renamed by this same patch).
- `services/orion-sql-writer/app/main.py`: review finding (HIGH) — removed the `drive_audits` `CREATE TABLE IF NOT EXISTS`/`ALTER TABLE`/`CREATE INDEX` boot DDL from `lifespan()`. This ran unconditionally on every startup; leaving it in place would have silently resurrected the just-dropped table (empty, but present) on the next `orion-sql-writer` restart, undoing the drop. `DriveAuditSQL`'s write path itself (worker.py's `MODEL_MAP` entry, settings.py's route map, the model class) was deliberately left wired — see Risks/concerns.
- `services/orion-sql-writer/tests/test_drive_audit_sql_shape.py`: replaced `test_boot_ddl_create_and_alter_include_attribution_columns` (asserted the now-removed DDL text existed) with `test_boot_ddl_no_longer_creates_drive_audits` (asserts it's gone). The other 16 tests in this file, which exercise the still-live `DriveAuditSQL` model/derivation/insert-only logic, are unchanged.
- `orion/core/schemas/drives.py`: review finding — removed `DRIVE_KEYS`, a tuple constant whose only stated justification ("real historical-data readers still need it: ... `drives_analytics*.py` reading `drive_audits`") no longer holds; confirmed via repo-wide grep that nothing actually imports it (only two files *mention* it in comments, both updated).
- `orion/field/channel_glossary.py`: review finding — docstring cited `drives_analytics.py`'s `_repo_root_candidates()` as an existing-pattern precedent; updated to note that file's since been removed while keeping the (still-accurate) pattern description.
- `orion/bus/channels.yaml`: review finding — `orion:memory:drives:audit` entry's comment still said the (now-removed) Hub page read `drive_audits`; updated with the full current picture (page/table removed, DDL removed, `DriveAuditSQL` write-mapping deliberately left as a disclosed follow-up).

## Schema / bus / API changes

- Removed: `GET /drives-analytics`, `GET /api/drives-analytics/subjects`, `GET /api/drives-analytics/snapshot`, `GET /api/drives-analytics/window`, `GET /api/drives-analytics/series`, `GET /api/drives-analytics/goal-alignment`, `GET /api/drives-analytics/divergence`.
- Removed: Postgres `drive_audits` table (dropped, not soft-deleted).
- No bus channel changes (this surface was Postgres-read-only; the 3 bus channels this data ultimately traced back to — `orion:memory:drives:state`, `orion:memory:tension:event`, `orion:memory:drives:audit` — were already marked `producer_services: []` since 2026-07-30, unaffected by this patch).
- Compatibility notes: any bookmark/link to `/drives-analytics` or `#drives` now 404s / lands on the default Hub tab. No known external consumers of the removed API endpoints (all were Hub-internal, iframe-embedded only).

## Env/config changes

- None.

## Tests run

```text
python3 -c "import ast; ast.parse(open('services/orion-hub/scripts/api_routes.py').read())"  -> OK
python3 -c "import ast; ast.parse(open('services/orion-sql-writer/app/main.py').read())"  -> OK
node --check services/orion-hub/static/js/app.js  -> OK

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-hub/tests/ -q
1080 passed, 32 failed, 5 skipped  (reproduced twice, identical count both times)

/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-sql-writer/tests/ -q \
  --ignore=services/orion-sql-writer/tests/test_dream_model_constraints.py
195 passed, 18 failed, 3 errors
```

Both sets of failures are pre-existing and unrelated — confirmed by running the same test files against an unmodified `main` checkout: identical failure counts AND identical failure names reproduce there too. Hub: spot-checked the two most drives-adjacent-sounding failures directly (`test_llm_route_selector.py`, `test_substrate_review_runtime_hub_debug.py`'s "shell tab switching" test) — both fail identically on `main`; `grep -i drive` across the full output confirms zero drives-related failures among the rest. Sql-writer: full failure-name set matches `main` exactly (18 failed/195 passed both places); `test_dream_model_constraints.py` is excluded from both runs because collecting the whole `tests/` directory together hits a pre-existing, unrelated SQLAlchemy `Table 'dreams' is already defined for this MetaData instance` collection error, reproduced identically on `main`.

The one sql-writer test that *did* need a real code change — `test_drive_audit_sql_shape.py::test_boot_ddl_create_and_alter_include_attribution_columns` — is a review finding, not a pre-existing failure; see Files changed and Review findings fixed.

## Evals run

No dedicated eval harness exists for `orion-hub`'s UI surfaces beyond its pytest suite — not applicable here (pure removal, no new behavior to eval).

## Docker/build/smoke checks

Not run — no local Docker daemon access in this session for a full rebuild/redeploy smoke. Syntax-checked both changed source files directly (`ast.parse`, `node --check`); the full pytest suite exercises `api_routes.py`'s route registration at FastAPI app-construction time (would fail loudly on a broken route decorator), so route-level breakage would already have surfaced.

## Review findings fixed

- Finding (HIGH): the `drive_audits` table drop (done earlier this session, snapshotted first) was not durable — `orion-sql-writer`'s boot DDL still had `CREATE TABLE IF NOT EXISTS drive_audits (...)` running unconditionally on every startup. The very next restart would have silently resurrected an empty table, undoing the drop, with no error or signal that it happened.
  - Fix: removed the `CREATE TABLE`/`ALTER TABLE`/`CREATE INDEX` block from `services/orion-sql-writer/app/main.py`'s `lifespan()`. `DriveAuditSQL`'s write-path wiring (worker.py `MODEL_MAP`, settings.py route map, the model class itself) was deliberately left in place — it shares a `_JSONB` type declaration with other still-live sql-writer models, and fully untangling it is a bigger, separate task than this patch's scope. With the table gone and the boot DDL removed, the worst case if `DriveAuditV1` were ever somehow republished is a loud, clean "relation does not exist" error, not a silent resurrection.
  - Evidence: `test_drive_audit_sql_shape.py::test_boot_ddl_no_longer_creates_drive_audits` (new) asserts the DDL text is gone; the rest of that file's 16 tests confirm the deliberately-kept write path still works.
- Finding (MEDIUM): `services/orion-spark-concept-induction/README.md` still described the Hub Drives tab as live ("still renders it, relabeled (historical)") and linked to an `orion/autonomy/README.md` anchor this same patch renamed, breaking the link.
  - Fix: updated the prose to describe the actual current state (tab and table both removed) and fixed the link.
  - Evidence: re-read the updated section; anchor now matches the renamed heading.
- Finding (LOW): PR report's test-failure count needed independent reproduction, not just a first-run number.
  - Fix: reran the hub suite twice (1080/32/5 both times) and the sql-writer suite once more after the DDL fix (195/18/3), both matched against a fresh `main` comparison run in the same session.
  - Evidence: see Tests run above.
- Finding (LOW): three stale comments across the repo cited now-deleted files (`orion/core/schemas/drives.py`'s `DRIVE_KEYS` justification, `orion/field/channel_glossary.py`'s pattern-precedent docstring, `orion/bus/channels.yaml`'s channel comment) as if they still existed.
  - Fix: `DRIVE_KEYS` itself removed (confirmed zero real importers, only comment mentions); the other two comments updated to note the removal while keeping their still-accurate underlying content.
  - Evidence: repo-wide grep for `DRIVE_KEYS` importers (zero hits outside comments) before removing it; both updated docstrings re-read for accuracy.

## Restart required

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: low
- Concern: no live click-through of the actual running Hub UI to visually confirm the tab is gone and no console errors fire (no Docker rebuild done this session).
- Mitigation: `node --check` confirms `app.js` is syntactically valid; grep confirms zero remaining `drivesAnalytics*`/`#drives`/`drives-analytics` references anywhere in `services/orion-hub/`; the full pytest suite (which does import and construct the FastAPI app, exercising route registration) passes with no new failures.

- Severity: low
- Concern: `DriveAuditSQL`'s write-path wiring in `orion-sql-writer` (worker.py's `MODEL_MAP` entry, settings.py's route map, the model class + its `_JSONB` type declaration shared with other live models) is still fully in place — only the boot-time table creation was removed. This is dead-but-wired code now, the same shape as the ~27 files' worth of now-redundant local workarounds disclosed (not removed) in the prior `scripts/platform` rename PR.
- Mitigation: with the table dropped and its boot DDL gone, the failure mode if this ever mattered again is a loud SQL error (relation does not exist), not silent data resurrection — the actual danger this review finding identified. Fully untangling `DriveAuditSQL` (shared `_JSONB` type, worker dispatch table, settings route map) is flagged as a real, separate follow-up, not done here to keep this patch's diff proportionate to what was asked.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1614
