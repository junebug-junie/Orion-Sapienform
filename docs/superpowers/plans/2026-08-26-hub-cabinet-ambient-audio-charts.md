# Hub Cabinet ambient audio charts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Multi-day ambient RMS + activity charts on Hub `#cabinet`, reading `orion_biometrics_summary`, with live chips from `/run/orion-audio`.

**Architecture:** Postgres history API + host snapshot latest API + SVG charts in `cabinet-sensors.js`. No new writer/table. No Chart.js.

**Tech Stack:** FastAPI Hub, asyncpg/sqlalchemy patterns already in Hub, Postgres `orion_biometrics_summary`, vanilla JS SVG polylines.

**Spec:** `docs/superpowers/specs/2026-08-26-hub-cabinet-ambient-audio-charts-design.md`

**Worktree:** `/mnt/scripts/Orion-Sapienform-hub-cabinet-ambient-charts` on `feat/hub-cabinet-ambient-charts`

## Global Constraints

- Ambient only — do not add Nano sparklines
- History from `orion_biometrics_summary` only (no new timeseries table)
- Windows: `24h` (default), `3d`, `7d`
- Downsample to ≤ `CABINET_AMBIENT_HISTORY_MAX_POINTS` (default 800)
- Absent ambient keys → empty points, never zero-fill
- Caption: biometrics summary ~30s grain
- Index `(node, timestamp)` ships with the history API
- Bind `/run/orion-audio:ro` on Hub; sync `.env` from `.env_example`
- Poll latest ~1s while tab visible; history only on activate / window change / Refresh
- Commit from this worktree; never stage `.env`

## File map

| File | Responsibility |
|---|---|
| sql-writer boot ALTER + `scripts/sql/` | `orion_biometrics_summary (node, timestamp)` index |
| `cabinet_ambient_routes.py` | `/api/cabinet/ambient/latest` + `/history` |
| Hub settings/compose/env | ambient path, stale, history knobs, volume bind |
| `cabinet-sensors.js` + `index.html` | Ambient section UI |
| Tests | API + panel wiring |

---

### Task 1: Postgres index on biometrics summary

**Files:**
- Modify: `services/orion-sql-writer/app/main.py` — add `CREATE INDEX IF NOT EXISTS orion_biometrics_summary_node_ts_idx ON orion_biometrics_summary (node, timestamp);` next to other boot ALTERs
- Create: `scripts/sql/2026-08-26_biometrics_summary_node_ts_idx.sql` — same statement for out-of-band apply
- Create: `services/orion-sql-writer/tests/test_biometrics_summary_node_ts_index.py` — assert boot SQL source contains the index name

**Interfaces:**
- Produces: index name `orion_biometrics_summary_node_ts_idx`

- [ ] **Step 1:** Failing test that greps main.py / script for index name
- [ ] **Step 2:** Add ALTER + SQL script
- [ ] **Step 3:** `pytest services/orion-sql-writer/tests/test_biometrics_summary_node_ts_index.py -q`
- [ ] **Step 4: Commit** `feat(sql): index orion_biometrics_summary(node, timestamp) for ambient history`

---

### Task 2: Hub ambient latest + history APIs

**Files:**
- Create: `services/orion-hub/scripts/cabinet_ambient_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py` — include router
- Modify: `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml` — `AMBIENT_AUDIO_PATH`, `AMBIENT_AUDIO_STALE_AFTER_SEC`, `CABINET_AMBIENT_HISTORY_NODE=athena`, `CABINET_AMBIENT_HISTORY_MAX_POINTS=800`; volume `/run/orion-audio:/run/orion-audio:ro`
- Create: `services/orion-hub/tests/test_cabinet_ambient_api.py`
- Run: `python scripts/sync_local_env_from_example.py`

**Interfaces:**
- `GET /api/cabinet/ambient/latest` → shape in spec
- `GET /api/cabinet/ambient/history?window=24h|3d|7d` → points + stats; invalid window → 400 or `ok=false`
- Pure helpers testable without live DB: `parse_window`, `downsample_points`, `rows_to_points`
- History DB access: use Hub's existing Postgres pattern (sqlalchemy sync or asyncpg — match nearby Hub read routes; prefer injectable connection/factory for tests)

Mirror snapshot load/stale logic from `cabinet_sensors_routes.py` / biometrics `ambient_audio_snapshot.py`.

- [ ] **Step 1:** Failing unit tests for latest + history helpers + route fixtures
- [ ] **Step 2:** Implement routes + settings + compose bind
- [ ] **Step 3:** `pytest services/orion-hub/tests/test_cabinet_ambient_api.py -q` + env sync
- [ ] **Step 4: Commit** `feat(hub): cabinet ambient latest and multi-day history APIs`

---

### Task 3: Cabinet tab ambient charts UI

**Files:**
- Modify: `services/orion-hub/templates/index.html` — ambient mounts (`cabinetAmbientStatus`, chips, window toggles, `cabinetAmbientRmsChart`, `cabinetAmbientActivityChart`)
- Modify: `services/orion-hub/static/js/cabinet-sensors.js` — fetch latest on poll; fetch history on activate/window/Refresh; SVG polylines; caption
- Modify: `services/orion-hub/tests/test_cabinet_sensors_panel.py` (or new `test_cabinet_ambient_panel.py`) — wiring asserts
- Bump / rely on `HUB_UI_ASSET_VERSION` as Hub already does for cache bust

**UI contract:**
- Window buttons: 24h / 3d / 7d
- Charts: RMS hero, activity secondary
- Live chips from latest
- Do not poll history every 1s
- Cap client state to server response (no unbounded local history array beyond current series)
- Preserve last good on transient errors
- Nano grid unchanged

Reuse SVG style from `cocreation-signals.js` / `attention-organ.js` (inline small helpers in cabinet-sensors.js — do not create a shared cathedral unless trivial copy).

- [ ] **Step 1:** Failing panel wiring tests
- [ ] **Step 2:** HTML + JS
- [ ] **Step 3:** panel tests pass
- [ ] **Step 4: Commit** `feat(hub): Cabinet tab ambient RMS/activity day charts`

---

### Task 4: Docs cross-links + spec status

**Files:**
- Update design status → implemented on branch awaiting merge
- Short Hub README / biometrics README pointer to Cabinet ambient charts
- Commit: `docs: Hub Cabinet ambient charts status and operator notes`

---

## Acceptance (branch done when)

1. API tests green
2. Panel wiring tests green
3. Index statement present in sql-writer boot path
4. Manual Athena check after Hub rebuild (operator): `#cabinet` shows chips + 24h chart from Postgres

## Spec coverage

| Spec item | Task |
|---|---|
| Index | 1 |
| latest + history APIs | 2 |
| downsample / windows | 2 |
| Charts + chips + poll rules | 3 |
| Docs | 4 |
