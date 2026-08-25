# Hub Cabinet Sensors Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a top-level Hub `#cabinet` tab that polls `/api/cabinet/sensors/latest` while visible and shows raw Athena Nano channels plus Hub-local `cabinet_*` activity pressures.

**Architecture:** Hub reads `/run/orion-sensors/latest.json` and optional `boot.json` via a bind mount, reuses `orion.telemetry.cabinet_sensors` helpers, and mirrors the Field Attention panel’s activate/deactivate poll lifecycle.

**Tech Stack:** FastAPI, Hub settings/env/compose, vanilla JS panel module, pytest static UI wiring tests.

## Global Constraints

- Top-level Hub nav tab `#cabinet` only (not biometrics strip expand).
- Poll ~1000 ms **only while tab visible**; stop on hide.
- Raw frame + pressures; absent sensors show `absent`, never zero-filled.
- Hub reads host snapshot directly; no biometrics HTTP proxy; no WebSocket for v1.
- API: `GET /api/cabinet/sensors/latest` with `ok`, `age_sec`, `snapshot`, `boot`, `measurements`, `pressures`.
- Env defaults: `CABINET_SENSORS_PATH=/run/orion-sensors/latest.json`, `CABINET_BOOT_PATH=/run/orion-sensors/boot.json`.
- Compose bind: `/run/orion-sensors:/run/orion-sensors:ro`.
- Pressure strip labeled as Hub operator-debug activity (local `CabinetSensorTracker`).
- No firmware / biometrics cognition pipeline changes.
- Mirror Field Attention wiring pattern (`field-attention.js` + `test_field_attention_operator_panel.py`).
- Sync local `.env` from `.env_example` if env keys change.

---

### Task 1: Cabinet sensors API + Hub mount/config

**Files:**
- Create: `services/orion-hub/scripts/cabinet_sensors_routes.py`
- Create: `services/orion-hub/tests/test_cabinet_sensors_api.py`
- Modify: `services/orion-hub/scripts/api_routes.py` (include router)
- Modify: `services/orion-hub/app/settings.py`
- Modify: `services/orion-hub/.env_example`
- Modify: `services/orion-hub/docker-compose.yml`
- Run: `python scripts/sync_local_env_from_example.py` from worktree root (or document skipped keys)

**Interfaces:**
- Consumes: host JSON files; `extract_cabinet_measurements`, `compute_cabinet_pressures`, `CabinetSensorTracker` from `orion.telemetry.cabinet_sensors`
- Produces: `GET /api/cabinet/sensors/latest` →
  ```python
  {
    "ok": bool,
    "age_sec": float | None,
    "snapshot": dict | None,  # full latest.json object when readable
    "boot": dict | None,
    "measurements": dict[str, float],
    "pressures": dict[str, float],
  }
  ```

- [ ] **Step 1: Write failing API tests** covering missing snapshot, unreadable JSON, fresh frame with magnetic/uv present, stale status/age (`ok=false` but snapshot returned), missing boot (`boot=null`), absent sensor keys not zero-filled, and `cabinet_sensor_staleness` 0 fresh / 1 stale when sensors present.

- [ ] **Step 2: Run tests → expect FAIL**

Run: `cd services/orion-hub && python -m pytest tests/test_cabinet_sensors_api.py -q`

- [ ] **Step 3: Implement settings, compose mount, route, and wire into `api_routes.py`**

Implementation notes:
- Settings fields: `CABINET_SENSORS_PATH`, `CABINET_BOOT_PATH`, `CABINET_SENSORS_STALE_AFTER_SEC` (default 10.0 unless biometrics already shares a named constant you can reuse).
- Load snapshot without raising; missing file → `ok=false`, `snapshot=null`, empty measurements/pressures.
- When readable: return full host snapshot dict (status/device/received_at/frame).
- Build sensors dict `{frame, received_at, stale}` for helpers.
- Module-level `CabinetSensorTracker` for Hub-local pressures; on fresh sensors call extract+compute; always set `cabinet_sensor_staleness` to `0.0` or `1.0` when a sensors payload exists.
- Boot load best-effort; invalid/missing → `boot=null`.

- [ ] **Step 4: Run tests → expect PASS**; sync env; commit

```bash
git add services/orion-hub/scripts/cabinet_sensors_routes.py \
  services/orion-hub/tests/test_cabinet_sensors_api.py \
  services/orion-hub/scripts/api_routes.py \
  services/orion-hub/app/settings.py \
  services/orion-hub/.env_example \
  services/orion-hub/docker-compose.yml
# do not stage .env
git commit -m "feat(hub): add cabinet sensors latest API and host mount"
```

---

### Task 2: Cabinet tab UI + poll lifecycle

**Files:**
- Create: `services/orion-hub/static/js/cabinet-sensors.js`
- Create: `services/orion-hub/tests/test_cabinet_sensors_panel.py` (mirror `test_field_attention_operator_panel.py`)
- Modify: `services/orion-hub/templates/index.html` (nav link, panel section, script tag)
- Modify: `services/orion-hub/static/js/app.js` (all Field Attention-style registration points)
- Modify: `services/orion-hub/README.md` (short operator note: mount, restart, tab)

**Interfaces:**
- Consumes: `GET /api/cabinet/sensors/latest`
- Produces: `window.OrionCabinetSensors` with `activate()` / `deactivate()`; panel IDs:
  - nav: `cabinetTabButton`, hash `#cabinet`
  - panel: `id="cabinet" data-panel="cabinet"`
  - mounts: `cabinetStatus`, `cabinetSensorGrid`, `cabinetPressureStrip`, `cabinetRefreshBtn`

- [ ] **Step 1: Write failing UI wiring tests** asserting nav/panel/script tag, app.js element binding + visibility + hash routing + activate/deactivate hooks, JS poll URL `/api/cabinet/sensors/latest`, `POLL_MS = 1000`, and absent-not-zero rendering helpers/comments.

- [ ] **Step 2: Run tests → expect FAIL**

Run: `cd services/orion-hub && python -m pytest tests/test_cabinet_sensors_panel.py -q`

- [ ] **Step 3: Implement panel markup + JS + app.js wiring**

UI requirements:
- Status strip: reader status, device, seq/uptime, age, I2C addresses from boot when present.
- Sensor grid tiles: environment, uv, magnetic, particulate, lidar, imu — values or `absent`.
- Pressure strip labeled “activity (Hub)” with the seven cabinet pressure keys when present.
- Poll only while active; keep last good render on transient fetch error with poll-error badge.
- No-snapshot message names `orion-cabinet-sensors.service`.

- [ ] **Step 4: Run panel + API tests → expect PASS**; commit

```bash
git commit -m "feat(hub): add Cabinet tab for realtime Nano sensors"
```

---

### Task 3: Integration gate + README restart note

**Files:**
- Modify: `services/orion-hub/README.md` if Task 2 left restart commands incomplete
- Verify only

- [ ] **Step 1: Run focused suite**

```bash
cd services/orion-hub && python -m pytest tests/test_cabinet_sensors_api.py tests/test_cabinet_sensors_panel.py -q
```

- [ ] **Step 2: Confirm README lists Hub restart after compose/env mount change**

Exact restart for Juniper (do not run sudo yourself):

```bash
scripts/safe_docker_build.sh orion-hub up -d --build
```

- [ ] **Step 3: Commit only if README needed a fix**; otherwise no-op

---

## Self-review notes for controller

- Spec coverage: API, mount/env, tab UI, poll lifecycle, absent-is-not-zero, pressures strip, tests.
- Placeholders: none intentional.
- Ordering: Task 1 before Task 2 (UI depends on API path); Task 3 is verification.
