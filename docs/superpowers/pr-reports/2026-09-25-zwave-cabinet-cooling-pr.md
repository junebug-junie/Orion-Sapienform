## Summary

- Adds read-only Z-Wave cabinet cooling telemetry: `home.cooling.sample.v1` on `orion:home:cooling:sample`, thin `orion-zwave` websocket client, sql-writer persistence, and Hub Cabinet Cooling strip.
- Host Z-Wave JS owns the USB stick; Orion never mounts the dongle. Default safe mode: `ORION_ZWAVE_ENABLED=false` (heartbeat-only).
- Absent-is-not-zero throughout: missing meter → omit watts; Hub shows `ok: false` / stale, not `0.0`.
- AC cooling watts are isolated from chassis `peak_pressure`, strain, and PDU totals.

## Outcome moved

Operators can see portable AC wall load beside Nano environment data in Biometrics → Cabinet once the Shelly Wave plug is paired and enabled. Until pair, the feature is inert but deployed (heartbeats + absent UI state).

## Current architecture

Cabinet Biometrics showed Nano sensors and ambient audio only. No Z-Wave stack, no cooling bus channel, no AC watt read path.

## Architecture touched

- **Contract:** `orion/schemas/telemetry/home_cooling.py`, `orion/bus/channels.yaml`, `orion/schemas/registry.py`
- **Producer:** `services/orion-zwave/` (WS client → bus)
- **Consumer:** `services/orion-sql-writer/` (`home_cooling_sample` table), `services/orion-hub/` (`/api/cabinet/cooling/*`, Cabinet strip JS)
- **Host (ops):** Z-Wave JS UI at `/home/athena/zwave-js/docker-compose.yml`

## Files changed

- `orion/schemas/telemetry/home_cooling.py`: `HomeCoolingSampleV1` schema
- `orion/bus/channels.yaml`, `orion/schemas/registry.py`: channel + registry
- `services/orion-zwave/`: new read-only poller service
- `services/orion-sql-writer/`: subscribe + SQL model for cooling samples
- `services/orion-hub/`: cooling routes, Cabinet UI strip, tests
- `tests/test_home_cooling_bus_catalog.py`: catalog gate
- `docs/superpowers/specs/2026-09-25-zwave-cabinet-cooling-design.md`: status → implemented

## Schema / bus / API changes

- Added: `home.cooling.sample.v1` / `HomeCoolingSampleV1` on `orion:home:cooling:sample`
- Added: Hub `GET /api/cabinet/cooling/latest`, `GET /api/cabinet/cooling/history`
- Removed: —
- Renamed: —
- Behavior changed: new read-only cooling telemetry path; no control channel
- Compatibility notes: feature off until `ORION_ZWAVE_ENABLED=true` after pair

## Env/config changes

- Added keys: `ORION_ZWAVE_ENABLED`, `ZWAVE_JS_WS_URL`, `ZWAVE_NODE_ID`, `ZWAVE_DEVICE_ID`, `ZWAVE_DEVICE_NAME`, `COOLING_SAMPLE_CHANNEL`, `COOLING_POLL_INTERVAL_SEC` (`orion-zwave`); sql-writer cooling channel subscribe
- Removed keys: —
- Renamed keys: —
- `.env_example` updated: yes (`orion-zwave`, `orion-sql-writer`)
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (worktree-local `orion-zwave/.env` from example; hub/sql-writer symlinked from primary checkout)
- skipped keys requiring operator action: none

## Tests run

```text
git diff --check  # pass
pytest tests/test_home_cooling_bus_catalog.py -q  # 2 passed
PYTHONPATH=services/orion-zwave:. pytest services/orion-zwave/tests -q  # 4 passed
PYTHONPATH=services/orion-sql-writer:. pytest services/orion-sql-writer/tests/test_home_cooling_sample_sql_shape.py -q  # 4 passed
pytest services/orion-hub/tests/test_cabinet_cooling_routes.py -q  # 11 passed
Total: 21 passed
```

## Evals run

```text
No eval harness for this feature slice; gate tests cover contract + route shape + absent-safe UI/API behavior.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-zwave up -d --build  # OK — heartbeat-only (ORION_ZWAVE_ENABLED=false)
scripts/safe_docker_build.sh orion-sql-writer up -d --build  # OK — subscribes orion:home:cooling:sample
scripts/safe_docker_build.sh orion-hub up -d --build  # OK
docker exec orion-athena-hub curl -fsS http://127.0.0.1:8080/api/cabinet/cooling/latest
  → {"ok": false, "age_sec": null, "sample": null}  # expected pre-pair
docker logs athena-zwave-js-ui → "Z-Wave driver not inited, no port configured"; nodes.json not found
Live bus watts + Hub strip with real AC: UNVERIFIED — blocked on Shelly pair (see Concerns)
```

## Review findings fixed

- Finding: Hub empty cooling state could read as zero watts
  - Fix: absent API + calm empty-state copy in Cabinet strip
  - Evidence: `test_cabinet_cooling_routes.py`, hub commit `5bd6035c0`

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform-zwave-cabinet-cooling
# After Juniper pairs Shelly and sets ORION_ZWAVE_ENABLED=true:
scripts/safe_docker_build.sh orion-zwave up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Already restarted during Task 6 smoke (heartbeat-only mode).

## Risks / concerns

- Severity: **High (ops blocker)**
- Concern: Shelly Wave plug not paired; Z-Wave JS driver not configured on serial port
- Mitigation: Follow pair checklist in `services/orion-zwave/README.md`; do not enable until node id confirmed

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2359

## Concerns (post-merge operator steps)

Z-Wave JS on Athena (`athena-zwave-js-ui`) is up but **not configured**:

1. `cd /home/athena/zwave-js && docker compose up -d`
2. Open `http://athena:8091` → Settings → serial port `/dev/zwave`, enable websocket on port `3000`
3. Control panel → Inclusion → include Shelly Wave plug (Shelly button procedure)
4. Note node id → set `ZWAVE_NODE_ID` in `services/orion-zwave/.env`
5. Set `ORION_ZWAVE_ENABLED=true`, restart `orion-zwave`
6. Run live smoke: bus subscribe, Hub Cabinet strip, failure test (stop Z-Wave JS → stale/absent)
