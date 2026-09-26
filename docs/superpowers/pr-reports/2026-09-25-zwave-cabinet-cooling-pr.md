## Summary

- Adds read-only Z-Wave cabinet cooling telemetry: `home.cooling.sample.v1` on `orion:home:cooling:sample`, thin `orion-zwave` websocket client, sql-writer persistence, and Hub Cabinet Cooling strip.
- Host Z-Wave JS owns the USB stick; Orion never mounts the dongle. No on/off control path.
- Absent-is-not-zero throughout: missing meter → omit watts; Hub shows calm absent / stale, not `0.0`.
- AC cooling watts are isolated from chassis `peak_pressure`, strain, and PDU totals.

## Outcome moved

Athena’s portable AC (Shelly Wave Plug US, Z-Wave node 2) now publishes live wall watts into Orion. Operators can see cooling load beside Nano environment in Biometrics → Cabinet.

## Current architecture

Cabinet Biometrics showed Nano sensors and ambient audio only. No Z-Wave stack, no cooling bus channel, no AC watt read path.

## Architecture touched

- **Contract:** `orion/schemas/telemetry/home_cooling.py`, `orion/bus/channels.yaml`, `orion/schemas/registry.py`
- **Producer:** `services/orion-zwave/` (WS client → bus)
- **Consumer:** `services/orion-sql-writer/` (`home_cooling_sample`), `services/orion-hub/` (`/api/cabinet/cooling/*`, Cabinet strip)
- **Host (ops, not in git secrets):** `/home/athena/zwave-js/` — Z-Wave JS UI, S2 keys in store, USA RF region, WS on `:3000`

## Files changed

- `orion/schemas/telemetry/home_cooling.py`: `HomeCoolingSampleV1`
- `orion/bus/channels.yaml`, `orion/schemas/registry.py`, `config/metrics/metric_definitions.lock.json`
- `services/orion-zwave/`: new read-only poller (+ seed fix for `start_listening` snapshot)
- `services/orion-sql-writer/`: subscribe + SQL model
- `services/orion-hub/`: cooling routes, Cabinet UI, tests
- `docs/superpowers/specs|plans|pr-reports/2026-09-25-zwave-cabinet-cooling-*`

## Schema / bus / API changes

- Added: `home.cooling.sample.v1` / `HomeCoolingSampleV1` on `orion:home:cooling:sample`
- Added: Hub `GET /api/cabinet/cooling/latest`, `GET /api/cabinet/cooling/history`
- Behavior: read-only cooling telemetry only; no control channel
- Compatibility: safe if flag off; Athena currently enabled with node 2

## Env/config changes

- Added (`orion-zwave`): `ORION_ZWAVE_ENABLED`, `ZWAVE_JS_WS_URL`, `ZWAVE_NODE_ID`, `ZWAVE_DEVICE_ID`, `ZWAVE_DEVICE_NAME`, `COOLING_SAMPLE_CHANNEL`, `COOLING_POLL_INTERVAL_SEC`, …
- sql-writer: subscribe `orion:home:cooling:sample`
- `.env_example` updated; local `.env` synced (`ORION_ZWAVE_ENABLED=true`, `ZWAVE_NODE_ID=2` on Athena)

## Tests run

```text
pytest tests/test_home_cooling_bus_catalog.py -q  # 2 passed
PYTHONPATH=services/orion-zwave:. pytest services/orion-zwave/tests -q  # 9 passed
PYTHONPATH=services/orion-sql-writer:. pytest …/test_home_cooling_sample_sql_shape.py -q  # 4 passed
pytest services/orion-hub/tests/test_cabinet_cooling_routes.py … panel contracts  # 42 passed (cooling + related)
```

## Evals run

```text
No dedicated eval harness; gate tests + live Athena smoke below.
```

## Docker/build/smoke checks

```text
Host: athena-zwave-js-ui — driver READY, USA RF, Shelly Wave Plug US = Node 002 (security None)
orion-zwave logs: Published cooling sample watts=33.11 switch_on=True
Hub Cabinet Cooling strip: live path after rebuild
```

## Review findings fixed

- Finding: connect() leaked WS on bootstrap failure → `74d28967a`
- Finding: Hub empty DB looked like live error → `5bd6035c0`
- Finding: watts stayed null after pair (no start_listening ingest) → `f9f60a809`
- Finding: metric definition drift for new channel → `7404feca7`

## Restart required

```bash
# Athena (already done for live path):
scripts/safe_docker_build.sh orion-zwave up -d --build
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: Low
- Concern: Node included with **security None** (no S2 PIN available at pair time). Fine for read-only watts; re-include with S2 later if desired.
- Concern: Host Z-Wave JS keys/ops live under `/home/athena/zwave-js/` (not in repo) — do not rotate keys after devices are paired.
- Follow-up (non-blocking): WS reconnect if Z-Wave JS restarts mid-session; optional retention on `home_cooling_sample`.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2359
