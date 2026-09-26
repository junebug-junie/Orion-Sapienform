# orion-zwave

Reads the portable AC’s Shelly Wave plug (watts only) and publishes that as
**cabinet cooling** telemetry so Hub Biometrics → Cabinet can show heat beside
cooling load.

This service is a **websocket client**. It never mounts the SONOFF USB stick
and it never turns the plug on or off.

## What it does

1. Connects to the host Z-Wave JS UI websocket (`ZWAVE_JS_WS_URL`).
2. Watches the configured Z-Wave node (`ZWAVE_NODE_ID`) for Meter (watts) and
   Binary Switch (reported on/off — observational only).
3. Every `COOLING_POLL_INTERVAL_SEC` (default 5s), publishes
   `home.cooling.sample.v1` on `orion:home:cooling:sample`.
4. Always publishes a bus-native `SystemHealthV1` heartbeat on
   `orion:system:health` so a dead container is visible even when Z-Wave is
   disabled or unreachable.

Absent meter readings stay absent — the poller never invents `0.0` watts when
the value map has no meter entry.

## What it does not do

- Switch the plug / any actuator command
- Mount `/dev/ttyUSB0` or the SONOFF stick
- Fold AC watts into biometrics `chassis_watts`, `peak_pressure`, or strain
- Own pairing UI (that’s Z-Wave JS)

## Architecture

```text
SONOFF Z-Wave stick (USB on Athena)
  → athena-zwave-js-ui  (owns /dev/zwave + S2 keys; UI :8091; WS 127.0.0.1:3000)
  → orion-zwave         (this service; WS client via host.docker.internal)
  → orion:home:cooling:sample
  → orion-sql-writer    → Postgres home_cooling_sample
  → Hub Biometrics → Cabinet → Cooling strip
```

Hard rule: **only** `/home/athena/zwave-js` compose may map the stick.
`docker-compose.yml` here has **no** `devices:` block.

## Host Z-Wave JS (Athena)

Ops compose: `/home/athena/zwave-js/` (see that directory’s README too).

| Surface | Value |
|---|---|
| UI | `http://athena:8091` (or `http://192.168.1.43:8091`) |
| Websocket | `ws://127.0.0.1:3000` on the host (`127.0.0.1:3000:3000` in compose) |
| Stick inside Z-Wave JS | `/dev/zwave` |
| Stick on host (by-id) | `/dev/serial/by-id/usb-SONOFF_SONOFF_ZWave_Dongle-PZG23_a6e1256f698cf011906026b9d9065118-if00-port0` |

Bring up:

```bash
cd /home/athena/zwave-js
docker compose up -d
docker logs --tail 50 athena-zwave-js-ui
```

## Pair checklist (operator)

Do this once before cooling samples can be real:

1. `cd /home/athena/zwave-js && docker compose up -d`
2. Open the UI → Settings → Z-Wave → serial port **`/dev/zwave`** → start driver
3. Settings → Home Assistant → enable **WS Server** on port **3000**
4. Control panel → Inclusion → include the Shelly Wave plug (button on the plug
   per Shelly docs)
5. Note the plug’s Z-Wave **node id**
6. In `services/orion-zwave/.env` set:
   - `ZWAVE_NODE_ID=<node id>`
   - `ORION_ZWAVE_ENABLED=true` (already the example default on this branch)
7. Recreate this service (see Deploy below)

If step 3 is skipped, `orion-zwave` logs
`ConnectionRefusedError ... ('172.17.0.1', 3000)` and retries — flag on is not
enough; the WS server must be listening.

## Environment

Provenance: `.env_example` → `docker-compose.yml` → `app/settings.py`.
After editing `.env_example`, sync local `.env`:

```bash
python scripts/sync_local_env_from_example.py orion-zwave
```

| Variable | Default (settings) | Meaning |
|---|---|---|
| `ORION_BUS_URL` | Tailscale Redis URL | Orion bus — never `bus-core` / bare `redis` |
| `ORION_BUS_ENABLED` | `true` | Disable all bus publish when `false` |
| `ORION_ZWAVE_ENABLED` | `false` in settings; **`true` in `.env_example`** | Master switch for WS poll + cooling samples |
| `ZWAVE_JS_WS_URL` | `ws://host.docker.internal:3000` | Host zwave-js-server from Docker |
| `ZWAVE_NODE_ID` | `2` | Shelly Wave node id from Z-Wave JS UI |
| `ZWAVE_DEVICE_ID` | `shelly-wave-plug-ac` | Stable id in published samples |
| `ZWAVE_DEVICE_NAME` | `portable_ac` | Human label (`portable_ac`) |
| `COOLING_SAMPLE_CHANNEL` | `orion:home:cooling:sample` | Bus channel |
| `COOLING_POLL_INTERVAL_SEC` | `5` | Sample cadence |
| `HEARTBEAT_INTERVAL_SEC` | `10` | Health heartbeat cadence |
| `ORION_HEALTH_CHANNEL` | `orion:system:health` | Health channel |

When `ORION_ZWAVE_ENABLED=false`, the process publishes heartbeats only (no WS).

## Bus / Hub contracts

| Channel / API | Schema / shape | Role |
|---|---|---|
| `orion:home:cooling:sample` | `home.cooling.sample.v1` (`HomeCoolingSampleV1`) | Cooling sample |
| `orion:system:health` | `SystemHealthV1` | Service liveness |
| `GET /api/cabinet/cooling/latest` | Hub | Latest row (or calm absent) |
| `GET /api/cabinet/cooling/history?window=` | Hub | Chart series (`24h` / `3d` / `7d`) |

Sample role is always `cabinet_cooling` so consumers do not treat this as a
fleet compute node.

## Deploy

From a **worktree** (not the shared primary checkout):

```bash
cd /mnt/scripts/Orion-Sapienform-zwave-cabinet-cooling   # or your worktree
python scripts/sync_local_env_from_example.py orion-zwave
scripts/safe_docker_build.sh orion-zwave up -d --build
# history + Hub strip need these once after first landing:
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
```

Container name: `${PROJECT}-orion-zwave` (e.g. `orion-athena-orion-zwave`).

## Live smoke

1. **Logs:** `docker logs --tail 50 orion-athena-orion-zwave`  
   Expect `enabled=True` and successful connect (not connection refused).
2. **Bus:** subscribe to `orion:home:cooling:sample` — expect
   `home.cooling.sample.v1` with `cooling_watts` while the AC runs.
3. **Hub:** Biometrics → Cabinet → **Cooling** strip next to Nano Environment;
   reported state is text only (no on/off button).
4. **Isolation:** Athena biometrics `peak_pressure` / strain unchanged when AC
   watts move.
5. **Failure:** `docker stop athena-zwave-js-ui` briefly → Cooling goes
   stale/absent, not fake `0.0` watts. Start it again afterward.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ConnectionRefusedError ... 3000` | WS server not enabled, or Z-Wave JS down | UI → enable WS on 3000; `docker ps` for `athena-zwave-js-ui` |
| Heartbeats only, no cooling samples | `ORION_ZWAVE_ENABLED=false` | Set `true` in `.env`, recreate container |
| Samples with `cooling_watts: null` forever | Wrong `ZWAVE_NODE_ID`, or plug not included | Re-check node id in Z-Wave JS; re-include if needed |
| Hub “no samples yet” | sql-writer not subscribed / not rebuilt | Rebuild sql-writer; confirm `orion:home:cooling:sample` in subscribe list |
| Stick busy / driver fails | Another process holds `ttyUSB0` | Only Z-Wave JS may own the stick |

## Tests

```bash
PYTHONPATH="services/orion-zwave:." pytest services/orion-zwave/tests -q
pytest tests/test_home_cooling_bus_catalog.py -q
```

## Spec / plan

- Design: `docs/superpowers/specs/2026-09-25-zwave-cabinet-cooling-design.md`
- Plan: `docs/superpowers/plans/2026-09-25-zwave-cabinet-cooling.md`
- PR report: `docs/superpowers/pr-reports/2026-09-25-zwave-cabinet-cooling-pr.md`
