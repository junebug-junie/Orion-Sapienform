# orion-zwave

Read-only Z-Wave poller for Athena cabinet cooling. Connects to the host Z-Wave JS UI websocket server, reads Shelly Wave plug meter values, and publishes `home.cooling.sample.v1` on `orion:home:cooling:sample`.

This service is a **websocket client only**. It never mounts the USB stick.

## Host Z-Wave JS (Athena)

Z-Wave JS UI runs on the host as `athena-zwave-js-ui`:

- UI: `http://athena:8091`
- Websocket server: `ws://127.0.0.1:3000` (bound to localhost on the host)
- Stick device path **inside the Z-Wave JS container**: `/dev/zwave`
- Host stick by-id path:
  `/dev/serial/by-id/usb-SONOFF_SONOFF_ZWave_Dongle-PZG23_a6e1256f698cf011906026b9d9065118-if00-port0`

Host compose lives at `/home/athena/zwave-js/docker-compose.yml`. Only that container may map the stick.

## Pair checklist (operator)

1. Bring up host Z-Wave JS: `cd /home/athena/zwave-js && docker compose up -d`
2. Open the UI and confirm the driver starts on serial port `/dev/zwave`
3. Settings → enable websocket server on port `3000`
4. Control panel → Inclusion → include the Shelly Wave plug (follow Shelly button procedure)
5. Note the plug's Z-Wave **node id** from the UI
6. Set `ZWAVE_NODE_ID=<node id>` in `services/orion-zwave/.env`
7. Set `ORION_ZWAVE_ENABLED=true` after pairing is verified
8. Restart `orion-zwave`

## Orion container wiring

- `ZWAVE_JS_WS_URL=ws://host.docker.internal:3000` reaches the host websocket from Docker
- `docker-compose.yml` uses `extra_hosts: host.docker.internal:host-gateway`
- **No `devices:` block** in this service — never mount `/dev/ttyUSB0` or the stick here

## Default safe mode

`ORION_ZWAVE_ENABLED=false` by default. The service publishes system health heartbeats only until the Shelly plug is paired and the operator flips the flag.

## Bus output

- Channel: `orion:home:cooling:sample`
- Kind: `home.cooling.sample.v1`
- Health: `orion:system:health` via `HeartbeatOnly`

Absent meter readings stay absent — the poller never publishes `0.0` watts when the value map lacks a meter entry.

## Local dev / tests

```bash
PYTHONPATH="services/orion-zwave:." pytest services/orion-zwave/tests -q
```

After editing `.env_example`:

```bash
python scripts/sync_local_env_from_example.py orion-zwave
```
