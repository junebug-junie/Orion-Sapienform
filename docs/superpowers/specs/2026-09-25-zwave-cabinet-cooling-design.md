# Z-Wave cabinet cooling telemetry — design

**Date:** 2026-09-25  
**Status:** Draft for Juniper review (approved verbally; awaiting spec sign-off)  
**Branch intent:** `docs/zwave-cabinet-cooling` → `feat/zwave-cabinet-cooling`  
**Worktree:** `/mnt/scripts/Orion-Sapienform-zwave-cabinet-cooling`

## Arsonist summary

Give Athena a **read-only cooling sense** for the portable AC that dumps heat Orion produces: a host-owned Z-Wave JS stack talks to the SONOFF stick and Shelly Wave plug; a thin `orion-zwave` service publishes watts/state on the bus; Hub’s Biometrics **Cabinet** panel shows that cooling load **inside** the same view as Nano sensors so heat and cooling sit side by side. No on/off. No Hub Z-Wave management tab. No folding AC watts into chassis `peak_pressure` / strain.

## Decisions locked

| Topic | Choice |
|---|---|
| Purpose | Cooling load for Athena heat (portable AC on Shelly Wave), not generic home IoT |
| Control | **Read-only** — no toggle, no power-intent, no auto-on |
| Pairing / stick admin | Z-Wave JS UI only (option B) — not a Hub tab |
| Hub surface | **Inside Cabinet** Biometrics subview (`#cabinet`), not a sibling subtab and not a top-level Z-Wave tab |
| Aggregation | Labeled cooling series; **must not** enter `chassis_watts`, `peak_pressure`, cluster strain, or fake a fleet node |
| USB ownership | Host **Z-Wave JS** owns the SONOFF stick; Orion service is a websocket client |
| Service boundary | New thin `services/orion-zwave/` — not biometrics, not power-guard, not Hub |
| Stick path (Athena) | `/dev/serial/by-id/usb-SONOFF_SONOFF_ZWave_Dongle-PZG23_a6e1256f698cf011906026b9d9065118-if00-port0` → `ttyUSB0` |

## Current architecture (grounded)

### Hardware (verified on Athena 2026-09-25)

- SONOFF Z-Wave 800 Dongle Plus (Dongle-PZG23, EFR32ZG23) present as Silicon Labs CP210x (`10c4:ea60`).
- Stable by-id symlink exists; `athena` is in `dialout` / `plugdev`.
- No Home Assistant / Z-Wave JS running on Athena or reachable Tailscale peers at design time.
- Port `8123` on Athena is Orion consolidation-runtime, not Home Assistant.

### Closest Orion patterns

| Pattern | Lesson |
|---|---|
| `orion-power-guard` | Host owns USB/UPS daemon; thin service polls and publishes bus events. Do **not** reuse this service — UPS shutdown ≠ cooling telemetry. |
| Cabinet Nano + ambient audio | Host reader / host daemon → snapshot or bus → Hub Cabinet reads for operator view; biometrics may also consume for measurements. |
| Hub Cabinet UI | `#cabinet` inside `#biometricsModalRoot`; `cabinet-sensors.js` polls while visible. Ambient + Nano already live there. |
| Biometrics PDU / iLO | Chassis/BMC watts for **compute** nodes — wrong instrument for wall AC. |

Existing Cabinet path (do not bypass for Nano/ambient):

```text
Nano / mic → host readers → /run/orion-sensors|audio
  → biometrics (+ Hub file APIs)
  → Biometrics modal → Cabinet subview
```

## End-to-end data flow (proposed)

```text
SONOFF Z-Wave stick (USB)
  → Z-Wave JS UI (Docker on Athena; owns /dev/serial/by-id/...PZG23...; S2 keys; pairing UI)
  → WebSocket API (localhost)
  → orion-zwave (new service; read-only client)
       → orion:home:cooling:sample  (schema: home.cooling.sample.v1)
       → SystemHealthV1 heartbeat
  → sql-writer (optional v1 / same patch if cheap) → Postgres history
  → Hub GET /api/cabinet/cooling/latest (+ history)
  → Cabinet panel: Cooling strip beside Environment / ambient
```

Hard rules:

1. Only Z-Wave JS may open the stick. `orion-zwave` and `orion-biometrics` must not mount `/dev/ttyUSB0`.
2. Missing / stale / unpaired plug → omit cooling keys and show `absent` / stale in UI — never zero-fill watts.
3. AC watts never enter host `peak_pressure`, `strain`, `chassis_watts`, or `pdu_watts` totals.
4. No control channel in v1 (no `orion:home:cooling:command`).

## Proposed schema / bus / API changes

### Bus

| Channel | Schema | Direction | Notes |
|---|---|---|---|
| `orion:home:cooling:sample` | `home.cooling.sample.v1` | publish | Periodic sample from `orion-zwave` |
| (existing) `orion:system:health` | `SystemHealthV1` | publish | Service liveness |

Register channel in `orion/bus/channels.yaml` and schema in `orion/schemas/` + `registry.py` in the same changeset. Do not copy the unregistered `orion:power:events` gap.

### `home.cooling.sample.v1` (sketch)

```json
{
  "schema": "home.cooling.sample.v1",
  "ts": "2026-09-25T...",
  "node": "athena",
  "role": "cabinet_cooling",
  "controller": {
    "ready": true,
    "driver": "zwave-js",
    "device_path": "/dev/serial/by-id/usb-SONOFF_...PZG23..."
  },
  "device": {
    "id": "shelly-wave-plug-ac",
    "name": "portable_ac",
    "product": "Shelly Wave Plug",
    "online": true
  },
  "measurements": {
    "cooling_watts": 412.5,
    "cooling_volts": 120.1,
    "cooling_amps": 3.4,
    "cooling_power_factor": 0.98
  },
  "state": {
    "switch_on": true
  },
  "provenance": {
    "zwave_node_id": 2,
    "source": "zwave-js",
    "sample_age_sec": 0.4
  }
}
```

Rules:

- `measurements` keys present only when the meter value is actually known.
- `state.switch_on` is **observational** (what the plug reports), not an actuator affordance. Hub may show “on/off as reported” as text; no button.
- `role` stays `cabinet_cooling` so consumers do not treat this as a fleet compute node.

### Hub HTTP

| Route | Role |
|---|---|
| `GET /api/cabinet/cooling/latest` | Latest sample (from Hub bus cache and/or Postgres), Cabinet poll while visible |
| `GET /api/cabinet/cooling/history?window=` | Optional history for a small watts chart (mirror ambient window buttons) |

Absent-is-not-zero: same invariant as cabinet sensors.

### Persistence

Prefer sql-writer consumer for history charts (same grain usefulness as cabinet ambient). Exact table name left to implementation plan; must be queryable by Hub history route.

## Files likely to touch

### New

- `services/orion-zwave/` — `app/main.py`, `settings.py`, `.env_example`, `docker-compose.yml`, `Dockerfile`, `README.md`, `tests/`, `evals/` (minimal)
- Host ops compose for Z-Wave JS (Athena): e.g. `/home/athena/zwave-js/docker-compose.yml` + volume for network keys (not committed secrets). Document path in service README; do **not** put S2 keys in Hub or biometrics `.env`.
- `orion/schemas/.../home_cooling.py` (or under telemetry)
- Hub: `scripts/cabinet_cooling_routes.py`, Cabinet markup + `cabinet-sensors.js` Cooling strip
- Docs: this spec; service README; optional PR report

### Likely edits

- `orion/bus/channels.yaml`
- `orion/schemas/registry.py`
- `services/orion-hub/templates/index.html` (`#cabinet` section only)
- `services/orion-hub/static/js/cabinet-sensors.js`
- `services/orion-hub/scripts/api_routes.py` (wire routes)
- `services/orion-sql-writer/` consumer if history is in v1
- Mesh allowlist for Athena if required (`mesh-utilities/...`)

### Explicitly not touched

- `services/orion-biometrics` USB / Z-Wave client code
- `services/orion-power-guard` (UPS only)
- Chassis measurement extractors that feed `peak_pressure`

## UI design (Cabinet)

Inside `#cabinet`, add a **Cooling** block (title along the lines of “Cooling — portable AC (Shelly Wave)”) that shows:

- live `cooling_watts` (and optional V/A if present)
- reported on/off as read-only status text
- controller ready / device online / age
- optional small history chart sharing the same window controls pattern as ambient (24h / 3d / 7d)

Layout goal: Environment (temp) and Cooling (watts) readable **in the same Cabinet scroll** without switching Biometrics subtabs.

No primary-nav Z-Wave item. No on/off button.

## Env / config (service)

Provenance: `services/orion-zwave/.env_example` → compose → `settings.py`. After keys land, run `python scripts/sync_local_env_from_example.py`.

| Variable (sketch) | Purpose |
|---|---|
| `ORION_BUS_URL` | Tailscale Redis bus (never bus-core hostname) |
| `ZWAVE_JS_URL` | Websocket URL to local Z-Wave JS |
| `COOLING_SAMPLE_CHANNEL` | default `orion:home:cooling:sample` |
| `COOLING_DEVICE_ID` / name filter | Which Z-Wave node is the AC plug |
| `COOLING_POLL_INTERVAL_SEC` | Sample cadence |
| `ORION_ZWAVE_ENABLED` | Master switch (default safe/off until stick + plug paired) |

## Non-goals (v1)

- Switching the plug / Orion-driven AC automation
- Home Assistant
- Folding cooling watts into biometrics host pressures or cluster aggregate
- Mounting the SONOFF stick into any Orion cognition container
- Multi-site Z-Wave mesh beyond Athena’s one stick + Shelly plug
- Keyword detectors or chat personality changes

## Risks / concerns

| Severity | Concern | Mitigation |
|---|---|---|
| High | Two processes claim the USB stick | Only Z-Wave JS gets the device; document and refuse stick mounts in Orion compose |
| Med | S2 / network keys leak into Hub `.env` | Keys stay in Z-Wave JS volume; Orion holds only WS URL + device id |
| Med | Zero-filled watts look like “AC off / calm” | Absent-is-not-zero; stale badge |
| Med | Operators confuse cooling watts with chassis load | Explicit `cabinet_cooling` role + Cabinet labeling |
| Low | Z-Wave JS not yet running at merge | Feature flag off; README onboarding steps; live smoke after pair |

## Acceptance checks

1. Z-Wave JS on Athena shows the SONOFF controller Ready; Shelly Wave plug included via Z-Wave JS UI.
2. `orion-zwave` publishes `home.cooling.sample.v1` with real `cooling_watts` while the AC runs (and omits/zeros-not when meter absent).
3. Channel + schema registered; gate tests cover parse/publish shape.
4. Hub Biometrics → Cabinet shows Cooling strip with live watts beside Nano environment; no on/off control.
5. Host biometrics `peak_pressure` / strain unchanged when AC watts move.
6. Kill Z-Wave JS or unplug stick → cooling UI goes stale/absent, not silent zero.
7. README documents pair flow (Z-Wave JS UI) and Athena device by-id path.

## Recommended next patch (implementation order)

1. Host Z-Wave JS compose on Athena + pair Shelly (ops; unblocks live values).
2. Contract: schema + channel + registry + fixture tests.
3. `orion-zwave` read-only poller → bus (+ health).
4. Hub Cabinet Cooling strip + latest API (history if sql-writer is cheap in same PR, else follow-up).
5. Focused tests + eval/smoke + PR report.

## Metric quality gate (cooling_watts)

1. **Provenance:** Shelly Wave meter via Z-Wave JS node values → `orion-zwave` sample field `measurements.cooling_watts`.
2. **Independence:** Not a transform of `chassis_watts` / PDU / RAPL; separate physical meter on the AC wall circuit.
3. **Theory:** Wall electrical power of the cooler removing cabinet/compute heat — cooling *effort*, not compute load.
4. **Live sanity:** Confirm varies with AC on/off/fan; can read near-zero when plug reports off; not decayed-to-zero by an unrelated loop.
5. **Existing mechanism:** No Orion Z-Wave producer today; PDU/iLO are chassis-only.
6. **Reversibility:** Feature-flag service off; channel unused → Hub strip absent. Do not bake into `ACTIVE_INFERENCE` / strain formulas in v1.

## Open items for implementation plan (not blocking this spec)

- Exact Z-Wave JS image tag and auth mode for localhost WS.
- Whether history ships in the first PR or immediately after.
- Stable device selector (node id vs name) once the Shelly is paired.
