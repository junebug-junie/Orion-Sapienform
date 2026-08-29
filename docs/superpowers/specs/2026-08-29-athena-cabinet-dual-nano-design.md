# Athena cabinet dual-Nano — design

Date: 2026-08-29  
Status: approved for implementation  
Branch: `feat/cabinet-dual-nano`

## Arsonist summary

Second Arduino Nano ESP32 on USB carries MMC5603 + BNO085 on its own I2C bus. Host runs two pinned serial readers; biometrics and Hub merge both snapshots into one logical cabinet frame. Same firmware sketch on both boards; no bus/schema registry changes.

## Decisions locked

| Topic | Choice |
|---|---|
| Nano 1 sensors | BME680, LTR390, VL53L1X (environment, UV, lidar) |
| Nano 2 sensors | MMC5603, BNO085 (magnetic, IMU) |
| Firmware | Same `athena-cabinet-nano.ino` on both; omit missing sub-objects |
| Readers | Two systemd units, each pinned via `ORION_CABINET_DEVICE_GLOB` |
| Primary paths | `/run/orion-sensors/latest.json`, `boot.json` (unchanged) |
| Secondary paths | `/run/orion-sensors/b/latest.json`, `b/boot.json` |
| Merge owner | `orion/telemetry/cabinet_snapshot_merge.py` (shared) |
| Cognition | One merged frame → existing biometrics / field `athena` channels |
| Hub | `GET /api/cabinet/sensors/latest` returns merged frame + `sources` debug |
| PMSA003I | Not in hardware kit — ignore |

## Merge rules

1. Load primary and optional secondary snapshot files independently.
2. For each frame channel (`environment`, `uv`, `magnetic`, `particulate`, `lidar`, `imu`), take the block from the **newest non-stale** source that has it.
3. If both sources have the same channel (misconfiguration), newer `received_at` wins.
4. Merged `stale=true` only when **no** non-stale channel contributed (all sources stale/missing/empty).
5. Absent-is-not-zero preserved — never zero-fill missing channels.

## Reader pinning

When two Nanos are present, **each service MUST set** `ORION_CABINET_DEVICE_GLOB` to a unique `/dev/serial/by-id/usb-Arduino_Nano_ESP32_*` path. Default glob (`sorted` first match) is unsafe with multiple devices.

## Non-goals

- Second field node
- Redis/bus payload changes
- Dedicated per-Nano firmware builds
- 3D lidar visualization

## Acceptance checks

- [ ] Both readers write distinct snapshot paths with pinned globs
- [ ] `./scripts/diagnose_athena_cabinet_sensors.sh` shows both nodes
- [ ] Biometrics sample includes env+uv+lidar from A and mag+imu from B when both fresh
- [ ] Hub Cabinet tab shows merged tiles; `sources` exposes per-Nano debug
- [ ] Single-Nano deploy unchanged when secondary paths unset
