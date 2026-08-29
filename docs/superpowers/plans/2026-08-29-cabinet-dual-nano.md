# Dual-Nano cabinet sensors — implementation plan

> **Goal:** Merge two USB Nanos (climate/UV/lidar + mag/IMU) into one cabinet telemetry stream.

**Architecture:** Second systemd reader → `/run/orion-sensors/b/*.json`; shared merge in `orion/telemetry/cabinet_snapshot_merge.py`; biometrics + Hub consume merged frame.

**Branch:** `feat/cabinet-dual-nano` in worktree `/mnt/scripts/Orion-Sapienform-cabinet-dual-nano`

## Tomorrow (hardware)

1. Flash **same firmware** on both Nanos: `./scripts/flash_athena_cabinet_nano.sh`
2. List by-id paths: `ls /dev/serial/by-id/usb-Arduino_Nano_ESP32_*`
3. Pin paths in env files:
   - `/etc/default/orion-cabinet-sensors` → Nano A `ORION_CABINET_DEVICE_GLOB=...`
   - `/etc/default/orion-cabinet-sensors-b` → Nano B `ORION_CABINET_DEVICE_GLOB=...`
4. `sudo ./scripts/setup_athena_cabinet_sensors.sh` (installs both units)
5. `sudo systemctl start orion-cabinet-sensors-b.service`
6. Set `CABINET_SENSORS_B_PATH` / `CABINET_BOOT_B_PATH` in biometrics + Hub `.env`
7. `./scripts/diagnose_athena_cabinet_sensors.sh` — expect merged `environment`+`uv`+`lidar`+`magnetic`+`imu`

## Tests run

```text
pytest tests/test_cabinet_snapshot_merge.py \
  services/orion-biometrics/tests/test_cabinet_snapshot.py \
  services/orion-hub/tests/test_cabinet_sensors_api.py -q
→ 26 passed
```
