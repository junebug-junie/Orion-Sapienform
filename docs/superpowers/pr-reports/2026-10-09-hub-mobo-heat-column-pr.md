## Summary

- The biometrics pipeline now keeps the hottest motherboard chipset / voltage-regulator temperature from each node's BMC as `board_temp_c_max` (it was collected as `ilo_thermal_c` and dropped).
- Hub biometrics card: a third "mobo" tile per node (°C), grid 2 → 3 columns.
- Hub biometrics modal: mobo tile and 24h chart per node (`board_temp_c_max` added to `_CHANNEL_COLUMN`).

## Outcome moved

Juniper can see motherboard heat per node in Hub. Before this, the only "max temp" (`temp_c_max`) was lm-sensors' hottest reading, which is the CPU package.

## Current architecture

`services/orion-biometrics/app/ilo.py` polls Redfish Thermal every 60s into `ilo_thermal_c`. `extract_measurements` used only fan and power from it. Hub card showed strain and power per node.

## Architecture touched

- `orion/telemetry/biometrics_pipeline.py`: new `board_temp_c_max` measurement (raw °C, absent when no board sensor).
- `services/orion-hub`: route channel map, `biometrics-view.js`, `index.html`.

## Files changed

- `orion/telemetry/biometrics_pipeline.py`: `_BOARD_SENSOR_RE` + `board_temp_c_max`.
- `tests/test_biometrics_measurements.py`: real live sensor names from HPE iLO (athena) and AMI BMC (circe).
- `services/orion-hub/scripts/biometrics_preview_routes.py`: history channel.
- `services/orion-hub/static/js/biometrics-view.js`: `boardTempFor`, card tile, modal raw channel via `RAW_UNITS`.
- `services/orion-hub/static/js/biometrics-view.test.js`, `services/orion-hub/tests/test_biometrics_preview_api.py`: tests.
- `services/orion-hub/templates/index.html`: `grid-cols-3`.

## Schema / bus / API changes

- Added: `board_temp_c_max` key in `BiometricsSummaryV1.measurements` (free `Dict[str, float]`, no schema change); `/api/biometrics/preview/history{,_multi}` accept `channel=board_temp_c_max`.
- Removed / Renamed: none.
- Compatibility: additive; consumers read named keys only.

## Metric gate

1. Provenance: BMC Redfish `/Chassis/*/Thermal` `Temperatures[].ReadingCelsius` (`ilo.py` `fetch_ilo_snapshot`), filtered to chipset/PCH/VR names.
2. Independence: separate instrument from `temp_c_max` (lm-sensors, CPU-dominated), `gpu*_temp_c` (nvidia-smi) and `cabinet_temp_c` (cabinet sensor). Correlated with load, but measures different parts.
3. Theory: chipset and VRMs are the board's own heat sources; VRM overheating causes throttling and board failure, independent of CPU die temp.
4. Live data 2026-10-09: athena chipset 45 °C, VR 36–44 °C → 45; circe PCH 41 °C, VR 42–50 °C → 50. Not degenerate, rest state well above 0 (ambient ~29–32 °C), as expected for a physical temperature.
5. Existing mechanism: none; `ilo_thermal_c` had no consumer.
6. Reversibility: one dict key, nothing builds on it.

## Env/config changes

None.

## Tests run

```text
node --test services/orion-hub/static/js/biometrics-view.test.js   9 pass
pytest services/orion-hub/tests/test_biometrics_preview_api.py     39 passed
pytest tests/test_biometrics_measurements.py + biometrics suites    167 passed, 12 failed
  (same 12 fail on clean origin/main — pre-existing, unrelated)
```

## Evals run

No eval: display-only measurement, no behavior change in cognition.

## Docker/build/smoke checks

UNVERIFIED live. Needs biometrics rebuilt on athena and circe, and hub on athena, after merge.

## Review findings fixed

- Reviewer subagent: no material findings. Low notes not taken: "VRM" sensor naming (no such vendor in the fleet), fleet-wide board max (no consumer).

## Restart required

From the primary checkout on main after merge:

```bash
scripts/safe_docker_build.sh orion-biometrics up -d --build   # on athena AND on circe
scripts/safe_docker_build.sh orion-hub up -d --build          # on athena
```

## Risks / concerns

- Severity: low. Concern: a new BMC vendor with different sensor names → tile shows "—". Mitigation: absent-not-zero; add names when hardware appears.
