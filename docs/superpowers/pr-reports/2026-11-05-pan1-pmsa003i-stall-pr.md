## Summary

- Fixes office particulates (PMSA003I) going silent after a few minutes and never coming back until the Nano rebooted.
- Root cause: Adafruit PM25 AQI 2.0 `begin_I2C()` returns false on every call after the first (it refuses when its driver object already exists). The firmware called it again after any transient read miss, so every retry failed.
- PM is now read between frames; the newest good reading is sent while under 3 s old. One miss never counts as a lost sensor.
- Review found an older bug in the same loop: the loop clock was read before the re-inits, so a CCS811 or SCD40 that dropped mid-run could never recover. Fixed for all sensors.

## Outcome moved

Before (live, after PR #3124 deployed): 0 of 20 stored office samples had PM; raw frames showed no `particulate` for 30+ min while the sensor still answered on the bus. Raw 32-byte reads dumped on the board were valid (header `42 4D`, checksum OK) about 3 in 5 times; the rest were cut short with `FF` padding mid-update.

After: PM present in every frame right after flashing (`pm25_ug_m3` 0..1); stored sample `office_pm25_ug_m3=1, office_pm10_ug_m3=2` at 01:31Z. 15-minute stored-row check: see "Evals run".

## Current architecture

`firmware/pan-1-office-nano` (both pan-1 Nanos, PR #3124) read the PMSA003I once per 1 s frame; a failed read plus a failed probe set `pm_ok=false`, and the 10 s retry called `begin_I2C()` again.

## Architecture touched

Firmware only. No host, schema, bus, env, or Docker change.

## Files changed

- `firmware/pan-1-office-nano/pan-1-office-nano.ino`: PM poll + cache + stall retry; `begin_I2C()` once; fan gate resets only when the sensor left the bus, latched against millis() wrap; loop clock taken after re-inits.
- `firmware/pan-1-office-nano/README.md`: PM row documents the behaviour and the library trap.
- this report.

## Schema / bus / API changes

- Added / Removed / Renamed: none.
- Behavior changed: particulates are sent from a cached reading (max 3 s old) instead of only the same-frame read.
- Compatibility notes: none; the frame shape is unchanged.

## Env/config changes

None. `.env_example` not touched; no sync needed.

## Tests run

```text
arduino-cli compile (arduino:esp32 2.0.18, PM25 AQI 2.0.0) ... OK, 323133 bytes
Library check: SHT4x, CCS811, TSL2591 begin() delete+recreate their I2C device, so only PM25 AQI has the re-init trap.
```

No firmware unit-test harness exists in this repo (follow-up below).

## Evals run

```text
Live: raw PMSA003I reads dumped on the office Nano (diagnostic sketch) -> valid frames 3/5, truncated 2/5.
Live: 30 stored office samples on the final firmware -> see PR comment / below.
```

## Docker/build/smoke checks

```text
Flashed both Nanos (DFU by serial). Office sample 01:31Z: pm1=0 pm25=1 pm10=2, lux=0.05, temp 25.77.
Closet sample 01:31Z: co2=472 temp=27.8 humidity=23.9.
```

## Review findings fixed

- Finding: `begin_I2C()` creates its driver object even when its own probe NACKs, so a failed first call still killed PM forever.
  - Fix: driver counts as created after the first call; our own probe gates success.
- Finding: `now` read before the retry block made `now - stamp` wrap after a re-init — PM recovery failed ~40%, and the 30 s fan gate could open at 0 s.
  - Fix: take `now` after the re-inits; retry timer uses `millis()`.
- Finding (pre-existing): the same wrap made CCS811 and SCD40 unable to recover after any mid-run drop.
  - Fix: same change.
  - Evidence: compile + reasoning; a real mid-run drop was not reproduced (UNVERIFIED on hardware).
- Finding: fan gate would close for 30 s every 49.7 days.
  - Fix: `pm_fan_settled` latch.

## Restart required

```text
No restart required. Firmware already flashed to both pan-1 Nanos (office E8F60AD53FB4, closet E8F60ABE42B4).
```

## Risks / concerns

- Severity: low. Concern: CCS811/SCD40 mid-run recovery is fixed by reasoning, not observed on hardware. Mitigation: unplug a sensor for 20 s and re-plug; it should be back within ~10 s (CCS811 then warms 20 min).
- Severity: low. Concern: no firmware test harness; these state machines are only compile-checked. Mitigation: follow-up to extract the timing logic into a host-testable header.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3126

🤖 Generated with [Claude Code](https://claude.com/claude-code)
