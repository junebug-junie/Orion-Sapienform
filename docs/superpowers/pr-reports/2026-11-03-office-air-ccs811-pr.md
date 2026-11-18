## Summary

- Onboards the SparkFun Zio Qwiic CCS811 air-quality sensor on pan-1's office node (Nano ESP32, TCA9548A mux **channel 2**, address `0x5A`, found by I2C scan 2026-11-03).
- Firmware emits an optional `air` object (`eco2_ppm`, `tvoc_ppb`, `since_init_ms`) and feeds the SHT40 reading to the CCS811 for humidity/temperature compensation. Absent sensor = no `air` key, never zeros.
- `orion-office-env` withholds air readings for the datasheet's 20-minute conditioning window after every sensor start (`state.air_warming=true`) and drops values outside the datasheet ranges.
- `orion-sql-writer` stores `office_eco2_ppm`, `office_tvoc_ppb`, `air_warming` (boot DDL adds the columns to the existing table).
- Hub Office strip: eCO2 and TVOC tiles ("warming up" during conditioning) and history charts.

## Outcome moved

Orion can now see the office's air quality (VOC load and an eCO2 estimate) next to temperature/humidity, stored and charted. Raw samples only: no pressure, alert, or cognition reading.

## Current architecture

Nano ESP32 + SHT40 (USB on pan-1) → `orion-office-env` → `orion:office:env:sample` (`office.env.sample.v1`) → `orion-sql-writer` → `office_env_sample` → Hub Office strip / prediction (`office_temp_c` only).

## Architecture touched

Same path, additive fields. No new channel, service, or env key.

## Files changed

- `firmware/pan-1-office-nano/*`: CCS811 init/poll/compensation, per-sensor mux channels, silent-stall watchdog, poll throttle, wrap-safe conditioning age.
- `orion/schemas/telemetry/office_env.py`: frame `air`; measurements `office_eco2_ppm`/`office_tvoc_ppb`; state `air_warming`; provenance `air_since_init_ms`.
- `services/orion-office-env/app/{sample,main,smoke}.py`: warm-up gate, range drop, log line, smoke exit 4 = no CCS811 reading. Tests + `evals/test_air_warmup_replay_eval.py`. README (incl. deploy order).
- `services/orion-sql-writer/app/{models/office_env_sample.py,worker.py,main.py}`: columns, normalizer, boot DDL. Test. README.
- `services/orion-hub/{scripts/office_env_routes.py,static/js/office-env.js,templates/index.html}`: fields, per-field `gap_fields`, tiles, charts. Tests. README.

## Schema / bus / API changes

- Added: `OfficeEnvFrameAirV1`; `OfficeEnvMeasurementsV1.office_eco2_ppm`, `.office_tvoc_ppb`; `OfficeEnvStateV1.air_warming`; `OfficeEnvProvenanceV1.air_since_init_ms`. Hub `/api/office/env/latest` sample gains `office_eco2_ppm`, `office_tvoc_ppb`, `air_warming`; `/history` points gain the two fields and optional `gap_fields`; stats gain `eco2_*`, `tvoc_*`.
- Behavior changed: `stale` now means "no usable reading from **either** sensor" (was SHT40 only).
- Compatibility: models are `extra="forbid"`. The only bus consumer is `orion-sql-writer`. **sql-writer must run the new schema before pan-1's `orion-office-env` is rebuilt**, or every office sample (temperature included) fails validation. New firmware with the old host is safe (frame model is `extra="ignore"`); it is already flashed.

## Env/config changes

- None. `.env_example` unchanged; no sync needed.

## Metric quality gate (raw storage only; not wired into any model)

1. Provenance: CCS811 ALG_RESULT_DATA registers via `Adafruit_CCS811::geteCO2/getTVOC`, firmware `poll_ccs811`.
2. Independence: TVOC is the sensor's VOC reading; eCO2 is the CCS811's own monotonic-ish transform of the same VOC signal, so **eCO2 and TVOC are not independent of each other**. Both are compensated by the SHT40, so they partly depend on temperature/humidity too.
3. Theory anchor: metal-oxide VOC sensing (ams CCS811 datasheet). eCO2 is a VOC-derived CO2 equivalent and is not a CO2 measurement.
4. Live data: UNVERIFIED as signal. Live frames 2026-11-03 03:33Z read `eco2=400, tvoc=0` (warm-up floor). The 48 h first-use burn-in runs until ~2026-11-05. The rest state is 400 ppm / 0 ppb (the sensor's floor), so it can read calm.
5. Existing mechanism: no other air-quality sensor in the repo.
6. Reversibility: nullable columns and optional fields. Cheap to drop.

## Tests run

```text
services/orion-office-env: pytest tests evals            -> 32 passed
services/orion-sql-writer: pytest tests/test_office_env_sample_sql_shape.py -> 7 passed
services/orion-hub: pytest tests/test_office_env_routes.py tests/test_office_env_panel.py tests/test_cabinet_climate_overlay_panel.py -> passed
services/orion-hub: node --test static/js/office-env.test.js -> 8 pass, 0 fail
tests/test_office_env_bus_catalog.py -> 2 passed
```

## Evals run

```text
services/orion-office-env/evals/test_air_warmup_replay_eval.py (sensor restart mid-session; air only from conditioned stretches, temp uninterrupted) -> passed
services/orion-office-env/evals/test_outage_replay_eval.py -> passed
```

## Docker/build/smoke checks

```text
arduino-cli compile (arduino:esp32:nano_nora, pan-1) -> Sketch uses 314197 bytes, no warnings
Flashed 2026-11-03 03:33Z. Raw frames: {"environment":{"temp_c":21.03,...},"air":{"eco2_ppm":400,"tvoc_ppb":0,"since_init_ms":12000}}
Old pan-1 service after flash: Published office sample temp_c=21.01 humidity_pct=29.36 stale=False
```

## Review findings fixed

- Finding: rebuilding pan-1 before sql-writer stops ALL office storage (forbid models).
  - Fix: deploy-order section in `services/orion-office-env/README.md` + this report.
  - Evidence: README "Deploy order for a schema change".
- Finding: a CCS811 that resets on its own (boot mode, no error) never recovers.
  - Fix: firmware watchdog re-inits after 10 s without a new reading.
  - Evidence: `poll_ccs811`, compiled and flashed.
- Finding: I2C polled in a tight loop.
  - Fix: 250 ms poll throttle.
- Finding: `since_init_ms` wraps at ~49.7 days → false warming.
  - Fix: sticky `ccs_conditioned`; reported age held at ≥ 20 min once conditioned.
- Finding: downsampled charts draw eCO2 across a warm-up hole while temp rows continue.
  - Fix: per-field `gap_fields` in `downsample_points`; `splitRuns` honors it. Tests in Python and JS.
- Finding: Hub queries new columns before boot DDL may have run.
  - Fix: covered by the deploy-order note (sql-writer first).
- Finding: `sensor.model` / `online` semantics stale.
  - Fix: comment updated (either sensor).

## Restart required

After merge, in order:

```bash
# 1. athena, primary main checkout (post-pull rebuild normally does this)
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
# 2. pan-1, primary main checkout, only after sql-writer is up
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && scripts/safe_docker_build.sh orion-office-env up -d --build
```

## Risks / concerns

- Severity: low. Concern: 48 h first-use burn-in; first two days of eCO2/TVOC drift. Mitigation: raw storage only; nothing consumes them yet.
- Severity: low. Concern: no baseline save/restore, so every Nano reboot restarts the baseline learning. Mitigation: documented; follow-up if the values prove useful.
- Severity: low. Concern: eCO2 is not CO2. Mitigation: labelled "estimated" in the Hub and docs.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3045

🤖 Generated with [Claude Code](https://claude.com/claude-code)
