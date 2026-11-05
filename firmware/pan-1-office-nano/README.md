# pan-1 sensor nodes — Arduino Nano ESP32 (office + server closet)

One firmware for both Nanos plugged into **pan-1** by USB (same pattern as Athena's two
cabinet Nanos: same sketch on every board, a missing sensor's object is just never sent).
One NDJSON line per second at 115200 baud. Host contract:
`orion/schemas/telemetry/office_env.py` (`orion.office_env_frame.v1`).

| Board | USB by-id serial | Sensors | Bus sample |
|-------|------------------|---------|------------|
| office | `usb-Arduino_Nano_ESP32_E8F60AD53FB4-if01` | SHT40, CCS811, PMSA003I, TSL2591 (windowsill) | `office.env.sample.v1` |
| closet | `usb-Arduino_Nano_ESP32_E8F60ABE42B4-if01` | SCD40 (server closet, above the server cabinet) | `closet.env.sample.v1` |

```json
{"schema":"orion.office_env_frame.v1","seq":42,"uptime_ms":43012,
 "environment":{"temp_c":21.43,"humidity_pct":33.1},
 "air":{"eco2_ppm":612,"tvoc_ppb":31,"since_init_ms":1260000},
 "particulate":{"pm1_ug_m3":1,"pm25_ug_m3":3,"pm10_ug_m3":4},
 "light":{"lux":412.5,"full":5000,"ir":1200,"gain_x":25,"saturated":false},
 "co2":{"co2_ppm":640,"temp_c":28.0,"humidity_pct":26.0}}
```

A failed or missing sensor **omits its object** — never zero-filled. Each sensor is retried
every 10 s, so plugging one in late still works.

| Object | Sensor | Address | Notes |
|--------|--------|---------|-------|
| `environment` | SHT40 | `0x44` | Heater never enabled. Its reading compensates the CCS811 every frame. |
| `air` | CCS811 (SparkFun Zio Qwiic) | `0x5A` | **eCO2** is a CO2 *estimate* from VOCs, plus TVOC. Not valid for 20 min after each start; the host withholds it until `since_init_ms` passes 20 min. 48-hour first-use burn-in. No baseline save/restore. |
| `particulate` | PMSA003I | `0x12` | CF=1 "standard" mass concentrations, same as Athena. Read between frames; the newest good reading is sent while under 3 s old. Sent only 30 s after its fan starts (datasheet). If it still answers on the bus but has not read for 10 s, the driver is restarted (2026-11-05: it went silent for 30 min while the old firmware only skipped failed frames). |
| `light` | TSL2591 | `0x29` | Gain auto-ranges (1x/25x/428x/9876x), 100 ms integration. `saturated=true` (and no `lux`) only when it overflows at the lowest gain: too bright to measure, not dark. |
| `co2` | SCD40 | `0x62` | Photoacoustic CO2 — a real CO2 measurement. New reading every 5 s; the frame repeats the newest one for up to 15 s. Temperature/humidity are the SCD40's own. Automatic self-calibration is left at the factory default (on). |

## Multiplexer channels are found, not configured

Sensors sit behind a TCA9548A I2C multiplexer (STEMMA QT board, `0x70`). At every (re)init
the firmware probes each sensor's address with the multiplexer off (wired directly to the
Nano) and then on channels 0..7, and uses wherever it answers. Moving a cable to another
channel needs no reflash — the sensor comes back within ~10 s.

Live scans 2026-11-04: closet Nano — SCD40 on channel 7. Office Nano — SHT40 channel 3,
CCS811 channel 2 (earlier scans); PMSA003I / TSL2591 positions not yet scanned.

## Wiring (STEMMA QT / Qwiic or jumpers)

| Sensor pin | Nano ESP32 |
|-------|------------|
| VIN | 3V3 |
| GND | GND |
| SDA | A4 |
| SCL | A5 |

## Build and flash (on pan-1)

**Two Nanos means two DFU devices.** The Nano ESP32 uploads over DFU, and with both boards
plugged in `arduino-cli upload` fails: `dfu-util: More than one DFU capable USB device
found!`. Select the board by its USB serial (the 12 hex digits in its by-id name; the DFU
serial is the same string, verified 2026-11-04):

```bash
arduino-cli core install arduino:esp32
arduino-cli lib install "ArduinoJson@6.21.5" "Adafruit SHT4x Library" "Adafruit BusIO" \
  "Adafruit Unified Sensor" "Adafruit CCS811 Library" "Adafruit PM25 AQI Sensor" \
  "Adafruit TSL2591 Library" "Sensirion I2C SCD4x@0.4.0"
arduino-cli compile --fqbn arduino:esp32:nano_nora --output-dir /tmp/pan1-fw firmware/pan-1-office-nano
DFU=$(dirname "$(find ~/.arduino15/packages/arduino/tools/dfu-util -name dfu-util -type f | head -1)")/dfu-util
$DFU -l                                    # lists both boards with serial="..."
$DFU --device 0x2341:0x0070 -S <serial> -D /tmp/pan1-fw/pan-1-office-nano.ino.bin -Q
```

Stop the container that reads a board before flashing it (`docker stop
orion-pan-1-orion-office-env` or `orion-pan-1-orion-closet-env`), then start it again.
Flashing new firmware before the host service is updated is safe: the host ignores frame
keys it does not know. Verify with the service smoke (`services/orion-office-env/README.md`).

A brand-new Nano can enumerate as `usb-Espressif_Systems_ARDUINO_NANO_NORA_*` until its
first Arduino flash; after it, it shows as `usb-Arduino_Nano_ESP32_*` with the same serial.

## Later sensors

A sound meter gets its own optional key in the frame (and its own schema fields), not a
reuse of any `cabinet_*` name.
