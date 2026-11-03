# pan-1 office sensor node — Arduino Nano ESP32 + SHT40 + CCS811

Firmware for the office temperature/humidity node plugged into **pan-1** by USB.
One NDJSON line per second at 115200 baud. Host contract:
`orion/schemas/telemetry/office_env.py` (`orion.office_env_frame.v1`).

```json
{"schema":"orion.office_env_frame.v1","seq":42,"uptime_ms":43012,
 "environment":{"temp_c":21.43,"humidity_pct":33.1},
 "air":{"eco2_ppm":612,"tvoc_ppb":31,"since_init_ms":1260000}}
```

A failed or missing SHT40 **omits `environment`**, and a failed or missing CCS811 **omits
`air`** — never zero-filled. Each sensor is retried every 10 s, so plugging one in late
still works. The SHT40 heater is never enabled.

The CCS811 (SparkFun Zio Qwiic Air Quality Sensor, address `0x5A`) reports **eCO2**, a CO2
*estimate* derived from VOCs (not a CO2 measurement), and **TVOC**. Its readings are not
valid for 20 minutes after each sensor start; the firmware reports `since_init_ms` and the
host withholds readings until it passes 20 min. The datasheet also asks for a 48-hour
burn-in on first use, so the first two days of a brand-new sensor drift. The SHT40's
reading is written to the CCS811 every frame for humidity/temperature compensation.
Baseline save/restore is not implemented.

## Wiring (STEMMA QT / Qwiic or jumpers)

| SHT40 | Nano ESP32 |
|-------|------------|
| VIN | 3V3 |
| GND | GND |
| SDA | A4 |
| SCL | A5 |

I2C addresses: SHT40 `0x44`, CCS811 `0x5A`.

**Bench setup:** both sensors plug into a TCA9548A I2C multiplexer (STEMMA QT board, address
`0x70`): the SHT40 on **channel 3** (verified 2026-10-28) and the CCS811 on **channel 2**
(verified 2026-11-03). The firmware selects the right channel before every sensor access
(`SHT_MUX_CHANNEL` / `CCS_MUX_CHANNEL` in the sketch). For a sensor wired directly to the
Nano, set its constant to `-1`. If a sensor moves, change the constant and reflash. To find
it, scan each multiplexer channel for its address.

## Build and flash (run on the machine the Nano is plugged into)

```bash
arduino-cli core install arduino:esp32
arduino-cli lib install "ArduinoJson@6.21.5" "Adafruit SHT4x Library" "Adafruit BusIO" "Adafruit Unified Sensor" "Adafruit CCS811 Library"
arduino-cli compile --fqbn arduino:esp32:nano_nora firmware/pan-1-office-nano
ls /dev/serial/by-id/          # find the stable path; never use /dev/ttyACM0 directly
arduino-cli upload --fqbn arduino:esp32:nano_nora -p /dev/serial/by-id/<nano> firmware/pan-1-office-nano
```

Flashing is a manual operator step. Verify with the service smoke
(`services/orion-office-env/README.md`).

## Later sensors

A sound meter gets its own optional key in the frame (and its own schema fields), not a
reuse of any `cabinet_*` name.
