# pan-1 office sensor node — Arduino Nano ESP32 + SHT40

Firmware for the office temperature/humidity node plugged into **pan-1** by USB.
One NDJSON line per second at 115200 baud. Host contract:
`orion/schemas/telemetry/office_env.py` (`orion.office_env_frame.v1`).

```json
{"schema":"orion.office_env_frame.v1","seq":42,"uptime_ms":43012,
 "environment":{"temp_c":21.43,"humidity_pct":33.1}}
```

A failed or missing SHT40 **omits `environment`** — never zero-filled. The sensor is
retried every 10 s, so plugging it in late still works. The heater is never enabled.

## Wiring (STEMMA QT / Qwiic or jumpers)

| SHT40 | Nano ESP32 |
|-------|------------|
| VIN | 3V3 |
| GND | GND |
| SDA | A4 |
| SCL | A5 |

I2C address `0x44`.

**Bench setup (verified 2026-10-28):** the SHT40 is plugged into a TCA9548A I2C multiplexer
(STEMMA QT board, address `0x70`) on **channel 3**. The firmware selects that channel before every
sensor access (`MUX_CHANNEL` in the sketch). For a sensor wired directly to the Nano, set
`MUX_CHANNEL = -1`. If the sensor moves to another channel, change the constant and reflash.
To find it, scan each channel of the multiplexer for a device at `0x44`.

## Build and flash (run on the machine the Nano is plugged into)

```bash
arduino-cli core install arduino:esp32
arduino-cli lib install "ArduinoJson@6.21.5" "Adafruit SHT4x Library" "Adafruit BusIO" "Adafruit Unified Sensor"
arduino-cli compile --fqbn arduino:esp32:nano_nora firmware/pan-1-office-nano
ls /dev/serial/by-id/          # find the stable path; never use /dev/ttyACM0 directly
arduino-cli upload --fqbn arduino:esp32:nano_nora -p /dev/serial/by-id/<nano> firmware/pan-1-office-nano
```

Flashing is a manual operator step. Verify with the service smoke
(`services/orion-office-env/README.md`).

## Later sensors

A sound meter gets its own optional key in the frame (and its own schema fields), not a
reuse of any `cabinet_*` name.
