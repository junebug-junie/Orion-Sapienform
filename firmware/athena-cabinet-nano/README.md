# Athena cabinet sensor node — Arduino Nano ESP32

Firmware for the Athena cabinet sensory board (Arduino Nano ESP32 / ABX00083). Emits one NDJSON line per second on USB serial at 115200 baud.

Host contract: `orion/schemas/telemetry/cabinet_sensor_frame.py` and `docs/superpowers/specs/2026-08-23-athena-cabinet-sensor-node-design.md`.

## Frame schema (`orion.sensor_frame.v1`)

Each line is a single JSON object. Failed sensors **omit their entire sub-object** — never zero-fill.

Example (all sensors healthy):

```json
{
  "schema": "orion.sensor_frame.v1",
  "seq": 4812,
  "uptime_ms": 992813,
  "environment": {
    "temp_c": 24.6,
    "humidity_pct": 31.8,
    "pressure_hpa": 857.2,
    "gas_resistance_ohm": 138421
  },
  "uv": {"raw": 17, "als_raw": 1292},
  "magnetic": {
    "x_ut": 31.2,
    "y_ut": -8.4,
    "z_ut": 42.1,
    "magnitude_ut": 53.0
  },
  "particulate": {
    "pm1_ug_m3": 2,
    "pm25_ug_m3": 4,
    "pm10_ug_m3": 5
  },
  "lidar": {"distance_mm": 438, "status": 0},
  "imu": {
    "accel_x": 0.01,
    "accel_y": -0.02,
    "accel_z": 9.79,
    "yaw_deg": 12.4,
    "pitch_deg": 0.8,
    "roll_deg": -1.1
  }
}
```

Notes:

- No `audio` block — MAX9814 is not on this board.
- LiDAR `distance_mm` is always emitted with `status`; host trusts distance only when `status == 0`.
- Partial frames (missing sub-objects) are valid when individual sensors fail init or read.

## Sensors

| Sensor | Bus | Role |
|--------|-----|------|
| BME680 | I2C | `environment` |
| LTR390 | I2C | `uv` (raw UV + ALS counts) |
| LIS3MDL | I2C | `magnetic` (µT + magnitude) — legacy breakout |
| MMC5603 | I2C | `magnetic` (µT + magnitude) — dual-Nano mag board |
| PMSA003I | I2C | `particulate` (µg/m³) |
| VL53L1X | I2C | `lidar` (mm + range status) |
| BNO085 | **I2C default** (0x4A/0x4B); UART-RVC fallback | `imu` (accel m/s², yaw/pitch/roll °) |

## Two-hub STEMMA topology (Athena cabinet)

Observed live wiring:

```text
Nano A4/A5 ──► Hub A (5-port) ──► 3 sensors          ← I2C scan sees these
                    │
                    └── daisy ──► Hub B (5-port) ──► 2+ sensors  ← NOT on bus
```

Live pin-matrix scan only ever finds `0x29` / `0x4A` / `0x77` on **A4/A5**.
So Hub A is fine; Hub B (or the daisy cable into it) is not passing SDA/SCL.

### Isolation test (do this once)

1. Unplug the daisy cable Hub A → Hub B.
2. Move **one** “missing” sensor (LTR390 or LIS3MDL) onto an empty port on **Hub A**.
3. Reboot Nano / wait for boot JSON.
4. Interpret:
   - **Appears on Hub A** → that breakout is fine; fix Hub B daisy (cable / hub / length).
   - **Still missing on Hub A** → bad QT cable into that sensor, wrong part (UART PMS5003 vs I2C PMSA003I), or dead breakout.

### Daisy link checklist

- Cable Hub A → Hub B must be a real **4-pin STEMMA QT / Qwiic** cable (not power-only).
- Prefer Hub A **port → Hub B IN**, not “out the side of a sensor” if that run is long.
- Keep the inter-hub cable **short** (<20 cm if you can).
- Firmware runs the bus at **25 kHz** for multi-hub capacitance.

### Shared I2C (`Wire`)

| Signal | Nano ESP32 pin |
|--------|----------------|
| SDA | A4 |
| SCL | A5 |

All I2C sensors share this bus. Typical 7-bit addresses:

| Sensor | Address |
|--------|---------|
| BME680 | 0x76 or 0x77 |
| LTR390 | 0x53 |
| LIS3MDL | 0x1C or 0x1E |
| MMC5603 | 0x30 |
| PMSA003I | 0x12 |
| VL53L1X | 0x29 |

3.3 V and GND to each breakout. Keep I2C leads short inside the cabinet.

### PMSA003I SET pin

| Signal | Nano ESP32 pin |
|--------|----------------|
| SET (wake) | D2 (driven HIGH to keep sensor awake) |

### BNO085 (I2C default — Adafruit STEMMA)

Adafruit BNO085 breakouts default to **I2C** (PS0/PS1 pulled low). Address **0x4A** (or **0x4B** if DI tied high). Put it on the same STEMMA/I2C bus as the other sensors.

| BNO085 | Nano ESP32 |
|--------|------------|
| SDA / SCL / 3V3 / GND | A4 / A5 / 3.3 V / GND |

**UART-RVC fallback** (only if jumpers set PS0=1 PS1=0 and not on I2C):

| BNO085 | Nano ESP32 |
|--------|------------|
| TXO → | D7 (Serial1 RX) |
| RXI ← | D6 (Serial1 TX) |

## Arduino libraries

Install via Library Manager or `arduino-cli lib install`:

- Adafruit BME680 Library
- Adafruit LTR390 Library
- Adafruit LIS3MDL
- Adafruit MMC5603 (Adafruit_MMC56x3 library)
- Adafruit VL53L1X
- Adafruit BNO08x
- Adafruit BNO08x RVC (UART-RVC fallback)
- Adafruit PM25 AQI Sensor
- ArduinoJson

## Board target

- **Board:** Arduino Nano ESP32 (`arduino:esp32:nano_nora`)
- **USB serial:** 115200 baud
- **Frame rate:** ~1 Hz

## Build / flash

From repo root:

```bash
./scripts/flash_athena_cabinet_nano.sh
```

Compile only (no upload):

```bash
./scripts/flash_athena_cabinet_nano.sh --compile-only
```

Requires `arduino-cli` with the `arduino:esp32` core and libraries above. The flash script discovers `/dev/serial/by-id/usb-Arduino_Nano_ESP32_*` — never hard-code `ttyACM0`.

## Soft-fail behavior

Each sensor initializes independently in `setup()`. On init or per-frame read failure, that sensor's JSON sub-object is omitted for that frame. One dead sensor does not halt the loop or zero-fill other channels.

## Boot diagnostic (`orion.sensor_boot.v1`)

Once at boot, before data frames, the sketch emits one NDJSON line:

```json
{
  "schema": "orion.sensor_boot.v1",
  "uptime_ms": 4200,
  "i2c": {"sda_pin": "A4", "scl_pin": "A5", "addresses": ["0x29", "0x76"]},
  "sensors": {
    "bme680": {"ok": true, "addr": "0x76"},
    "ltr390": {"ok": false, "detail": "not_on_bus"},
    "lis3mdl": {"ok": false, "detail": "not_on_bus"},
    "pmsa003i": {"ok": false, "detail": "not_on_bus", "set_pin": "D2"},
    "vl53l1x": {"ok": true, "addr": "0x29"},
    "bno085": {"ok": false, "detail": "uart_no_sync", "rx_pin": "D7", "tx_pin": "D6", "baud": 115200, "mode": "uart_rvc"}
  }
}
```

The host reader captures this to `/run/orion-sensors/boot.json`. Inspect with:

```bash
./scripts/diagnose_athena_cabinet_sensors.sh
```

`detail` values:

| detail | Meaning |
|--------|---------|
| `not_on_bus` | No I2C ACK — unwired, unpowered, or wrong sensor variant |
| `begin_failed` | ACK present but driver init failed |
| `uart_no_sync` | BNO085 not sending RVC packets — UART wiring or PS0/PS1 mode |
