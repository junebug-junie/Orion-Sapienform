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
| LIS3MDL | I2C | `magnetic` (µT + magnitude) |
| PMSA003I | I2C | `particulate` (µg/m³) |
| VL53L1X | I2C | `lidar` (mm + range status) |
| BNO085 | **UART-RVC only** | `imu` (accel m/s², yaw/pitch/roll °) |

## Wiring (default sketch pins)

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
| PMSA003I | 0x12 |
| VL53L1X | 0x29 |

3.3 V and GND to each breakout. Keep I2C leads short inside the cabinet.

### PMSA003I SET pin

| Signal | Nano ESP32 pin |
|--------|----------------|
| SET (wake) | D2 (driven HIGH to keep sensor awake) |

### BNO085 UART-RVC (not I2C)

Configure the BNO085 breakout for **UART-RVC** (PS0=1, PS1=0 per Hillcrest / SparkFun jumper table).

| BNO085 | Nano ESP32 |
|--------|------------|
| TXO → | D7 (Serial1 RX) |
| RXI ← | D6 (Serial1 TX) |
| 3V3, GND | 3.3 V, GND |

Do **not** connect BNO085 to the ESP32 I2C bus for this firmware.

## Arduino libraries

Install via Library Manager or `arduino-cli lib install`:

- Adafruit BME680 Library
- Adafruit LTR390 Library
- Adafruit LIS3MDL
- Adafruit VL53L1X
- Adafruit BNO08x RVC
- ArduinoJson

PMSA003I is read with inline I2C code (no separate library).

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
