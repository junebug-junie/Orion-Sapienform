/*
 * pan-1 office sensor node — Arduino Nano ESP32 (ABX00083) + SHT40
 *
 * Emits orion.office_env_frame.v1 at ~1 Hz on USB serial (115200 baud), one JSON
 * object per line. Contract: orion/schemas/telemetry/office_env.py.
 *
 * Absent is not zero: if the SHT40 failed to initialise or a read fails, the
 * `environment` object is omitted from that frame entirely. The heater is never
 * enabled (it would bias the temperature reading).
 *
 * I2C: Wire on the Nano ESP32 default pins, SDA=A4, SCL=A5. SHT40 address 0x44,
 * behind a TCA9548A multiplexer (0x70) on channel 3; see MUX_CHANNEL.
 *
 * Libraries: ArduinoJson (v6), Adafruit SHT4x (pulls Adafruit BusIO / Unified Sensor).
 */

#include <Wire.h>
#include <ArduinoJson.h>
#include <Adafruit_SHT4x.h>

static constexpr uint32_t USB_BAUD = 115200;
static constexpr uint32_t FRAME_INTERVAL_MS = 1000;
static constexpr uint32_t SHT_REINIT_INTERVAL_MS = 10000;

// SHT40 sits behind a TCA9548A I2C multiplexer (STEMMA QT board) at 0x70, on channel 3
// (found with an I2C scan 2026-10-28). Set MUX_CHANNEL to -1 if the sensor is wired directly.
static constexpr int MUX_CHANNEL = 3;
static constexpr uint8_t MUX_ADDR = 0x70;

static Adafruit_SHT4x sht4;
static bool sht_ok = false;
static uint32_t seq = 0;

// Point the multiplexer at the SHT40's channel. Called before every sensor access.
static bool select_sensor_channel() {
  if (MUX_CHANNEL < 0) {
    return true;
  }
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(static_cast<uint8_t>(1 << MUX_CHANNEL));
  return Wire.endTransmission() == 0;
}

static bool init_sht40() {
  if (!select_sensor_channel() || !sht4.begin()) {
    return false;
  }
  sht4.setPrecision(SHT4X_HIGH_PRECISION);
  sht4.setHeater(SHT4X_NO_HEATER);
  return true;
}

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial && millis() < 3000) {
    delay(10);
  }
  Wire.begin();
  sht_ok = init_sht40();
}

void loop() {
  static uint32_t last_frame_ms = 0;
  static uint32_t last_init_try_ms = 0;
  const uint32_t now = millis();

  // A sensor that was missing at boot (or dropped off the bus) is retried, not given up on.
  if (!sht_ok && now - last_init_try_ms >= SHT_REINIT_INTERVAL_MS) {
    last_init_try_ms = now;
    sht_ok = init_sht40();
  }

  if (now - last_frame_ms < FRAME_INTERVAL_MS) {
    return;
  }
  last_frame_ms = now;

  StaticJsonDocument<256> doc;
  doc["schema"] = "orion.office_env_frame.v1";
  doc["seq"] = seq++;
  doc["uptime_ms"] = now;

  if (sht_ok) {
    sensors_event_t humidity, temp;
    if (select_sensor_channel() && sht4.getEvent(&humidity, &temp)) {
      JsonObject env = doc.createNestedObject("environment");
      env["temp_c"] = temp.temperature;
      env["humidity_pct"] = humidity.relative_humidity;
    } else {
      sht_ok = false;  // drop to the retry path above; frame carries no environment
    }
  }

  serializeJson(doc, Serial);
  Serial.println();
}
