/*
 * pan-1 office sensor node — Arduino Nano ESP32 (ABX00083) + SHT40 + CCS811
 *
 * Emits orion.office_env_frame.v1 at ~1 Hz on USB serial (115200 baud), one JSON
 * object per line. Contract: orion/schemas/telemetry/office_env.py.
 *
 * Absent is not zero: if a sensor failed to initialise or a read fails, its
 * object (`environment` for the SHT40, `air` for the CCS811) is omitted from that
 * frame entirely. The SHT40 heater is never enabled (it would bias the temperature).
 *
 * I2C: Wire on the Nano ESP32 default pins, SDA=A4, SCL=A5, through a TCA9548A
 * multiplexer (0x70). SHT40 0x44 on channel 3, CCS811 0x5A on channel 2.
 *
 * CCS811 (SparkFun Zio Qwiic): eCO2 is an estimate derived from VOCs, not a CO2
 * measurement. `air.since_init_ms` lets the host drop readings from the datasheet's
 * 20-minute conditioning period after every sensor start. The SHT40's reading is
 * written to the CCS811 each frame for humidity/temperature compensation.
 *
 * Libraries: ArduinoJson (v6), Adafruit SHT4x (pulls Adafruit BusIO / Unified Sensor),
 * Adafruit CCS811 Library.
 */

#include <Wire.h>
#include <ArduinoJson.h>
#include <Adafruit_SHT4x.h>
#include <Adafruit_CCS811.h>

static constexpr uint32_t USB_BAUD = 115200;
static constexpr uint32_t FRAME_INTERVAL_MS = 1000;
static constexpr uint32_t SHT_REINIT_INTERVAL_MS = 10000;
// The CCS811 makes a new reading every 1 s; a reading older than this is not reported.
static constexpr uint32_t CCS_MAX_READING_AGE_MS = 3000;

// Both sensors sit behind a TCA9548A I2C multiplexer (STEMMA QT board) at 0x70.
// Channels found with I2C scans: SHT40 2026-10-28, CCS811 2026-11-03. Set a channel
// to -1 if that sensor is wired directly to the Nano.
static constexpr int SHT_MUX_CHANNEL = 3;
static constexpr int CCS_MUX_CHANNEL = 2;
static constexpr uint8_t MUX_ADDR = 0x70;
static constexpr uint8_t CCS_ADDR = 0x5A;

static Adafruit_SHT4x sht4;
static Adafruit_CCS811 ccs;
static bool sht_ok = false;
static bool ccs_ok = false;
static uint32_t ccs_init_ms = 0;
static uint32_t ccs_read_ms = 0;
static bool ccs_have_reading = false;
static uint16_t ccs_eco2 = 0;
static uint16_t ccs_tvoc = 0;
static uint32_t seq = 0;

// Point the multiplexer at one sensor's channel. Called before every sensor access.
static bool select_channel(int channel) {
  if (channel < 0) {
    return true;
  }
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(static_cast<uint8_t>(1 << channel));
  return Wire.endTransmission() == 0;
}

static bool init_ccs811() {
  // begin() resets the sensor, so its 20-minute conditioning clock restarts here.
  if (!select_channel(CCS_MUX_CHANNEL) || !ccs.begin(CCS_ADDR)) {
    return false;
  }
  ccs_init_ms = millis();
  ccs_have_reading = false;
  return true;
}

// Pull a new reading if the sensor has one. A sensor error drops to the retry path.
static void poll_ccs811(uint32_t now) {
  if (!ccs_ok || !select_channel(CCS_MUX_CHANNEL)) {
    return;
  }
  if (!ccs.available()) {
    return;
  }
  if (ccs.readData() != 0 || ccs.checkError()) {
    ccs_ok = false;
    ccs_have_reading = false;
    return;
  }
  ccs_eco2 = ccs.geteCO2();
  ccs_tvoc = ccs.getTVOC();
  ccs_read_ms = now;
  ccs_have_reading = true;
}

static bool init_sht40() {
  if (!select_channel(SHT_MUX_CHANNEL) || !sht4.begin()) {
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
  ccs_ok = init_ccs811();
}

void loop() {
  static uint32_t last_frame_ms = 0;
  static uint32_t last_init_try_ms = 0;
  const uint32_t now = millis();

  // A sensor that was missing at boot (or dropped off the bus) is retried, not given up on.
  if ((!sht_ok || !ccs_ok) && now - last_init_try_ms >= SHT_REINIT_INTERVAL_MS) {
    last_init_try_ms = now;
    if (!sht_ok) sht_ok = init_sht40();
    if (!ccs_ok) ccs_ok = init_ccs811();
  }

  poll_ccs811(now);

  if (now - last_frame_ms < FRAME_INTERVAL_MS) {
    return;
  }
  last_frame_ms = now;

  StaticJsonDocument<384> doc;
  doc["schema"] = "orion.office_env_frame.v1";
  doc["seq"] = seq++;
  doc["uptime_ms"] = now;

  if (sht_ok) {
    sensors_event_t humidity, temp;
    if (select_channel(SHT_MUX_CHANNEL) && sht4.getEvent(&humidity, &temp)) {
      JsonObject env = doc.createNestedObject("environment");
      env["temp_c"] = temp.temperature;
      env["humidity_pct"] = humidity.relative_humidity;
      // Compensation: the CCS811 assumes 50 %RH / 25 C unless told otherwise.
      if (ccs_ok && select_channel(CCS_MUX_CHANNEL)) {
        ccs.setEnvironmentalData(humidity.relative_humidity, temp.temperature);
      }
    } else {
      sht_ok = false;  // drop to the retry path above; frame carries no environment
    }
  }

  if (ccs_ok && ccs_have_reading && now - ccs_read_ms <= CCS_MAX_READING_AGE_MS) {
    JsonObject air = doc.createNestedObject("air");
    air["eco2_ppm"] = ccs_eco2;
    air["tvoc_ppb"] = ccs_tvoc;
    air["since_init_ms"] = now - ccs_init_ms;
  }

  serializeJson(doc, Serial);
  Serial.println();
}
