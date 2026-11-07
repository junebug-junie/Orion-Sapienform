/*
 * pan-1 sensor node firmware — Arduino Nano ESP32 (ABX00083).
 *
 * One sketch for both Nanos on pan-1 (same pattern as Athena's two cabinet Nanos):
 *   office Nano: SHT40 + CCS811 + PMSA003I + TSL2591 + sound level meter (analog, A0)
 *   closet Nano: SCD40
 * A sensor that is not on a board is simply never found, and its object is never sent.
 *
 * Emits orion.office_env_frame.v1 at ~1 Hz on USB serial (115200 baud), one JSON
 * object per line. Contract: orion/schemas/telemetry/office_env.py.
 *
 * Absent is not zero: if a sensor failed to initialise or a read fails, its object
 * (`environment` SHT40, `air` CCS811, `particulate` PMSA003I, `light` TSL2591,
 * `sound` sound meter, `co2` SCD40) is omitted from that frame entirely. The SHT40 heater
 * is never enabled.
 *
 * I2C: Wire on the Nano ESP32 default pins (SDA=A4, SCL=A5), usually through a
 * TCA9548A multiplexer at 0x70. Each sensor's channel is FOUND, not configured: at
 * every (re)init the firmware probes the sensor's address with the multiplexer off
 * (wired directly) and then on channels 0..7. Moving a cable needs no reflash.
 *
 * CCS811: eCO2 is an estimate derived from VOCs, not a CO2 measurement.
 *   `air.since_init_ms` lets the host drop the 20-minute conditioning period.
 * PMSA003I: read between frames; the newest good reading is sent while < 3 s old. Values
 *   are sent only 30 s after its fan starts (datasheet). A sensor that answers but stops
 *   reading for 10 s is restarted. CF=1 "standard" values, same as Athena.
 * TSL2591: gain is auto-ranged. A reading saturated even at the lowest gain sends
 *   `light.saturated=true` with no lux (too bright to measure, not zero light). A reading
 *   too dark for the current gain (full == 0) also has no lux; the gain steps up.
 * Sound meter: DFRobot SEN0232 on A0 (analog, NOT on the I2C bus or the multiplexer; powered
 *   from VBUS). Output 0.6..2.6 V = 30..130 dBA. The frame carries `sound.level_mv`, the
 *   mean pin voltage over the frame interval; the host converts to dBA and range-checks.
 *   The meter is "found" by a pull-down test: a powered meter drives the pin to >= 0.6 V,
 *   a pin with nothing on it (the closet Nano) reads ~0 and is never reported.
 * SCD40: photoacoustic CO2 (a real CO2 measurement), new reading every 5 s.
 *   Automatic self-calibration stays at the factory default (on).
 *
 * Libraries: ArduinoJson (v6), Adafruit SHT4x, Adafruit CCS811, Adafruit PM25 AQI,
 * Adafruit TSL2591, Sensirion I2C SCD4x (0.4.x), Adafruit BusIO / Unified Sensor.
 */

#include <Wire.h>
#include <math.h>
#include <ArduinoJson.h>
#include <Adafruit_SHT4x.h>
#include <Adafruit_CCS811.h>
#include <Adafruit_PM25AQI.h>
#include <Adafruit_TSL2591.h>
#include <SensirionI2CScd4x.h>

static constexpr uint32_t USB_BAUD = 115200;
static constexpr uint32_t FRAME_INTERVAL_MS = 1000;
static constexpr uint32_t REINIT_INTERVAL_MS = 10000;
// The CCS811 makes a new reading every 1 s; a reading older than this is not reported.
static constexpr uint32_t CCS_MAX_READING_AGE_MS = 3000;
// No new reading for this long means the sensor reset itself (boot mode, idle): re-init it.
static constexpr uint32_t CCS_SILENT_REINIT_MS = 10000;
// Datasheet conditioning period; past it the reported age is held at >= this value so a
// millis() wrap (~49.7 days) cannot make a conditioned sensor look like it is warming again.
static constexpr uint32_t CCS_CONDITIONING_MS = 20UL * 60UL * 1000UL;
// The CCS811 updates once a second; polling faster only loads the bus.
static constexpr uint32_t CCS_POLL_INTERVAL_MS = 250;
// PMSA003I datasheet: data is stable 30 s after the fan starts.
static constexpr uint32_t PM_STABLE_MS = 30000;
// The PMSA003I makes a new reading about once a second. A good reading older than this is
// not reported; no good reading for PM_STALL_REINIT_MS restarts the driver.
static constexpr uint32_t PM_MAX_READING_AGE_MS = 3000;
static constexpr uint32_t PM_STALL_REINIT_MS = 10000;
static constexpr uint32_t PM_POLL_INTERVAL_MS = 250;
// SCD40 makes a reading every 5 s; one older than this is not reported.
static constexpr uint32_t SCD_MAX_READING_AGE_MS = 15000;
static constexpr uint32_t SCD_POLL_INTERVAL_MS = 1000;

// Sound meter on A0. The meter never outputs below 0.6 V, so a mean under SOUND_PRESENT_MIN_MV
// means nothing is driving the pin (not connected, or it lost power): no `sound` object.
static constexpr uint8_t SOUND_PIN = A0;
static constexpr uint32_t SOUND_POLL_INTERVAL_MS = 10;
static constexpr uint16_t SOUND_PRESENT_MIN_MV = 400;
static constexpr uint16_t SOUND_MIN_SAMPLES = 20;

static constexpr uint8_t MUX_ADDR = 0x70;
static constexpr uint8_t SHT_ADDR = 0x44;
static constexpr uint8_t CCS_ADDR = 0x5A;
static constexpr uint8_t PM_ADDR = 0x12;
static constexpr uint8_t TSL_ADDR = 0x29;
static constexpr uint8_t SCD_ADDR = 0x62;

// Channel result: >= 0 multiplexer channel, DIRECT = on the Nano's bus, NOT_FOUND.
static constexpr int8_t DIRECT = -1;
static constexpr int8_t NOT_FOUND = -2;

static Adafruit_SHT4x sht4;
static Adafruit_CCS811 ccs;
static Adafruit_PM25AQI pm;
static Adafruit_TSL2591 tsl(2591);
static SensirionI2CScd4x scd;

static int8_t sht_ch = NOT_FOUND, ccs_ch = NOT_FOUND, pm_ch = NOT_FOUND, tsl_ch = NOT_FOUND,
              scd_ch = NOT_FOUND;
static bool sht_ok = false, ccs_ok = false, pm_ok = false, tsl_ok = false, scd_ok = false;

static uint32_t ccs_init_ms = 0, ccs_read_ms = 0, ccs_last_poll_ms = 0;
static bool ccs_have_reading = false, ccs_conditioned = false;
static uint16_t ccs_eco2 = 0, ccs_tvoc = 0;

// Fan start (power-up or the sensor reappearing on the bus). A restart of a sensor that
// stayed powered does NOT reset it: the fan never stopped.
static uint32_t pm_fan_start_ms = 0;
static bool pm_fan_known = false;
static bool pm_driver = false;
// Latched once the fan has run PM_STABLE_MS, so a millis() wrap (~49.7 days) cannot close
// the gate again (same idea as ccs_conditioned).
static bool pm_fan_settled = false;
static uint32_t pm_read_ms = 0, pm_last_try_ms = 0;
static bool pm_have_reading = false;
static uint16_t pm_1 = 0, pm_25 = 0, pm_10 = 0;

// TSL2591 auto-range ladder, low to high sensitivity.
static const tsl2591Gain_t kTslGains[] = {TSL2591_GAIN_LOW, TSL2591_GAIN_MED, TSL2591_GAIN_HIGH,
                                          TSL2591_GAIN_MAX};
static const uint16_t kTslGainX[] = {1, 25, 428, 9876};
static uint8_t tsl_gain_idx = 1;
// ADC full scale at the 100 ms integration time (ams datasheet). The Adafruit library only
// flags 0xFFFF, which a 100 ms channel never reaches, so saturation is checked against this.
static constexpr uint16_t TSL_MAX_COUNT_100MS = 36863;

static uint32_t scd_read_ms = 0, scd_last_poll_ms = 0;
static bool scd_have_reading = false;
static uint16_t scd_co2 = 0;
static float scd_temp = 0, scd_hum = 0;

static bool sound_ok = false;
static uint32_t sound_sum_mv = 0, sound_count = 0, sound_last_poll_ms = 0;

static uint32_t seq = 0;

static bool mux_write(uint8_t mask) {
  Wire.beginTransmission(MUX_ADDR);
  Wire.write(mask);
  return Wire.endTransmission() == 0;
}

static bool probe(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

// Point the multiplexer at one sensor's channel. Called before every sensor access.
static bool select_channel(int8_t ch) {
  if (ch == NOT_FOUND) return false;
  if (ch == DIRECT) return true;  // direct devices answer whatever the mux is set to
  return mux_write(static_cast<uint8_t>(1 << ch));
}

// Where is this address? Direct bus first (mux off), then mux channels 0..7.
static int8_t find_channel(uint8_t addr) {
  const bool mux = probe(MUX_ADDR);
  if (mux) mux_write(0);
  if (probe(addr)) return DIRECT;
  if (!mux) return NOT_FOUND;
  for (int8_t ch = 0; ch < 8; ch++) {
    if (mux_write(static_cast<uint8_t>(1 << ch)) && probe(addr)) return ch;
  }
  return NOT_FOUND;
}

static bool init_sht40() {
  sht_ch = find_channel(SHT_ADDR);
  if (!select_channel(sht_ch) || !sht4.begin()) return false;
  sht4.setPrecision(SHT4X_HIGH_PRECISION);
  sht4.setHeater(SHT4X_NO_HEATER);
  return true;
}

static bool init_ccs811() {
  // begin() resets the sensor, so its 20-minute conditioning clock restarts here.
  ccs_ch = find_channel(CCS_ADDR);
  if (!select_channel(ccs_ch) || !ccs.begin(CCS_ADDR)) return false;
  ccs_init_ms = millis();
  ccs_have_reading = false;
  ccs_conditioned = false;
  return true;
}

static bool init_pmsa003i() {
  pm_ch = find_channel(PM_ADDR);
  if (!select_channel(pm_ch)) return false;
  // Adafruit PM25 AQI 2.0 begin_I2C() returns false on EVERY call after the first (it
  // refuses when its driver object already exists), so the driver is created once and a
  // restart only re-finds the channel. Calling it again killed particulates until the Nano
  // rebooted (live 2026-11-05).
  if (!pm_driver) {
    // The driver object is created even when begin_I2C's own probe NACKs and it returns
    // false, so it counts as created either way; our probe below decides.
    pm.begin_I2C(&Wire);
    pm_driver = true;
  }
  if (!probe(PM_ADDR)) return false;
  if (!pm_fan_known) {
    pm_fan_start_ms = millis();
    pm_fan_known = true;
    pm_fan_settled = false;
  }
  pm_have_reading = false;
  pm_read_ms = millis();  // the stall clock starts now
  return true;
}

static void tsl_apply_gain() {
  tsl.setGain(kTslGains[tsl_gain_idx]);
  tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);
}

static bool init_tsl2591() {
  tsl_ch = find_channel(TSL_ADDR);
  if (!select_channel(tsl_ch) || !tsl.begin(&Wire, TSL_ADDR)) return false;
  tsl_apply_gain();
  return true;
}

static bool init_scd40() {
  scd_ch = find_channel(SCD_ADDR);
  if (!select_channel(scd_ch)) return false;
  scd.begin(Wire);
  // A running measurement (e.g. from before a Nano reset) blocks configuration; stop first.
  scd.stopPeriodicMeasurement();
  delay(500);
  if (scd.startPeriodicMeasurement() != 0) return false;
  scd_have_reading = false;
  scd_last_poll_ms = millis();
  scd_read_ms = millis();
  return true;
}

// Pull a new CCS811 reading if the sensor has one. A sensor error drops to the retry path.
static void poll_ccs811(uint32_t now) {
  if (!ccs_ok || now - ccs_last_poll_ms < CCS_POLL_INTERVAL_MS) return;
  ccs_last_poll_ms = now;
  // Silent stall watchdog: a CCS811 that reset on its own sits in boot mode with no
  // error and no data. Drop it to the retry path so begin() runs again.
  if (now - (ccs_have_reading ? ccs_read_ms : ccs_init_ms) > CCS_SILENT_REINIT_MS) {
    ccs_ok = false;
    ccs_have_reading = false;
    return;
  }
  if (!select_channel(ccs_ch) || !ccs.available()) return;
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

// Read the PMSA003I between frames, so one failed read (bad header or checksum) does not
// drop a frame's particulates. A read that keeps failing while the sensor still answers
// on the bus (seen live 2026-11-05: 30 min of no particulate) restarts the driver.
static void poll_pmsa003i(uint32_t now) {
  if (!pm_ok || now - pm_last_try_ms < PM_POLL_INTERVAL_MS) return;
  pm_last_try_ms = now;
  PM25_AQI_Data data;
  // About 2 in 5 reads land mid-update and come back cut short (0xFF padding); the sensor
  // can NACK a probe at that moment too. One miss is never treated as a lost sensor.
  if (select_channel(pm_ch) && pm.read(&data)) {
    pm_1 = data.pm10_standard;
    pm_25 = data.pm25_standard;
    pm_10 = data.pm100_standard;
    pm_read_ms = now;
    pm_have_reading = true;
  } else if (now - pm_read_ms > PM_STALL_REINIT_MS) {
    pm_ok = false;  // no good read for 10 s: back to the retry path
    pm_have_reading = false;
    // Gone from the bus means unplugged or unpowered, so its fan restarts too.
    if (!select_channel(pm_ch) || !probe(PM_ADDR)) pm_fan_known = false;
  }
}

static void poll_scd40(uint32_t now) {
  if (!scd_ok || now - scd_last_poll_ms < SCD_POLL_INTERVAL_MS) return;
  scd_last_poll_ms = now;
  // No reading for a long time: the sensor stopped measuring. Re-init it.
  if (now - scd_read_ms > SCD_MAX_READING_AGE_MS * 2) {
    scd_ok = false;
    scd_have_reading = false;
    return;
  }
  bool ready = false;
  if (!select_channel(scd_ch) || scd.getDataReadyFlag(ready) != 0 || !ready) return;
  uint16_t co2 = 0;
  float t = 0, h = 0;
  if (scd.readMeasurement(co2, t, h) != 0 || co2 == 0) return;  // 0 ppm = invalid sample
  scd_co2 = co2;
  scd_temp = t;
  scd_hum = h;
  scd_read_ms = now;
  scd_have_reading = true;
}

// Pull the pin low and see whether something drives it back up. Run at boot and on the retry
// path, so a meter plugged in later is found without a reflash.
static bool init_sound() {
  pinMode(SOUND_PIN, INPUT_PULLDOWN);
  delay(20);
  uint32_t sum = 0;
  for (uint8_t i = 0; i < 8; i++) {
    sum += analogReadMilliVolts(SOUND_PIN);
    delay(2);
  }
  sound_sum_mv = 0;
  sound_count = 0;
  return sum / 8 >= SOUND_PRESENT_MIN_MV;
}

static void poll_sound(uint32_t now) {
  if (!sound_ok || now - sound_last_poll_ms < SOUND_POLL_INTERVAL_MS) return;
  sound_last_poll_ms = now;
  sound_sum_mv += analogReadMilliVolts(SOUND_PIN);
  sound_count++;
}

// Mean pin voltage since the last frame. A window with too few samples, or one that reads
// below what a powered meter can output, is not a reading: omitted, and the retry path
// re-tests the pin. Always restarts the window.
static void add_sound(JsonDocument &doc) {
  if (!sound_ok) return;
  const uint32_t n = sound_count;
  const uint32_t mean_mv = n > 0 ? sound_sum_mv / n : 0;
  sound_sum_mv = 0;
  sound_count = 0;
  if (n < SOUND_MIN_SAMPLES) return;
  if (mean_mv < SOUND_PRESENT_MIN_MV) {
    sound_ok = false;  // nothing driving the pin: retry path, no sound object
    return;
  }
  JsonObject sound = doc.createNestedObject("sound");
  sound["level_mv"] = mean_mv;
}

static void add_light(JsonDocument &doc) {
  if (!select_channel(tsl_ch) || !probe(TSL_ADDR)) {
    tsl_ok = false;  // gone from the bus: retry path, no light object
    return;
  }
  const uint32_t lum = tsl.getFullLuminosity();  // blocks for the 100 ms integration
  const uint16_t ir = lum >> 16;
  const uint16_t full = lum & 0xFFFF;
  const uint8_t used_gain = tsl_gain_idx;
  const float lux = tsl.calculateLux(full, ir);
  const bool saturated = lux < 0 || full >= TSL_MAX_COUNT_100MS || ir >= TSL_MAX_COUNT_100MS;
  // full == 0 (darker than this gain can see) divides by zero, and ir > full is low-count
  // noise that yields a positive garbage lux. Both are unmeasurable: lux is omitted.
  const bool measurable = !saturated && full > 0 && ir <= full && isfinite(lux);
  JsonObject light = doc.createNestedObject("light");
  light["full"] = full;
  light["ir"] = ir;
  light["gain_x"] = kTslGainX[used_gain];
  // Saturated at a higher gain just means "step down"; only the lowest gain means too bright.
  light["saturated"] = saturated && used_gain == 0;
  if (measurable) light["lux"] = lux;
  // Auto-range for the next frame.
  if (saturated || full > 30000) {
    if (tsl_gain_idx > 0) tsl_gain_idx--;
  } else if (full < 200 && tsl_gain_idx < 3) {
    tsl_gain_idx++;
  }
  if (tsl_gain_idx != used_gain) tsl_apply_gain();
}

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial && millis() < 3000) delay(10);
  Wire.begin();
  Wire.setTimeOut(50);
  sht_ok = init_sht40();
  ccs_ok = init_ccs811();
  pm_ok = init_pmsa003i();
  tsl_ok = init_tsl2591();
  // The SCD40 needs ~1 s after power-up before it accepts commands; the Serial wait above
  // returns at once when the host already has the port open.
  while (millis() < 1000) delay(10);
  scd_ok = init_scd40();
  sound_ok = init_sound();
}

void loop() {
  static uint32_t last_frame_ms = 0;
  static uint32_t last_init_try_ms = 0;

  // A sensor missing at boot (or dropped off the bus) is retried, never given up on.
  if ((!sht_ok || !ccs_ok || !pm_ok || !tsl_ok || !scd_ok || !sound_ok) &&
      millis() - last_init_try_ms >= REINIT_INTERVAL_MS) {
    last_init_try_ms = millis();
    if (!sht_ok) sht_ok = init_sht40();
    if (!ccs_ok) ccs_ok = init_ccs811();
    if (!pm_ok) pm_ok = init_pmsa003i();
    if (!tsl_ok) tsl_ok = init_tsl2591();
    if (!scd_ok) scd_ok = init_scd40();
    if (!sound_ok) sound_ok = init_sound();
  }

  // Read the clock AFTER the inits: they stamp ccs_init_ms / pm_read_ms / scd_read_ms with
  // millis(), and a `now` taken before them makes `now - stamp` wrap to ~4e9, so a sensor
  // that just recovered was dropped again in the same pass, forever (review 2026-11-05).
  const uint32_t now = millis();

  poll_ccs811(now);
  poll_pmsa003i(now);
  poll_scd40(now);
  poll_sound(now);

  if (now - last_frame_ms < FRAME_INTERVAL_MS) return;
  last_frame_ms = now;

  StaticJsonDocument<768> doc;
  doc["schema"] = "orion.office_env_frame.v1";
  doc["seq"] = seq++;
  doc["uptime_ms"] = now;

  if (sht_ok) {
    sensors_event_t humidity, temp;
    if (select_channel(sht_ch) && sht4.getEvent(&humidity, &temp)) {
      JsonObject env = doc.createNestedObject("environment");
      env["temp_c"] = temp.temperature;
      env["humidity_pct"] = humidity.relative_humidity;
      // Compensation: the CCS811 assumes 50 %RH / 25 C unless told otherwise.
      if (ccs_ok && select_channel(ccs_ch)) {
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
    uint32_t since_init = now - ccs_init_ms;
    if (since_init >= CCS_CONDITIONING_MS) ccs_conditioned = true;
    if (ccs_conditioned && since_init < CCS_CONDITIONING_MS) since_init = CCS_CONDITIONING_MS;
    air["since_init_ms"] = since_init;
  }

  if (pm_ok && pm_have_reading && now - pm_read_ms <= PM_MAX_READING_AGE_MS &&
      (pm_fan_settled || (pm_fan_settled = now - pm_fan_start_ms >= PM_STABLE_MS))) {
    JsonObject part = doc.createNestedObject("particulate");
    part["pm1_ug_m3"] = pm_1;
    part["pm25_ug_m3"] = pm_25;
    part["pm10_ug_m3"] = pm_10;
  }

  if (tsl_ok) add_light(doc);

  add_sound(doc);

  if (scd_ok && scd_have_reading && now - scd_read_ms <= SCD_MAX_READING_AGE_MS) {
    JsonObject co2 = doc.createNestedObject("co2");
    co2["co2_ppm"] = scd_co2;
    co2["temp_c"] = scd_temp;
    co2["humidity_pct"] = scd_hum;
  }

  serializeJson(doc, Serial);
  Serial.println();
}
