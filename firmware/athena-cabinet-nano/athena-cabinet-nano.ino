/*
 * Athena cabinet sensor node — Arduino Nano ESP32 (ABX00083)
 *
 * Emits orion.sensor_boot.v1 (with pin-matrix I2C probe) then
 * orion.sensor_frame.v1 at ~1 Hz. Failed sensors omit their sub-object.
 *
 * I2C: probes common Nano pin pairs at boot. Primary bus (Wire) hosts
 * BME680 / VL53 / BNO. If LTR390 / LIS3MDL / PMSA003I ACKs on another
 * pair, they run on Wire1.
 */

#include <Wire.h>
#include <math.h>
#include <ArduinoJson.h>

#include <Adafruit_Sensor.h>
#include <Adafruit_BME680.h>
#include <Adafruit_LTR390.h>
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_VL53L1X.h>
#include <Adafruit_PM25AQI.h>
#include <Adafruit_BNO08x.h>
#include <Adafruit_BNO08x_RVC.h>

static constexpr uint8_t BNO085_RX_PIN = D7;
static constexpr uint8_t BNO085_TX_PIN = D6;
static constexpr uint32_t BNO085_BAUD = 115200;
static constexpr uint32_t BNO085_SYNC_MS = 1500;

static constexpr uint8_t PMSA003I_SET_PIN = D2;
static constexpr uint32_t PMSA003I_BOOT_MS = 3000;

static constexpr uint32_t FRAME_INTERVAL_MS = 1000;
static constexpr uint32_t USB_BAUD = 115200;
static constexpr uint8_t I2C_SCAN_MAX = 24;
static constexpr uint8_t I2C_PROBE_MAX = 8;

struct PinPair {
  uint8_t sda;
  uint8_t scl;
  const char *label;
};

// Common Nano ESP32 pairs (skip D2 SET, D6/D7 UART).
static const PinPair kPinPairs[] = {
    {A4, A5, "A4/A5"}, {A2, A3, "A2/A3"}, {A6, A7, "A6/A7"},
    {A0, A1, "A0/A1"}, {D4, D5, "D4/D5"}, {D8, D9, "D8/D9"},
    {D10, D11, "D10/D11"}, {D12, D13, "D12/D13"},
};
static constexpr size_t kPinPairCount = sizeof(kPinPairs) / sizeof(kPinPairs[0]);

static constexpr uint8_t kPrimaryTargets[] = {0x29, 0x4A, 0x4B, 0x76, 0x77};
static constexpr uint8_t kMissingTargets[] = {0x12, 0x1C, 0x1E, 0x53};

Adafruit_BME680 bme680;
Adafruit_LTR390 ltr390;
Adafruit_LIS3MDL lis3mdl;
Adafruit_VL53L1X vl53l1x;
Adafruit_PM25AQI pm25;
Adafruit_BNO08x bno08x_i2c(-1);
Adafruit_BNO08x_RVC bno08x_rvc;

bool have_bme680 = false;
bool have_ltr390 = false;
bool have_lis3mdl = false;
bool have_pmsa003i = false;
bool have_vl53l1x = false;
bool have_bno085_i2c = false;
bool have_bno085_rvc = false;
bool have_wire1 = false;

uint8_t bme680_addr = 0;
uint8_t lis3mdl_addr = 0;
uint8_t bno085_addr = 0;
const char *bno085_mode = "none";
const char *primary_label = "A4/A5";
const char *secondary_label = "none";

uint8_t i2c_sda = A4;
uint8_t i2c_scl = A5;
uint8_t i2c2_sda = 0;
uint8_t i2c2_scl = 0;

uint8_t i2c_found[I2C_SCAN_MAX];
uint8_t i2c_found_count = 0;
uint8_t i2c2_found[I2C_SCAN_MAX];
uint8_t i2c2_found_count = 0;

struct ProbeHit {
  const char *label;
  uint8_t count;
  uint8_t addrs[I2C_SCAN_MAX];
};
ProbeHit probe_hits[I2C_PROBE_MAX];
uint8_t probe_hit_count = 0;

TwoWire *bus_ltr = &Wire;
TwoWire *bus_lis = &Wire;
TwoWire *bus_pms = &Wire;

uint32_t frame_seq = 0;

float imu_accel_x = 0, imu_accel_y = 0, imu_accel_z = 0;
float imu_yaw = 0, imu_pitch = 0, imu_roll = 0;
bool imu_have_accel = false;
bool imu_have_orient = false;

static void append_i2c_addr_hex(JsonArray out, uint8_t addr) {
  char buf[8];
  snprintf(buf, sizeof(buf), "0x%02X", addr);
  out.add(buf);
}

static bool addr_in_list(const uint8_t *addrs, uint8_t n, uint8_t want) {
  for (uint8_t i = 0; i < n; i++) {
    if (addrs[i] == want) {
      return true;
    }
  }
  return false;
}

static uint8_t count_targets(const uint8_t *found, uint8_t nfound,
                             const uint8_t *targets, size_t ntargets) {
  uint8_t score = 0;
  for (size_t t = 0; t < ntargets; t++) {
    if (addr_in_list(found, nfound, targets[t])) {
      score++;
    }
  }
  return score;
}

static void scan_on(TwoWire *bus, uint8_t *out, uint8_t *out_count) {
  *out_count = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    bus->beginTransmission(addr);
    if (bus->endTransmission() == 0) {
      if (*out_count < I2C_SCAN_MAX) {
        out[(*out_count)++] = addr;
      }
    }
  }
}

static void i2c_scan_bus() {
  scan_on(&Wire, i2c_found, &i2c_found_count);
  if (have_wire1) {
    scan_on(&Wire1, i2c2_found, &i2c2_found_count);
  } else {
    i2c2_found_count = 0;
  }
}

static bool i2c_has_addr(uint8_t addr) {
  return addr_in_list(i2c_found, i2c_found_count, addr) ||
         addr_in_list(i2c2_found, i2c2_found_count, addr);
}

static TwoWire *bus_for_addr(uint8_t addr) {
  if (addr_in_list(i2c2_found, i2c2_found_count, addr)) {
    return &Wire1;
  }
  return &Wire;
}

static void pmsa003i_wake() {
  pinMode(PMSA003I_SET_PIN, OUTPUT);
  digitalWrite(PMSA003I_SET_PIN, HIGH);
}

static void probe_pin_matrix() {
  probe_hit_count = 0;
  uint8_t best_primary_score = 0;
  uint8_t best_missing_score = 0;
  int best_primary_idx = 0;
  int best_missing_idx = -1;

  for (size_t i = 0; i < kPinPairCount; i++) {
    const PinPair &p = kPinPairs[i];
    Wire.end();
    delay(5);
    Wire.begin(p.sda, p.scl);
    Wire.setClock(50000);
    delay(20);

    uint8_t found[I2C_SCAN_MAX];
    uint8_t n = 0;
    scan_on(&Wire, found, &n);

    if (n > 0 && probe_hit_count < I2C_PROBE_MAX) {
      ProbeHit &hit = probe_hits[probe_hit_count++];
      hit.label = p.label;
      hit.count = n;
      for (uint8_t j = 0; j < n; j++) {
        hit.addrs[j] = found[j];
      }
    }

    const uint8_t primary_score =
        count_targets(found, n, kPrimaryTargets, sizeof(kPrimaryTargets));
    const uint8_t missing_score =
        count_targets(found, n, kMissingTargets, sizeof(kMissingTargets));

    if (primary_score > best_primary_score ||
        (primary_score == best_primary_score && i == 0)) {
      best_primary_score = primary_score;
      best_primary_idx = static_cast<int>(i);
    }
    if (missing_score > best_missing_score) {
      best_missing_score = missing_score;
      best_missing_idx = static_cast<int>(i);
    }
  }

  Wire.end();
  delay(5);

  i2c_sda = kPinPairs[best_primary_idx].sda;
  i2c_scl = kPinPairs[best_primary_idx].scl;
  primary_label = kPinPairs[best_primary_idx].label;

  have_wire1 = false;
  secondary_label = "none";
  if (best_missing_idx >= 0 && best_missing_score > 0 &&
      best_missing_idx != best_primary_idx) {
    i2c2_sda = kPinPairs[best_missing_idx].sda;
    i2c2_scl = kPinPairs[best_missing_idx].scl;
    secondary_label = kPinPairs[best_missing_idx].label;
    have_wire1 = true;
  }
}

static void init_i2c_buses() {
  // Two STEMMA hubs + long daisy = high capacitance; keep the bus slow.
  Wire.begin(i2c_sda, i2c_scl);
  Wire.setClock(25000);
  if (have_wire1) {
    Wire1.begin(i2c2_sda, i2c2_scl);
    Wire1.setClock(25000);
  }
}

static void reinit_primary_i2c() {
  Wire.end();
  delay(10);
  Wire.begin(i2c_sda, i2c_scl);
  Wire.setClock(25000);
}

static void quaternion_to_euler(float qr, float qi, float qj, float qk,
                                float *yaw, float *pitch, float *roll) {
  const float sqr = qr * qr;
  const float sqi = qi * qi;
  const float sqj = qj * qj;
  const float sqk = qk * qk;
  *yaw = atan2f(2.0f * (qi * qj + qk * qr), (sqi - sqj - sqk + sqr)) * RAD_TO_DEG;
  *pitch = asinf(-2.0f * (qi * qk - qj * qr) / (sqi + sqj + sqk + sqr)) * RAD_TO_DEG;
  *roll = atan2f(2.0f * (qj * qk + qi * qr), (-sqi - sqj + sqk + sqr)) * RAD_TO_DEG;
}

static void init_bme680() {
  if (bme680.begin(0x76, &Wire)) {
    have_bme680 = true;
    bme680_addr = 0x76;
  } else if (bme680.begin(0x77, &Wire)) {
    have_bme680 = true;
    bme680_addr = 0x77;
  }
  if (!have_bme680) {
    return;
  }
  bme680.setTemperatureOversampling(BME680_OS_8X);
  bme680.setHumidityOversampling(BME680_OS_2X);
  bme680.setPressureOversampling(BME680_OS_4X);
  bme680.setIIRFilterSize(BME680_FILTER_SIZE_3);
  bme680.setGasHeater(320, 150);
}

static void init_ltr390() {
  bus_ltr = bus_for_addr(0x53);
  have_ltr390 = ltr390.begin(bus_ltr);
  if (bus_ltr == &Wire) {
    reinit_primary_i2c();
  }
}

static void init_lis3mdl() {
  TwoWire *candidates[2] = {&Wire, have_wire1 ? &Wire1 : nullptr};
  for (uint8_t c = 0; c < 2; c++) {
    TwoWire *bus = candidates[c];
    if (bus == nullptr) {
      continue;
    }
    if (lis3mdl.begin_I2C(0x1C, bus)) {
      have_lis3mdl = true;
      lis3mdl_addr = 0x1C;
      bus_lis = bus;
      break;
    }
    if (lis3mdl.begin_I2C(0x1E, bus)) {
      have_lis3mdl = true;
      lis3mdl_addr = 0x1E;
      bus_lis = bus;
      break;
    }
  }
  if (!have_lis3mdl) {
    return;
  }
  lis3mdl.setPerformanceMode(LIS3MDL_ULTRAHIGHMODE);
  lis3mdl.setOperationMode(LIS3MDL_CONTINUOUSMODE);
  lis3mdl.setDataRate(LIS3MDL_DATARATE_155_HZ);
}

static void init_pmsa003i() {
  TwoWire *candidates[2] = {&Wire, have_wire1 ? &Wire1 : nullptr};
  for (uint8_t c = 0; c < 2; c++) {
    TwoWire *bus = candidates[c];
    if (bus == nullptr) {
      continue;
    }
    if (pm25.begin_I2C(bus)) {
      have_pmsa003i = true;
      bus_pms = bus;
      return;
    }
  }
}

static void init_vl53l1x() {
  if (!vl53l1x.begin(0x29, &Wire)) {
    return;
  }
  if (!vl53l1x.startRanging()) {
    return;
  }
  vl53l1x.setTimingBudget(50);
  have_vl53l1x = true;
}

static bool bno_enable_reports() {
  bool ok = true;
  ok = bno08x_i2c.enableReport(SH2_ACCELEROMETER, 50000) && ok;
  ok = bno08x_i2c.enableReport(SH2_ARVR_STABILIZED_RV, 50000) && ok;
  return ok;
}

static void init_bno085_i2c() {
  if (bno08x_i2c.begin_I2C(0x4A, &Wire)) {
    have_bno085_i2c = true;
    bno085_addr = 0x4A;
  } else if (bno08x_i2c.begin_I2C(0x4B, &Wire)) {
    have_bno085_i2c = true;
    bno085_addr = 0x4B;
  }
  if (!have_bno085_i2c) {
    return;
  }
  if (!bno_enable_reports()) {
    have_bno085_i2c = false;
    return;
  }
  bno085_mode = "i2c";
}

static bool bno085_rvc_wait_for_sync(uint32_t timeout_ms) {
  const uint32_t start = millis();
  while (millis() - start < timeout_ms) {
    if (Serial1.available() > 0 && Serial1.peek() == 0xAA) {
      BNO08x_RVC_Data sample;
      if (bno08x_rvc.read(&sample)) {
        return true;
      }
    }
    delay(10);
  }
  return false;
}

static void init_bno085_rvc() {
  Serial1.begin(BNO085_BAUD, SERIAL_8N1, BNO085_RX_PIN, BNO085_TX_PIN);
  delay(50);
  if (bno08x_rvc.begin(&Serial1) && bno085_rvc_wait_for_sync(BNO085_SYNC_MS)) {
    have_bno085_rvc = true;
    bno085_mode = "uart_rvc";
  }
}

static void init_bno085() {
  init_bno085_i2c();
  if (!have_bno085_i2c) {
    init_bno085_rvc();
  }
}

static void sensor_json(JsonObject obj, bool ok, const char *detail, uint8_t addr = 0) {
  obj["ok"] = ok;
  if (detail != nullptr) {
    obj["detail"] = detail;
  }
  if (addr != 0) {
    char buf[8];
    snprintf(buf, sizeof(buf), "0x%02X", addr);
    obj["addr"] = buf;
  }
}

static void emit_boot_diagnostic() {
  StaticJsonDocument<2048> doc;
  doc["schema"] = "orion.sensor_boot.v1";
  doc["uptime_ms"] = millis();

  JsonObject i2c = doc.createNestedObject("i2c");
  i2c["primary"] = primary_label;
  i2c["secondary"] = secondary_label;
  JsonArray addrs = i2c.createNestedArray("addresses");
  for (uint8_t i = 0; i < i2c_found_count; i++) {
    append_i2c_addr_hex(addrs, i2c_found[i]);
  }
  JsonArray addrs2 = i2c.createNestedArray("addresses_secondary");
  for (uint8_t i = 0; i < i2c2_found_count; i++) {
    append_i2c_addr_hex(addrs2, i2c2_found[i]);
  }

  JsonArray probes = i2c.createNestedArray("pin_probe");
  for (uint8_t i = 0; i < probe_hit_count; i++) {
    JsonObject row = probes.createNestedObject();
    row["pins"] = probe_hits[i].label;
    JsonArray a = row.createNestedArray("addresses");
    for (uint8_t j = 0; j < probe_hits[i].count; j++) {
      append_i2c_addr_hex(a, probe_hits[i].addrs[j]);
    }
  }

  JsonObject sensors = doc.createNestedObject("sensors");
  sensor_json(sensors.createNestedObject("bme680"), have_bme680,
              have_bme680 ? nullptr
                          : (i2c_has_addr(0x76) || i2c_has_addr(0x77) ? "begin_failed"
                                                                     : "not_on_bus"),
              bme680_addr);
  sensor_json(sensors.createNestedObject("ltr390"), have_ltr390,
              have_ltr390 ? nullptr
                          : (i2c_has_addr(0x53) ? "begin_failed" : "not_on_bus"));
  sensor_json(sensors.createNestedObject("lis3mdl"), have_lis3mdl,
              have_lis3mdl ? nullptr
                           : (i2c_has_addr(0x1C) || i2c_has_addr(0x1E) ? "begin_failed"
                                                                      : "not_on_bus"),
              lis3mdl_addr);
  sensor_json(sensors.createNestedObject("pmsa003i"), have_pmsa003i,
              have_pmsa003i ? nullptr
                            : (i2c_has_addr(0x12) ? "begin_failed" : "not_on_bus"));
  sensors["pmsa003i"]["set_pin"] = "D2";

  sensor_json(sensors.createNestedObject("vl53l1x"), have_vl53l1x,
              have_vl53l1x ? nullptr
                           : (i2c_has_addr(0x29) ? "begin_failed" : "not_on_bus"),
              have_vl53l1x ? 0x29 : 0);

  JsonObject bno = sensors.createNestedObject("bno085");
  const bool bno_ok = have_bno085_i2c || have_bno085_rvc;
  bno["ok"] = bno_ok;
  bno["mode"] = bno085_mode;
  if (bno085_addr != 0) {
    char buf[8];
    snprintf(buf, sizeof(buf), "0x%02X", bno085_addr);
    bno["addr"] = buf;
  }
  if (!bno_ok) {
    bno["detail"] = i2c_has_addr(0x4A) || i2c_has_addr(0x4B) ? "i2c_begin_failed"
                                                             : "not_on_bus_or_uart";
  }

  serializeJson(doc, Serial);
  Serial.println();
}

static bool read_environment(JsonObject out) {
  if (!have_bme680 || !bme680.performReading()) {
    return false;
  }
  out["temp_c"] = roundf(bme680.temperature * 100.0f) / 100.0f;
  out["humidity_pct"] = roundf(bme680.humidity * 100.0f) / 100.0f;
  out["pressure_hpa"] = roundf((bme680.pressure / 100.0f) * 100.0f) / 100.0f;
  out["gas_resistance_ohm"] = static_cast<uint32_t>(bme680.gas_resistance);
  return true;
}

static bool read_uv(JsonObject out) {
  if (!have_ltr390) {
    return false;
  }
  ltr390.setMode(LTR390_MODE_UVS);
  uint32_t uv_raw = ltr390.readUVS();
  ltr390.setMode(LTR390_MODE_ALS);
  uint32_t als_raw = ltr390.readALS();
  out["raw"] = uv_raw;
  out["als_raw"] = als_raw;
  return true;
}

static bool read_magnetic(JsonObject out) {
  if (!have_lis3mdl) {
    return false;
  }
  sensors_event_t event;
  if (!lis3mdl.getEvent(&event)) {
    return false;
  }
  const float x_ut = event.magnetic.x * 100.0f;
  const float y_ut = event.magnetic.y * 100.0f;
  const float z_ut = event.magnetic.z * 100.0f;
  const float magnitude_ut = sqrtf(x_ut * x_ut + y_ut * y_ut + z_ut * z_ut);
  out["x_ut"] = roundf(x_ut * 100.0f) / 100.0f;
  out["y_ut"] = roundf(y_ut * 100.0f) / 100.0f;
  out["z_ut"] = roundf(z_ut * 100.0f) / 100.0f;
  out["magnitude_ut"] = roundf(magnitude_ut * 100.0f) / 100.0f;
  return true;
}

static bool read_particulate(JsonObject out) {
  if (!have_pmsa003i) {
    return false;
  }
  PM25_AQI_Data data;
  if (!pm25.read(&data)) {
    return false;
  }
  out["pm1_ug_m3"] = data.pm10_standard;
  out["pm25_ug_m3"] = data.pm25_standard;
  out["pm10_ug_m3"] = data.pm100_standard;
  return true;
}

static bool read_lidar(JsonObject out) {
  if (!have_vl53l1x || !vl53l1x.dataReady()) {
    return false;
  }
  const int16_t distance_mm = vl53l1x.distance();
  if (distance_mm < 0) {
    vl53l1x.clearInterrupt();
    return false;
  }
  uint8_t status = 255;
  vl53l1x.VL53L1X_GetRangeStatus(&status);
  vl53l1x.clearInterrupt();
  out["distance_mm"] = distance_mm;
  out["status"] = status;
  return true;
}

static void poll_bno_i2c_events() {
  if (!have_bno085_i2c) {
    return;
  }
  if (bno08x_i2c.wasReset()) {
    bno_enable_reports();
    imu_have_accel = false;
    imu_have_orient = false;
  }
  for (uint8_t i = 0; i < 8; i++) {
    sh2_SensorValue_t value;
    if (!bno08x_i2c.getSensorEvent(&value)) {
      break;
    }
    if (value.sensorId == SH2_ACCELEROMETER) {
      imu_accel_x = value.un.accelerometer.x;
      imu_accel_y = value.un.accelerometer.y;
      imu_accel_z = value.un.accelerometer.z;
      imu_have_accel = true;
    } else if (value.sensorId == SH2_ARVR_STABILIZED_RV) {
      quaternion_to_euler(value.un.arvrStabilizedRV.real, value.un.arvrStabilizedRV.i,
                          value.un.arvrStabilizedRV.j, value.un.arvrStabilizedRV.k,
                          &imu_yaw, &imu_pitch, &imu_roll);
      imu_have_orient = true;
    }
  }
}

static bool read_imu(JsonObject out) {
  if (have_bno085_i2c) {
    poll_bno_i2c_events();
    if (!imu_have_accel && !imu_have_orient) {
      return false;
    }
    out["accel_x"] = roundf(imu_accel_x * 1000.0f) / 1000.0f;
    out["accel_y"] = roundf(imu_accel_y * 1000.0f) / 1000.0f;
    out["accel_z"] = roundf(imu_accel_z * 1000.0f) / 1000.0f;
    out["yaw_deg"] = roundf(imu_yaw * 100.0f) / 100.0f;
    out["pitch_deg"] = roundf(imu_pitch * 100.0f) / 100.0f;
    out["roll_deg"] = roundf(imu_roll * 100.0f) / 100.0f;
    return true;
  }
  if (!have_bno085_rvc) {
    return false;
  }
  BNO08x_RVC_Data heading;
  if (!bno08x_rvc.read(&heading)) {
    return false;
  }
  out["accel_x"] = roundf(heading.x_accel * 1000.0f) / 1000.0f;
  out["accel_y"] = roundf(heading.y_accel * 1000.0f) / 1000.0f;
  out["accel_z"] = roundf(heading.z_accel * 1000.0f) / 1000.0f;
  out["yaw_deg"] = roundf(heading.yaw * 100.0f) / 100.0f;
  out["pitch_deg"] = roundf(heading.pitch * 100.0f) / 100.0f;
  out["roll_deg"] = roundf(heading.roll * 100.0f) / 100.0f;
  return true;
}

static void emit_frame() {
  StaticJsonDocument<768> doc;
  doc["schema"] = "orion.sensor_frame.v1";
  doc["seq"] = frame_seq++;
  doc["uptime_ms"] = millis();

  JsonObject env = doc.createNestedObject("environment");
  if (!read_environment(env)) {
    doc.remove("environment");
  }
  JsonObject uv = doc.createNestedObject("uv");
  if (!read_uv(uv)) {
    doc.remove("uv");
  }
  JsonObject magnetic = doc.createNestedObject("magnetic");
  if (!read_magnetic(magnetic)) {
    doc.remove("magnetic");
  }
  JsonObject particulate = doc.createNestedObject("particulate");
  if (!read_particulate(particulate)) {
    doc.remove("particulate");
  }
  JsonObject lidar = doc.createNestedObject("lidar");
  if (!read_lidar(lidar)) {
    doc.remove("lidar");
  }
  JsonObject imu = doc.createNestedObject("imu");
  if (!read_imu(imu)) {
    doc.remove("imu");
  }

  serializeJson(doc, Serial);
  Serial.println();
}

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial && millis() < 3000) {
    delay(10);
  }

  pmsa003i_wake();
  delay(PMSA003I_BOOT_MS);

  probe_pin_matrix();
  init_i2c_buses();
  // Fresh scan on the selected buses before driver init.
  i2c_scan_bus();

  init_bme680();
  init_ltr390();
  init_lis3mdl();
  init_pmsa003i();
  init_vl53l1x();
  init_bno085();

  i2c_scan_bus();
  emit_boot_diagnostic();
}

void loop() {
  static uint32_t last_frame_ms = 0;
  static uint32_t last_boot_ms = 0;
  const uint32_t now = millis();
  if (have_bno085_i2c) {
    poll_bno_i2c_events();
  }
  if (last_boot_ms == 0 || now - last_boot_ms >= 10000) {
    last_boot_ms = now;
    i2c_scan_bus();
    emit_boot_diagnostic();
  }
  if (now - last_frame_ms >= FRAME_INTERVAL_MS) {
    last_frame_ms = now;
    emit_frame();
  }
}
