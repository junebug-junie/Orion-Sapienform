/*
 * Athena cabinet sensor node — Arduino Nano ESP32 (ABX00083)
 *
 * Emits one NDJSON boot diagnostic (orion.sensor_boot.v1) then one data frame
 * per second (orion.sensor_frame.v1). Failed sensors omit their sub-object.
 */

#include <Wire.h>
#include <math.h>
#include <ArduinoJson.h>

#include <Adafruit_Sensor.h>
#include <Adafruit_BME680.h>
#include <Adafruit_LTR390.h>
#include <Adafruit_LIS3MDL.h>
#include <Adafruit_VL53L1X.h>
#include <Adafruit_BNO08x_RVC.h>

static constexpr uint8_t I2C_SDA_PIN = A4;
static constexpr uint8_t I2C_SCL_PIN = A5;

static constexpr uint8_t BNO085_RX_PIN = D7;
static constexpr uint8_t BNO085_TX_PIN = D6;
static constexpr uint32_t BNO085_BAUD = 115200;
static constexpr uint32_t BNO085_SYNC_MS = 3000;

static constexpr uint8_t PMSA003I_SET_PIN = D2;
static constexpr uint8_t PMSA003I_I2C_ADDR = 0x12;
static constexpr uint32_t PMSA003I_BOOT_MS = 3000;

static constexpr uint32_t FRAME_INTERVAL_MS = 1000;
static constexpr uint32_t USB_BAUD = 115200;
static constexpr uint8_t I2C_SCAN_MAX = 16;

Adafruit_BME680 bme680;
Adafruit_LTR390 ltr390;
Adafruit_LIS3MDL lis3mdl;
Adafruit_VL53L1X vl53l1x;
Adafruit_BNO08x_RVC bno085;

bool have_bme680 = false;
bool have_ltr390 = false;
bool have_lis3mdl = false;
bool have_pmsa003i = false;
bool have_vl53l1x = false;
bool have_bno085 = false;

uint8_t bme680_addr = 0;
uint8_t lis3mdl_addr = 0;
uint8_t i2c_found[I2C_SCAN_MAX];
uint8_t i2c_found_count = 0;

uint32_t frame_seq = 0;

static void append_i2c_addr_hex(JsonArray out, uint8_t addr) {
  char buf[8];
  snprintf(buf, sizeof(buf), "0x%02X", addr);
  out.add(buf);
}

static void i2c_scan_bus() {
  i2c_found_count = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      if (i2c_found_count < I2C_SCAN_MAX) {
        i2c_found[i2c_found_count++] = addr;
      }
    }
  }
}

static bool i2c_has_addr(uint8_t addr) {
  for (uint8_t i = 0; i < i2c_found_count; i++) {
    if (i2c_found[i] == addr) {
      return true;
    }
  }
  return false;
}

static bool pmsa003i_wake() {
  pinMode(PMSA003I_SET_PIN, OUTPUT);
  digitalWrite(PMSA003I_SET_PIN, HIGH);
  return true;
}

static bool pmsa003i_probe() {
  Wire.beginTransmission(PMSA003I_I2C_ADDR);
  return Wire.endTransmission() == 0;
}

static bool pmsa003i_request_frame() {
  const uint8_t cmd[] = {0x42, 0x4D, 0xE2, 0x00, 0x00, 0x01, 0x71};
  Wire.beginTransmission(PMSA003I_I2C_ADDR);
  Wire.write(cmd, sizeof(cmd));
  if (Wire.endTransmission() != 0) {
    return false;
  }
  delay(10);
  return true;
}

static bool pmsa003i_read(uint16_t *pm1, uint16_t *pm25, uint16_t *pm10) {
  if (!pmsa003i_request_frame()) {
    return false;
  }

  uint8_t buf[32];
  Wire.requestFrom((int)PMSA003I_I2C_ADDR, 32);
  if (Wire.available() < 32) {
    return false;
  }
  for (uint8_t i = 0; i < 32; i++) {
    buf[i] = Wire.read();
  }

  if (buf[0] != 0x42 || buf[1] != 0x4d) {
    return false;
  }

  *pm1 = (uint16_t(buf[4]) << 8) | buf[5];
  *pm25 = (uint16_t(buf[6]) << 8) | buf[7];
  *pm10 = (uint16_t(buf[8]) << 8) | buf[9];
  return true;
}

static void init_i2c_bus() {
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(100000);
}

static void init_bme680() {
  if (bme680.begin(0x76)) {
    have_bme680 = true;
    bme680_addr = 0x76;
  } else if (bme680.begin(0x77)) {
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
  have_ltr390 = ltr390.begin(&Wire);
}

static void init_lis3mdl() {
  if (lis3mdl.begin_I2C(0x1C, &Wire)) {
    have_lis3mdl = true;
    lis3mdl_addr = 0x1C;
  } else if (lis3mdl.begin_I2C(0x1E, &Wire)) {
    have_lis3mdl = true;
    lis3mdl_addr = 0x1E;
  }
  if (!have_lis3mdl) {
    return;
  }
  lis3mdl.setPerformanceMode(LIS3MDL_ULTRAHIGHMODE);
  lis3mdl.setOperationMode(LIS3MDL_CONTINUOUSMODE);
  lis3mdl.setDataRate(LIS3MDL_DATARATE_155_HZ);
}

static void init_pmsa003i() {
  have_pmsa003i = pmsa003i_probe();
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

static bool bno085_wait_for_sync(uint32_t timeout_ms) {
  const uint32_t start = millis();
  while (millis() - start < timeout_ms) {
    if (Serial1.available() > 0 && Serial1.peek() == 0xAA) {
      BNO08x_RVC_Data sample;
      if (bno085.read(&sample)) {
        return true;
      }
    }
    delay(10);
  }
  return false;
}

static void init_bno085() {
  Serial1.begin(BNO085_BAUD, SERIAL_8N1, BNO085_RX_PIN, BNO085_TX_PIN);
  delay(100);
  if (bno085.begin(&Serial1) && bno085_wait_for_sync(BNO085_SYNC_MS)) {
    have_bno085 = true;
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
  StaticJsonDocument<1024> doc;
  doc["schema"] = "orion.sensor_boot.v1";
  doc["uptime_ms"] = millis();

  JsonObject i2c = doc.createNestedObject("i2c");
  i2c["sda_pin"] = "A4";
  i2c["scl_pin"] = "A5";
  JsonArray addrs = i2c.createNestedArray("addresses");
  for (uint8_t i = 0; i < i2c_found_count; i++) {
    append_i2c_addr_hex(addrs, i2c_found[i]);
  }

  JsonObject sensors = doc.createNestedObject("sensors");
  sensor_json(sensors.createNestedObject("bme680"), have_bme680,
              have_bme680 ? nullptr : (i2c_has_addr(0x76) || i2c_has_addr(0x77)
                                           ? "begin_failed"
                                           : "not_on_bus"),
              bme680_addr);
  sensor_json(sensors.createNestedObject("ltr390"), have_ltr390,
              have_ltr390 ? nullptr
                          : (i2c_has_addr(0x53) ? "begin_failed" : "not_on_bus"));
  sensor_json(sensors.createNestedObject("lis3mdl"), have_lis3mdl,
              have_lis3mdl ? nullptr
                           : (i2c_has_addr(0x1C) || i2c_has_addr(0x1E)
                                  ? "begin_failed"
                                  : "not_on_bus"),
              lis3mdl_addr);
  sensor_json(sensors.createNestedObject("pmsa003i"), have_pmsa003i,
              have_pmsa003i ? nullptr
                            : (i2c_has_addr(PMSA003I_I2C_ADDR) ? "probe_nack"
                                                               : "not_on_bus"));
  sensors["pmsa003i"]["set_pin"] = "D2";

  sensor_json(sensors.createNestedObject("vl53l1x"), have_vl53l1x,
              have_vl53l1x ? nullptr
                           : (i2c_has_addr(0x29) ? "begin_failed" : "not_on_bus"),
              have_vl53l1x ? 0x29 : 0);

  JsonObject bno = sensors.createNestedObject("bno085");
  bno["ok"] = have_bno085;
  if (!have_bno085) {
    bno["detail"] = "uart_no_sync";
  }
  bno["rx_pin"] = "D7";
  bno["tx_pin"] = "D6";
  bno["baud"] = BNO085_BAUD;
  bno["mode"] = "uart_rvc";

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
  const float magnitude_ut =
      sqrtf(x_ut * x_ut + y_ut * y_ut + z_ut * z_ut);

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

  uint16_t pm1 = 0;
  uint16_t pm25 = 0;
  uint16_t pm10 = 0;
  if (!pmsa003i_read(&pm1, &pm25, &pm10)) {
    return false;
  }

  out["pm1_ug_m3"] = pm1;
  out["pm25_ug_m3"] = pm25;
  out["pm10_ug_m3"] = pm10;
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

static bool read_imu(JsonObject out) {
  if (!have_bno085) {
    return false;
  }

  BNO08x_RVC_Data heading;
  if (!bno085.read(&heading)) {
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

  init_i2c_bus();
  init_bme680();
  init_ltr390();
  init_lis3mdl();
  init_vl53l1x();

  delay(PMSA003I_BOOT_MS);
  init_pmsa003i();

  init_bno085();

  i2c_scan_bus();
  emit_boot_diagnostic();
}

void loop() {
  static uint32_t last_frame_ms = 0;
  const uint32_t now = millis();

  if (now - last_frame_ms >= FRAME_INTERVAL_MS) {
    last_frame_ms = now;
    emit_frame();
  }
}
