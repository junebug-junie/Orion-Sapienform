/*
 * Athena cabinet sensor node — Arduino Nano ESP32 (ABX00083)
 *
 * Emits one NDJSON line per second on USB Serial matching schema
 * "orion.sensor_frame.v1". Failed sensors omit their entire sub-object;
 * never zero-fill absent readings.
 *
 * Sensors:
 *   BME680, LTR390, LIS3MDL, PMSA003I, VL53L1X — shared I2C (Wire)
 *   BNO085 — UART-RVC on Serial1 only (NOT on ESP32 I2C)
 *
 * No audio / MAX9814 on this board.
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

// ---------------------------------------------------------------------------
// Pin / bus configuration (see README for wiring)
// ---------------------------------------------------------------------------

static constexpr uint8_t I2C_SDA_PIN = A4;
static constexpr uint8_t I2C_SCL_PIN = A5;

// BNO085 UART-RVC: breakout TX -> Nano RX1 (D7), breakout RX -> Nano TX1 (D6).
static constexpr uint8_t BNO085_RX_PIN = D7;
static constexpr uint8_t BNO085_TX_PIN = D6;
static constexpr uint32_t BNO085_BAUD = 115200;

// PMSA003I SET pin (active low sleep); tie high or drive from MCU to wake.
static constexpr uint8_t PMSA003I_SET_PIN = D2;
static constexpr uint8_t PMSA003I_I2C_ADDR = 0x12;

static constexpr uint32_t FRAME_INTERVAL_MS = 1000;
static constexpr uint32_t USB_BAUD = 115200;

// ---------------------------------------------------------------------------
// Sensor drivers and availability flags (init-time only; never zero-fill)
// ---------------------------------------------------------------------------

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

uint32_t frame_seq = 0;

// ---------------------------------------------------------------------------
// PMSA003I — minimal I2C reader (no external library)
// ---------------------------------------------------------------------------

static bool pmsa003i_wake() {
  pinMode(PMSA003I_SET_PIN, OUTPUT);
  digitalWrite(PMSA003I_SET_PIN, HIGH);
  delay(100);
  return true;
}

static bool pmsa003i_probe() {
  Wire.beginTransmission(PMSA003I_I2C_ADDR);
  return Wire.endTransmission() == 0;
}

static bool pmsa003i_read(uint16_t *pm1, uint16_t *pm25, uint16_t *pm10) {
  uint8_t buf[32];
  Wire.requestFrom((int)PMSA003I_I2C_ADDR, 32);
  if (Wire.available() < 32) {
    return false;
  }
  for (uint8_t i = 0; i < 32; i++) {
    buf[i] = Wire.read();
  }

  // Standard Plantower frame: 0x42 0x4d header, PM values at bytes 4-9 (BE).
  if (buf[0] != 0x42 || buf[1] != 0x4d) {
    return false;
  }

  *pm1 = (uint16_t(buf[4]) << 8) | buf[5];
  *pm25 = (uint16_t(buf[6]) << 8) | buf[7];
  *pm10 = (uint16_t(buf[8]) << 8) | buf[9];
  return true;
}

// ---------------------------------------------------------------------------
// Per-sensor init (soft-fail)
// ---------------------------------------------------------------------------

static void init_i2c_bus() {
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(100000);
}

static void init_bme680() {
  if (bme680.begin(0x76)) {
    have_bme680 = true;
  } else if (bme680.begin(0x77)) {
    have_bme680 = true;
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
  have_ltr390 = ltr390.begin();
}

static void init_lis3mdl() {
  if (lis3mdl.begin(0x1C)) {
    have_lis3mdl = true;
  } else if (lis3mdl.begin(0x1E)) {
    have_lis3mdl = true;
  }
  if (have_lis3mdl) {
    lis3mdl.setPerformanceMode(LIS3MDL_ULTRAHIGHPERFORMANCE);
    lis3mdl.setOperationMode(LIS3MDL_CONTINUOUSMODE);
    lis3mdl.setDataRate(LIS3MDL_DATARATE_155_HZ);
  }
}

static void init_pmsa003i() {
  pmsa003i_wake();
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

static void init_bno085() {
  Serial1.begin(BNO085_BAUD, SERIAL_8N1, BNO085_RX_PIN, BNO085_TX_PIN);
  // UART-RVC only (PS0=1 PS1=0 on breakout). Not on ESP32 I2C.
  if (bno085.begin(&Serial1)) {
    have_bno085 = true;
  }
}

// ---------------------------------------------------------------------------
// Per-sensor read helpers — return false to omit sub-object for this frame
// ---------------------------------------------------------------------------

static bool read_environment(JsonObject out) {
  if (!have_bme680) {
    return false;
  }
  if (!bme680.performReading()) {
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

  // Adafruit LIS3MDL reports gauss; schema expects microtesla (uT).
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
  if (!have_vl53l1x) {
    return false;
  }

  if (!vl53l1x.dataReady()) {
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

  // VL53L1X RangeStatus 0 == valid (see host schema / ST user manual).
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

// ---------------------------------------------------------------------------
// Frame emission
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Arduino lifecycle
// ---------------------------------------------------------------------------

void setup() {
  Serial.begin(USB_BAUD);
  while (!Serial && millis() < 3000) {
    delay(10);
  }

  init_i2c_bus();
  init_bme680();
  init_ltr390();
  init_lis3mdl();
  init_pmsa003i();
  init_vl53l1x();
  init_bno085();

  delay(500);
}

void loop() {
  static uint32_t last_frame_ms = 0;
  const uint32_t now = millis();

  if (now - last_frame_ms >= FRAME_INTERVAL_MS) {
    last_frame_ms = now;
    emit_frame();
  }

}
