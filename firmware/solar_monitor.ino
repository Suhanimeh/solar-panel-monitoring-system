/*
 * IoT-Based Solar Panel Monitoring System
 * Firmware for Arduino Uno / Nano
 *
 * Reads panel voltage, current, ambient temperature, humidity, and light
 * intensity, validates readings against expected sensor ranges, and streams
 * timestamped CSV records over serial to a host computer running the
 * companion Python data logger (see /data_logger).
 *
 * Sensors:
 *   - Resistive voltage divider  -> Panel voltage      (Analog A0)
 *   - ACS712 current sensor       -> Panel current       (Analog A1)
 *   - DHT22                       -> Ambient temp/humidity (Digital D2)
 *   - LDR + fixed resistor        -> Light intensity     (Analog A2)
 *
 * Output protocol (one line per sample, CSV):
 *   timestamp_ms,voltage_v,current_a,power_w,temp_c,humidity_pct,light_pct,status
 *
 * status is OK, or a comma-free error code (SENSOR_FAULT, OUT_OF_RANGE)
 * so the host-side parser can flag bad rows without breaking the CSV shape.
 */

#include <DHT.h>

// ---- Pin configuration ----
#define VOLTAGE_PIN   A0
#define CURRENT_PIN   A1
#define LIGHT_PIN     A2
#define DHT_PIN       2
#define DHT_TYPE      DHT22
#define STATUS_LED    13

// ---- Calibration constants (adjust after bench calibration) ----
const float VOLTAGE_DIVIDER_RATIO = 5.0;     // (R1+R2)/R2 for panel voltage divider
const float ADC_REF_VOLTAGE       = 5.0;
const int   ADC_RESOLUTION        = 1024;
const float ACS712_SENSITIVITY    = 0.185;   // V per A, for 5A module
const float ACS712_ZERO_OFFSET    = 2.5;     // V at 0A

// ---- Valid sensor ranges (used for basic validation / fault detection) ----
const float VOLTAGE_MIN = 0.0;
const float VOLTAGE_MAX = 24.0;   // adjust to panel spec
const float CURRENT_MIN = -1.0;   // small negative allowed for sensor noise
const float CURRENT_MAX = 10.0;
const float TEMP_MIN    = -20.0;
const float TEMP_MAX    = 60.0;
const float HUMIDITY_MIN = 0.0;
const float HUMIDITY_MAX = 100.0;

const unsigned long SAMPLE_INTERVAL_MS = 2000;
unsigned long lastSampleTime = 0;

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(9600);
  pinMode(STATUS_LED, OUTPUT);
  dht.begin();

  // Header row so the host parser can validate the schema on connect
  Serial.println("timestamp_ms,voltage_v,current_a,power_w,temp_c,humidity_pct,light_pct,status");
}

void loop() {
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleTime = now;

  float voltage = readPanelVoltage();
  float current = readPanelCurrent();
  float power = voltage * current;
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  float lightPct = readLightIntensity();

  bool dhtFault = isnan(temperature) || isnan(humidity);
  String status = classifyReading(voltage, current, temperature, humidity, dhtFault);

  digitalWrite(STATUS_LED, status == "OK" ? HIGH : LOW);

  Serial.print(now);
  Serial.print(",");
  Serial.print(voltage, 2);
  Serial.print(",");
  Serial.print(current, 2);
  Serial.print(",");
  Serial.print(power, 2);
  Serial.print(",");
  Serial.print(dhtFault ? -999 : temperature, 1);
  Serial.print(",");
  Serial.print(dhtFault ? -999 : humidity, 1);
  Serial.print(",");
  Serial.print(lightPct, 1);
  Serial.print(",");
  Serial.println(status);
}

float readPanelVoltage() {
  int raw = analogRead(VOLTAGE_PIN);
  float vAtPin = (raw / (float)ADC_RESOLUTION) * ADC_REF_VOLTAGE;
  return vAtPin * VOLTAGE_DIVIDER_RATIO;
}

float readPanelCurrent() {
  int raw = analogRead(CURRENT_PIN);
  float vAtPin = (raw / (float)ADC_RESOLUTION) * ADC_REF_VOLTAGE;
  return (vAtPin - ACS712_ZERO_OFFSET) / ACS712_SENSITIVITY;
}

float readLightIntensity() {
  int raw = analogRead(LIGHT_PIN);
  return (raw / (float)ADC_RESOLUTION) * 100.0;
}

String classifyReading(float v, float c, float t, float h, bool dhtFault) {
  if (dhtFault) return "SENSOR_FAULT";
  if (v < VOLTAGE_MIN || v > VOLTAGE_MAX) return "OUT_OF_RANGE";
  if (c < CURRENT_MIN || c > CURRENT_MAX) return "OUT_OF_RANGE";
  if (t < TEMP_MIN || t > TEMP_MAX) return "OUT_OF_RANGE";
  if (h < HUMIDITY_MIN || h > HUMIDITY_MAX) return "OUT_OF_RANGE";
  return "OK";
}
