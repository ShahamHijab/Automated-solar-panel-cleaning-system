
#include <Wire.h>
#include <Adafruit_INA219.h>

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "logistic_model.h"  // Your converted model as C array

// ===== Sensor Pins =====
const int dustLEDPin = 4;
const int dustAnalogPin = 34;

// ===== Normalization values (from your Python training) =====
#define DUST_MEAN  312.4
#define DUST_STD   58.3
#define VOLT_MEAN  4.96
#define VOLT_STD   0.44

// ===== TFLite Setup =====
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

Adafruit_INA219 ina219;

void setup() {
  Serial.begin(115200);

  // Dust LED pin
  pinMode(dustLEDPin, OUTPUT);
  digitalWrite(dustLEDPin, LOW);

  // INA219 Init
  if (!ina219.begin()) {
    Serial.println("INA219 not found!");
    while (1);
  }

  // Load TFLite model
  const tflite::Model* model = tflite::GetModel(logistic_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch");
    while (1);
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("System ready!");
}

// ===== GP2Y1010AU0F Read Function =====
float readDustSensor() {
  digitalWrite(dustLEDPin, LOW);
  delayMicroseconds(280);
  int rawADC = analogRead(dustAnalogPin);
  delayMicroseconds(40);
  digitalWrite(dustLEDPin, HIGH);
  delayMicroseconds(9680);  // Complete 10ms cycle

  float voltage = rawADC * (3.3 / 4095.0);
  float dustDensity = (voltage - 0.1) * 1000.0 / 0.5;
  return dustDensity;  // in µg/m³ approx.
}

// ===== Inference and Control =====
void loop() {
  float dust = readDustSensor();
  float voltage = ina219.getBusVoltage_V();  // from INA219

  // Normalize inputs
  float normDust = (dust - DUST_MEAN) / DUST_STD;
  float normVolt = (voltage - VOLT_MEAN) / VOLT_STD;

  // Feed input
  input->data.f[0] = normDust;
  input->data.f[1] = normVolt;

  // Run inference
  TfLiteStatus status = interpreter->Invoke();
  if (status != kTfLiteOk) {
    Serial.println("Model inference failed!");
    return;
  }

  float result = output->data.f[0];

  // Show results
  Serial.print("Dust: "); Serial.print(dust);
  Serial.print(" µg/m³, Voltage: "); Serial.print(voltage);
  Serial.print(" V, Model Output: "); Serial.println(result);

  if (result > 0.5) {
    Serial.println("→ CLEANING REQUIRED!");
    // digitalWrite(motorPin, HIGH);
  } else {
    Serial.println("→ CLEANING NOT NEEDED.");
    // digitalWrite(motorPin, LOW);
  }

  delay(3000);
}
