#include <Wire.h>
#include <Adafruit_INA219.h>

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Your converted model
#include "logistic_model.h"  // Should define 'model_tflite'

// ======= Pins & Constants =======
const int dustLEDPin = 2;
const int dustAnalogPin = 4;
const int SDA_PIN = 21;
const int SCL_PIN = 19;

#define DUST_MEAN 2.65    // Mean dust from training data (0 to 5)
#define DUST_STD  1.58    // Adjust as per CSV
#define VOLT_MEAN 18.0    // Mean voltage (16 to 20)
#define VOLT_STD  1.15

// ======= TensorFlow Lite Micro Setup =======
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

Adafruit_INA219 ina219;

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(dustLEDPin, OUTPUT);
  digitalWrite(dustLEDPin, HIGH);  // LED off

  // I2C init
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!ina219.begin(&Wire)) {
    Serial.println("INA219 not found!");
    while (1);
  }

  // Load model
  const tflite::Model* model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    while (1);
  }

  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddFullyConnected();
  resolver.AddLogistic();
  resolver.AddReshape();
  resolver.AddQuantize();
  resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("System ready!");
}

// ======= Dust Sensor Reading =======
float readDustSensor() {
  digitalWrite(dustLEDPin, LOW);
  delayMicroseconds(280);
  int rawADC = analogRead(dustAnalogPin);
  delayMicroseconds(40);
  digitalWrite(dustLEDPin, HIGH);
  delayMicroseconds(9680);

  float voltage = rawADC * (3.3 / 4095.0);  // ADC to voltage
  float dustDensity = 0.17 * voltage - 0.1;  // Adjusted formula

  if (dustDensity < 0) dustDensity = 0;
  return dustDensity;
}

// ======= Main Loop =======
void loop() {
  float dust = readDustSensor();

  float busVoltage = ina219.getBusVoltage_V();         // Load side
  float shuntVoltage = ina219.getShuntVoltage_mV() / 1000.0; // In volts
  float loadVoltage = busVoltage + shuntVoltage;       // Total voltage

  // Normalize based on training data
  float normDust = (dust - DUST_MEAN) / DUST_STD;
  float normVolt = (loadVoltage - VOLT_MEAN) / VOLT_STD;

  input->data.f[0] = normDust;
  input->data.f[1] = normVolt;

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed!");
    return;
  }

  float prob = output->data.f[0];
  const char* result = (prob < 0.5) ? "Needs_cleaning" : "Clean";

  Serial.println("-----------------------");
  Serial.print("Dust: ");
  Serial.print(dust);
  Serial.print(" µg/m³, Voltage: ");
  Serial.print(loadVoltage);
  Serial.println(" V");

  Serial.print("Predicted probability: ");
  Serial.println(prob, 2);
  Serial.print("Prediction: ");
  Serial.println(result);
  Serial.println("-----------------------");

  delay(1000);
}
