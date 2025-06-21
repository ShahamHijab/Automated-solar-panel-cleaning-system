#include <Wire.h>
#include <Adafruit_INA219.h>
#include <WiFi.h>
#include <WebServer.h>
// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Your converted model
#include "logistic_model.h"  // Should define 'model_tflite'
// wifi crediantilas
const char* ssid = "YOUR_SSID";
const char* password = "YOUR_PASSWORD";

WebServer server(80);

String htmlPage = "<h1>Initializing...</h1>";

// ======= WiFi and Server Setup =======
void handleRoot() {
  server.send(200, "text/html", htmlPage);
}
// ======= Pins & Constants =======
const int dustLEDPin = 2;
const int dustAnalogPin = 4;
const int RELAY_PIN = 5;     // Relay control pin
const int SDA_PIN = 21;
const int SCL_PIN = 19;

#define DUST_MEAN 2.65
#define DUST_STD  1.58
#define VOLT_MEAN 18.0
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

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);  // Make sure solenoid is off at start

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
   // Web server setup
  server.on("/", handleRoot);
  server.begin();
  Serial.println("HTTP server started");

  // Serial.println("System ready!");
  Serial.println("System ready!");
}

float readDustSensor() {
  digitalWrite(dustLEDPin, LOW);
  delayMicroseconds(280);
  int rawADC = analogRead(dustAnalogPin);
  delayMicroseconds(40);
  digitalWrite(dustLEDPin, HIGH);
  delayMicroseconds(9680);

  float voltage = rawADC * (3.3 / 4095.0);
  float dustDensity = 0.17 * voltage - 0.1;
  if (dustDensity < 0) dustDensity = 0;
  return dustDensity;
}

void loop() {
  float dust = readDustSensor();

  float busVoltage = ina219.getBusVoltage_V();
  float shuntVoltage = ina219.getShuntVoltage_mV() / 1000.0;
  float loadVoltage = busVoltage + shuntVoltage;

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

  // === Solenoid Control ===
  if (prob < 0.5) {
    Serial.println("Triggering solenoid for cleaning...");
    digitalWrite(RELAY_PIN, HIGH);  // Activate solenoid
    delay(5000);                   // Keep it open for 5 seconds
    digitalWrite(RELAY_PIN, LOW);   // Close it
    Serial.println("Cleaning complete.");
  }
   // Web Page Content
 htmlPage = "<!DOCTYPE html><html><head><meta charset='UTF-8'>";
  htmlPage += "<meta http-equiv='refresh' content='5'>";
  htmlPage += "<title>ESP32 Sensor Monitor</title>";
  htmlPage += "<style>";
  htmlPage += "body { font-family: Arial, sans-serif; background: #f2f2f2; padding: 20px; }";
  htmlPage += "h2 { color: #333; }";
  htmlPage += "table { border-collapse: collapse; width: 100%; max-width: 600px; margin-top: 20px; }";
  htmlPage += "th, td { text-align: left; padding: 12px; }";
  htmlPage += "th { background-color: #4CAF50; color: white; }";
  htmlPage += "tr:nth-child(even) { background-color: #f9f9f9; }";
  htmlPage += ".status-clean { color: green; font-weight: bold; }";
  htmlPage += ".status-dirty { color: red; font-weight: bold; }";
  htmlPage += "</style></head><body>";

  htmlPage += "<h2>ESP32 Sensor Dashboard</h2>";
  htmlPage += "<table border='1'>";
  htmlPage += "<tr><th>Parameter</th><th>Value</th></tr>";
  htmlPage += "<tr><td>Dust Level</td><td>" + String(dust) + " µg/m³</td></tr>";
  htmlPage += "<tr><td>Voltage</td><td>" + String(loadVoltage) + " V</td></tr>";
  htmlPage += "<tr><td>Probability</td><td>" + String(prob, 2) + "</td></tr>";

  // Styled prediction cell
  String statusClass = (prob < 0.5) ? "status-dirty" : "status-clean";
  htmlPage += "<tr><td>Prediction</td><td class='" + statusClass + "'>" + String(result) + "</td></tr>";

  htmlPage += "</table>";
  htmlPage += "<p>Page refreshes every 5 seconds.</p>";
  htmlPage += "</body></html>";

  server.handleClient();
  Serial.println("-----------------------");
  delay(5000);  // Wait 5 seconds before next reading
}
