#include <Wire.h>
// #include <Adafruit_INA219.h>  // Commented out
#include <HTTPClient.h>  
#include <esp_heap_caps.h>
#include <WiFi.h>

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define WIFI_SSID "Wifi-79J"
#define WIFI_PASSWORD "797979jjjj"

// Active-Low Relay Definitions
#define RELAY_ON  LOW     // LOW = motor ON
#define RELAY_OFF HIGH    // HIGH = motor OFF

// === Pins & Constants ===
const int dustLEDPin = 2;
const int dustAnalogPin = 4;
const int RELAY_PIN = 5;
const int DEBUG_LED = 13;     // Optional: for testing relay status
const int SDA_PIN = 21;
const int SCL_PIN = 19;

#define DUST_MEAN 2.6
#define DUST_STD 1.6
#define VOLT_MEAN 17.7
#define VOLT_STD 1.2

// TensorFlow Model
#include "logistic_model.h"

// TensorFlow Lite Micro setup
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t* tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Connect to WiFi
void connectWiFi() {
  WiFi.disconnect(true);
  delay(100);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.println("Connecting to WiFi...");

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("Failed to connect to WiFi!");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, RELAY_OFF);

  pinMode(DEBUG_LED, OUTPUT); // Optional debug LED
  digitalWrite(DEBUG_LED, LOW);

  connectWiFi();

  if (!tensor_arena) {
    Serial.println("Failed to allocate tensor arena in PSRAM!");
    while (1);
  }
  if (!tensor_arena) {
  Serial.println(" Failed to allocate tensor arena in PSRAM!");
  while (1);
}

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

// Dust sensor reading
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

  // Simulated voltage
  float loadVoltage = 17.7;

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
  Serial.printf("Dust: %.2f µg/m³, Voltage: %.2f V\n", dust, loadVoltage);
  Serial.printf("Predicted probability: %.2f\n", prob);
  Serial.printf("Prediction: %s\n", result);

  // ========== motor CONTROL ==========
  if (prob < 0.5) { // or use: if (true) to force test
    Serial.println("Triggering motor for cleaning...");
    digitalWrite(RELAY_PIN, RELAY_ON);     // Turn motor ON
    digitalWrite(DEBUG_LED, HIGH);         // Optional LED ON
    delay(10000);
    digitalWrite(RELAY_PIN, RELAY_OFF);    // Turn motor OFF
    digitalWrite(DEBUG_LED, LOW);          // Optional LED OFF
    Serial.println("Cleaning complete.");
  }

  // ========== Firebase ==========
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String firebaseUrl = String("https://solarsystem-2babe-default-rtdb.firebaseio.com//solarData.json?auth=ACOZE3BvabpPdNGuu83DAyVm2NkRlEzUg3bPgZWr");

    String payload = "{";
    payload += "\"dustDensity\":" + String(dust) + ",";
    payload += "\"voltage\":" + String(loadVoltage) + ",";
    payload += "\"probability\":" + String(prob) + ",";
    payload += "\"prediction\":\"" + String(result) + "\"";
    payload += "}";

    http.begin(firebaseUrl);
    http.addHeader("Content-Type", "application/json");

    int responseCode = http.POST(payload);
    Serial.print("Firebase Response: ");
    Serial.println(responseCode);
    http.end();
  } else {
    Serial.println("WiFi not connected.");
  }

  // ========== ThingSpeak ==========
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String thingSpeakAPIKey = "UFJX6PHYTC441XUE";
    int predictionCode = (String(result) == "Clean") ? 1 : 0;

    String url = "http://api.thingspeak.com/update?api_key=" + thingSpeakAPIKey;
    url += "&field1=" + String(dust, 2);
    url += "&field2=" + String(loadVoltage, 2);
    url += "&field3=" + String(prob, 4);
    url += "&field4=" + String(predictionCode);

    http.begin(url);
    int httpCode = http.GET();
    Serial.print("ThingSpeak Response: ");
    Serial.println(httpCode);
    http.end();
  }

  Serial.println("-----------------------");
  delay(7000);
}
