
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <HTTPClient.h>  
#include <esp_heap_caps.h>
#include <WiFi.h>



// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"


#define WIFI_SSID "Sbain"
#define WIFI_PASSWORD "cant7301"

#define RELAY_ON  LOW     // Active-low relay
#define RELAY_OFF HIGH

// Variables to store the latest sensor values
float lastDust = 0.0;
float lastVoltage = 0.0;
float lastProbability = 0.0;
String lastPrediction = "Unknown";


//  converted model
#include "logistic_model.h"  // Should define 'model_tflite'

// ======= Pins & Constants =======
const int dustLEDPin = 2;
const int dustAnalogPin = 4;
const int RELAY_PIN = 8;     // Relay control pin
const int SDA_PIN = 21;
const int SCL_PIN = 19;

#define DUST_MEAN 2.6
#define DUST_STD 1.6
#define VOLT_MEAN 17.7
#define VOLT_STD 1.2






// ======= TensorFlow Lite Micro Setup =======
constexpr int kTensorArenaSize = 10 * 1024;  // Increase if needed
uint8_t* tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);



tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

Adafruit_INA219 ina219;
void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ Failed to connect to WiFi!");
  }
}


void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, RELAY_OFF);  // Motor OFF at startup




  // Initialize I2C and INA219
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(500);
  if (!ina219.begin(&Wire)) {
    Serial.println("INA219 not found!");
    while (1);
  }
  if (!tensor_arena) {
  Serial.println("❌ Failed to allocate tensor arena in PSRAM!");
  while (1);
}


  // Load ML model
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

  // INA219 readings
  float busVoltage = ina219.getBusVoltage_V();
  float shuntVoltage = ina219.getShuntVoltage_mV() / 1000.0;
  float loadVoltage = busVoltage + shuntVoltage;

  // Normalize inputs
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

  if (prob < 0.5) {
  Serial.println("Triggering solenoid for cleaning...");
  digitalWrite(RELAY_PIN, RELAY_ON);
  delay(10000);
  digitalWrite(RELAY_PIN, RELAY_OFF);
  Serial.println("Cleaning complete.");
}

  // Send to Firebase
if (WiFi.status() == WL_CONNECTED) {
  HTTPClient http;

  // Firebase URL (change YOUR_PROJECT_ID)
  String firebaseUrl = String("https://solarsystem-2babe-default-rtdb.firebaseio.com//solarData.json?auth=ACOZE3BvabpPdNGuu83DAyVm2NkRlEzUg3bPgZWr");

  // Format payload as JSON
  String payload = "{";
  payload += "\"dustDensity\":" + String(dust) + ",";
  payload += "\"voltage\":" + String(loadVoltage) + ",";
  payload += "\"probability\":" + String(prob) + ",";
  payload += "\"prediction\":\"" + String(result) + "\"";
  payload += "}";

  // Start connection
  http.begin(firebaseUrl);
  http.addHeader("Content-Type", "application/json");

  // Send POST
  int responseCode = http.POST(payload);
  Serial.print("Firebase Response: ");
  Serial.println(responseCode);

  if (responseCode > 0) {
    Serial.println("Data sent successfully!");
  } else {
    Serial.print("Error sending to Firebase: ");
    Serial.println(http.errorToString(responseCode));
  }

  http.end();
} else {
  Serial.println("WiFi not connected.");
}
// ===== Send to ThingSpeak =====
if (WiFi.status() == WL_CONNECTED) {
  HTTPClient http;

  String thingSpeakAPIKey = "UFJX6PHYTC441XUE";  // Your Write API key

  // Convert prediction to numeric: 1 = Clean, 0 = Needs_cleaning
  int predictionCode = (String(result) == "Clean") ? 1 : 0;

  String url = "http://api.thingspeak.com/update?api_key=" + thingSpeakAPIKey;
  url += "&field1=" + String(dust, 2);
  url += "&field2=" + String(loadVoltage, 2);
  url += "&field3=" + String(prob, 4);
  url += "&field4=" + String(predictionCode);  // ✅ Send numeric value now

  http.begin(url);
  int httpCode = http.GET();

  Serial.print("ThingSpeak Response: ");
  Serial.println(httpCode);

  if (httpCode > 0) {
    Serial.println("ThingSpeak update success!");
  } else {
    Serial.println("ThingSpeak update failed.");
  }

  http.end();
}







  Serial.println("-----------------------");
  delay(5000);
}
