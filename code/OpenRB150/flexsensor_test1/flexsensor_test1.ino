// Read the test code of three Flex sensors (A0, A1, A2) at the same time. 
// A0, A1 and A2 read Flex Sensor of thumb, index finger and middle finger respectively. 
// It will output the raw ADC value, voltage, and resistance value of each sensor in the serial port.

const float Vcc = 5.0;              // Power supply voltage
const float R_fixed_0 = 100000;     // A0: short Flex sensor, uses 100kΩ fixed resistor
const float R_fixed_1 = 100000;     // A1: short Flex sensor, uses 100kΩ fixed resistor
const float R_fixed_2 = 22000;      // A2: long Flex sensor, uses 22kΩ fixed resistor

const int flexPin0 = A0;            // thumb
const int flexPin1 = A1;            // index
const int flexPin2 = A2;            // middle

void setup() {
  Serial.begin(9600);
  Serial.println("Three Flex Sensor Test (A0, A1, A2)");
}

void loop() {
  // Read sensors
  int raw0 = analogRead(flexPin0);
  int raw1 = analogRead(flexPin1);
  int raw2 = analogRead(flexPin2);

  float v0 = raw0 * Vcc / 1023.0;
  float v1 = raw1 * Vcc / 1023.0;
  float v2 = raw2 * Vcc / 1023.0;

  // Calculate resistance
  float r0 = (v0 > 0 && v0 < Vcc) ? R_fixed_0 * (v0 / (Vcc - v0)) : -1;
  float r1 = (v1 > 0 && v1 < Vcc) ? R_fixed_1 * (v1 / (Vcc - v1)) : -1;
  float r2 = (v2 > 0 && v2 < Vcc) ? R_fixed_2 * (v2 / (Vcc - v2)) : -1;

  // Serial output
  Serial.print("Finger1 (A0) - Raw: "); Serial.print(raw0);
  Serial.print(" | V: "); Serial.print(v0, 2);
  Serial.print(" V | R: "); Serial.print(r0, 1); Serial.println(" ohms");

  Serial.print("Finger2 (A1) - Raw: "); Serial.print(raw1);
  Serial.print(" | V: "); Serial.print(v1, 2);
  Serial.print(" V | R: "); Serial.print(r1, 1); Serial.println(" ohms");

  Serial.print("Finger3 (A2) - Raw: "); Serial.print(raw2);
  Serial.print(" | V: "); Serial.print(v2, 2);
  Serial.print(" V | R: "); Serial.print(r2, 1); Serial.println(" ohms");

  Serial.println("-------------------------------");

  delay(200); 
}
