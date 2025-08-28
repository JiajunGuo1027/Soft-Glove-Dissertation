#include <Dynamixel2Arduino.h>

#define DXL_SERIAL     Serial1
#define DEBUG_SERIAL   Serial
#define DXL_DIR_PIN    -1
#define BUZZER_PIN     10

const float DXL_PROTOCOL_VERSION = 2.0;
const int NUM_STEPS = 60;
const float STEP_DELAY = 66;
const int NUM_CYCLES = 5;

// Motor groups
const uint8_t flex_ids[3]   = {1, 2, 3};  // Thumb, Index, Middle (Flexion)
const uint8_t extend_ids[3] = {4, 5, 6};  // Thumb, Index, Middle (Extension)

// Per-motor angle settings
const float flex_deltas[3]   = {-188.1, -320.5, -332.8};  // Flexion angles
const float extend_deltas[3] = {-204.6, -200.2, -221.8};      // Extension angles

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

void setup() {
  delay(9000);
  DEBUG_SERIAL.begin(115200);
  while (!DEBUG_SERIAL);

  DXL_SERIAL.begin(57600);
  dxl.begin();
  dxl.setPortProtocolVersion(DXL_PROTOCOL_VERSION);

  pinMode(BUZZER_PIN, OUTPUT);

  for (int i = 0; i < 3; i++) {
    initMotor(flex_ids[i]);
    initMotor(extend_ids[i]);
  }

  DEBUG_SERIAL.println("Time(ms),MotorID,Phase,Direction,Position,Angle(deg)");
  buzz(2);
  delay(2000);
}

void loop() {
  for (int cycle = 1; cycle <= NUM_CYCLES; cycle++) {
    DEBUG_SERIAL.print("=== Starting Cycle ");
    DEBUG_SERIAL.print(cycle);
    DEBUG_SERIAL.println(" ===");

    buzz(1);
    runMotorCycleRelativeTicks(flex_ids, flex_deltas, "Flex");
    delay(1000);

    buzz(1);
    runMotorCycleRelativeTicks(extend_ids, extend_deltas, "Extend");
    delay(2000);
  }

  DEBUG_SERIAL.println("=== All cycles completed ===");
  buzz(3);
  while (true);
}

// ========== Motor Initialization ==========
void initMotor(uint8_t id) {
  dxl.torqueOff(id);
  dxl.setOperatingMode(id, OP_EXTENDED_POSITION);
  int32_t pos = dxl.getPresentPosition(id);
  dxl.setGoalPosition(id, pos);
  dxl.torqueOn(id);
}

// ========== Relative Motion ==========
void runMotorCycleRelativeTicks(const uint8_t motor_ids[], const float deltaAngles[], const char* phase) {
  int32_t deltaTicks[3], stepTicks[3], baseTicks[3];

  for (int j = 0; j < 3; j++) {
    deltaTicks[j] = (int32_t)(deltaAngles[j] / 360.0 * 4095);
    stepTicks[j]  = deltaTicks[j] / NUM_STEPS;
    baseTicks[j]  = dxl.getPresentPosition(motor_ids[j]);
  }

  // Forward (Flex/Extend)
  for (int i = 0; i <= NUM_STEPS; i++) {
    for (int j = 0; j < 3; j++) {
      int32_t targetTick = baseTicks[j] + stepTicks[j] * i;
      dxl.setGoalPosition(motor_ids[j], targetTick);
      logMotorStateTicks(motor_ids[j], phase, ">>", targetTick);
    }
    delay(STEP_DELAY);
  }

  // Backward (Return)
  for (int i = NUM_STEPS; i >= 0; i--) {
    for (int j = 0; j < 3; j++) {
      int32_t targetTick = baseTicks[j] + stepTicks[j] * i;
      dxl.setGoalPosition(motor_ids[j], targetTick);
      logMotorStateTicks(motor_ids[j], phase, "<<", targetTick);
    }
    delay(STEP_DELAY);
  }
}

// ========== Serial Logging ==========
void logMotorStateTicks(uint8_t id, const char* phase, const char* dirSymbol, int32_t targetTick) {
  float angle = valueToAngle(dxl.getPresentPosition(id));
  unsigned long t = millis();

  DEBUG_SERIAL.print(t);
  DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(id);
  DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(phase);
  DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(dirSymbol);
  DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(targetTick);
  DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.println(angle, 2);
}

// ========== Tick to Angle ==========
float valueToAngle(int32_t val) {
  return (float)val * 360.0 / 4095.0;
}

// ========== Buzzer ==========
void buzz(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(100);
    digitalWrite(BUZZER_PIN, LOW);
    delay(150);
  }
}
