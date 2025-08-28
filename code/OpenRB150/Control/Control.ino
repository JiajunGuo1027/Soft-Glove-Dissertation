/* =========================================================================
 * OpenRB150 Soft Glove — Stable Closed-loop v2 (step-and-hold, 60*66ms)
 *
 * - Two modes: single finger / three-finger synchronous (non-interleaved)
 * - Trajectory rhythm aligned with old code: Flex(60 steps) → Hold → Extend(60 steps) → Hold, repeated NUM_CYCLES times
 * - Buzzer: 1 beep at the start of each segment; 2 beeps before starting; 3 beeps at the very end
 * - Closed-loop: θ_target(t) -> inverse solve L_target -> dL→ticks increment; low-gain PID + rate limiting + deadband + LPF
 * - Direction reversal: configurable sign for each phase depending on wiring
 * - Only torque is enabled for the "current phase" motor, the other side torque is disabled
 * - Logging: CSV (column names compatible with eval_closedloop.py; evaluation works even without θ_gt)
 * =========================================================================*/


#include <Dynamixel2Arduino.h>

/* ---------------- Board & DXL ---------------- */
#define DXL_SERIAL   Serial1
#define DEBUG_SERIAL Serial
#define DXL_DIR_PIN  -1  
#define BUZZER_PIN   10

const float PROTOCOL_VER = 2.0;
const uint32_t USB_BAUD  = 115200;
const uint32_t DXL_BAUD  = 57600;

/* ---------------- Modes ---------------------- */

// #define MODE_SINGLE   1
#define MODE_MULTI    1

// Single finger selection（0=Thumb(A0,1/4), 1=Index(A1,2/5), 2=Middle(A2,3/6)）
const int SINGLE_FINGER = 1;

/* ---------------- Pattern (like old) --------- */
const int   NUM_STEPS   = 60;     // Each section consists of 60 steps
const float STEP_DELAY  = 66;     // Each step 66ms
const int   NUM_CYCLES  = 5;      
const uint16_t HOLD_MS  = 800;    //  break

/* ---------------- Geometry ------------------- */
const float R_CM          = 1.00f;    // Spool radius cm
const float TICKS_PER_REV = 4096.0f;
const float L_PER_REV     = 6.2831853f * R_CM;

/* ---------------- Gains & Limits ------------- */
float Kp = 0.35f, Ki = 0.01f, Kd = 0.0f;

// Maximum tick change allowed every 66ms (soft velocity limit)
const int   MAX_STEP_TICKS   = 40;  
const int   SOFT_TICK_MIN    = -200000;
const int   SOFT_TICK_MAX    =  200000;

// Deadband (if θ error is smaller, ignore to reduce jitter)
const float PHASE_DEADZONE_DEG = 2.0f;

// Low-pass filter (θ measurement, first-order IIR: y+=α(x-y))
const float LPF_ALPHA_THETA = 0.25f;

/* ---------------- Finger map ----------------- */
// Motor IDs：flex(1/2/3), extend(4/5/6)
const uint8_t FLEX_ID[3]   = {1,2,3};
const uint8_t EXTEND_ID[3] = {4,5,6};
// Sensor analog pins
const uint8_t ADC_PIN [3]  = {A0, A1, A2};

int SIGN_FLEX [3] = { -1, -1, -1 };   // Thumb/Index/Middle movement direction in Flex phase
int SIGN_EXT  [3] = { +1, +1, +1 };   // Extend phase

// Target angles (total joint angle). Thumb=∠ABD，Index/Middle=θ_total(=ABC+BCD)
float THETA_FLEX[3] = {45.f, 45.f, 40.f};
float THETA_EXT [3] = { 5.f,  5.f,  5.f};

/* ---------------- Models (R->L, L->θ) --------
 * Note:
 *  - This runtime keeps the “composed model” simplification: measurement uses R->L + L->θ to obtain θ_meas;
 *  - Inverse θ->L uses phase quadratic polynomial 
 *  - Coefficients fill in
 */
struct RLPhase { float lo, hi, qa, qb, qc; };     // L = qa*z^2 + qb*z + qc, z=(R-lo)/(hi-lo)
struct LTPhase { float a, b, c; };                // θ = aL^2 + bL + c

// Thumb
RLPhase RL_T_Flex = { 392.0f, 631.95f,  -0.2769907f, -2.8749299f, -0.1582192f };
RLPhase RL_T_Ext  = { 292.0f, 490.0f,    1.4815102f,  1.8496427f, -3.4713462f };
LTPhase LT_T_Flex = { -3.1875943f, -3.7138699f,  52.790895f };
LTPhase LT_T_Ext  = {  3.8955807f,  9.9750736f,  -0.7871132f };
// Index
RLPhase RL_I_Flex = { 312.0f, 575.19f,   2.6447835f, -6.7967072f, -1.1273738f };
RLPhase RL_I_Ext  = { 263.28f, 409.0f,  -0.2458797f,  3.3552034f, -3.5165549f };
LTPhase LT_I_Flex = { -6.0396793f, -41.807521f, 33.224357f };
LTPhase LT_I_Ext  = { 12.7013474f,  63.726495f, 78.346393f };
// Middle(θ only as proxy)
RLPhase RL_M_Flex = { 440.0f, 779.73f,   2.2007213f, -6.8444605f, -0.8640967f };
RLPhase RL_M_Ext  = { 358.0f, 566.65f,  -0.3096960f,  3.6076787f, -3.5262206f };
LTPhase LT_M_Flex = { 0.f, 1.f, 0.f };
LTPhase LT_M_Ext  = { 0.f, 1.f, 0.f };

// affine (according to active-phase suggested values; not used for Middle)
float AFFINE_A[3] = { 0.9906f, 0.9877f, 1.0f };
float AFFINE_B[3] = { 0.3030f, 0.6574f, 0.0f };

/* ---------------- Runtime -------------------- */
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

enum Phase { Flex=0, Extend=1 };
struct LoopState {
  // State of each finger
  float integ[3]; float prev_err[3]; float theta_lpf[3];
  int   present_tick[3]; int goal_tick[3];
  // Trajectory counters
  int   step_idx; Phase phase;
  unsigned long seg_start_ms;
  int   cycle_idx;
} S;

/* ---------------- Utils ---------------------- */
inline void buzz(int n){
  for(int i=0;i<n;i++){ digitalWrite(BUZZER_PIN, HIGH); delay(100); digitalWrite(BUZZER_PIN, LOW); delay(150); }
}
inline float clampf(float v,float lo,float hi){ return v<lo?lo:(v>hi?hi:v); }
inline int   clampi(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }

inline float z_from_raw(int raw, const RLPhase& ph){
  float den = (ph.hi>ph.lo)?(ph.hi-ph.lo):1e-6f;
  return (raw - ph.lo)/den;
}
inline float R_to_L_cm(int raw, const RLPhase& ph){
  float z = z_from_raw(raw, ph);
  return ph.qa*z*z + ph.qb*z + ph.qc;
}
inline float L_to_theta(float L, const LTPhase& ph){ return ph.a*L*L + ph.b*L + ph.c; }
inline float dtheta_dL (float L, const LTPhase& ph){ return 2.f*ph.a*L + ph.b; }

//  Quadratic inverse θ->L (choose root close to previous target)
bool invert_theta_to_L(float theta_deg, const LTPhase& ph, float L_hint, float& L_sol){
  float A=ph.a, B=ph.b, C=ph.c - theta_deg;
  if (fabs(A) < 1e-7f){ if (fabs(B)<1e-7f) return false; L_sol = -C/B; return true; }
  float D = B*B - 4*A*C; if (D<0) return false;
  float sD = sqrtf(D);
  float L1 = (-B + sD)/(2*A), L2 = (-B - sD)/(2*A);
  L_sol = (fabs(L1-L_hint)<=fabs(L2-L_hint))?L1:L2;
  return true;
}

void torque_only(uint8_t on_id, uint8_t off_id){
  dxl.torqueOff(off_id);
  dxl.torqueOn(on_id);
}

/* ---------------- Selection helpers ---------- */
inline const RLPhase& RL_of(int f, Phase ph){
  if (f==0) return (ph==Flex)?RL_T_Flex:RL_T_Ext;
  if (f==1) return (ph==Flex)?RL_I_Flex:RL_I_Ext;
  return           (ph==Flex)?RL_M_Flex:RL_M_Ext;
}
inline const LTPhase& LT_of(int f, Phase ph){
  if (f==0) return (ph==Flex)?LT_T_Flex:LT_T_Ext;
  if (f==1) return (ph==Flex)?LT_I_Flex:LT_I_Ext;
  return           (ph==Flex)?LT_M_Flex:LT_M_Ext;
}

/* ---------------- Logging header -------------- */
void print_header(){
  DEBUG_SERIAL.println("Time(ms),Finger,Phase,MotorID,SensorRaw,L_meas_cm,Theta_meas_deg,Theta_target_deg,L_target_cm,GoalTick,PresentTick,Err_theta_deg");
}

/* ---------------- Init ------------------------ */
void initMotor(uint8_t id){
  dxl.torqueOff(id);
  dxl.setOperatingMode(id, OP_EXTENDED_POSITION);
  int32_t pos = dxl.getPresentPosition(id, UNIT_RAW);
  dxl.setGoalPosition(id, pos, UNIT_RAW);
  dxl.torqueOn(id);
}

void setup(){
  delay(2000);
  pinMode(BUZZER_PIN, OUTPUT);

  DEBUG_SERIAL.begin(USB_BAUD);
  while(!DEBUG_SERIAL){}

  DXL_SERIAL.begin(DXL_BAUD);
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(PROTOCOL_VER);

  // init all 6
  for(int i=0;i<3;i++){ initMotor(FLEX_ID[i]); initMotor(EXTEND_ID[i]); }

  memset(&S, 0, sizeof(S));
  S.phase = Flex; S.step_idx = 0; S.cycle_idx = 1;
  S.seg_start_ms = millis();
  for(int f=0; f<3; ++f){
    S.present_tick[f] = dxl.getPresentPosition(FLEX_ID[f], UNIT_RAW); 
    S.goal_tick[f]    = S.present_tick[f];
    S.theta_lpf[f]    = 0.f;
  }

  buzz(2); delay(1000);
  print_header();
}

/* ---------------- Single step (one finger) ---- */
void one_finger_step(int f, float theta_tar, Phase ph){
  const RLPhase& RL = RL_of(f, ph);
  const LTPhase& LT = LT_of(f, ph);
  const uint8_t  mid = (ph==Flex)?FLEX_ID[f]:EXTEND_ID[f];
  const uint8_t  mid_other = (ph==Flex)?EXTEND_ID[f]:FLEX_ID[f];
  const int      sgn = (ph==Flex)?SIGN_FLEX[f]:SIGN_EXT[f];

// Only torque to current phase
  torque_only(mid, mid_other);

  // Read sensor → L_meas → θ_meas (with affine & LPF)
  int   raw    = analogRead(ADC_PIN[f]);
  float L_meas = R_to_L_cm(raw, RL);
  float th_raw = L_to_theta(L_meas, LT);
  float th_est = AFFINE_A[f]*th_raw + AFFINE_B[f];
  S.theta_lpf[f] += LPF_ALPHA_THETA * (th_est - S.theta_lpf[f]);

  // Inverse θ_target → L_target (use previous L_meas as hint)
  static float L_hint[3] = {0,0,0};
  float L_tar = L_hint[f];
  bool ok = invert_theta_to_L(theta_tar, LT, L_hint[f], L_tar);
  if (!ok) L_tar = L_hint[f];

  // Error (θ domain) + PID 
  float err = theta_tar - S.theta_lpf[f];
 // Deadband
  if (fabs(err) < PHASE_DEADZONE_DEG) { err = 0.f; }

  float dt = STEP_DELAY/1000.0f;
  S.integ[f]    += err * dt;
  float deriv    = (err - S.prev_err[f]) / dt;
  S.prev_err[f]  = err;

  // Use dθ/dL for unit conversion (avoid jitter)
  float slope = dtheta_dL(L_meas, LT); if (fabs(slope) < 1e-3f) slope = (slope>=0?1e-3f:-1e-3f);
  float dL_pid = (Kp*err + Ki*S.integ[f] + Kd*deriv) / slope;

 // Target displacement (adjust around L_target)
  float L_cmd = L_tar + dL_pid;

 // ΔL → Δticks (apply sign)
  float dL = L_cmd - L_meas;
  int   delta_tick = (int)( sgn * (dL / L_PER_REV) * TICKS_PER_REV );

 // Single-step clamp + soft limits
  delta_tick           = clampi(delta_tick, -MAX_STEP_TICKS, MAX_STEP_TICKS);
  S.present_tick[f]    = dxl.getPresentPosition(mid, UNIT_RAW);
  S.goal_tick[f]       = clampi(S.present_tick[f] + delta_tick, SOFT_TICK_MIN, SOFT_TICK_MAX);
  dxl.setGoalPosition(mid, S.goal_tick[f], UNIT_RAW);

  L_hint[f] = L_tar;


  unsigned long now = millis();
  DEBUG_SERIAL.print(now);DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print((f==0)?"Thumb":(f==1)?"Index":"Middle");DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print((ph==Flex)?"Flex":"Extend");DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(mid);DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(raw);DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(L_meas,6);DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(S.theta_lpf[f],6);DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(theta_tar,6);DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(L_tar,6);DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(S.goal_tick[f]);DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.print(S.present_tick[f]);DEBUG_SERIAL.print(",");
  DEBUG_SERIAL.println(theta_tar - S.theta_lpf[f],6);
}

/* ---------------- Runner --------------------- */
void run_single_finger_cycle(){
  const int f = SINGLE_FINGER;
  // Flex phase
  buzz(1);
  for(int i=0;i<=NUM_STEPS;i++){
    float theta_tar = THETA_EXT[f] + (THETA_FLEX[f]-THETA_EXT[f]) * (float)i/NUM_STEPS;
    one_finger_step(f, theta_tar, Flex);
    delay((uint32_t)STEP_DELAY);
  }
  delay(HOLD_MS);

  // Extend phase
  buzz(1);
  for(int i=0;i<=NUM_STEPS;i++){
    float theta_tar = THETA_FLEX[f] + (THETA_EXT[f]-THETA_FLEX[f]) * (float)i/NUM_STEPS;
    one_finger_step(f, theta_tar, Extend);
    delay((uint32_t)STEP_DELAY);
  }
  delay(HOLD_MS);
}

void run_multi_fingers_cycle(){
  // Flex phase: three fingers synchronous
  buzz(1);
  for(int i=0;i<=NUM_STEPS;i++){
    for(int f=0; f<3; ++f){
      float theta_tar = THETA_EXT[f] + (THETA_FLEX[f]-THETA_EXT[f]) * (float)i/NUM_STEPS;
      one_finger_step(f, theta_tar, Flex);
    }
    delay((uint32_t)STEP_DELAY);
  }
  delay(HOLD_MS);

  // Extend phase: three fingers synchronous
  buzz(1);
  for(int i=0;i<=NUM_STEPS;i++){
    for(int f=0; f<3; ++f){
      float theta_tar = THETA_FLEX[f] + (THETA_EXT[f]-THETA_FLEX[f]) * (float)i/NUM_STEPS;
      one_finger_step(f, theta_tar, Extend);
    }
    delay((uint32_t)STEP_DELAY);
  }
  delay(HOLD_MS);
}

/* ---------------- Loop ----------------------- */
void loop(){
#if defined(MODE_SINGLE)
  for(int c=1; c<=NUM_CYCLES; ++c){
    run_single_finger_cycle();
  }
#elif defined(MODE_MULTI)
  for(int c=1; c<=NUM_CYCLES; ++c){
    run_multi_fingers_cycle();
  }
#endif

  buzz(3);
  while(1){ delay(1000); }
}
