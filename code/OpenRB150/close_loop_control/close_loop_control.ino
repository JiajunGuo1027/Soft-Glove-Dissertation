/* OpenRB-150 + Dynamixel XL330 + Flex sensors
 * - Index/Thumb: θ-domain closed-loop (R->L->θ feedback, θ_target->L_target feedforward)
 * - Middle: L-domain closed-loop (R->L feedback)
 * - Phase beeps on transitions, 20 ms loop, CSV streaming log
 *
 */

#include <Dynamixel2Arduino.h>

// ---------- Board / Ports ----------
#define USB_BAUD     115200
#define DXL_BAUD     1000000

#define DXL_PORT     Serial3
#define DXL_DIR_PIN  -1  

// ---------- Pins ----------
#define BUZZER_PIN   10
#define PIN_A0       A0   // Thumb
#define PIN_A1       A1   // Index
#define PIN_A2       A2   // Middle

// ---------- Timing ----------
#define LOOP_DT_MS   20UL
#define BEEP_MS      120

// ---------- Dynamixel ----------
Dynamixel2Arduino dxl(DXL_PORT, DXL_DIR_PIN);
const float TICKS_PER_RAD = 4096.0f / (2.0f * 3.1415926535f);
const uint8_t ID_T_FLEX = 1, ID_I_FLEX = 2, ID_M_FLEX = 3;
const uint8_t ID_T_EXT  = 4, ID_I_EXT  = 5, ID_M_EXT  = 6;

int8_t SIGN_T_FLEX = +1, SIGN_I_FLEX = +1, SIGN_M_FLEX = +1;
int8_t SIGN_T_EXT  = -1, SIGN_I_EXT  = -1, SIGN_M_EXT  = -1;

// Home offsets (ticks) captured at start
int32_t HOME_T_FLEX=0, HOME_I_FLEX=0, HOME_M_FLEX=0;
int32_t HOME_T_EXT =0, HOME_I_EXT =0, HOME_M_EXT =0;

// ---------- Model structs ----------
struct RLPhase {
  float a, b, c;       // L = a*z^2 + b*z + c  (cm);  z = (R - lo)/(hi - lo)
  float lo, hi;        // sensor normalization bounds
  float Lmin, Lmax;    // safe L range (cm)
};

struct LThetaPhase {
  float a, b, c;       // theta = a*L^2 + b*L + c  (deg)
  float Tmin, Tmax;    // safe theta range (deg)
};

struct FingerModel {
  // R->L per phase
  RLPhase flex_rl, ext_rl;
  // L->theta per phase (index/thumb); for middle no need but keep for proxy
  LThetaPhase flex_lt, ext_lt;
  // Affine theta correction (optional, default 1,0)
  float aff_a, aff_b;
  // Spool radius (cm)
  float spool_r_cm;
};


struct MotorTargets { int32_t flex_ticks; int32_t ext_ticks; };


// Global models for three fingers
FingerModel F_IDX, F_THU, F_MID;

// ---------- Control / State ----------
enum Phase { PH_FLEX=0, PH_EXT=1 };
struct FingerState {
  Phase phase;
  float t_in_phase;       // s
  float period;           // s, full cycle = 2*period for triangular profile
  float theta_lo, theta_hi;  // deg for index/thumb
  float L_lo_cm, L_hi_cm;    // cm for middle
  // PID (θ-domain for index/thumb; L-domain for middle)
  float Kp, Ki, Kd;
  float integ;
  float prev_err;
};

FingerState S_IDX, S_THU, S_MID;

// trajectory type: 0=triangle, 1=sine, 2=step
uint8_t TRAJ_TYPE = 0;
uint16_t CYCLES_TO_RUN = 6;
uint16_t cycles_done = 0;

// ---------- Helpers ----------
float clampf(float v, float lo, float hi){ if(v<lo) return lo; if(v>hi) return hi; return v; }
float radiansf(float deg){ return deg*3.1415926535f/180.0f; }
float degreesf(float rad){ return rad*180.0f/3.1415926535f; }

float z_norm(float R, float lo, float hi){
  float den = (hi>lo)? (hi-lo) : 1e-9f;
  return (R - lo)/den;
}

float sensor_to_L(const RLPhase& ph, float Rraw){
  float z = z_norm(Rraw, ph.lo, ph.hi);
  float L = ph.a*z*z + ph.b*z + ph.c;
  return clampf(L, ph.Lmin, ph.Lmax);
}

float L_to_theta(const LThetaPhase& ph, float L){
  float th = ph.a*L*L + ph.b*L + ph.c;
  return clampf(th, ph.Tmin, ph.Tmax);
}

// Invert theta = a*L^2 + b*L + c  ->  L 
bool theta_to_L(const LThetaPhase& ph, float theta, float lastL, float* Lout){
  float A = ph.a, B = ph.b, C = ph.c - theta;
  float disc = B*B - 4*A*C;
  if(disc < 0) return false;
  float sqrtD = sqrtf(disc);
  float L1 = (-B + sqrtD)/(2*A);
  float L2 = (-B - sqrtD)/(2*A);
  // pick root closest to lastL
  float Lpick = (fabsf(L1 - lastL) < fabsf(L2 - lastL)) ? L1 : L2;
  if(Lpick < -1000 || Lpick > 1000) return false;
  *Lout = clampf(Lpick, ph.Tmin, ph.Tmax); 
  return true;
}

// safer clamp for L using model ranges
float clampL(float L, const RLPhase& rl){ return clampf(L, rl.Lmin, rl.Lmax); }

// θ estimate from sensor: θ = aff( L(R) -> θ(L) )
float theta_from_R(const FingerModel& M, Phase ph, float Rraw){
  const RLPhase& rlp = (ph==PH_FLEX)? M.flex_rl : M.ext_rl;
  const LThetaPhase& ltp = (ph==PH_FLEX)? M.flex_lt : M.ext_lt;
  float L = sensor_to_L(rlp, Rraw);
  float th = L_to_theta(ltp, L);
  return M.aff_a*th + M.aff_b;
}

// move DXL to absolute ticks with simple saturation of goal velocity via Profile params
void set_goal_ticks(uint8_t id, int32_t ticks){
  dxl.setGoalPosition(id, (float)ticks);
}

int32_t read_present_ticks(uint8_t id){
  return (int32_t)dxl.getPresentPosition(id);
}

// Simple beep
void beep(uint16_t dur_ms=BEEP_MS){
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, HIGH); delay(dur_ms);
  digitalWrite(BUZZER_PIN, LOW);
}



String sline;

FingerModel* selFinger(const String& tag){
  if(tag=="IDX") return &F_IDX;
  if(tag=="THU") return &F_THU;
  if(tag=="MID") return &F_MID;
  return nullptr;
}

void applyRL(FingerModel* F, bool isFlex, float a,float b,float c,float lo,float hi,float Lmin,float Lmax){
  RLPhase& ph = isFlex? F->flex_rl : F->ext_rl;
  ph.a=a; ph.b=b; ph.c=c; ph.lo=lo; ph.hi=hi; ph.Lmin=Lmin; ph.Lmax=Lmax;
}
void applyLTH(FingerModel* F, bool isFlex, float a,float b,float c,float Tmin,float Tmax){
  LThetaPhase& ph = isFlex? F->flex_lt : F->ext_lt;
  ph.a=a; ph.b=b; ph.c=c; ph.Tmin=Tmin; ph.Tmax=Tmax;
}

void process_line(const String& line){

  const int MAXTOK=16;
  String tok[MAXTOK]; int n=0;
  int start=0;
  for(int i=0;i<line.length() && n<MAXTOK;i++){
    if(line[i]==','){ tok[n++] = line.substring(start,i); start=i+1; }
    else if(i==line.length()-1){ tok[n++] = line.substring(start,i+1); }
  }
  if(n==0) return;
  if(tok[0]!="CFG") return;

  if(n>=3 && tok[2]=="RUN"){
    CYCLES_TO_RUN = (uint16_t)tok[3].toInt();
    cycles_done = 0;
    // beep twice to mark start
    beep(80); delay(80); beep(80);
    return;
  }

  if(n<4) return;
  FingerModel* F = selFinger(tok[1]);
  String type = tok[3];

  if(type=="RLQ" && n>=12){
    bool isFlex = (tok[2]=="FLEX");
    applyRL(F,isFlex, tok[4].toFloat(),tok[5].toFloat(),tok[6].toFloat(),
                 tok[7].toFloat(),tok[8].toFloat(),tok[9].toFloat(),tok[10].toFloat());
  } else if(type=="LTHQ" && n>=11){
    bool isFlex = (tok[2]=="FLEX");
    applyLTH(F,isFlex, tok[4].toFloat(),tok[5].toFloat(),tok[6].toFloat(),
                    tok[7].toFloat(),tok[8].toFloat());
  } else if(type=="AFF" && n>=6){
    F->aff_a = tok[4].toFloat(); F->aff_b = tok[5].toFloat();
  } else if(type=="SPOOL" && n>=5){
    F->spool_r_cm = tok[4].toFloat();
  } else if(type=="PID" && n>=7){
    FingerState* S = (tok[1]=="IDX")? &S_IDX : (tok[1]=="THU")? &S_THU : &S_MID;
    S->Kp = tok[4].toFloat(); S->Ki = tok[5].toFloat(); S->Kd = tok[6].toFloat();
  } else if(type=="TRAJ" && n>=9){
    FingerState* S = (tok[1]=="IDX")? &S_IDX : &S_THU;
    TRAJ_TYPE = (uint8_t)tok[4].toInt();
    S->period   = tok[5].toFloat();
    S->theta_lo = tok[6].toFloat();
    S->theta_hi = tok[7].toFloat();
    S->t_in_phase = 0; S->phase = PH_FLEX; S->integ=0; S->prev_err=0;
  } else if(type=="TRAJ_L" && n>=9){
    FingerState* S = &S_MID;
    TRAJ_TYPE = (uint8_t)tok[4].toInt();
    S->period   = tok[5].toFloat();
    S->L_lo_cm  = tok[6].toFloat();
    S->L_hi_cm  = tok[7].toFloat();
    S->t_in_phase = 0; S->phase = PH_FLEX; S->integ=0; S->prev_err=0;
  } else if(type=="SIGN" && n>=7){
    int8_t fs = (int8_t)tok[4].toInt();
    int8_t es = (int8_t)tok[5].toInt();
    if(tok[1]=="IDX"){ SIGN_I_FLEX=fs; SIGN_I_EXT=es; }
    else if(tok[1]=="THU"){ SIGN_T_FLEX=fs; SIGN_T_EXT=es; }
    else { SIGN_M_FLEX=fs; SIGN_M_EXT=es; }
  }
}

// ---------- Setup ----------
void setup(){
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.begin(USB_BAUD);

  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(2.0);

  uint8_t ids[6] = {ID_T_FLEX, ID_I_FLEX, ID_M_FLEX, ID_T_EXT, ID_I_EXT, ID_M_EXT};

  for (int i = 0; i < 6; i++) {
    uint8_t cur = ids[i];
    dxl.torqueOff(cur);
    dxl.setOperatingMode(cur, OP_POSITION);
    dxl.writeControlTableItem(ControlTableItem::PROFILE_VELOCITY,     cur, 90);
    dxl.writeControlTableItem(ControlTableItem::PROFILE_ACCELERATION, cur,  20);
    dxl.torqueOn(cur);
  }

  // capture homes
  HOME_T_FLEX = read_present_ticks(ID_T_FLEX);
  HOME_I_FLEX = read_present_ticks(ID_I_FLEX);
  HOME_M_FLEX = read_present_ticks(ID_M_FLEX);
  HOME_T_EXT  = read_present_ticks(ID_T_EXT);
  HOME_I_EXT  = read_present_ticks(ID_I_EXT);
  HOME_M_EXT  = read_present_ticks(ID_M_EXT);

  // defaults
  F_IDX.aff_a=1; F_IDX.aff_b=0; F_THU.aff_a=1; F_THU.aff_b=0; F_MID.aff_a=1; F_MID.aff_b=0;
  F_IDX.spool_r_cm=1.0f; F_THU.spool_r_cm=1.0f; F_MID.spool_r_cm=1.0f;
  S_IDX.period=1.5f; S_THU.period=1.5f; S_MID.period=1.5f;
  S_IDX.theta_lo=0; S_IDX.theta_hi=45;
  S_THU.theta_lo=0; S_THU.theta_hi=40;
  S_MID.L_lo_cm=0;  S_MID.L_hi_cm=1.0f;
  S_IDX.Kp=0.8f; S_IDX.Ki=0.0f; S_IDX.Kd=0.02f;
  S_THU.Kp=0.8f; S_THU.Ki=0.0f; S_THU.Kd=0.02f;
  S_MID.Kp=0.6f; S_MID.Ki=0.0f; S_MID.Kd=0.02f;

  // Intro beep
  beep(80); delay(80); beep(80);
}

// ---------- Trajectory generators ----------
float tri_u01(float t, float T){ // triangle 0..1 with period T
  float x = fmodf(t, T) / T;
  return (x<0.5f)? (2*x) : (2*(1-x));
}
float sin_u01(float t, float T){
  return 0.5f*(1.0f - cosf(2.0f*3.1415926535f*(t/T)));
}
float step_u01(float t, float T){
  float x = fmodf(t, T) / T;
  return (x<0.5f)? 0.0f : 1.0f;
}
float u01_by_type(float t, float T){
  if(TRAJ_TYPE==1) return sin_u01(t,T);
  if(TRAJ_TYPE==2) return step_u01(t,T);
  return tri_u01(t,T);
}

// ---------- Control core (per finger) ----------
// struct MotorTargets { int32_t flex_ticks; int32_t ext_ticks; };
MotorTargets compute_targets_idxthumb(const FingerModel& M, FingerState& S, uint8_t id_flex, uint8_t id_ext, int32_t home_flex, int32_t home_ext, int8_t sign_flex, int8_t sign_ext, float Rraw){
  // trajectory
  float u = u01_by_type(S.t_in_phase, S.period); // 0..1
  float theta_tgt = S.theta_lo + (S.theta_hi - S.theta_lo)*u; // deg

  // determine current phase by u slope (triangle/step) or time halves
  Phase ph = (u<0.5f)? PH_FLEX : PH_EXT;
  if(ph != S.phase){ S.phase = ph; S.integ=0; S.prev_err=0; beep(); }

  // feedforward: theta -> L
  const RLPhase& rlp = (ph==PH_FLEX)? M.flex_rl : M.ext_rl;
  const LThetaPhase& ltp = (ph==PH_FLEX)? M.flex_lt : M.ext_lt;
  static float lastL=0;
  float L_ff = 0;
  // invert θ(L) robustly
  {
    // quadratic inversion with trend: use previous L
    float A=ltp.a, B=ltp.b, C=ltp.c - theta_tgt;
    float disc=B*B-4*A*C;
    if(disc>0){
      float sqrtD=sqrtf(disc);
      float L1=(-B+sqrtD)/(2*A);
      float L2=(-B-sqrtD)/(2*A);
      L_ff = (fabsf(L1-lastL) < fabsf(L2-lastL))? L1 : L2;
    }
    lastL = L_ff;
  }
  L_ff = clampL(L_ff, rlp);

  // feedback: sensor -> theta_meas
  float theta_meas = theta_from_R(M, ph, Rraw);
  float err = theta_tgt - theta_meas;

  // PID
  S.integ += err * (LOOP_DT_MS/1000.0f);
  float deriv = (err - S.prev_err) / (LOOP_DT_MS/1000.0f);
  S.prev_err = err;
  float u_corr = S.Kp*err + S.Ki*S.integ + S.Kd*deriv;

  // convert: θ correction -> L correction using local slope dθ/dL ≈ b + 2a L_ff
  float dtheta_dL = ltp.b + 2.0f*ltp.a*L_ff;
  if(fabsf(dtheta_dL) < 1e-3f) dtheta_dL = (dtheta_dL<0)? -1e-3f:1e-3f;
  float L_corr = u_corr / dtheta_dL;

  float L_cmd = clampL(L_ff + L_corr, rlp);

  // convert L to ticks for active motor; inactive holds home
  float rad = (L_cmd / M.spool_r_cm); // since L(cm)=r(cm)*phi(rad)
  int32_t delta_ticks = (int32_t)(rad * TICKS_PER_RAD);
  MotorTargets mt;
  if(ph==PH_FLEX){
    mt.flex_ticks = home_flex + sign_flex * delta_ticks;
    mt.ext_ticks  = home_ext; 
  }else{
    mt.flex_ticks = home_flex;
    mt.ext_ticks  = home_ext + sign_ext * delta_ticks;
  }

  // --- CSV log ---
  static uint32_t t0 = millis();
  uint32_t t = millis() - t0;
  Serial.print("LOG,IDXTH,"); // common for index/thumb
  Serial.print((ph==PH_FLEX)?"FLEX":"EXT"); Serial.print(',');
  Serial.print(t); Serial.print(',');
  Serial.print(theta_tgt,3); Serial.print(',');   // theta_target
  Serial.print(theta_meas,3); Serial.print(',');  // theta_meas
  Serial.print(Rraw,1);        Serial.print(','); // sensor raw
  Serial.print(L_ff,3);        Serial.print(','); // L_ff
  Serial.print(L_cmd,3);       Serial.print(','); // L_cmd
  Serial.print(mt.flex_ticks); Serial.print(',');
  Serial.print(mt.ext_ticks);  Serial.println();
  return mt;
}

MotorTargets compute_targets_middle(const FingerModel& M, FingerState& S, uint8_t id_flex, uint8_t id_ext, int32_t home_flex, int32_t home_ext, int8_t sign_flex, int8_t sign_ext, float Rraw){
  float u = u01_by_type(S.t_in_phase, S.period);
  Phase ph = (u<0.5f)? PH_FLEX : PH_EXT;
  if(ph != S.phase){ S.phase = ph; S.integ=0; S.prev_err=0; beep(); }

  const RLPhase& rlp = (ph==PH_FLEX)? M.flex_rl : M.ext_rl;

  float L_tgt = S.L_lo_cm + (S.L_hi_cm - S.L_lo_cm)*u;  // cm
  L_tgt = clampL(L_tgt, rlp);

  // feedback: sensor -> L_meas
  float L_meas = sensor_to_L(rlp, Rraw);
  float err = L_tgt - L_meas;

  // PID in L-domain
  S.integ += err * (LOOP_DT_MS/1000.0f);
  float deriv = (err - S.prev_err) / (LOOP_DT_MS/1000.0f);
  S.prev_err = err;
  float L_cmd = clampL(L_tgt + (S.Kp*err + S.Ki*S.integ + S.Kd*deriv), rlp);

  float rad = (L_cmd / M.spool_r_cm);
  int32_t delta_ticks = (int32_t)(rad * TICKS_PER_RAD);
  MotorTargets mt;
  if(ph==PH_FLEX){
    mt.flex_ticks = home_flex + sign_flex * delta_ticks;
    mt.ext_ticks  = home_ext;
  }else{
    mt.flex_ticks = home_flex;
    mt.ext_ticks  = home_ext + sign_ext * delta_ticks;
  }

  // CSV log
  static uint32_t t0 = millis();
  uint32_t t = millis() - t0;
  Serial.print("LOG,MID,");
  Serial.print((ph==PH_FLEX)?"FLEX":"EXT"); Serial.print(',');
  Serial.print(t); Serial.print(',');
  Serial.print(L_tgt,3);  Serial.print(',');   // L_target
  Serial.print(L_meas,3); Serial.print(',');   // L_meas
  Serial.print(Rraw,1);   Serial.print(',');   // sensor raw
  Serial.print(mt.flex_ticks); Serial.print(',');
  Serial.print(mt.ext_ticks);  Serial.println();
  return mt;
}

// ---------- Main loop ----------
uint32_t last_ms = 0;

void loop(){
  // serial receive (non-blocking)
  while(Serial.available()){
    char c = Serial.read();
    if(c=='\r') continue;
    if(c=='\n'){
      process_line(sline);
      sline = "";
    }else{
      sline += c;
      if(sline.length()>180) sline="";
    }
  }

  uint32_t now = millis();
  if(now - last_ms < LOOP_DT_MS) return;
  last_ms = now;
  S_IDX.t_in_phase += (LOOP_DT_MS/1000.0f);
  S_THU.t_in_phase += (LOOP_DT_MS/1000.0f);
  S_MID.t_in_phase += (LOOP_DT_MS/1000.0f);

  // read sensors (raw ADC)
  float R_thu = analogRead(PIN_A0);   // A0 thumb
  float R_idx = analogRead(PIN_A1);   // A1 index
  float R_mid = analogRead(PIN_A2);   // A2 middle

  // compute targets
  MotorTargets mt_idx = compute_targets_idxthumb(F_IDX, S_IDX, ID_I_FLEX, ID_I_EXT, HOME_I_FLEX, HOME_I_EXT, SIGN_I_FLEX, SIGN_I_EXT, R_idx);
  MotorTargets mt_thu = compute_targets_idxthumb(F_THU, S_THU, ID_T_FLEX, ID_T_EXT, HOME_T_FLEX, HOME_T_EXT, SIGN_T_FLEX, SIGN_T_EXT, R_thu);
  MotorTargets mt_mid = compute_targets_middle (F_MID, S_MID, ID_M_FLEX, ID_M_EXT, HOME_M_FLEX, HOME_M_EXT, SIGN_M_FLEX, SIGN_M_EXT, R_mid);

  // send to motors
  set_goal_ticks(ID_I_FLEX, mt_idx.flex_ticks);
  set_goal_ticks(ID_I_EXT,  mt_idx.ext_ticks);
  set_goal_ticks(ID_T_FLEX, mt_thu.flex_ticks);
  set_goal_ticks(ID_T_EXT,  mt_thu.ext_ticks);
  set_goal_ticks(ID_M_FLEX, mt_mid.flex_ticks);
  set_goal_ticks(ID_M_EXT,  mt_mid.ext_ticks);
}
