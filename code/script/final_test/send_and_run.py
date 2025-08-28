# send_and_run.py
# - Read existing model JSON
# - Send to OpenRB via serial using the "CFG,..." protocol
# - Configure trajectory / PID / period / amplitude / sign direction and spool radius
# - Trigger run and save board-side log as CSV


import os, json, time, serial
from pathlib import Path

# Configuration
COM_PORT = "COM3"
BAUD     = 115200
OUT_LOG  = r"D:\Soft_glove\control_results\run_logs\log_run.csv"

BASE = Path(r"D:\Soft_glove\models")

# Index
IDX_RL  = BASE/"index"/"R_L"/"R_L_model_index.json"
IDX_LT  = BASE/"index"/"L_theta"/"L_theta_model_index.json"
IDX_RT  = BASE/"index"/"R_theta_compose_active"/"R_theta_composed_model_ACTIVE_index.json"

# Thumb
THU_RL  = BASE/"thumb"/"R_L"/"R_L_model_thumb.json"
THU_LT  = BASE/"thumb"/"L_theta"/"L_theta_model_thumb.json"
THU_RT  = BASE/"thumb"/"R_theta_compose_active"/"R_theta_composed_model_ACTIVE_thumb.json"

# Middle
MID_RL  = BASE/"middle"/"R_L"/"R_L_model_middle.json"

SPOOL_R = 1.0

TRAJ_TYPE = 0
PERIOD_S  = 1.5
CYCLES    = 8

IDX_THETA = (0.0, 45.0)
THU_THETA = (0.0, 40.0)
MID_L     = (0.0, 1.0)

PID_IDX = (0.8, 0.0, 0.02)
PID_THU = (0.8, 0.0, 0.02)
PID_MID = (0.6, 0.0, 0.02)

SIGN_IDX = (+1, -1)
SIGN_THU = (+1, -1)
SIGN_MID = (+1, -1)
# =================

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def send_line(ser, line, *, wait_echo=True):
    """Send one line, optionally wait for board echo/ACK; always use CRLF, flush after writing."""
    msg = (line + "\r\n").encode("utf-8")
    ser.write(msg)
    ser.flush()
    time.sleep(0.02)  # 节流，避免淹没固件
    if wait_echo:
        t0 = time.time()
        while time.time() - t0 < 0.6:  # allow 0.6s waiting
            got = ser.readline().decode("utf-8", "ignore").strip()
            if got:
                print("<<", got)  # print to confirm
                # If firmware has "ACK" or "OK" you can detect and return early
                if got.startswith("ACK") or got.startswith("OK"):
                    return
                # Some firmwares echo the command, also treat it as received
                if got.find(line.split(",")[0]) >= 0:
                    return
        # Timeout: still continue (some firmware may not echo), but at least log it
        print(".. no echo for:", line)

def push_RL(ser, tag, rlj):
    cnt = 0
    for ph in ["Flex", "Extend"]:
        phj = rlj["phases"][ph]
        a = phj["quadratic"]["a"]; b = phj["quadratic"]["b"]; c = phj["quadratic"]["c"]
        lo = phj["sensor_norm_lo"]; hi = phj["sensor_norm_hi"]
        Lmin = phj.get("L_range", {}).get("min", -2.0)
        Lmax = phj.get("L_range", {}).get("max", +2.0)
        send_line(ser, f"CFG,{tag},{ph.upper()},RLQ,{a},{b},{c},{lo},{hi},{Lmin},{Lmax}")
        cnt += 1
        if cnt % 12 == 0:
            time.sleep(0.1)

def push_LTH(ser, tag, ltj, ycol="θ_total"):
    cnt = 0
    for ph in ["Flex", "Extend"]:
        pj = ltj["phases"].get(ph, {})
        tg_all = pj.get("targets", {})
        tg = tg_all.get(ycol)
        if tg is None and tg_all:
            tg = tg_all[list(tg_all.keys())[0]]
        if tg is None:
            continue
        q = tg.get("quadratic")
        if q:
            a, b, c = q["a"], q["b"], q["c"]
        else:
            lin = tg.get("linear", {"a":1.0, "b":0.0})
            a, b, c = 0.0, lin["a"], lin["b"]
        Tmin = tg.get("theta_range", {}).get("min", -10.0)
        Tmax = tg.get("theta_range", {}).get("max",  90.0)
        send_line(ser, f"CFG,{tag},{ph.upper()},LTHQ,{a},{b},{c},{Tmin},{Tmax}")
        cnt += 1
        if cnt % 12 == 0:
            time.sleep(0.1)

def push_AFF(ser, tag, rtj):
    aff = rtj.get("affine", {"a":1.0, "b":0.0})
    send_line(ser, f"CFG,{tag},FLEX,AFF,{aff.get('a',1.0)},{aff.get('b',0.0)}")
    send_line(ser, f"CFG,{tag},EXT,AFF,{aff.get('a',1.0)},{aff.get('b',0.0)}")

def handshake(ser):
    """Reset → wait for READY/any start line; if nothing received, try sending PING."""
    try:
        ser.setDTR(False); ser.setRTS(False)
        time.sleep(0.05)
        ser.setDTR(True); ser.setRTS(True)
    except Exception:
        pass
    ser.reset_input_buffer()
    t0 = time.time()
    print(".. waiting for board (READY/hello)...")
    while time.time() - t0 < 2.5:
        line = ser.readline().decode("utf-8", "ignore").strip()
        if line:
            print("<<", line)
            if "READY" in line.upper():
                break
    send_line(ser, "PING", wait_echo=True)

def main():
    for p in [IDX_RL, IDX_LT, IDX_RT, THU_RL, THU_LT, THU_RT, MID_RL]:
        if not os.path.isfile(p):
            print("[ERR] missing file:", p)
    os.makedirs(os.path.dirname(OUT_LOG), exist_ok=True)

    # Open serial
    ser = serial.Serial(COM_PORT, BAUD, timeout=0.5)
    time.sleep(0.8)
    handshake(ser)

    # Load models
    idx_rl = load_json(IDX_RL);  idx_lt = load_json(IDX_LT);  idx_rt = load_json(IDX_RT)
    thu_rl = load_json(THU_RL);  thu_lt = load_json(THU_LT);  thu_rt = load_json(THU_RT)
    mid_rl = load_json(MID_RL)

    # Send configuration
    print(".. push INDEX")
    push_RL(ser, "IDX", idx_rl)
    push_LTH(ser, "IDX", idx_lt, ycol="θ_total")
    push_AFF(ser, "IDX", idx_rt)
    send_line(ser, f"CFG,IDX,FLEX,SPOOL,{SPOOL_R}")
    send_line(ser, f"CFG,IDX,EXT,SPOOL,{SPOOL_R}")
    send_line(ser, f"CFG,IDX,FLEX,PID,{PID_IDX[0]},{PID_IDX[1]},{PID_IDX[2]}")
    send_line(ser, f"CFG,IDX,EXT,PID,{PID_IDX[0]},{PID_IDX[1]},{PID_IDX[2]}")
    send_line(ser, f"CFG,IDX,FLEX,SIGN,{SIGN_IDX[0]},{SIGN_IDX[1]}")

    print(".. push THUMB")
    push_RL(ser, "THU", thu_rl)
    push_LTH(ser, "THU", thu_lt, ycol="∠ABD_smooth")
    push_AFF(ser, "THU", thu_rt)
    send_line(ser, f"CFG,THU,FLEX,SPOOL,{SPOOL_R}")
    send_line(ser, f"CFG,THU,EXT,SPOOL,{SPOOL_R}")
    send_line(ser, f"CFG,THU,FLEX,PID,{PID_THU[0]},{PID_THU[1]},{PID_THU[2]}")
    send_line(ser, f"CFG,THU,EXT,PID,{PID_THU[0]},{PID_THU[1]},{PID_THU[2]}")
    send_line(ser, f"CFG,THU,FLEX,SIGN,{SIGN_THU[0]},{SIGN_THU[1]}")

    print(".. push MIDDLE")
    push_RL(ser, "MID", mid_rl)
    send_line(ser, f"CFG,MID,FLEX,SPOOL,{SPOOL_R}")
    send_line(ser, f"CFG,MID,EXT,SPOOL,{SPOOL_R}")
    send_line(ser, f"CFG,MID,FLEX,PID,{PID_MID[0]},{PID_MID[1]},{PID_MID[2]}")
    send_line(ser, f"CFG,MID,FLEX,SIGN,{SIGN_MID[0]},{SIGN_MID[1]}")

    # Trajectories
    send_line(ser, f"CFG,IDX,FLEX,TRAJ,{TRAJ_TYPE},{PERIOD_S},{IDX_THETA[0]},{IDX_THETA[1]}")
    send_line(ser, f"CFG,THU,FLEX,TRAJ,{TRAJ_TYPE},{PERIOD_S},{THU_THETA[0]},{THU_THETA[1]}")
    send_line(ser, f"CFG,MID,FLEX,TRAJ_L,{TRAJ_TYPE},{PERIOD_S},{MID_L[0]},{MID_L[1]}")

    # Trigger run
    print(".. RUN")
    send_line(ser, f"CFG,IDX,RUN,{CYCLES}", wait_echo=True)
    time.sleep(0.2)

    # Logging
    print(".. logging")
    t0 = time.time()
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("src,phase,t_ms,theta_tgt,theta_meas,R,L_ff,L_cmd,flex_ticks,ext_ticks\n")
        while True:
            line = ser.readline().decode("utf-8", "ignore").strip()
            if line:
                print("<<", line) 
            if line.startswith("LOG,"):
                if line.startswith("LOG,MID,"):
                    parts = line.split(',')
                    out = ["MID", parts[2], parts[3], "", "", parts[6], "", "", parts[7], parts[8]]
                    f.write(",".join(out) + "\n")
                else:
                    parts = line.split(',')
                    out = ["IDXTH", parts[2], parts[3], parts[4], parts[5], parts[6], parts[7], parts[8], parts[9], parts[10]]
                    f.write(",".join(out) + "\n")
            if time.time() - t0 > CYCLES * 2 * PERIOD_S + 10:
                break

    print(f"[DONE] log saved -> {OUT_LOG}")

if __name__ == "__main__":
    main()
