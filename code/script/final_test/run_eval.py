# run_eval.py
# # 1) First connect the board and let the firmware run and output via serial
# # 2) Capture 12s serial log + evaluate
# python run_eval.py --port COM7 --baud 115200 --duration 12
# python run_eval.py --csv experiment_log.csv

import argparse, time, json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import serial
except:
    serial = None
    print("[WARN] pyserial not installed; reading existing CSV only.", file=sys.stderr)

def read_serial_to_csv(port, baud, out_csv, duration=None):
    if serial is None:
        raise RuntimeError("pyserial not available. pip install pyserial")
    ser = serial.Serial(port, baud, timeout=1)
    print(f"[INFO] Listening {port} @ {baud}. Press Ctrl+C to stop.")
    lines = []
    t0 = time.time()
    try:
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            if line and line[0].isdigit():  # skip header; firmware prints header first
                lines.append(line)
            if duration and (time.time()-t0)>=duration: break
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
    # Write
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("Time(ms),Finger,Phase,MotorID,SensorRaw,SensorNorm,L_est_cm,Theta_est_deg,Theta_target_deg,GoalTick,PresentTick,Err_deg\n")
        for l in lines:
            f.write(l+"\n")
    print(f"[DONE] Saved {out_csv}")
    return out_csv

def compute_active_mask(df, dmotor_min_tick=2, phase=None):
    #Use motor position derivative + phase filtering
    g = df.copy()
    g["dpos"] = g["PresentTick"].diff()
    m = g["dpos"].abs() >= dmotor_min_tick
    if phase is not None:
        m &= (g["Phase"]==phase)
    return m

def rmse_r2(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum()<3: return np.nan, np.nan
    rmse = float(np.sqrt(np.mean((y[m]-yhat[m])**2)))
    ss_res = float(np.sum((y[m]-yhat[m])**2))
    ss_tot = float(np.sum((y[m]-np.mean(y[m]))**2))
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    return rmse, r2

def step_metrics(t, y, u):
    # For step input: u is the target, y is the measurement
    t = np.asarray(t); y=np.asarray(y); u=np.asarray(u)
    # Find the step moment (largest rising edge of u)
    du = np.diff(u)
    if len(du)==0: return {}
    k = int(np.argmax(np.abs(du))) + 1
    u0, u1 = u[k-1], u[k]
    A = u1 - u0
    if abs(A) < 1e-6: return {}
    yseg = y[k:]
    tseg = t[k:] - t[k]
    # 10-90% rise time
    y10 = u0 + 0.1*A
    y90 = u0 + 0.9*A
    try:
        t10 = tseg[np.where(np.sign(yseg - y10)>0)[0][0]]
        t90 = tseg[np.where(np.sign(yseg - y90)>0)[0][0]]
        rise = float(max(0.0, t90 - t10))
    except:
        rise = np.nan
    # Overshoot
    overshoot = float((np.max(yseg)-u1)/abs(A)*100.0)
    #Settling time (±2% band)
    band_lo, band_hi = u1 - 0.02*abs(A), u1 + 0.02*abs(A)
    settling = np.nan
    for i in range(len(yseg)-1, -1, -1):
        if (yseg[i]<band_lo) or (yseg[i]>band_hi):
            if i < len(yseg)-1:
                settling = float(tseg[i+1])
            break
    #  Steady-state error
    ss_err = (float(np.mean(yseg[-int(0.5/np.median(np.diff(t))):]) - u1)
          if len(yseg) > 5 else np.nan)
    return {"rise_time_s":rise, "overshoot_pct":overshoot, "settling_time_s":settling, "steady_state_error_deg":ss_err}

def xcorr_lag_ms(t, ref, sig):
    # Simple cross-correlation delay estimate (ms); positive means sig lags behind ref
    ref = np.asarray(ref); sig=np.asarray(sig)
    m = np.isfinite(ref) & np.isfinite(sig)
    ref = ref[m] - np.mean(ref[m]); sig = sig[m] - np.mean(sig[m])
    if len(ref)<3: return np.nan
    c = np.correlate(sig, ref, mode="full")
    lags = np.arange(-len(ref)+1, len(ref))
    i = int(np.argmax(c))
    dt = np.median(np.diff(t[m]))
    return float(lags[i]*dt*1000.0)

def suggest_affine(yhat, y):
    m = np.isfinite(yhat) & np.isfinite(y)
    if m.sum()<10: return None
    A = np.c_[yhat[m], np.ones(m.sum())]
    a, b = np.linalg.lstsq(A, y[m], rcond=None)[0]
    rmse0, r20 = rmse_r2(y[m], yhat[m])
    y2 = a*yhat[m] + b
    rmse1, r21 = rmse_r2(y[m], y2)
    return dict(a=float(a), b=float(b), rmse_before=rmse0, rmse_after=rmse1, r2_before=r20, r2_after=r21)

def plot_timeseries(df, out_png):
    t = (df["Time(ms)"].to_numpy()-df["Time(ms)"].iloc[0])/1000.0
    plt.figure(figsize=(10,5))
    plt.plot(t, df["Theta_target_deg"], label="θ target")
    plt.plot(t, df["Theta_est_deg"], label="θ estimate")
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def plot_error(df, out_png):
    t = (df["Time(ms)"].to_numpy()-df["Time(ms)"].iloc[0])/1000.0
    err = df["Err_deg"].to_numpy()
    plt.figure(figsize=(10,3.5))
    plt.plot(t, err)
    plt.xlabel("Time (s)"); plt.ylabel("Error (deg)"); plt.grid(True); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="COMx / /dev/ttyUSB0 ; if omitted, read --csv only")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--csv", default="experiment_log.csv")
    ap.add_argument("--duration", type=float, default=None, help="seconds to capture; omit to Ctrl+C")
    ap.add_argument("--active-dpos-min", type=int, default=2)
    ap.add_argument("--phase", default=None, choices=[None,"Flex","Extend"])
    args = ap.parse_args()

    if args.port:
        read_serial_to_csv(args.port, args.baud, args.csv, args.duration)

    df = pd.read_csv(args.csv)
    # Basic plots
    plot_timeseries(df, "timeseries.png")
    plot_error(df, "error.png")

    # Metrics
    rmse_all, r2_all = rmse_r2(df["Theta_target_deg"], df["Theta_est_deg"])
    active = compute_active_mask(df, dmotor_min_tick=args.active_dpos_min, phase=args.phase)
    rmse_act, r2_act = rmse_r2(df.loc[active,"Theta_target_deg"], df.loc[active,"Theta_est_deg"])
    t = (df["Time(ms)"].to_numpy()-df["Time(ms)"].iloc[0])/1000.0
    step = step_metrics(t, df["Theta_est_deg"].to_numpy(), df["Theta_target_deg"].to_numpy())
    lag_ms = xcorr_lag_ms(t, df["Theta_target_deg"], df["Theta_est_deg"])

    metrics = {
        "RMSE_all_deg": rmse_all, "R2_all": r2_all,
        "RMSE_active_deg": rmse_act, "R2_active": r2_act,
        "lag_ms": lag_ms, "step_metrics": step
    }

    aff = suggest_affine(df.loc[active,"Theta_est_deg"], df.loc[active,"Theta_target_deg"])
    if aff: metrics["suggest_affine"] = aff

    with open("metrics.json","w",encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
