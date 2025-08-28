# eval_closedloopf.py
#   python eval_closedloop.py --port COM3 --baud 115200 --duration 12
#   python eval_closedloop.py --csv "D:\Soft_glove\output\control_results\experiment_log_multi.csv"

import argparse, time, json, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fixed output directory
OUT_DIR = r"D:\Soft_glove\output\control_results\5"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    import serial
except:
    serial = None
    print("[WARN] pyserial not installed; reading CSV only.", file=sys.stderr)

HEADER = "Time(ms),Finger,Phase,MotorID,SensorRaw,SensorNorm,L_meas_cm,Theta_meas_deg,Theta_target_deg,L_target_cm,L_cmd_cm,GoalTick,PresentTick,Err_theta_deg\n"

def read_serial(port, baud, out_csv, duration=None):
    if serial is None:
        raise RuntimeError("pyserial not available")
    ser = serial.Serial(port, baud, timeout=1)
    print(f"[INFO] Listening {port} @ {baud}")
    lines=[]
    t0=time.time()
    try:
        while True:
            s=ser.readline().decode(errors="ignore").strip()
            # firmware prints header once; keep only numeric lines
            if s and s[0].isdigit():
                lines.append(s)
            if duration and (time.time()-t0)>=duration:
                break
    except KeyboardInterrupt:
        pass
    ser.close()
    with open(out_csv,"w",encoding="utf-8") as f:
        f.write(HEADER)
        for l in lines: f.write(l+"\n")
    print(f"[DONE] {out_csv} saved.")
    return out_csv

def rmse_r2(y, yhat):
    y=np.asarray(y); yhat=np.asarray(yhat)
    m=np.isfinite(y)&np.isfinite(yhat)
    if m.sum()<5: return np.nan, np.nan
    rmse=float(np.sqrt(np.mean((y[m]-yhat[m])**2)))
    ss_res=float(np.sum((y[m]-yhat[m])**2))
    ss_tot=float(np.sum((y[m]-np.mean(y[m]))**2))
    r2=1-ss_res/ss_tot if ss_tot>0 else np.nan
    return rmse, r2

def step_metrics(t, y, u):
    """Basic step metrics; if there is no strict step input, will still try to give steady-state error at the end."""
    t=np.asarray(t); y=np.asarray(y); u=np.asarray(u)
    if len(t)<3: return {}
    du=np.diff(u)
    if len(du)==0: return {}
    k=int(np.argmax(np.abs(du)))+1
    u0,u1=u[k-1],u[k]; A=u1-u0
    yseg=y[k:]; tseg=t[k:]-t[k]
    # rise 10->90%
    if abs(A)>1e-6:
        y10=u0+0.1*A; y90=u0+0.9*A
        try:
            t10=tseg[np.where(np.sign(yseg-y10)>0)[0][0]]
            t90=tseg[np.where(np.sign(yseg-y90)>0)[0][0]]
            rise=float(max(0.0,t90-t10))
        except: rise=np.nan
        overshoot=float((np.max(yseg)-u1)/abs(A)*100.0)
        band_lo,band_hi=u1-0.02*abs(A), u1+0.02*abs(A)
        settling=np.nan
        for i in range(len(yseg)-1,-1,-1):
            if (yseg[i]<band_lo) or (yseg[i]>band_hi):
                if i<len(yseg)-1:
                    settling=float(tseg[i+1])
                break
    else:
        rise=overshoot=settling=np.nan
    # steady-state error (last 0.5s)
    dt=np.median(np.diff(t)) if len(t)>1 else 0.02
    tail=max(1, int(0.5/max(dt,1e-3)))
    ss_err=float(np.mean(y[-tail:]-u[-tail:])) if len(y)>tail else np.nan
    return {"rise_time_s":rise,"overshoot_pct":overshoot,"settling_time_s":settling,"steady_state_error_deg":ss_err}

def active_mask(df, dpos_min=2, phase=None):
    g=df.copy()
    # Derivative is computed along the sample order of this finger only
    g["dpos"]=g["PresentTick"].diff().abs()
    m=g["dpos"]>=dpos_min
    if phase is not None:
        m &= (g["Phase"]==phase)
    return m

def xcorr_lag_ms(t, ref, sig):
    ref=np.asarray(ref); sig=np.asarray(sig)
    m=np.isfinite(ref)&np.isfinite(sig)
    if m.sum()<3: return np.nan
    ref=ref[m]-np.mean(ref[m]); sig=sig[m]-np.mean(sig[m])
    c=np.correlate(sig, ref, mode="full")
    lags=np.arange(-len(ref)+1, len(ref))
    i=int(np.argmax(c))
    dt=np.median(np.diff(np.asarray(t)[m])) if m.sum()>1 else 0.02
    return float(lags[i]*dt*1000.0)

def plot_ts(df, out_png, title=None):
    t=(df["Time(ms)"]-df["Time(ms)"].iloc[0])/1000.0
    plt.figure(figsize=(10,5))
    plt.plot(t, df["Theta_target_deg"], label="θ_target")
    plt.plot(t, df["Theta_meas_deg"],   label="θ_measured")
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
    if title: plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def plot_err(df, out_png, title=None):
    t=(df["Time(ms)"]-df["Time(ms)"].iloc[0])/1000.0
    plt.figure(figsize=(10,3.5))
    plt.plot(t, df["Err_theta_deg"])
    plt.xlabel("Time (s)"); plt.ylabel("Error (deg)")
    if title: plt.title(title)
    plt.grid(True); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def plot_overview(groups, out_png):
    """Multi-finger overview: overlay θ_target/θ_meas of each finger (to check if relative motion is synchronized)"""
    plt.figure(figsize=(11,6))
    for name, df in groups.items():
        t=(df["Time(ms)"]-df["Time(ms)"].iloc[0])/1000.0
        plt.plot(t, df["Theta_target_deg"], label=f"{name} θ_target", alpha=0.7)
        plt.plot(t, df["Theta_meas_deg"],   label=f"{name} θ_meas",   alpha=0.7, linestyle="--")
    plt.xlabel("Time (s)"); plt.ylabel("Angle (deg)")
    plt.title("Multi-finger overview")
    plt.grid(True); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def per_finger_metrics(df, dpos_min=2, phase=None):
    t=(df["Time(ms)"]-df["Time(ms)"].iloc[0])/1000.0
    rmse_all, r2_all = rmse_r2(df["Theta_target_deg"], df["Theta_meas_deg"])
    m_act = active_mask(df, dpos_min=dpos_min, phase=phase)
    rmse_act, r2_act = rmse_r2(df.loc[m_act,"Theta_target_deg"], df.loc[m_act,"Theta_meas_deg"])
    step = step_metrics(t, df["Theta_meas_deg"], df["Theta_target_deg"])
    lag  = xcorr_lag_ms(t, df["Theta_target_deg"], df["Theta_meas_deg"])
    return {
        "RMSE_all_deg": rmse_all, "R2_all": r2_all,
        "RMSE_active_deg": rmse_act, "R2_active": r2_act,
        "lag_ms": lag, "step_metrics": step
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--csv", default="experiment_log.csv")
    ap.add_argument("--phase", default=None, choices=[None,"Flex","Extend"])
    ap.add_argument("--active-dpos-min", type=int, default=2)
    args=ap.parse_args()

    if args.port:
        read_serial(args.port, args.baud, args.csv, args.duration)

    df=pd.read_csv(args.csv)

    # Multi-finger or single-finger
    fingers = sorted(df["Finger"].unique()) if "Finger" in df.columns else ["Unknown"]
    metrics_all = {}
    groups = {}

    if len(fingers) == 1:
        # Single-finger
        plot_ts(df, os.path.join(OUT_DIR, "timeseries.png"))
        plot_err(df, os.path.join(OUT_DIR, "error.png"))
        metrics_all[fingers[0]] = per_finger_metrics(df, dpos_min=args.active_dpos_min, phase=args.phase)

    else:
        # Multi-finger: evaluate & plot separately
        for f in fingers:
            dff = df[df["Finger"]==f].reset_index(drop=True)
            groups[f] = dff
            plot_ts(dff, os.path.join(OUT_DIR, f"timeseries_{f}.png"), title=f)
            plot_err(dff, os.path.join(OUT_DIR, f"error_{f}.png"), title=f)
            metrics_all[f] = per_finger_metrics(dff, dpos_min=args.active_dpos_min, phase=args.phase)
        # Overlay overview
        plot_overview(groups, os.path.join(OUT_DIR, "timeseries_overview.png"))

    metrics_all["_aggregate"] = per_finger_metrics(df, dpos_min=args.active_dpos_min, phase=args.phase)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics_all, indent=2))
    print(f"[INFO] Outputs saved to: {OUT_DIR}")

if __name__=="__main__":
    main()
