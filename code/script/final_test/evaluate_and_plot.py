# evaluate_and_plot.py
# - Read board-side CSV log
# - For index/thumb (θ-domain) and middle finger (L-domain), compute: RMSE, steady-state error, overshoot, settling time
# - Plot time-series overlays and scatter

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

LOG_CSV = r"D:\Soft_glove\control_results\run_logs\log_run.csv"
OUT_DIR = r"D:\Soft_glove\control_results\run_eval"

def rmse(a,b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum()<5: return np.nan
    return float(np.sqrt(np.mean((a[m]-b[m])**2)))

def steady_state_err(t, y, yref, tail_frac=0.2):
    n = len(y); 
    if n<10: return np.nan
    s = int(n*(1-tail_frac))
    return float(np.nanmean(y[s:]-yref[s:]))

def overshoot(y, yref):
    m = np.isfinite(y) & np.isfinite(yref)
    if m.sum()<5: return np.nan
    peak = np.nanmax(y[m]); final = np.nanmedian(yref[m])
    if final==0: return np.nan
    return float( (peak-final) / max(1e-6, abs(final)) )

def settling_time(t, y, yref, band=0.05):
    m = np.isfinite(y) & np.isfinite(yref)
    if m.sum()<5: return np.nan
    yy = y[m]; rr = yref[m]; tt = t[m]
    bandv = np.abs(rr)*band + 1e-6
    inside = np.abs(yy-rr) <= bandv
    last_out = np.where(~inside)[0]
    if len(last_out)==0: return float(tt[0])
    i = last_out[-1]
    if i>=len(tt)-1: return np.nan
    return float(tt[i+1]-tt[0])

def plot_idxthumb(df, out_prefix):
    t  = df["t_ms"].to_numpy()/1000.0
    th = pd.to_numeric(df["theta_meas"], errors="coerce").to_numpy()
    ref= pd.to_numeric(df["theta_tgt"],  errors="coerce").to_numpy()
    r  = pd.to_numeric(df["R"], errors="coerce").to_numpy()

    plt.figure(figsize=(12,5))
    plt.plot(t, ref, label="θ target")
    plt.plot(t, th,  label="θ measured")
    plt.xlabel("Time (s)"); plt.ylabel("θ (deg)")
    plt.title("Index/Thumb closed-loop tracking")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_prefix+"_timeseries.png", dpi=300); plt.close()

    # Metrics
    metrics = {
        "RMSE_deg": rmse(ref, th),
        "SteadyStateError_deg": steady_state_err(t, th, ref),
        "Overshoot_rel": overshoot(th, ref),
        "SettlingTime_s": settling_time(t, th, ref)
    }
    pd.DataFrame([metrics]).to_csv(out_prefix+"_metrics.csv", index=False)
    return metrics

def plot_middle(df, out_prefix):
    t  = df["t_ms"].to_numpy()/1000.0
    L_meas = pd.to_numeric(df["R"], errors="coerce").to_numpy()
    plt.figure(figsize=(12,5))
    plt.plot(t, pd.to_numeric(df["theta_meas"], errors="coerce").to_numpy(), label="L_meas (cm)")
    plt.xlabel("Time (s)"); plt.ylabel("L (cm)")
    plt.title("Middle finger (L-domain) monitoring")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_prefix+"_timeseries.png", dpi=300); plt.close()

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LOG_CSV)
    # Split into two: IDX/THU (recorded together as IDXTH), and MID
    d_it = df[df["src"]=="IDXTH"].copy()
    d_md = df[df["src"]=="MID"].copy()

    if len(d_it):
        m = plot_idxthumb(d_it, str(Path(OUT_DIR)/"index_thumb"))
        print("[IDX/THU]", m)
    if len(d_md):
        plot_middle(d_md, str(Path(OUT_DIR)/"middle"))
        print("[MID] plotted.")

if __name__ == "__main__":
    main()
