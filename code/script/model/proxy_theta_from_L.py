# proxy_theta_from_L.py
# ------------------------------------------------------------
# Build a "proxy" joint angle from sensor via:
#   R --(R->L JSON)--> L_est --(Index θ(L))--> θ_proxy
# Optionally apply an affine tweak: θ_proxy = a*θ_proxy + b
#
# Inputs:
#   - aligned_data_*.csv  (has Time, MotorID, Angle(deg), SensorAx)
#   - R_L_model_*.json    (exported by model_R_L_middle.py)
#   - L_theta_summary.csv (θ(L) quadratic coeffs per phase from *Index* or same finger)
#
# Outputs:
#   - proxy_theta.csv  (adds L_est_cm, theta_proxy)
#   - proxy_theta_timeseries.png

# CMD
# python proxy_theta_from_L.py ^
#   --aligned-csv "D:\Soft_glove\output\middle_angles\aligned_data_middle.csv" ^
#   --r-l-json    "D:\Soft_glove\models\middle\R_L\R_L_model_middle.json" ^
#   --index-summary "D:\Soft_glove\models\index\L_theta\L_theta_summary.csv" ^
#   --sensor-col SensorA2 ^
#   --out "D:\Soft_glove\models\middle\proxy_theta" ^
#   --affine-a 1.00 --affine-b 0.0
# ------------------------------------------------------------

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_r_l_json(path):
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    return js

def load_theta_L_quadratic_from_summary(summary_csv, ycol="θ_total"):
    df = pd.read_csv(summary_csv, encoding="utf-8-sig")
    def pick(phase):
        row = df[(df["ycol"] == ycol) & (df["phase"] == phase)]
        if row.empty:
            raise ValueError(f"Cannot find {ycol} {phase} in {summary_csv}")
        coeffs = [float(x) for x in str(row.iloc[0]["quadratic_coeffs"]).split(",")]
        return coeffs
    return {"Flex": pick("Flex"), "Extend": pick("Extend")}

def estimate_L_from_R(row, js_r_l, sensor_col):
    phase = "Flex" if row["MotorID"] in [1, 2, 3] else "Extend"
    phase_cfg = js_r_l["phases"][phase]
    lo, hi = phase_cfg["sensor_norm_lo"], phase_cfg["sensor_norm_hi"]
    z = (row[sensor_col] - lo) / (hi - lo if hi > lo else 1e-9)
    # quadratic by default
    a = phase_cfg["quadratic"]["a"]
    b = phase_cfg["quadratic"]["b"]
    c = phase_cfg["quadratic"]["c"]
    L = a * (z * z) + b * z + c
    return L, phase

def theta_from_L(L, phase, thetaL_coeffs):
    c = thetaL_coeffs[phase]  # [a,b,c]
    return float(np.polyval(c, L))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-csv", required=True, help="aligned_data_*.csv of the target finger")
    ap.add_argument("--r-l-json", required=True, help="R_L_model_*.json for the same finger")
    ap.add_argument("--index-summary", required=True, help="Index L_theta_summary.csv to provide θ(L)")
    ap.add_argument("--sensor-col", required=True, help="e.g., SensorA1/SensorA2/SensorA0")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--affine-a", type=float, default=1.0, help="optional scale a for proxy θ")
    ap.add_argument("--affine-b", type=float, default=0.0, help="optional offset b for proxy θ (deg)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # load data & models
    df = pd.read_csv(args.aligned_csv, encoding="utf-8-sig")
    js = load_r_l_json(args.r_l_json)
    thetaL = load_theta_L_quadratic_from_summary(args.index_summary, ycol="θ_total")

    # compute L_est & theta_proxy
    L_list, phase_list, theta_list = [], [], []
    for _, row in df.iterrows():
        if not np.isfinite(row.get(args.sensor_col, np.nan)):
            L_list.append(np.nan); phase_list.append(None); theta_list.append(np.nan); continue
        L, phase = estimate_L_from_R(row, js, args.sensor_col)
        th = theta_from_L(L, phase or "Flex", thetaL)  # default Flex if None
        th = args.affine_a * th + args.affine_b
        L_list.append(L); phase_list.append(phase); theta_list.append(th)

    df_out = df.copy()
    df_out["L_est_cm"] = L_list
    df_out["phase_est"] = phase_list
    df_out["theta_proxy(deg)"] = theta_list

    out_csv = os.path.join(args.out, "proxy_theta.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # -------------------- Plot (with distinct colors) --------------------
    t = df_out["Time(ms)"].to_numpy() / 1000.0
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()

    # use different colors for the two axes/lines
    c1, c2 = "C0", "C1"

    # left axis: L_est
    l1, = ax1.plot(t, df_out["L_est_cm"], label="L_est (cm)", color=c1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("L_est (cm)", color=c1)
    ax1.tick_params(axis="y", colors=c1)
    ax1.spines["left"].set_color(c1)
    ax1.grid(True)

    # right axis: theta_proxy
    ax2 = ax1.twinx()
    l2, = ax2.plot(t, df_out["theta_proxy(deg)"], label="theta_proxy (deg)", color=c2)
    ax2.set_ylabel("theta_proxy (deg)", color=c2)
    ax2.tick_params(axis="y", colors=c2)
    ax2.spines["right"].set_color(c2)

    plt.title("Proxy θ from L_est via Index θ(L)")
    plt.legend([l1, l2], ["L_est (cm)", "theta_proxy (deg)"], loc="upper right")
    plt.tight_layout()

    out_png = os.path.join(args.out, "proxy_theta_timeseries.png")
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[DONE] Saved:\n- {out_csv}\n- {out_png}")

if __name__ == "__main__":
    main()
