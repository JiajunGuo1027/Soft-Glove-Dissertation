# model_R_L.py
# ------------------------------------------------------------
# - Active-segment extraction by motor direction
# - Per-cycle zeroing to get L_rel (cm)
# - Sensor normalization (1%-99% quantiles)
# - Linear & quadratic fits per phase (Flex/Extend)
# - Export plots + CSV summary + JSON model (for controller)
#
# Usage (Windows CMD):
# Thumb:
# python model_R_L.py ^
#   --csv "D:\Soft_glove\output\thumb_angles\aligned_data_thumb.csv" ^
#   --out "D:\Soft_glove\models\thumb\R_L" ^
#   --flex-id 1 --extend-id 4 --sensor-col SensorA0

# Index:
# python model_R_L.py ^
#   --csv "D:\Soft_glove\output\index_angles\aligned_data_index.csv" ^
#   --out "D:\Soft_glove\models\index\R_L" ^
#   --flex-id 2 --extend-id 5 --sensor-col SensorA1

# Middle:
#   python model_R_L.py ^
#     --csv "D:\Soft_glove\output\index_angles\aligned_data_index.csv" ^
#     --out "D:\Soft_glove\models\middle\R_L" ^
#     --flex-id 3 --extend-id 6 --sensor-col SensorA2

# ------------------------------------------------------------

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def quantile_norm(x, qlo=0.01, qhi=0.99):
    x = np.asarray(x, float)
    lo = np.nanquantile(x, qlo)
    hi = np.nanquantile(x, qhi)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    z = (x - lo) / (hi - lo)
    return z, float(lo), float(hi)

def poly_metrics(x, y, deg):
    c = np.polyfit(x, y, deg=deg)
    yhat = np.polyval(c, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    mse = float(np.mean((y - yhat) ** 2))
    return c, r2, mse

def extract_active_segments(df, motor_id, sensor_col, r_cm=1.0,
                            rolling=5, gap_ms=5000, active_dir="neg",
                            dmotor_min=0.02):
    need = ["Time(ms)","MotorID","Angle(deg)", sensor_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    sub = df[df["MotorID"] == motor_id].copy().sort_values("Time(ms)").reset_index(drop=True)

    #  Motor angle smoothing and derivative
    sub["angle_smooth"] = sub["Angle(deg)"].rolling(window=rolling, min_periods=1).mean()
    sub["dmotor"] = sub["angle_smooth"].diff()

    # Active direction filtering (neg=clockwise=angle decreases)
    if active_dir == "neg":
        sub = sub[sub["dmotor"] < 0]
    else:
        sub = sub[sub["dmotor"] > 0]

    # Remove near-static points
    sub = sub[sub["dmotor"].abs() >= float(dmotor_min)]

    # Cable displacement & cycle segmentation & per-cycle zeroing
    sub["L_cm"] = np.radians(sub["angle_smooth"]) * float(r_cm)
    dt = sub["Time(ms)"].diff().fillna(0.0)
    sub["cycle"] = (dt > float(gap_ms)).cumsum()
    sub["L_rel_cm"] = sub["L_cm"] - sub.groupby("cycle")["L_cm"].transform("first")

    # Keep only required columns
    keep = ["Time(ms)","MotorID","angle_smooth","dmotor","cycle","L_cm","L_rel_cm", sensor_col]
    return sub[keep]

def fit_and_plot(df_act, sensor_col, out_png, title):
    # Keep only valid values
    m = np.isfinite(df_act["L_rel_cm"]) & np.isfinite(df_act[sensor_col])
    g = df_act[m].copy()
    if len(g) < 10:
        plt.figure(figsize=(10,6)); plt.title(f"{title}\n(no enough points: N={len(g)})")
        plt.xlabel(sensor_col); plt.ylabel("L_rel (cm)"); plt.grid(True); plt.tight_layout()
        plt.savefig(out_png, dpi=300); plt.close()
        return None

    # Normalize sensor
    z, lo, hi = quantile_norm(g[sensor_col].to_numpy())
    g["_R_norm"] = z

    # Fit
    x = g["_R_norm"].to_numpy()
    y = g["L_rel_cm"].to_numpy()
    c1, r2_1, mse_1 = poly_metrics(x, y, 1)
    c2, r2_2, mse_2 = poly_metrics(x, y, 2)

    # Plot
    plt.figure(figsize=(10,6))
    for cyc, gg in g.groupby("cycle"):
        plt.scatter(gg["_R_norm"], gg["L_rel_cm"], s=12, alpha=0.7, label=f"cycle {int(cyc)}")
    xs = np.linspace(np.min(x), np.max(x), 300)
    plt.plot(xs, np.polyval(c1, xs), linewidth=2, label=f"Linear R²={r2_1:.3f}")
    plt.plot(xs, np.polyval(c2, xs), linewidth=2, label=f"Quadratic R²={r2_2:.3f}")
    plt.xlabel(f"{sensor_col} (normalized)"); plt.ylabel("L_rel (cm)")
    plt.title(title); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

    return {
        "sensor_norm_lo": lo, "sensor_norm_hi": hi,
        "linear_coeffs": [float(c1[0]), float(c1[1])],           # L = a*z + b
        "quadratic_coeffs": [float(c2[0]), float(c2[1]), float(c2[2])],
        "R2_linear": float(r2_1), "MSE_linear": float(mse_1),
        "R2_quadratic": float(r2_2), "MSE_quadratic": float(mse_2),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="aligned_data_middle.csv")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--flex-id", type=int, required=True)
    ap.add_argument("--extend-id", type=int, required=True)
    ap.add_argument("--sensor-col", default="SensorA2")
    ap.add_argument("--r-cm", type=float, default=1.0)
    ap.add_argument("--rolling", type=int, default=5)
    ap.add_argument("--gap-ms", type=float, default=5000)
    ap.add_argument("--dmotor-min", type=float, default=0.02)
    ap.add_argument("--flex-dir", default="neg", choices=["neg","pos"])
    ap.add_argument("--extend-dir", default="neg", choices=["neg","pos"])
    args = ap.parse_args()

    ensure_dir(args.out)
    df = pd.read_csv(args.csv, encoding="utf-8-sig")

    rows = []
    model_json = {"finger":"middle", "r_cm":float(args.r_cm), "sensor_col":args.sensor_col, "phases":{}}

    for phase, motor_id, mdir in [
        ("Flex", args.flex_id, args.flex_dir),
        ("Extend", args.extend_id, args.extend_dir),
    ]:
        act = extract_active_segments(
            df, motor_id, args.sensor_col,
            r_cm=args.r_cm, rolling=args.rolling, gap_ms=args.gap_ms,
            active_dir=mdir, dmotor_min=args.dmotor_min
        )
        act_csv = os.path.join(args.out, f"R_L_{phase}_active.csv")
        act.to_csv(act_csv, index=False, encoding="utf-8-sig")

        png = os.path.join(args.out, f"R_L_{phase}.png")
        title = f"Middle {phase}: {args.sensor_col} (norm) → L_rel(cm)"
        res = fit_and_plot(act, args.sensor_col, png, title)

        if res is None:
            continue

        # Save summary
        row = {"phase":phase, "plot":png, "active_csv":act_csv, **res}
        rows.append(row)

        # Build JSON
        model_json["phases"][phase] = {
            "sensor_norm_lo": res["sensor_norm_lo"],
            "sensor_norm_hi": res["sensor_norm_hi"],
            "quadratic": {"a":res["quadratic_coeffs"][0], "b":res["quadratic_coeffs"][1], "c":res["quadratic_coeffs"][2]},
            "linear":    {"a":res["linear_coeffs"][0],    "b":res["linear_coeffs"][1]},
            "metrics":   {"R2_lin":res["R2_linear"], "MSE_lin":res["MSE_linear"],
                          "R2_quad":res["R2_quadratic"], "MSE_quad":res["MSE_quadratic"]}
        }

    # Save summary CSV
    summ = pd.DataFrame(rows)
    summ_csv = os.path.join(args.out, "R_L_summary.csv")
    summ.to_csv(summ_csv, index=False, encoding="utf-8-sig")

    # Export JSON
    js_path = os.path.join(args.out, "R_L_model.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved:\n- {summ_csv}\n- {js_path}")
    for r in rows:
        print(f"  [{r['phase']}] R2_quad={r['R2_quadratic']:.3f}  -> {r['plot']}")

if __name__ == "__main__":
    main()
