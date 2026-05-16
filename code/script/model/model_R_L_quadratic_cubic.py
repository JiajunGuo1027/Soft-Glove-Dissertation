# model_R_L_quadratic_cubic.py
# ------------------------------------------------------------
# Sensor (R) -> Cable displacement (L_rel) modeling with Linear/Quadratic/Cubic fits (per phase).
# Steps:
#   - Active-segment extraction by motor direction
#   - Per-cycle zeroing to get L_rel (cm)
#   - 1%~99% quantile normalization for sensor
#   - Fit & plot L = f(R_norm) with linear / quadratic / cubic
# Output: R_L_summary.csv, R_L_model.json, and plots per phase
#


# Thumb:
# python model_R_L_cubic.py ^
#   --csv "D:\Soft_glove\output\thumb_angles\aligned_data_thumb.csv" ^
#   --out "D:\Soft_glove\models\cubic\thumb\R_L" ^
#   --flex-id 1 --extend-id 4 --sensor-col SensorA0

# Index:
# python model_R_L_cubic.py ^
#   --csv "D:\Soft_glove\output\index_angles\aligned_data_index.csv" ^
#   --out "D:\Soft_glove\models\cubic\index\R_L" ^
#   --flex-id 2 --extend-id 5 --sensor-col SensorA1
#
# Middle:
#   python model_R_L_quadratic_cubic.py ^
#     --csv "D:\Soft_glove\output\middle_angles\aligned_data_middle.csv" ^
#     --out "D:\Soft_glove\models\qu_cubic\middle\R_L" ^
#     --flex-id 3 --extend-id 6 --sensor-col SensorA2

# python model_R_L_quadratic_cubic.py --csv "D:\Soft_glove\output\middle_angles\aligned_data_middle.csv" --out "D:\Soft_glove\models\qu_cubic\middle\R_L" --flex-id 3 --extend-id 6 --sensor-col SensorA2

# ------------------------------------------------------------

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 18,
})

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

    # Motor angle smoothing and derivative
    sub["angle_smooth"] = sub["Angle(deg)"].rolling(window=rolling, min_periods=1).mean()
    sub["dmotor"] = sub["angle_smooth"].diff()

    # Direction filter
    if active_dir == "neg":
        sub = sub[sub["dmotor"] < 0]
    else:
        sub = sub[sub["dmotor"] > 0]

    # Remove near-static points
    sub = sub[sub["dmotor"].abs() >= float(dmotor_min)]
    if sub.empty: return sub

    # Cable displacement & per-cycle zero
    sub["L_cm"] = np.radians(sub["angle_smooth"]) * float(r_cm)
    dt = sub["Time(ms)"].diff().fillna(0.0)
    sub["cycle"] = (dt > float(gap_ms)).cumsum()
    sub["L_rel_cm"] = sub["L_cm"] - sub.groupby("cycle")["L_cm"].transform("first")

    keep = ["Time(ms)","MotorID","angle_smooth","dmotor","cycle","L_cm","L_rel_cm", sensor_col]
    return sub[keep]

def fit_and_plot_with_cubic(df_act, sensor_col, out_png, title):
    m = np.isfinite(df_act["L_rel_cm"]) & np.isfinite(df_act[sensor_col])
    g = df_act[m].copy()
    if len(g) < 10:
        plt.figure(figsize=(10, 6))
        plt.title(f"{title}\n(not enough points: N={len(g)})")
        plt.xlabel("Normalized sensor output")
        plt.ylabel("Relative tendon displacement, $L_{rel}$ (cm)")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.close()
        return None

    # Sensor normalization
    z, lo, hi = quantile_norm(g[sensor_col].to_numpy())
    g["_R_norm"] = z
    x = g["_R_norm"].to_numpy()
    y = g["L_rel_cm"].to_numpy()

    # Fit: linear / quadratic / cubic
    c1, r2_1, mse_1 = poly_metrics(x, y, 1)
    c2, r2_2, mse_2 = poly_metrics(x, y, 2)
    c3, r2_3, mse_3 = poly_metrics(x, y, 3)

    # Plot with quadratic and cubic curves only
    plt.figure(figsize=(10, 6))

    first = True
    for cyc, gg in g.groupby("cycle"):
        plt.scatter(
            gg["_R_norm"], gg["L_rel_cm"],
            s=12, alpha=0.55,
            label="Experimental data" if first else None
        )
        first = False

    xs = np.linspace(np.min(x), np.max(x), 300)

    # Linear model is intentionally not shown.
    # plt.plot(xs, np.polyval(c1, xs), linewidth=2,
    #          label=f"Linear, $R^2$={r2_1:.3f}")

    plt.plot(xs, np.polyval(c2, xs), linewidth=2.0,
             label=f"Quadratic, $R^2$={r2_2:.3f}")
    plt.plot(xs, np.polyval(c3, xs), linewidth=2.0,
             label=f"Cubic, $R^2$={r2_3:.3f}")

    plt.xlabel("Normalized sensor output, $R_{norm}$")
    plt.ylabel("Relative tendon displacement, $L_{rel}$ (cm)")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()

    return {
        "sensor_norm_lo": lo, "sensor_norm_hi": hi,
        "linear_coeffs":    [float(c1[0]), float(c1[1])],
        "quadratic_coeffs": [float(c2[0]), float(c2[1]), float(c2[2])],
        "cubic_coeffs":     [float(c3[0]), float(c3[1]), float(c3[2]), float(c3[3])],
        "R2_linear": float(r2_1), "MSE_linear": float(mse_1),
        "R2_quadratic": float(r2_2), "MSE_quadratic": float(mse_2),
        "R2_cubic": float(r2_3), "MSE_cubic": float(mse_3),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="aligned_data_*.csv")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--flex-id", type=int, required=True)
    ap.add_argument("--extend-id", type=int, required=True)
    ap.add_argument("--sensor-col", default="SensorA1")
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
    model_json = {
        "sensor_col": args.sensor_col,
        "r_cm": float(args.r_cm),
        "phases": {}
    }

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
        title = f"{phase}: {args.sensor_col}(norm) → L_rel(cm)"
        res = fit_and_plot_with_cubic(act, args.sensor_col, png, title)
        if res is None: continue

        rows.append({"phase": phase, "plot": png, "active_csv": act_csv, **res})

        model_json["phases"][phase] = {
            "sensor_norm_lo": res["sensor_norm_lo"],
            "sensor_norm_hi": res["sensor_norm_hi"],
            "linear":    {"a":res["linear_coeffs"][0],    "b":res["linear_coeffs"][1]},
            "quadratic": {"a":res["quadratic_coeffs"][0], "b":res["quadratic_coeffs"][1], "c":res["quadratic_coeffs"][2]},
            "cubic":     {"a":res["cubic_coeffs"][0],     "b":res["cubic_coeffs"][1],    "c":res["cubic_coeffs"][2], "d":res["cubic_coeffs"][3]},
            "metrics":   {
                "R2_lin":res["R2_linear"], "MSE_lin":res["MSE_linear"],
                "R2_quad":res["R2_quadratic"], "MSE_quad":res["MSE_quadratic"],
                "R2_cubic":res["R2_cubic"], "MSE_cubic":res["MSE_cubic"]
            }
        }

    # Save
    summ = pd.DataFrame(rows)
    summ_csv = os.path.join(args.out, "R_L_summary.csv")
    summ.to_csv(summ_csv, index=False, encoding="utf-8-sig")

    js_path = os.path.join(args.out, "R_L_model.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved:\n- {summ_csv}\n- {js_path}")
    for r in rows:
        print(f"  [{r['phase']}] R2_lin={r['R2_linear']:.3f}  R2_quad={r['R2_quadratic']:.3f}  R2_cubic={r['R2_cubic']:.3f} -> {r['plot']}")

if __name__ == "__main__":
    main()
