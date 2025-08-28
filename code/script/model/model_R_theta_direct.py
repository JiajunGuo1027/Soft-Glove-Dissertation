# model_R_theta_direct.py
# ------------------------------------------------------------
# Directly fit R -> θ per phase (Flex/Extend), with robust filtering & normalization.
# - Fingers: index (θ_total = ∠ABC_smooth + ∠BCD_smooth), thumb (∠ABD_smooth), middle (same as index if θ exists)
# - Outputs: plots, CSV summary, JSON model usable by controller

# CMD 

#  Index（θ_total = ∠ABC_smooth + ∠BCD_smooth）
# python model_R_theta_direct.py ^
#   --csv "D:\Soft_glove\output\index_angles\aligned_data_index.csv" ^
#   --out "D:\Soft_glove\models\index\R_theta_direct" ^
#   --finger index --sensor-col SensorA1 --flex-id 2 --extend-id 5

#  Thumb（θ = ∠ABD_smooth）
# python model_R_theta_direct.py ^
#   --csv "D:\Soft_glove\output\thumb_angles\aligned_data_thumb.csv" ^
#   --out "D:\Soft_glove\models\thumb\R_theta_direct" ^
#   --finger thumb --sensor-col SensorA0 --flex-id 1 --extend-id 4

#  middle
# python model_R_theta_direct.py ^
#   --csv "D:\Soft_glove\output\middle_angles\aligned_data_middle.csv" ^
#   --out "D:\Soft_glove\models\middle\R_theta_direct" ^
#   --finger middle --sensor-col SensorA2 --flex-id 3 --extend-id 6

# ------------------------------------------------------------

import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def quantile_norm(x, qlo=0.01, qhi=0.99):
    x = np.asarray(x, float)
    lo = np.nanquantile(x, qlo)
    hi = np.nanquantile(x, qhi)
    if hi - lo < 1e-9: hi = lo + 1e-9
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

def extract_active_segments(df, motor_id, theta_col, sensor_col,
                            rolling=5, gap_ms=5000, active_dir="neg",
                            dmotor_min=0.02, dtheta_min=None):
    need = ["Time(ms)", "MotorID", "Angle(deg)", sensor_col]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    if theta_col not in df.columns:
        raise ValueError(f"Missing theta column: {theta_col}")

    sub = df[df["MotorID"] == motor_id].copy().sort_values("Time(ms)").reset_index(drop=True)

    # Motor angle smoothing & derivative
    sub["angle_smooth"] = sub["Angle(deg)"].rolling(window=rolling, min_periods=1).mean()
    sub["dmotor"] = sub["angle_smooth"].diff()

    #Active direction filtering (neg = clockwise = angle decreasing)
    if active_dir == "neg":
        sub = sub[sub["dmotor"] < 0]
    else:
        sub = sub[sub["dmotor"] > 0]

    # Remove near-static points
    sub = sub[sub["dmotor"].abs() >= float(dmotor_min)]

    # Split into cycles
    dt = sub["Time(ms)"].diff().fillna(0.0)
    sub["cycle"] = (dt > float(gap_ms)).cumsum()

    if dtheta_min is not None:
        sub["dtheta"] = sub[theta_col].diff().abs()
        sub = sub[sub["dtheta"] >= float(dtheta_min)]

    keep = ["Time(ms)", "MotorID", sensor_col, theta_col, "cycle"]
    return sub[keep]

def fit_phase(df_phase, sensor_col, theta_col, out_png, title):
    #Filter NaN
    m = np.isfinite(df_phase[sensor_col]) & np.isfinite(df_phase[theta_col])
    g = df_phase[m].copy()
    if len(g) < 10:
        plt.figure(figsize=(10,6))
        plt.title(f"{title}\n(no enough valid points: N={len(g)})")
        plt.xlabel(f"{sensor_col} (normalized)"); plt.ylabel(f"{theta_col} (deg)")
        plt.grid(True); plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
        return None

    # If θ is basically unchanged (flat segment), skip
    if np.nanvar(g[theta_col].to_numpy()) < 1e-6:
        plt.figure(figsize=(10,6))
        plt.title(f"{title}\n(flat θ segment, skip fitting)")
        plt.xlabel(f"{sensor_col} (normalized)"); plt.ylabel(f"{theta_col} (deg)")
        plt.grid(True); plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
        return {"note": "flat_theta"}

    # Sensor quantile normalization
    z, lo, hi = quantile_norm(g[sensor_col].to_numpy())
    g["_R_norm"] = z

    x = g["_R_norm"].to_numpy()
    y = g[theta_col].to_numpy()

    # Linear & quadratic fitting（θ = a*z^2 + b*z + c）
    c1, r2_1, mse_1 = poly_metrics(x, y, 1)
    c2, r2_2, mse_2 = poly_metrics(x, y, 2)

    # Plotting (color by cycle)
    plt.figure(figsize=(10,6))
    for cyc, gg in g.groupby("cycle"):
        plt.scatter(gg["_R_norm"], gg[theta_col], s=12, alpha=0.7, label=f"cycle {int(cyc)}")
    xs = np.linspace(np.min(x), np.max(x), 300)
    plt.plot(xs, np.polyval(c1, xs), linewidth=2, label=f"Linear R²={r2_1:.3f}")
    plt.plot(xs, np.polyval(c2, xs), linewidth=2, label=f"Quadratic R²={r2_2:.3f}")
    plt.xlabel(f"{sensor_col} (normalized)")
    plt.ylabel(f"{theta_col} (deg)")
    plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

    return {
        "sensor_norm_lo": lo, "sensor_norm_hi": hi,
        "linear_coeffs":   [float(c1[0]), float(c1[1])],           # θ = a*z + b
        "quadratic_coeffs":[float(c2[0]), float(c2[1]), float(c2[2])],  # θ = a*z^2 + b*z + c
        "R2_linear": float(r2_1), "MSE_linear": float(mse_1),
        "R2_quadratic": float(r2_2), "MSE_quadratic": float(mse_2),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="aligned_data_*.csv")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--finger", required=True, choices=["index","middle","thumb"])
    ap.add_argument("--sensor-col", required=True, help="SensorA0/1/2")
    ap.add_argument("--flex-id", type=int, required=True)
    ap.add_argument("--extend-id", type=int, required=True)
    # Filtering/robustness
    ap.add_argument("--flex-dir", default="neg", choices=["neg","pos"])
    ap.add_argument("--extend-dir", default="neg", choices=["neg","pos"])
    ap.add_argument("--rolling", type=int, default=5)
    ap.add_argument("--gap-ms", type=float, default=5000)
    ap.add_argument("--dmotor-min", type=float, default=0.02)
    ap.add_argument("--dtheta-min", type=float, default=None, help="可选：θ也要在动")
    args = ap.parse_args()

    ensure_dir(args.out)
    df = pd.read_csv(args.csv, encoding="utf-8-sig")

    # θ column definition
    if args.finger in ("index","middle"):
        if "∠ABC_smooth" in df.columns and "∠BCD_smooth" in df.columns:
            df["θ_total"] = df["∠ABC_smooth"] + df["∠BCD_smooth"]
            theta_col = "θ_total"
        else:
            print("[WARN] No θ columns found for index/middle. Skip fitting.")
            theta_col = None
    else:  # thumb
        theta_col = "∠ABD_smooth" if "∠ABD_smooth" in df.columns else None
        if theta_col is None:
            print("[WARN] No ∠ABD_smooth in CSV. Skip fitting.")

    if theta_col is None:
        # Export empty summary to avoid pipeline break
        pd.DataFrame([]).to_csv(os.path.join(args.out, "R_theta_direct_summary.csv"),
                                index=False, encoding="utf-8-sig")
        print("[DONE] No θ available; nothing fitted.")
        return

    rows = []
    model_json = {
        "finger": args.finger,
        "sensor_col": args.sensor_col,
        "ycol": theta_col,
        "phases": {}
    }

    for phase, motor_id, mdir in [
        ("Flex", args.flex_id, args.flex_dir),
        ("Extend", args.extend_id, args.extend_dir),
    ]:
        act = extract_active_segments(
            df, motor_id, theta_col, args.sensor_col,
            rolling=args.rolling, gap_ms=args.gap_ms, active_dir=mdir,
            dmotor_min=args.dmotor_min, dtheta_min=args.dtheta_min
        )
        act_csv = os.path.join(args.out, f"R_theta_{phase}_active.csv")
        act.to_csv(act_csv, index=False, encoding="utf-8-sig")

        png = os.path.join(args.out, f"R_theta_{phase}.png")
        title = f"{args.finger.capitalize()} {phase}: {args.sensor_col}(norm) → {theta_col}(deg)"
        res = fit_phase(act, args.sensor_col, theta_col, png, title)
        if res is None or res.get("note") == "flat_theta":
            continue

        row = {"phase": phase, "plot": png, "active_csv": act_csv, **res}
        rows.append(row)

        # Write into JSON
        model_json["phases"][phase] = {
            "sensor_norm_lo": res["sensor_norm_lo"],
            "sensor_norm_hi": res["sensor_norm_hi"],
            "quadratic": {"a": res["quadratic_coeffs"][0],
                          "b": res["quadratic_coeffs"][1],
                          "c": res["quadratic_coeffs"][2]},
            "linear":    {"a": res["linear_coeffs"][0],
                          "b": res["linear_coeffs"][1]},
            "metrics":   {"R2_lin": res["R2_linear"], "MSE_lin": res["MSE_linear"],
                          "R2_quad": res["R2_quadratic"], "MSE_quad": res["MSE_quadratic"]}
        }

    # Export
    summ = pd.DataFrame(rows)
    summ_csv = os.path.join(args.out, "R_theta_direct_summary.csv")
    summ.to_csv(summ_csv, index=False, encoding="utf-8-sig")

    js_path = os.path.join(args.out, "R_theta_direct_model.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved:\n- {summ_csv}\n- {js_path}")
    for r in rows:
        print(f"  [{r['phase']}] R2_quad={r['R2_quadratic']:.3f}  -> {r['plot']}")

if __name__ == "__main__":
    main()
