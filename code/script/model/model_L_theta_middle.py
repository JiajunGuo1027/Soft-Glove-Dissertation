# model_L_theta_middle.py
# ------------------------------------------------------------
# Direct modeling of L <-> θ for one finger (Index or Thumb).
# - Reads aligned_data_*.csv (already time-aligned)
# - Extracts active segments per phase (by motor direction)
# - Uses per-cycle zeroing to get L_rel (cm)
# - Fits Linear & Quadratic θ = f(L_rel) per phase
# - Exports plots, CSV summary, and a JSON model for controller use
#
# Usage (Windows CMD)：
#   Middle:
#   python model_L_theta_middle.py ^
#     --csv "D:\Soft_glove\output\middle_angles\aligned_data_middle.csv" ^
#     --flex-id 3 --extend-id 6 ^
#     --finger middle ^
#     --out "D:\Soft_glove\models\middle\L_theta"
# ------------------------------------------------------------

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def poly_metrics(x, y, deg):
    c = np.polyfit(x, y, deg=deg)
    yhat = np.polyval(c, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    mse = float(np.mean((y - yhat) ** 2))
    return c, r2, mse

def extract_active_segments(df, motor_id, angle_cols, r_cm=1.0,
                            rolling_window=5, time_gap_ms=5000,
                            active_dir="neg", deriv_abs_min=0.02):
    """Return a DataFrame containing only active segments, with L_rel_cm and cycle."""
    need = ["Time(ms)", "MotorID", "Angle(deg)"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    for a in angle_cols:
        if a not in df.columns:
            raise ValueError(f"Missing angle column: {a}")

    sub = df[df["MotorID"] == motor_id].copy()
    sub = sub.dropna(subset=["Angle(deg)"]).sort_values("Time(ms)").reset_index(drop=True)

    # Smooth motor angle & derivative
    sub["angle_smooth"] = sub["Angle(deg)"].rolling(window=rolling_window, min_periods=1).mean()
    sub["delta_motor"]  = sub["angle_smooth"].diff()

    # Direction filtering
    if active_dir == "neg":
        sub = sub[sub["delta_motor"] < 0]
    elif active_dir == "pos":
        sub = sub[sub["delta_motor"] > 0]
    else:
        raise ValueError("active_dir must be 'neg' or 'pos'")

    # Remove near-static points
    if deriv_abs_min is not None:
        sub = sub[sub["delta_motor"].abs() >= float(deriv_abs_min)]

    # Cable displacement L (cm) & cycle segmentation
    sub["L_cm"] = np.radians(sub["angle_smooth"]) * float(r_cm)
    dt = sub["Time(ms)"].diff().fillna(0.0)
    sub["cycle"] = (dt > float(time_gap_ms)).cumsum()

    # Zeroing within phase → L_rel
    sub["L_rel_cm"] = sub["L_cm"] - sub.groupby("cycle")["L_cm"].transform("first")

    # Keep only required angle columns
    keep = ["Time(ms)","MotorID","Angle(deg)","angle_smooth","delta_motor","cycle","L_cm","L_rel_cm"] + angle_cols
    return sub[keep]

def fit_and_plot(df_act, xcol, ycol, title, out_png):
    #  Keep only finite values
    x = df_act[xcol].to_numpy()
    y = df_act[ycol].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    plt.figure(figsize=(10,6))
    for cyc, g in df_act[m].groupby("cycle"):
        plt.scatter(g[xcol], g[ycol], s=10, alpha=0.6, label=f"cycle {int(cyc)}")

    # Exit if too few points
    if len(x) < 10:
        plt.title(f"{title}\n(no enough valid points: N={len(x)})")
        plt.xlabel("Relative cable displacement L_rel (cm)")
        plt.ylabel(f"{ycol} (deg)")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(out_png, dpi=300); plt.close()
        return None

    # If y is nearly constant (flat segment), print message and skip fitting
    y_var = float(np.var(y))
    if y_var < 1e-6:
        plt.title(f"{title}\n(flat segment: var≈0 → skip fitting)")
        plt.xlabel("Relative cable displacement L_rel (cm)")
        plt.ylabel(f"{ycol} (deg)")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(out_png, dpi=300); plt.close()
        return {
            "ycol": ycol, "linear_coeffs": [float("nan"), float("nan")],
            "quadratic_coeffs": [float("nan"), float("nan"), float("nan")],
            "R2_linear": float("nan"), "MSE_linear": float("nan"),
            "R2_quadratic": float("nan"), "MSE_quadratic": float("nan"),
            "plot": out_png
        }

    #Normal fitting
    c1 = np.polyfit(x, y, deg=1)
    c2 = np.polyfit(x, y, deg=2)
    def r2_mse(x, y, c):
        yhat = np.polyval(c, x)
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
        mse = float(np.mean((y - yhat)**2))
        return r2, mse
    r2_1, mse_1 = r2_mse(x, y, c1)
    r2_2, mse_2 = r2_mse(x, y, c2)

    xs = np.linspace(np.min(x), np.max(x), 300)
    plt.plot(xs, np.polyval(c1, xs), linewidth=2, label=f"Linear R²={r2_1:.3f}")
    plt.plot(xs, np.polyval(c2, xs), linewidth=2, label=f"Quadratic R²={r2_2:.3f}")
    plt.xlabel("Relative cable displacement L_rel (cm)")
    plt.ylabel(f"{ycol} (deg)")
    plt.title(title)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

    return {
        "ycol": ycol,
        "linear_coeffs": [float(c1[0]), float(c1[1])],
        "quadratic_coeffs": [float(c2[0]), float(c2[1]), float(c2[2])],
        "R2_linear": float(r2_1), "MSE_linear": float(mse_1),
        "R2_quadratic": float(r2_2), "MSE_quadratic": float(mse_2),
        "plot": out_png
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="aligned_data_*.csv")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--finger", required=True, choices=["index","middle","thumb"])
    ap.add_argument("--flex-id", type=int, required=True)
    ap.add_argument("--extend-id", type=int, required=True)
    # Direction rule
    ap.add_argument("--flex-dir",  default="neg", choices=["neg","pos"])
    ap.add_argument("--extend-dir",default="neg", choices=["neg","pos"])
    ap.add_argument("--r-cm", type=float, default=1.0)
    ap.add_argument("--rolling", type=int, default=5)
    ap.add_argument("--gap-ms", type=float, default=5000)
    ap.add_argument("--deriv-min", type=float, default=0.02)
    args = ap.parse_args()

    ensure_dir(args.out)
    df = pd.read_csv(args.csv, encoding="utf-8-sig")

    # Angle column definitions 
    if args.finger in ("index","middle"):
        # Index/Middle: provide PIP, DIP, and total angle as targets
        if "∠ABC_smooth" not in df.columns or "∠BCD_smooth" not in df.columns:
            raise ValueError("Expect ∠ABC_smooth and ∠BCD_smooth in CSV for index/middle.")
        df["θ_total"] = df["∠ABC_smooth"] + df["∠BCD_smooth"]
        angle_targets = ["θ_total", "∠ABC_smooth", "∠BCD_smooth"]  # 主推 θ_total，其余作对照
    else:
        # Thumb: IP angle
        if "∠ABD_smooth" not in df.columns:
            raise ValueError("Expect ∠ABD_smooth in CSV for thumb.")
        angle_targets = ["∠ABD_smooth"]

    rows = []
    # Fit models separately for two phases
    for motor_id, phase_label, rule in [
        (args.flex_id,   "Flex",   args.flex_dir),
        (args.extend_id, "Extend", args.extend_dir),
    ]:
        act = extract_active_segments(
            df, motor_id, angle_cols=angle_targets,
            r_cm=args.r_cm, rolling_window=args.rolling,
            time_gap_ms=args.gap_ms, active_dir=rule,
            deriv_abs_min=args.deriv_min
        )
        # Save active-segment data
        act_csv = os.path.join(args.out, f"{phase_label}_active.csv")
        act.to_csv(act_csv, index=False, encoding="utf-8-sig")

        for ycol in angle_targets:
            title = f"{args.finger.capitalize()} {phase_label}: {ycol} vs L_rel"
            png  = os.path.join(args.out, f"{args.finger}_{phase_label}_{ycol}_vs_Lrel.png")
            res = fit_and_plot(act, "L_rel_cm", ycol, title, png)
            if res:
                res.update({"phase": phase_label, "finger": args.finger, "N": int(len(act))})
                rows.append(res)

    # Save
    def cstr(v): 
        return ", ".join([f"{x:.6g}" for x in v]) if isinstance(v, (list,tuple)) else ""
    summ = pd.DataFrame(rows)
    if not summ.empty:
        summ["linear_coeffs"]    = summ["linear_coeffs"].apply(cstr)
        summ["quadratic_coeffs"] = summ["quadratic_coeffs"].apply(cstr)
    summ_path = os.path.join(args.out, "L_theta_summary.csv")
    summ.to_csv(summ_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved summary to: {summ_path}")
    for _, r in summ.iterrows():
        print(f"  [{r['finger']}/{r['phase']}/{r['ycol']}] "
              f"R2_lin={r['R2_linear']:.3f} R2_quad={r['R2_quadratic']:.3f} -> {r['plot']}")

if __name__ == "__main__":
    main()
