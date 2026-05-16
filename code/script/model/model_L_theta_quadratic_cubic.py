# model_L_theta_quadratic_cubic.py
# ------------------------------------------------------------
# Direct modeling of L -> θ with Linear / Quadratic / Cubic fits (per phase).
# Input: aligned_data_*.csv (index/middle: ∠ABC_smooth + ∠BCD_smooth; thumb: ∠ABD_smooth)
# Output: plots, L_theta_direct_summary.csv, L_theta_direct_model.json
#
# Index:
#   python model_L_theta_quadratic_cubic.py ^
#     --csv "D:\Soft_glove\output\index_angles\aligned_data_index.csv" ^
#     --finger index --flex-id 2 --extend-id 5 ^
#     --out "D:\Soft_glove\models\qu_cubic\index\L_theta"

# python model_L_theta_quadratic_cubic.py --csv "D:\Soft_glove\output\index_angles\aligned_data_index.csv" --finger index --flex-id 2 --extend-id 5 --out "D:\Soft_glove\models\qu_cubic\index\L_theta"

#   Middle:
    # python model_L_theta_cubic.py ^
    #   --csv "D:\Soft_glove\output\middle_angles\aligned_data_middle.csv" ^
    #   --finger middle --flex-id 3 --extend-id 6 ^
    #   --out "D:\Soft_glove\models\cubic\middle\L_theta"
#
#   Thumb:
    # python model_L_theta_cubic.py ^
    #   --csv "D:\Soft_glove\output\thumb_angles\aligned_data_thumb.csv" ^
    #   --finger thumb --flex-id 1 --extend-id 4 ^
    #   --out "D:\Soft_glove\models\cubic\thumb\L_theta"
# ------------------------------------------------------------

import os, argparse, json
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

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# ---------- metrics ----------
def _polyfit_metrics(x: np.ndarray, y: np.ndarray, deg: int):
    c = np.polyfit(x, y, deg=deg)
    yhat = np.polyval(c, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mse = float(np.mean((y - yhat) ** 2))
    return c, r2, mse

def _list2str(v) -> str:
    return ", ".join([f"{float(x):.6g}" for x in v])

# ---------- active-segment extraction ----------
def extract_active(
    df: pd.DataFrame,
    motor_id: int,
    theta_cols_exist,
    *,
    r_cm: float = 1.0,
    rolling: int = 5,
    gap_ms: float = 5000,
    active_dir: str = "neg",  # neg: angle decreases; pos: increases
    dmotor_min: float = 0.02,
    dtheta_min: float | None = None,
    theta_main: str | None = None,
) -> pd.DataFrame:
    need = ["Time(ms)", "MotorID", "Angle(deg)"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    for c in theta_cols_exist:
        if c not in df.columns:
            raise ValueError(f"Missing angle column: {c}")

    sub = df[df["MotorID"] == motor_id].copy().sort_values("Time(ms)").reset_index(drop=True)
    if sub.empty: return sub

    # Motor smoothing & derivative
    sub["angle_smooth"] = sub["Angle(deg)"].rolling(window=rolling, min_periods=1).mean()
    sub["dmotor"] = sub["angle_smooth"].diff()

    # Direction filter
    if active_dir == "neg":
        sub = sub[sub["dmotor"] < 0]
    elif active_dir == "pos":
        sub = sub[sub["dmotor"] > 0]
    else:
        raise ValueError("active_dir must be 'neg' or 'pos'")

    # Remove near-static motor points
    if dmotor_min is not None:
        sub = sub[sub["dmotor"].abs() >= float(dmotor_min)]
    if sub.empty: return sub

    # Cable length (cm), cycles, per-cycle zeroing
    sub["L_cm"] = np.radians(sub["angle_smooth"]) * float(r_cm)
    dt = sub["Time(ms)"].diff().fillna(0.0)
    sub["cycle"] = (dt > float(gap_ms)).cumsum()
    sub["L_rel_cm"] = sub["L_cm"] - sub.groupby("cycle")["L_cm"].transform("first")

    # (optional) also require |dθ| movement on main θ
    if dtheta_min is not None and theta_main is not None and theta_main in sub.columns:
        sub["dtheta"] = sub[theta_main].diff().abs()
        sub = sub[sub["dtheta"] >= float(dtheta_min)]

    return sub

# ---------- plotting ----------
def plot_fit_with_cubic(
    df_phase: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    out_png: str,
    c_lin, c_quad, c_cub,
    r2_lin: float, r2_quad: float, r2_cub: float,
) -> None:
    plt.figure(figsize=(10, 6))

    # Plot all cycle data with one legend entry only.
    first = True
    for cyc, g in df_phase.groupby("cycle"):
        plt.scatter(
            g[xcol], g[ycol],
            s=12, alpha=0.55,
            label="Experimental data" if first else None
        )
        first = False

    xs = np.linspace(np.nanmin(df_phase[xcol]), np.nanmax(df_phase[xcol]), 400)

    # Linear model is intentionally not shown.
    plt.plot(xs, np.polyval(c_quad, xs), linewidth=2.0,
             label=f"Quadratic, $R^2$={r2_quad:.3f}")
    plt.plot(xs, np.polyval(c_cub, xs), linewidth=2.0,
             label=f"Cubic, $R^2$={r2_cub:.3f}")

    plt.xlabel("$L_{rel}$ (cm)")
    plt.ylabel("Joint angle, $\\theta$ (deg)")
    plt.title(title)

    plt.grid(True, alpha=0.35)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()
# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="aligned_data_*.csv")
    ap.add_argument("--finger", required=True, choices=["index", "middle", "thumb"])
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--flex-id", type=int, required=True)
    ap.add_argument("--extend-id", type=int, required=True)
    # options
    ap.add_argument("--r-cm", type=float, default=1.0)
    ap.add_argument("--rolling", type=int, default=5)
    ap.add_argument("--gap-ms", type=float, default=5000)
    ap.add_argument("--flex-dir", default="neg", choices=["neg","pos"])
    ap.add_argument("--extend-dir", default="neg", choices=["neg","pos"])
    ap.add_argument("--dmotor-min", type=float, default=0.02)
    ap.add_argument("--dtheta-min", type=float, default=None)
    args = ap.parse_args()

    ensure_dir(args.out)
    df = pd.read_csv(args.csv, encoding="utf-8-sig")

    # Select θ columns by finger
    if args.finger in ("index", "middle"):
        if "∠ABC_smooth" not in df.columns or "∠BCD_smooth" not in df.columns:
            raise ValueError("Expect ∠ABC_smooth and ∠BCD_smooth in CSV for index/middle.")
        df["θ_total"] = df["∠ABC_smooth"] + df["∠BCD_smooth"]
        theta_targets = ["θ_total", "∠ABC_smooth", "∠BCD_smooth"]
        theta_main = "θ_total"
        theta_exist_cols = ["∠ABC_smooth", "∠BCD_smooth"]
    else:  # thumb
        if "∠ABD_smooth" not in df.columns:
            raise ValueError("Expect ∠ABD_smooth in CSV for thumb.")
        theta_targets = ["∠ABD_smooth"]
        theta_main = "∠ABD_smooth"
        theta_exist_cols = ["∠ABD_smooth"]

    rows = []
    model_json = {"finger": args.finger, "r_cm": float(args.r_cm), "phases": {}}

    for phase, motor_id, mdir in [
        ("Flex", args.flex_id, args.flex_dir),
        ("Extend", args.extend_id, args.extend_dir),
    ]:
        act = extract_active(
            df, motor_id, theta_exist_cols,
            r_cm=args.r_cm, rolling=args.rolling, gap_ms=args.gap_ms,
            active_dir=mdir, dmotor_min=args.dmotor_min,
            dtheta_min=args.dtheta_min, theta_main=theta_main,
        )

        act_csv = os.path.join(args.out, f"{args.finger}_{phase}_active.csv")
        act.to_csv(act_csv, index=False, encoding="utf-8-sig")

        if act.empty or act["L_rel_cm"].notna().sum() < 10:
            print(f"[WARN] {phase}: too few active points; skip."); continue

        model_json["phases"][phase] = {"targets": {}}

        for ycol in theta_targets:
            m = np.isfinite(act["L_rel_cm"]) & np.isfinite(act[ycol])
            g = act.loc[m, ["L_rel_cm", ycol, "cycle"]].copy()
            if len(g) < 10: continue
            if float(np.var(g[ycol].to_numpy())) < 1e-6:
                print(f"[INFO] {phase}/{ycol}: flat segment; skip."); continue

            x = g["L_rel_cm"].to_numpy()
            y = g[ycol].to_numpy()
            c1, r2_1, mse_1 = _polyfit_metrics(x, y, 1)
            c2, r2_2, mse_2 = _polyfit_metrics(x, y, 2)
            c3, r2_3, mse_3 = _polyfit_metrics(x, y, 3)

            png = os.path.join(args.out, f"{args.finger}_{phase}_{ycol}_vs_Lrel.png")
            plot_fit_with_cubic(
                g, "L_rel_cm", ycol,
                f"{args.finger.capitalize()} {phase}: {ycol} vs L_rel",
                png, c1, c2, c3, r2_1, r2_2, r2_3
            )

            row = {
                "finger": args.finger, "phase": phase, "ycol": ycol,
                "linear_coeffs(a,b)":      _list2str(c1),
                "quadratic_coeffs(a,b,c)": _list2str(c2),
                "cubic_coeffs(a,b,c,d)":   _list2str(c3),
                "R2_linear": r2_1,  "MSE_linear": mse_1,
                "R2_quadratic": r2_2, "MSE_quadratic": mse_2,
                "R2_cubic": r2_3,  "MSE_cubic": mse_3,
                "plot": png, "active_csv": act_csv, "N": int(len(g)),
            }
            rows.append(row)

            Lmin = float(np.nanmin(g["L_rel_cm"]))
            Lmax = float(np.nanmax(g["L_rel_cm"]))
            model_json["phases"][phase]["targets"][ycol] = {
                "linear":    {"a": float(c1[0]), "b": float(c1[1])},
                "quadratic": {"a": float(c2[0]), "b": float(c2[1]), "c": float(c2[2])},
                "cubic":     {"a": float(c3[0]), "b": float(c3[1]), "c": float(c3[2]), "d": float(c3[3])},
                "metrics": {
                    "R2_lin": float(r2_1),  "MSE_lin": float(mse_1),
                    "R2_quad": float(r2_2), "MSE_quad": float(mse_2),
                    "R2_cubic": float(r2_3),"MSE_cubic": float(mse_3),
                },
                "L_rel_range": {"min": Lmin, "max": Lmax},
            }

    summary = pd.DataFrame(rows)
    out_csv  = os.path.join(args.out, "L_theta_direct_summary.csv")
    out_json = os.path.join(args.out, "L_theta_direct_model.json")
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(model_json, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved:\n- {out_csv}\n- {out_json}")
    if not summary.empty:
        for _, r in summary.iterrows():
            print(f"  [{r['finger']}/{r['phase']}/{r['ycol']}] "
                  f"R2_lin={r['R2_linear']:.3f}  R2_quad={r['R2_quadratic']:.3f}  R2_cubic={r['R2_cubic']:.3f} -> {r['plot']}")
    else:
        print("[WARN] No fits produced. Check active-direction / thresholds / data coverage.")

if __name__ == "__main__":
    main()
