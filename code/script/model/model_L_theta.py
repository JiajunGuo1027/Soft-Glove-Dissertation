# model_L_theta.py — 三指通用（Index / Middle / Thumb）
# ------------------------------------------------------------
# Direct modeling of L <-> θ for one finger.
# - Reads aligned_data_*.csv (already time-aligned)
# - Extracts active segments per phase (by motor direction)
# - Uses per-cycle zeroing to get L_rel (cm)
# - Fits Linear & Quadratic θ = f(L_rel) per phase
# - Exports plots, CSV summary, and a JSON model for controller use
#
# Usage (Windows CMD)：
#   Index finger:
#     python model_L_theta.py ^
#       --csv "D:\Soft_glove\output\index_angles\aligned_data_index.csv" ^
#       --finger index --flex-id 2 --extend-id 5 ^
#       --out "D:\Soft_glove\models\index\L_theta"
#
#   Middle finger:
#     python model_L_theta.py ^
#       --csv "D:\Soft_glove\output\middle_angles\aligned_data_middle.csv" ^
#       --finger middle --flex-id 3 --extend-id 6 ^
#       --out "D:\Soft_glove\models\middle\L_theta"
#
#   Thumb:
#     python model_L_theta.py ^
#       --csv "D:\Soft_glove\output\thumb_angles\aligned_data_thumb.csv" ^
#       --finger thumb --flex-id 1 --extend-id 4 ^
#       --out "D:\Soft_glove\models\thumb\L_theta"
# ------------------------------------------------------------

import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ---------- utils ----------

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
    active_dir: str = "neg",
    dmotor_min: float = 0.02,
    dtheta_min: float | None = None,
    theta_main: str | None = None,
) -> pd.DataFrame:
    """
    - Filter MotorID == motor_id
    - Smooth motor angle & derivative; keep direction (neg: decreasing angle)
    - Remove near-static motor points
    - Compute cable length & per-cycle L_rel
    - (optional) also require |dθ| >= dtheta_min on theta_main
    """
    need = ["Time(ms)", "MotorID", "Angle(deg)"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    for c in theta_cols_exist:
        if c not in df.columns:
            raise ValueError(f"Missing angle column: {c}")

    sub = (
        df[df["MotorID"] == motor_id]
        .copy()
        .sort_values("Time(ms)")
        .reset_index(drop=True)
    )
    if sub.empty:
        return sub

    # motor smoothing & derivative
    sub["angle_smooth"] = sub["Angle(deg)"].rolling(window=rolling, min_periods=1).mean()
    sub["dmotor"] = sub["angle_smooth"].diff()

    # direction filter
    if active_dir == "neg":
        sub = sub[sub["dmotor"] < 0]
    elif active_dir == "pos":
        sub = sub[sub["dmotor"] > 0]
    else:
        raise ValueError("active_dir must be 'neg' or 'pos'")

    # motor derivative threshold
    if dmotor_min is not None:
        sub = sub[sub["dmotor"].abs() >= float(dmotor_min)]

    if sub.empty:
        return sub

    # cable length (cm) & cycles & per-cycle zeroing
    sub["L_cm"] = np.radians(sub["angle_smooth"]) * float(r_cm)
    dt = sub["Time(ms)"].diff().fillna(0.0)
    sub["cycle"] = (dt > float(gap_ms)).cumsum()
    sub["L_rel_cm"] = sub["L_cm"] - sub.groupby("cycle")["L_cm"].transform("first")

    # optional: require theta also moving
    if dtheta_min is not None and theta_main is not None and theta_main in sub.columns:
        sub["dtheta"] = sub[theta_main].diff().abs()
        sub = sub[sub["dtheta"] >= float(dtheta_min)]

    return sub


# ---------- plotting ----------

def plot_fit(
    df_phase: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    out_png: str,
    c_lin,
    c_quad,
    r2_lin: float,
    r2_quad: float,
) -> None:
    plt.figure(figsize=(10, 6))
    for cyc, g in df_phase.groupby("cycle"):
        plt.scatter(g[xcol], g[ycol], s=12, alpha=0.7, label=f"cycle {int(cyc)}")
    xs = np.linspace(np.nanmin(df_phase[xcol]), np.nanmax(df_phase[xcol]), 400)
    plt.plot(xs, np.polyval(c_lin, xs), linewidth=2, label=f"Linear R²={r2_lin:.3f}")
    plt.plot(xs, np.polyval(c_quad, xs), linewidth=2, label=f"Quadratic R²={r2_quad:.3f}")
    plt.xlabel("L_rel (cm)")
    plt.ylabel(f"{ycol} (deg)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="aligned_data_*.csv (index/middle/thumb)")
    ap.add_argument("--finger", required=True, choices=["index", "middle", "thumb"])
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--flex-id", type=int, required=True)
    ap.add_argument("--extend-id", type=int, required=True)
    # options
    ap.add_argument("--r-cm", type=float, default=1.0)
    ap.add_argument("--rolling", type=int, default=5)
    ap.add_argument("--gap-ms", type=float, default=5000)
    ap.add_argument("--flex-dir", default="neg", choices=["neg", "pos"], help="Flex active direction filter")
    ap.add_argument("--extend-dir", default="neg", choices=["neg", "pos"], help="Extend active direction filter")
    ap.add_argument("--dmotor-min", type=float, default=0.02, help="min |Δmotor| to keep a point")
    ap.add_argument(
        "--dtheta-min",
        type=float,
        default=None,
        help="optional θ derivative threshold on main θ to filter static periods",
    )
    args = ap.parse_args()

    ensure_dir(args.out)
    df = pd.read_csv(args.csv, encoding="utf-8-sig")

    # choose theta columns by finger
    if args.finger in ("index", "middle"):
        # Index/middle：PIP, DIP and total angle
        if "∠ABC_smooth" not in df.columns or "∠BCD_smooth" not in df.columns:
            raise ValueError("Expect ∠ABC_smooth and ∠BCD_smooth in CSV for index/middle.")
        df["θ_total"] = df["∠ABC_smooth"] + df["∠BCD_smooth"]
        theta_targets = ["θ_total", "∠ABC_smooth", "∠BCD_smooth"]  # Mainly promote θ_total, and use the rest for comparison
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
            df,
            motor_id,
            theta_exist_cols,
            r_cm=args.r_cm,
            rolling=args.rolling,
            gap_ms=args.gap_ms,
            active_dir=mdir,
            dmotor_min=args.dmotor_min,
            dtheta_min=args.dtheta_min,
            theta_main=theta_main,
        )

        act_csv = os.path.join(args.out, f"{args.finger}_{phase}_active.csv")
        act.to_csv(act_csv, index=False, encoding="utf-8-sig")

        if act.empty or act["L_rel_cm"].notna().sum() < 10:
            print(f"[WARN] {phase}: too few active points; skip.")
            continue

        model_json["phases"][phase] = {"targets": {}}

        for ycol in theta_targets:
            # Clean NaN
            m = np.isfinite(act["L_rel_cm"]) & np.isfinite(act[ycol])
            g = act.loc[m, ["L_rel_cm", ycol, "cycle"]].copy()
            if len(g) < 10:
                continue

            # flat detection (No fitting if the variance of θ is too small)
            if float(np.var(g[ycol].to_numpy())) < 1e-6:
                print(f"[INFO] {phase}/{ycol}: flat segment; skip fitting.")
                continue

            
            c1, r2_1, mse_1 = _polyfit_metrics(
                g["L_rel_cm"].to_numpy(), g[ycol].to_numpy(), 1
            )
            c2, r2_2, mse_2 = _polyfit_metrics(
                g["L_rel_cm"].to_numpy(), g[ycol].to_numpy(), 2
            )

            # drawing
            png = os.path.join(args.out, f"{args.finger}_{phase}_{ycol}_vs_Lrel.png")
            plot_fit(
                g,
                "L_rel_cm",
                ycol,
                f"{args.finger.capitalize()} {phase}: {ycol} vs L_rel",
                png,
                c1,
                c2,
                r2_1,
                r2_2,
            )

            # Record to the CSV line
            row = {
                "finger": args.finger,
                "phase": phase,
                "ycol": ycol,
                "linear_coeffs(a,b)": _list2str(c1),
                "quadratic_coeffs(a,b,c)": _list2str(c2),
                "R2_linear": r2_1,
                "MSE_linear": mse_1,
                "R2_quadratic": r2_2,
                "MSE_quadratic": mse_2,
                "plot": png,
                "active_csv": act_csv,
                "N": int(len(g)),
            }
            rows.append(row)

            # Write to JSON
            Lmin = float(np.nanmin(g["L_rel_cm"]))
            Lmax = float(np.nanmax(g["L_rel_cm"]))
            model_json["phases"][phase]["targets"][ycol] = {
                "linear": {"a": float(c1[0]), "b": float(c1[1])},
                "quadratic": {
                    "a": float(c2[0]),
                    "b": float(c2[1]),
                    "c": float(c2[2]),
                },
                "metrics": {
                    "R2_lin": float(r2_1),
                    "MSE_lin": float(mse_1),
                    "R2_quad": float(r2_2),
                    "MSE_quad": float(mse_2),
                },
                "L_rel_range": {"min": Lmin, "max": Lmax},
            }

    # Export summary CSV
    summary = pd.DataFrame(rows)
    out_csv = os.path.join(args.out, "L_theta_direct_summary.csv")
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # Export JSON
    out_json = os.path.join(args.out, "L_theta_direct_model.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(model_json, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved:\n- {out_csv}\n- {out_json}")
    if not summary.empty:
        for _, r in summary.iterrows():
            print(
                f"  [{r['finger']}/{r['phase']}/{r['ycol']}] "
                f"R2_lin={r['R2_linear']:.3f} R2_quad={r['R2_quadratic']:.3f} -> {r['plot']}"
            )
    else:
        print("[WARN] No fits produced. Check active-direction / thresholds / data coverage.")


if __name__ == "__main__":
    main()
