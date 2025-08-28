# model_L_theta_transfer_fit.py
# ------------------------------------------------------------
# Cross-finger transfer for Middle finger (θ-L):
#  - Use Index's θ(L) as PRIOR (from L_theta_summary.csv or fit on-the-fly)
#  - On Middle's aligned CSV, extract active segments per phase and compute L_rel
#  - Affine correction (two-point minmax OR auto least-squares if middle angles exist)
#  - Fit final quadratic θ_mid(L_rel) per phase; export plots + summary CSV
#
# CMD:
#   Method A: use Index summary as prior + auto (middle has angle columns)
#   python model_L_theta_middle.py ^
#     --middle-csv "C:\Users\nicey\Desktop\Soft_glove\output\middle_angles\aligned_data_middle.csv" ^
#     --index-summary "C:\Users\nicey\Desktop\Soft_glove\models\index\L_theta_summary.csv" ^
#     --out "C:\Users\nicey\Desktop\Soft_glove\models\middle" ^
#     --flex-id 3 --extend-id 6 --mode auto
#
#   Method B: no reliable middle angles → use two-point anchors (min/max)
#   python model_L_theta_middle.py ^
#     --middle-csv "C:\Users\nicey\Desktop\Soft_glove\output\middle_angles\aligned_data_middle.csv" ^
#     --index-summary "C:\Users\nicey\Desktop\Soft_glove\models\index\L_theta_summary.csv" ^
#     --out "C:\Users\nicey\Desktop\Soft_glove\models\middle" ^
#     --flex-id 3 --extend-id 6 --mode minmax ^
#     --theta-min 5 --theta-max 80
#
#   Method C: if no index summary, fit index prior online from index CSV
#   python model_L_theta_middle.py ^
#     --middle-csv "C:\...\aligned_data_middle.csv" ^
#     --index-csv "C:\Users\nicey\Desktop\Soft_glove\output\index_angles\aligned_data_index.csv" ^
#     --out "C:\Users\nicey\Desktop\Soft_glove\models\middle" ^
#     --flex-id 3 --extend-id 6 --mode auto
# ------------------------------------------------------------

import os, argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- utilities ----------
def parse_quad_coeff_str(s):
    # from "a, b, c" to [a,b,c]
    parts = [float(x.strip()) for x in str(s).split(",")]
    if len(parts) != 3:
        raise ValueError(f"Bad quadratic_coeffs string: {s}")
    return parts

def poly_metrics(x, y, coeffs):
    yhat = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    mse = float(np.mean((y - yhat) ** 2))
    return r2, mse

def fit_quad(x, y):
    c = np.polyfit(x, y, deg=2)
    r2, mse = poly_metrics(x, y, c)
    return c, r2, mse

def list_to_str(v):
    return ", ".join([f"{x:.6g}" for x in v])

# ---------- active-segment extraction ----------
def extract_active_segments(df, motor_id, angle_cols_for_exist_check,
                            r_cm=1.0, rolling_window=5, gap_ms=5000,
                            motor_dir="neg", dmotor_min=0.02, dtheta_min=None, theta_main_col=None):
    """
    - Filter by MotorID
    - Smooth motor angle, derivative, keep direction (neg/pos)
    - Remove near-static motor points by |dmotor| >= dmotor_min
    - Compute L_cm, segment cycles by time gaps; zero to L_rel_cm within each cycle
    - (optional) if dtheta_min provided and theta_main_col present, also require |dθ| >= dtheta_min
    """
    need = ["Time(ms)","MotorID","Angle(deg)"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    for c in angle_cols_for_exist_check:
        if c not in df.columns:
            raise ValueError(f"Missing angle column: {c}")

    sub = df[df["MotorID"] == motor_id].copy().sort_values("Time(ms)").reset_index(drop=True)
    sub["angle_smooth"] = sub["Angle(deg)"].rolling(window=rolling_window, min_periods=1).mean()
    sub["dmotor"] = sub["angle_smooth"].diff()

    if motor_dir == "neg":
        sub = sub[sub["dmotor"] < 0]
    elif motor_dir == "pos":
        sub = sub[sub["dmotor"] > 0]
    else:
        raise ValueError("motor_dir must be 'neg' or 'pos'")

    if dmotor_min is not None:
        sub = sub[sub["dmotor"].abs() >= float(dmotor_min)]

    # cable length & cycles
    sub["L_cm"] = np.radians(sub["angle_smooth"]) * float(r_cm)
    dt = sub["Time(ms)"].diff().fillna(0.0)
    sub["cycle"] = (dt > float(gap_ms)).cumsum()
    sub["L_rel_cm"] = sub["L_cm"] - sub.groupby("cycle")["L_cm"].transform("first")

    # (optional) require joint angle moving as well
    if dtheta_min is not None and theta_main_col is not None and theta_main_col in sub.columns:
        sub["dtheta"] = sub[theta_main_col].diff().abs()
        sub = sub[sub["dtheta"] >= float(dtheta_min)]

    return sub

# ---------- PRIOR: index θ(L) ----------
def load_index_prior_from_summary(summary_csv, ycol="θ_total", phase="Flex"):
    df = pd.read_csv(summary_csv, encoding="utf-8-sig")
    # Expected columns: phase, ycol, quadratic_coeffs
    df2 = df[(df["ycol"] == ycol) & (df["phase"] == phase)]
    if df2.empty:
        raise ValueError(f"Not found prior in {summary_csv} for ycol={ycol}, phase={phase}")
    coeff_str = df2.iloc[0]["quadratic_coeffs"]
    coeffs = parse_quad_coeff_str(coeff_str)
    return coeffs  # [a,b,c]

def fit_index_prior_from_csv(index_csv, motor_id, motor_dir="neg",
                             r_cm=1.0, rolling=5, gap_ms=5000, dmotor_min=0.02):
    df = pd.read_csv(index_csv, encoding="utf-8-sig")
    if "∠ABC_smooth" not in df.columns or "∠BCD_smooth" not in df.columns:
        raise ValueError("Index CSV needs ∠ABC_smooth and ∠BCD_smooth")
    df["θ_total"] = df["∠ABC_smooth"] + df["∠BCD_smooth"]
    act = extract_active_segments(df, motor_id, ["θ_total"], r_cm, rolling, gap_ms, motor_dir, dmotor_min)
    x = act["L_rel_cm"].to_numpy()
    y = act["θ_total"].to_numpy()
    if len(x) < 10:
        raise ValueError("Too few samples to fit index prior.")
    c, r2, mse = fit_quad(x, y)
    return c, r2, mse, act

# ---------- affine correction ----------
def affine_by_minmax(theta_hat0, theta_min, theta_max):
    # map [min_hat0, max_hat0] -> [theta_min, theta_max]
    t0_min = np.nanmin(theta_hat0); t0_max = np.nanmax(theta_hat0)
    denom = (t0_max - t0_min) if (t0_max - t0_min) != 0 else 1e-9
    a = (theta_max - theta_min) / denom
    b = theta_min - a * t0_min
    return float(a), float(b), dict(t0_min=float(t0_min), t0_max=float(t0_max))

def affine_by_leastsq(theta_hat0, theta_mid):
    m = np.isfinite(theta_hat0) & np.isfinite(theta_mid)
    x = theta_hat0[m]; y = theta_mid[m]
    if len(x) < 5:
        raise ValueError("Not enough paired samples for least-squares affine.")
    A = np.c_[x, np.ones_like(x)]
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b), dict(N=int(len(x)))

# ---------- plotting ----------
def plot_phase(ax, df_phase, xcol, ycol_scatter, prior_coeffs, a, b, title):
    # scatter per cycle
    for cyc, g in df_phase.groupby("cycle"):
        ax.scatter(g[xcol], g[ycol_scatter], s=12, alpha=0.7, label=f"cycle {int(cyc)}")
    xs = np.linspace(np.nanmin(df_phase[xcol]), np.nanmax(df_phase[xcol]), 400)
    # priorθ
    prior_line = np.polyval(prior_coeffs, xs)
    ax.plot(xs, prior_line, linewidth=2, label="Index prior θ(L)")
    # correctedθ
    corr_line = a*prior_line + b
    ax.plot(xs, corr_line, linewidth=2, label="After affine: θ_mid(L)")
    ax.set_xlabel("L_rel (cm)")
    ax.set_ylabel("θ (deg)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--middle-csv", required=True, help="aligned_data_middle.csv")
    ap.add_argument("--out", required=True, help="output folder for middle models")
    ap.add_argument("--flex-id", type=int, required=True, help="middle flex motor id (e.g., 3)")
    ap.add_argument("--extend-id", type=int, required=True, help="middle extend motor id (e.g., 6)")
    # PRIOR source: either summary or fit online from index CSV
    ap.add_argument("--index-summary", default="", help="Index L_theta_summary.csv (prior)")
    ap.add_argument("--index-csv", default="", help="aligned_data_index.csv (fit prior online if summary not given)")
    ap.add_argument("--index-flex-id", type=int, default=2)
    ap.add_argument("--index-extend-id", type=int, default=5)
    # Transfer correction mode
    ap.add_argument("--mode", choices=["auto","minmax"], default="auto",
                    help="auto: LS using middle θ; minmax: two-point anchors")
    ap.add_argument("--theta-min", type=float, default=None, help="middle ROM minimum angle (deg) for minmax")
    ap.add_argument("--theta-max", type=float, default=None, help="middle ROM maximum angle (deg) for minmax")
    # Preprocessing and filtering
    ap.add_argument("--r-cm", type=float, default=1.0)
    ap.add_argument("--rolling", type=int, default=5)
    ap.add_argument("--gap-ms", type=float, default=5000)
    ap.add_argument("--dmotor-min", type=float, default=0.02)
    ap.add_argument("--dtheta-min", type=float, default=None, help="optional θ derivative threshold")
    ap.add_argument("--flex-dir", default="neg", choices=["neg","pos"])
    ap.add_argument("--extend-dir", default="neg", choices=["neg","pos"])
    args = ap.parse_args()

    ensure_dir(args.out)

    # ---------- load middle ----------
    df_mid = pd.read_csv(args.middle_csv, encoding="utf-8-sig")
    # prepare θ_total if exists
    theta_cols_exist = [c for c in ["∠ABC_smooth","∠BCD_smooth"] if c in df_mid.columns]
    has_theta = len(theta_cols_exist) == 2
    if has_theta:
        df_mid["θ_total"] = df_mid["∠ABC_smooth"] + df_mid["∠BCD_smooth"]

    # ---------- load/fit index prior per phase ----------
    priors = {}
    prior_info = {}

    def get_prior(phase, idx_motor_id, idx_motor_dir):
        if args.index_summary:
            coeffs = load_index_prior_from_summary(args.index_summary, ycol="θ_total", phase=phase)
            pri_info = {"from": "summary", "phase": phase, "coeffs": coeffs}
            return coeffs, pri_info
        elif args.index_csv:
            coeffs, r2, mse, act = fit_index_prior_from_csv(
                args.index_csv, motor_id=idx_motor_id, motor_dir=idx_motor_dir,
                r_cm=args.r_cm, rolling=args.rolling, gap_ms=args.gap_ms, dmotor_min=args.dmotor_min
            )
            pri_info = {"from": "index_csv", "phase": phase, "coeffs": coeffs, "fit_r2": r2, "fit_mse": mse, "N": int(len(act))}
            return coeffs, pri_info
        else:
            raise ValueError("Provide either --index-summary or --index-csv for prior.")

    priors["Flex"], prior_info["Flex"] = get_prior("Flex", args.index_flex_id, args.flex_dir)
    priors["Extend"], prior_info["Extend"] = get_prior("Extend", args.index_extend_id, args.extend_dir)

    # ---------- per phase process on middle ----------
    rows = []
    for phase, motor_id, motor_dir in [("Flex", args.flex_id, args.flex_dir),
                                       ("Extend", args.extend_id, args.extend_dir)]:
        angle_exist_cols = ["∠ABC_smooth","∠BCD_smooth"] if has_theta else []
        theta_main_col = "θ_total" if has_theta else None
        act = extract_active_segments(
            df_mid, motor_id, angle_exist_cols,
            r_cm=args.r_cm, rolling_window=args.rolling, gap_ms=args.gap_ms,
            motor_dir=motor_dir, dmotor_min=args.dmotor_min,
            dtheta_min=args.dtheta_min, theta_main_col=theta_main_col
        )
        act_csv = os.path.join(args.out, f"middle_{phase}_active.csv")
        act.to_csv(act_csv, index=False, encoding="utf-8-sig")

        if act.empty or act["L_rel_cm"].notna().sum() < 10:
            print(f"[WARN] Middle {phase}: active points too few, skip.")
            continue

        L = act["L_rel_cm"].to_numpy()
        # prior θ from index
        prior_coeffs = priors[phase]
        theta_hat0 = np.polyval(prior_coeffs, L)

        # ----- affine correction -----
        if args.mode == "auto" and has_theta:
            theta_mid = act["θ_total"].to_numpy()
            a, b, info = affine_by_leastsq(theta_hat0, theta_mid)
            corr_source = "auto_ls"
        else:
            if args.theta_min is None or args.theta_max is None:
                raise ValueError("mode=minmax requires --theta-min and --theta-max")
            a, b, info = affine_by_minmax(theta_hat0, args.theta_min, args.theta_max)
            corr_source = "minmax"

        theta_mid_est = a * theta_hat0 + b

        # ----- final fit (middle's own θ vs L_rel) -----
        c_mid, r2_mid, mse_mid = fit_quad(L, theta_mid_est)

        # ----- plots -----
        fig, ax = plt.subplots(figsize=(10,6))
        # If ground-truth θ exists, scatter with true values; otherwise use estimated θ_mid_est
        y_scatter = act["θ_total"].to_numpy() if has_theta else theta_mid_est
        title = f"Middle {phase}: transfer & affine correction\n" \
                f"prior={list_to_str(prior_coeffs)}; a={a:.4f}, b={b:.4f}; final={list_to_str(c_mid)}"
        for cyc, g in act.groupby("cycle"):
            ax.scatter(g["L_rel_cm"], (g["θ_total"] if has_theta else (a*np.polyval(prior_coeffs, g['L_rel_cm']) + b)),
                       s=12, alpha=0.7, label=f"cycle {int(cyc)}")
        xs = np.linspace(np.nanmin(L), np.nanmax(L), 400)
        ax.plot(xs, np.polyval(prior_coeffs, xs), linewidth=2, label="Index prior θ(L)")
        ax.plot(xs, a*np.polyval(prior_coeffs, xs) + b, linewidth=2, label="Affine corrected θ_mid(L)")
        ax.plot(xs, np.polyval(c_mid, xs), linewidth=2, label=f"Final fit θ_mid(L) (R²={r2_mid:.3f})")
        ax.set_xlabel("L_rel (cm)"); ax.set_ylabel("θ (deg)"); ax.grid(True); ax.legend()
        ax.set_title(title)
        png = os.path.join(args.out, f"middle_{phase}_transfer_fit.png")
        plt.tight_layout(); plt.savefig(png, dpi=300); plt.close(fig)

        # ----- log row -----
        row = {
            "phase": phase,
            "prior_from": prior_info[phase]["from"],
            "prior_coeffs(a,b,c)": list_to_str(prior_coeffs),
            "affine_source": corr_source,
            "affine_a": a, "affine_b": b,
            "affine_info": json.dumps(info, ensure_ascii=False),
            "final_quad_coeffs(a,b,c)": list_to_str(c_mid),
            "final_R2": r2_mid, "final_MSE": mse_mid,
            "N_points": int(len(L)),
            "active_csv": act_csv,
            "plot": png
        }
        if "fit_r2" in prior_info[phase]:
            row["prior_fit_R2(on_index)"] = prior_info[phase]["fit_r2"]
            row["prior_fit_MSE(on_index)"] = prior_info[phase]["fit_mse"]
            row["prior_fit_N(on_index)"] = prior_info[phase]["N"]
        rows.append(row)

    # save summary
    summary = pd.DataFrame(rows)
    out_csv = os.path.join(args.out, "L_theta_middle_transfer_summary.csv")
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved summary to: {out_csv}")
    for _, r in summary.iterrows():
        print(f"  [{r['phase']}] prior={r['prior_coeffs(a,b,c)']}  affine a={r['affine_a']:.4f}, b={r['affine_b']:.4f}  "
              f"final={r['final_quad_coeffs(a,b,c)']}  R²={r['final_R2']:.3f}")

if __name__ == "__main__":
    main()
