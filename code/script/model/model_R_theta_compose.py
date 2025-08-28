# model_R_theta_compose.py 
# --------------------------------------------------------------------
# Compose R->L (from JSON) with L->θ (JSON/CSV) to get R->θ.
# - Works for index/middle/thumb. Per-phase (Flex/Extend).
# - If ground-truth θ exists in aligned CSV (index: θ_total, thumb: ∠ABD_smooth),
#   evaluates RMSE/R² and can apply optional affine tweak (a,b).
# - Exports plots + CSV summary + a JSON with the two-stage model for controller.
#
# - Input:
#   * R->L JSON:    exported by R_L modeling script (includes per-phase norm and quadratic coeffs)
#   * L->θ JSON:    exported by model_L_theta.py (L_theta_direct_model.json)
#   * L->θ CSV:     L_theta_summary.csv (optional fallback)
# - Output:
#   * R_theta_composed_timeseries.csv / .png
#   * R_theta_composed_model.json (directly usable by controller)
#   * R_theta_summary.csv (includes evaluation & suggested affine parameters)
#
# Usage (Windows CMD):
#   Index:
# python model_R_theta_compose.py ^
#   --aligned-csv "D:\Soft_glove\output\index_angles\aligned_data_index.csv" ^
#   --r-l-json    "D:\Soft_glove\models\index\R_L\R_L_model_middle.json" ^
#   --thetaL-json "D:\Soft_glove\models\index\L_theta\L_theta_model.json" ^
#   --finger index --sensor-col SensorA1 --flex-id 2 --extend-id 5 ^
#   --out "D:\Soft_glove\models\index\R_theta_compose" ^
#   --affine-a 0.5643 --affine-b 23.9132
#
#   Thumb:
#   python model_R_theta_compose.py ^
#     --aligned-csv "D:\Soft_glove\output\thumb_angles\aligned_data_thumb.csv" ^
#     --r-l-json    "D:\Soft_glove\models\thumb\R_L\R_L_model_middle.json" ^
#     --thetaL-json "D:\Soft_glove\models\thumb\L_theta\L_theta_direct_model.json" ^
#     --finger thumb --sensor-col SensorA0 --flex-id 1 --extend-id 4 ^
#     --out "D:\Soft_glove\models\thumb\R_theta_compose"
#     --affine-a 1.0 --affine-b 0.0
#
#   Middle (no ground-truth θ, only outputs proxy θ without evaluation):
#   python model_R_theta_compose.py ^
#     --aligned-csv "D:\Soft_glove\output\middle_angles\aligned_data_middle.csv" ^
#     --r-l-json    "D:\Soft_glove\models\middle\R_L\R_L_model_middle.json" ^
#     --thetaL-json "D:\Soft_glove\models\middle\L_theta\L_theta_direct_model.json" ^
#     --finger middle --sensor-col SensorA2 --flex-id 3 --extend-id 6 ^
#     --out "D:\Soft_glove\models\middle\R_theta_compose" ^
#     --affine-a 1.0 --affine-b 0.0
# --------------------------------------------------------------------

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ---------------- I/O helpers ----------------

def load_r_l_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_thetaL_from_json(path: str, finger: str):
    """Read JSON exported by model_L_theta.py：
    {
      "finger": "index",
      "r_cm": 1.0,
      "phases": {
        "Flex": {
          "targets": {
            "θ_total": {
              "linear": {"a":..., "b":...},
              "quadratic": {"a":..., "b":..., "c":...},
              "metrics": {...},
              "L_rel_range": {"min":..., "max":...}
            }, ...
          }
        },
        "Extend": { ... }
      }
    }
    Returns: ({phase: [a,b,c]}, ycol_used)
    Preference: index/middle -> "θ_total"; thumb -> "∠ABD_smooth"; if not present, take the first target.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    phases = obj.get("phases", {})

    # Target ycol selection strategy
    prefer = "θ_total" if finger in ("index", "middle") else "∠ABD_smooth"

    # If any phase contains the preferred target, use it; otherwise fall back to the first target in that phase.
    def pick_target_name(phase_dict):
        tg = phase_dict.get("targets", {})
        if prefer in tg:
            return prefer
        if len(tg) == 0:
            return None
        return list(tg.keys())[0]

    # Unified ycol selection: prefer the preferred one; otherwise take the first target of Flex; if Flex not available, check Extend.
    ycol_used = None
    if "Flex" in phases:
        ycol_used = pick_target_name(phases["Flex"])
    if ycol_used is None and "Extend" in phases:
        ycol_used = pick_target_name(phases["Extend"])
    if ycol_used is None:
        raise ValueError(f"No targets found in {path}")

    out = {}
    for ph in ("Flex", "Extend"):
        if ph not in phases:
            continue
        tg = phases[ph].get("targets", {})
        tgt = tg.get(ycol_used)
        if tgt is None:
            # If the chosen ycol is missing for that phase, fall back to the first available target.
            if len(tg) == 0:
                continue
            first_name = list(tg.keys())[0]
            tgt = tg[first_name]
        q = tgt.get("quadratic", None)
        if q is None:
            # If quadratic not available, degrade linear into quadratic [0,a,b]
            lin = tgt.get("linear", None)
            if lin is None:
                continue
            a2, b2, c2 = 0.0, float(lin["a"]), float(lin["b"])  # θ = a2*L^2 + b2*L + c2
        else:
            a2, b2, c2 = float(q.get("a", 0.0)), float(q.get("b", 0.0)), float(q.get("c", 0.0))
        out[ph] = [a2, b2, c2]

    if not out:
        raise ValueError(f"No usable L->θ model found in JSON: {path}")
    return out, ycol_used


def load_thetaL_from_csv(summary_csv: str, finger: str):
    """read quadratic coefficients per phase from CSV
    Expected column names: 'quadratic_coeffs' or 'quadratic_coeffs(a,b,c)'
    Return: ({phase: [a,b,c]}, ycol_used)
    """
    df = pd.read_csv(summary_csv, encoding="utf-8-sig")
    # Compatible with different column names
    quad_col = None
    for name in ("quadratic_coeffs", "quadratic_coeffs(a,b,c)"):
        if name in df.columns:
            quad_col = name
            break
    if quad_col is None:
        raise ValueError(f"CSV missing quadratic coeffs column: {summary_csv}")

    ypref = "θ_total" if finger in ("index", "middle") else "∠ABD_smooth"
    ycol_used = ypref if quad_col in df.columns else None

    out = {}
    for phase in ["Flex", "Extend"]:
        # First filter by ycol, then by phase
        if "ycol" in df.columns:
            row = df[(df["phase"] == phase) & (df["ycol"].fillna("") == ypref)]
        else:
            row = df[(df["phase"] == phase)]
        if row.empty:
            continue
        coeffs_str = str(row.iloc[0][quad_col])
        coeffs = [float(x) for x in coeffs_str.replace("[","").replace("]","").split(",")]
        out[phase] = coeffs[:3]
    if not out:
        raise ValueError(f"No usable phase rows in CSV: {summary_csv}")
    return out, (ypref if ycol_used is None else ycol_used)


# ---------------- compose helpers ----------------

def phase_of_row(row, flex_id, extend_id):
    mid = int(row["MotorID"]) if np.isfinite(row.get("MotorID", np.nan)) else None
    if mid == flex_id:
        return "Flex"
    if mid == extend_id:
        return "Extend"
    if mid in (1, 2, 3):
        return "Flex"
    if mid in (4, 5, 6):
        return "Extend"
    return "Flex"


def estimate_L_from_R(row, rlj, sensor_col, phase):
    ph = rlj.get("phases", {}).get(phase, None)
    if ph is None:
        vals = list(rlj.get("phases", {}).values())
        if not vals:
            return np.nan
        ph = vals[0]
    lo, hi = float(ph.get("sensor_norm_lo", 0.0)), float(ph.get("sensor_norm_hi", 1.0))
    z = (row[sensor_col] - lo) / (hi - lo if hi > lo else 1e-9)
    a = float(ph.get("quadratic", {}).get("a", 0.0))
    b = float(ph.get("quadratic", {}).get("b", 0.0))
    c = float(ph.get("quadratic", {}).get("c", 0.0))
    return a * z * z + b * z + c


def compose_theta(thetaL_dict, L_est, phase):
    coeffs = thetaL_dict.get(phase)
    if coeffs is None:
        coeffs = list(thetaL_dict.values())[0]
    return float(np.polyval(coeffs, L_est))


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-csv", required=True)
    ap.add_argument("--r-l-json", required=True)
    ap.add_argument("--thetaL-json", required=False, default=None,
                    help="Path to L_theta_direct_model.json (recommended)")
    ap.add_argument("--thetaL-summary", required=False, default=None,
                    help="(Optional fallback) L_theta_summary.csv")

    ap.add_argument("--finger", required=True, choices=["index", "middle", "thumb"])
    ap.add_argument("--sensor-col", required=True, help="SensorA0/1/2")
    ap.add_argument("--flex-id", type=int, required=True)
    ap.add_argument("--extend-id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--affine-a", type=float, default=1.0)
    ap.add_argument("--affine-b", type=float, default=0.0)
    args = ap.parse_args()

    ensure_dir(args.out)

    df = pd.read_csv(args.aligned_csv, encoding="utf-8-sig")
    rlj = load_r_l_json(args.r_l_json)

    # Read L->θ (JSON first; otherwise, fall back to CSV)
    thetaL = None
    ycol_used = None
    if args.thetaL_json and os.path.isfile(args.thetaL_json):
        thetaL, ycol_used = load_thetaL_from_json(args.thetaL_json, args.finger)
    elif args.thetaL_summary and os.path.isfile(args.thetaL_summary):
        thetaL, ycol_used = load_thetaL_from_csv(args.thetaL_summary, args.finger)
    else:
        raise FileNotFoundError("Please provide --thetaL-json (preferred) or --thetaL-summary (fallback).")

    # ground-truth θ column name
    has_theta = False
    if args.finger in ("index", "middle"):
        if "∠ABC_smooth" in df.columns and "∠BCD_smooth" in df.columns:
            df["θ_total"] = df["∠ABC_smooth"] + df["∠BCD_smooth"]
            # By default, only index finger is used for evaluation (middle finger usually lacks reliable annotation)
            has_theta = (args.finger == "index")
            gt_col = "θ_total"
        else:
            gt_col = None
    else:
        gt_col = "∠ABD_smooth" if "∠ABD_smooth" in df.columns else None
        has_theta = (gt_col is not None and args.finger == "thumb")

    # Row-wise composition: R -> (R_L JSON) -> L_est -> (L_theta JSON/CSV) -> θ
    L_list, Th_list, Ph_list = [], [], []
    for _, row in df.iterrows():
        sval = row.get(args.sensor_col, np.nan)
        if not np.isfinite(sval):
            L_list.append(np.nan); Th_list.append(np.nan); Ph_list.append(None); continue
        phase = phase_of_row(row, args.flex_id, args.extend_id)
        L_est = estimate_L_from_R(row, rlj, args.sensor_col, phase)
        th = compose_theta(thetaL, L_est, phase)
        th = args.affine_a * th + args.affine_b  # Optional affine adjustment
        L_list.append(L_est); Th_list.append(th); Ph_list.append(phase)

    df_out = df.copy()
    df_out["L_est_cm"] = L_list
    df_out["theta_from_R(compose)"] = Th_list
    df_out["phase_used"] = Ph_list

    # valuation (if ground truth exists) and suggest affine a,b
    metrics = {}
    if has_theta and (gt_col is not None) and (gt_col in df_out.columns):
        m = np.isfinite(df_out["theta_from_R(compose)"]) & np.isfinite(df_out[gt_col])
        if m.sum() >= 20:
            y = df_out.loc[m, gt_col].to_numpy()
            yhat = df_out.loc[m, "theta_from_R(compose)"].to_numpy()
            rmse0 = float(np.sqrt(np.mean((y - yhat) ** 2)))
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2_0 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            # Affine parameters suggested by least squares fitting
            A = np.c_[yhat, np.ones_like(yhat)]
            a_opt, b_opt = np.linalg.lstsq(A, y, rcond=None)[0]
            yhat_opt = a_opt * yhat + b_opt
            rmse1 = float(np.sqrt(np.mean((y - yhat_opt) ** 2)))
            ss_res1 = float(np.sum((y - yhat_opt) ** 2))
            r2_1 = 1 - ss_res1 / ss_tot if ss_tot > 0 else float("nan")
            metrics = {
                "has_gt": True,
                "gt_col": gt_col,
                "RMSE_before": rmse0,
                "R2_before": r2_0,
                "suggest_a": float(a_opt),
                "suggest_b": float(b_opt),
                "RMSE_after": rmse1,
                "R2_after": r2_1,
            }
        else:
            metrics = {"has_gt": True, "gt_col": gt_col, "note": "too few valid pairs"}
    else:
        metrics = {"has_gt": False, "note": "no reliable ground-truth theta (proxy only)"}

    # Export CSV
    out_csv = os.path.join(args.out, "R_theta_composed_timeseries.csv")
    df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if "Time(ms)" in df_out.columns:
        t = df_out["Time(ms)"].to_numpy() / 1000.0
    else:
        t = np.arange(len(df_out))
    plt.figure(figsize=(12, 6))
    plt.plot(t, df_out["theta_from_R(compose)"], label="theta_from_R(compose)")
    if metrics.get("has_gt", False) and (gt_col in df_out.columns):
        plt.plot(t, df_out[gt_col], label=f"ground truth ({gt_col})", alpha=0.7)
    ttl = f"R->θ composed ({args.finger})"
    if "RMSE_before" in metrics:
        ttl += f" | RMSE={metrics['RMSE_before']:.2f}, R²={metrics['R2_before']:.3f}"
    plt.title(ttl)
    plt.xlabel("Time (s)"); plt.ylabel("θ (deg)"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "R_theta_composed_timeseries.png"), dpi=300)
    plt.close()

    # Export JSON for controller use
    export = {
        "finger": args.finger,
        "sensor_col": args.sensor_col,
        "affine": {"a": float(args.affine_a), "b": float(args.affine_b)},
        "R_to_L_model": rlj, 
        "L_to_theta_model": {"phase_quadratic": thetaL, "ycol_used": ycol_used},
        "motor_ids": {"flex_id": int(args.flex_id), "extend_id": int(args.extend_id)},
        "metrics": metrics,
    }
    out_json = os.path.join(args.out, "R_theta_composed_model.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    # Summary CSV
    summ = pd.DataFrame([
        {
            "finger": args.finger,
            "sensor_col": args.sensor_col,
            "affine_a": float(args.affine_a),
            "affine_b": float(args.affine_b),
            "has_gt": metrics.get("has_gt", False),
            "gt_col": metrics.get("gt_col", ""),
            "RMSE_before": metrics.get("RMSE_before", np.nan),
            "R2_before": metrics.get("R2_before", np.nan),
            "suggest_a": metrics.get("suggest_a", np.nan),
            "suggest_b": metrics.get("suggest_b", np.nan),
            "RMSE_after": metrics.get("RMSE_after", np.nan),
            "R2_after": metrics.get("R2_after", np.nan),
            "ycol_used": ycol_used,
            "thetaL_source": "JSON" if (args.thetaL_json and os.path.isfile(args.thetaL_json)) else "CSV",
        }
    ])
    summ.to_csv(os.path.join(args.out, "R_theta_summary.csv"), index=False, encoding="utf-8-sig")

    print(
        f"[DONE] Saved:\n- {out_csv}\n- {out_json}\n- {os.path.join(args.out,'R_theta_summary.csv')}"
    )
    if metrics.get("has_gt", False) and "suggest_a" in metrics:
        print(
            f"[TIP] Suggested affine tweak: a={metrics['suggest_a']:.4f}, b={metrics['suggest_b']:.4f} "
            f"(would give RMSE={metrics['RMSE_after']:.2f}, R²={metrics['R2_after']:.3f})"
        )


if __name__ == "__main__":
    main()
