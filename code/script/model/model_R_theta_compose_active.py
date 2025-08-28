# model_R_theta_compose_active.py
# ------------------------------------------------------------
# - Compatible: L->θ comes from JSON (preferred) or CSV (fallback)
# - R->θ composed modeling (R->L + L->θ), with "active segment filtering"
# - Generate ALL time-series comparison plots/CSV
# - Evaluate on active segments, and export ACTIVE_* CSV/scatter plots/summary
# - Optionally print/write suggested affine (a,b)

# Index finger (evaluate only on active segments; save full composed curve separately)
# python model_R_theta_compose_active.py ^
#   --aligned-csv "D:\Soft_glove\output\index_angles\aligned_data_index.csv" ^
#   --r-l-json    "D:\Soft_glove\models\index\R_L\R_L_model_middle.json" ^
#   --thetaL-json "D:\Soft_glove\models\index\L_theta\L_theta_model.json" ^
#   --finger index --sensor-col SensorA1 --flex-id 2 --extend-id 5 ^
#   --out "D:\Soft_glove\models\index\R_theta_compose_active" ^
#   --affine-a 0.9877 --affine-b 0.6574 ^
#   --flex-dir neg --extend-dir neg --dmotor-min 0.02 --gap-ms 5000 

# Thumb
# python model_R_theta_compose_active.py ^
#   --aligned-csv "D:\Soft_glove\output\thumb_angles\aligned_data_thumb.csv" ^
#   --r-l-json    "D:\Soft_glove\models\thumb\R_L\R_L_model_middle.json" ^
#   --thetaL-json "D:\Soft_glove\models\thumb\L_theta\L_theta_direct_model.json" ^
#   --finger thumb --sensor-col SensorA0 --flex-id 1 --extend-id 4 ^
#   --out "D:\Soft_glove\models\thumb\R_theta_compose_active" ^
#   --affine-a 0.9906 --affine-b 0.3030 ^
#   --flex-dir neg --extend-dir neg --dmotor-min 0.02 --gap-ms 5000 

#   --affine-a 1.0 --affine-b 0.0

# Middle finger (no ground-truth θ, only generate proxy θ sequence & JSON for active segments)
# python model_R_theta_compose_active.py ^
#   --aligned-csv "D:\Soft_glove\output\middle_angles\aligned_data_middle.csv" ^
#   --r-l-json    "D:\Soft_glove\models\middle\R_L\R_L_model_middle.json" ^
#   --thetaL-json "D:\Soft_glove\models\middle\L_theta\L_theta_direct_model.json" ^
#   --finger middle --sensor-col SensorA2 --flex-id 3 --extend-id 6 ^
#   --out "D:\Soft_glove\models\middle\R_theta_compose_active" ^
#   --flex-dir neg --extend-dir neg --dmotor-min 0.02 --gap-ms 5000 

# ------------------------------------------------------------
# R->θ composed modeling (R->L + L->θ), with "active segment filtering"
# - Generate ALL time-series comparison plots/CSV
# - Evaluate on active segments, and export ACTIVE_* CSV/scatter plots/summary
# - Optionally print/write suggested affine (a,b)
# ------------------------------------------------------------

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def phase_of_row(mid, flex_id, extend_id):
    if mid == flex_id: return "Flex"
    if mid == extend_id: return "Extend"
    if mid in (1,2,3): return "Flex"
    if mid in (4,5,6): return "Extend"
    return "Flex"

def estimate_L_from_R(sensor_val, rlj_phase):
    lo = float(rlj_phase["sensor_norm_lo"])
    hi = float(rlj_phase["sensor_norm_hi"])
    z = (sensor_val - lo) / (hi - lo if hi > lo else 1e-9)
    qa = float(rlj_phase["quadratic"]["a"])
    qb = float(rlj_phase["quadratic"]["b"])
    qc = float(rlj_phase["quadratic"]["c"])
    return qa*z*z + qb*z + qc

def compose_theta(L, coeffs):
    # coeffs = [a, b, c]; polyval expects [a, b, c]
    return float(np.polyval(coeffs, L))

def load_thetaL_from_json(path, finger):
    """Return {phase: [a,b,c]}, ycol_used, source"""
    obj = load_json(path)
    phases = obj.get("phases", {})
    prefer = "θ_total" if finger in ("index", "middle") else "∠ABD_smooth"

    def pick_target_name(phase_dict):
        tg = phase_dict.get("targets", {})
        if prefer in tg: return prefer
        return next(iter(tg.keys()), None) if tg else None

    ycol = None
    if "Flex" in phases: ycol = pick_target_name(phases["Flex"])
    if ycol is None and "Extend" in phases: ycol = pick_target_name(phases["Extend"])
    if ycol is None: raise ValueError(f"No targets in {path}")

    out = {}
    for ph in ("Flex", "Extend"):
        if ph not in phases: continue
        tgt = phases[ph].get("targets", {}).get(ycol)
        if not tgt:
            tg = phases[ph].get("targets", {})
            if not tg: continue
            tgt = tg[next(iter(tg.keys()))]
        q = tgt.get("quadratic")
        if q:
            out[ph] = [float(q["a"]), float(q["b"]), float(q["c"])]
        else:
            lin = tgt.get("linear")
            if not lin: continue
            out[ph] = [0.0, float(lin["a"]), float(lin["b"])]
    if not out: raise ValueError("No usable L->θ model in JSON.")
    return out, ycol, "JSON"

def load_thetaL_from_csv(csv_path, finger):
    """Fallback: read quadratic coefficients from summary CSV"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    quad_col = None
    for name in ("quadratic_coeffs","quadratic_coeffs(a,b,c)"):
        if name in df.columns: quad_col = name; break
    if not quad_col: raise ValueError("CSV missing quadratic coeffs column.")

    ypref = "θ_total" if finger in ("index","middle") else "∠ABD_smooth"
    out = {}
    for ph in ("Flex","Extend"):
        row = df[(df["phase"]==ph) & (df.get("ycol", pd.Series([""])).fillna("")==ypref)] if "ycol" in df.columns else df[(df["phase"]==ph)]
        if row.empty: continue
        coeffs = [float(x) for x in str(row.iloc[0][quad_col]).replace("[","").replace("]","").split(",")[:3]]
        out[ph] = coeffs
    if not out: raise ValueError("No usable rows in CSV.")
    return out, ypref, "CSV"

def filter_active(df, motor_id, dir_flag="neg", rolling=5, dmotor_min=0.02, gap_ms=5000, dtheta_min=None, theta_col=None):
    """Extract active segments by motor direction/threshold, and segment cycles by gap_ms"""
    sub = df[df["MotorID"]==motor_id].copy().sort_values("Time(ms)").reset_index(drop=True)
    if sub.empty: return sub
    sub["angle_smooth"] = sub["Angle(deg)"].rolling(window=rolling, min_periods=1).mean()
    sub["dmotor"] = sub["angle_smooth"].diff()
    sub = sub[(sub["dmotor"] < 0) if dir_flag=="neg" else (sub["dmotor"] > 0)]
    sub = sub[sub["dmotor"].abs() >= float(dmotor_min)]
    if dtheta_min is not None and theta_col and theta_col in sub.columns:
        sub["dtheta"] = sub[theta_col].diff().abs()
        sub = sub[sub["dtheta"] >= float(dtheta_min)]
    dt = sub["Time(ms)"].diff().fillna(0.0)
    sub["cycle"] = (dt > float(gap_ms)).cumsum()
    return sub

def suggest_affine(yhat, y):
    A = np.c_[yhat, np.ones_like(yhat)]
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    ss_res0 = float(np.sum((y - yhat)**2))
    ss_tot  = float(np.sum((y - np.mean(y))**2))
    rmse0 = float(np.sqrt(ss_res0 / len(y)))
    r2_0  = 1 - ss_res0/ss_tot if ss_tot>0 else float("nan")
    yhat2 = a*yhat + b
    ss_res1 = float(np.sum((y - yhat2)**2))
    rmse1 = float(np.sqrt(ss_res1 / len(y)))
    r2_1  = 1 - ss_res1/ss_tot if ss_tot>0 else float("nan")
    return float(a), float(b), rmse0, rmse1, r2_0, r2_1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned-csv", required=True)
    ap.add_argument("--r-l-json", required=True)
    ap.add_argument("--thetaL-json", default=None)
    ap.add_argument("--thetaL-summary", default=None)
    ap.add_argument("--finger", required=True, choices=["index","middle","thumb"])
    ap.add_argument("--sensor-col", required=True)
    ap.add_argument("--flex-id", type=int, required=True)
    ap.add_argument("--extend-id", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--affine-a", type=float, default=1.0)
    ap.add_argument("--affine-b", type=float, default=0.0)
    # Active segment parameters
    ap.add_argument("--flex-dir", default="neg", choices=["neg","pos"])
    ap.add_argument("--extend-dir", default="neg", choices=["neg","pos"])
    ap.add_argument("--rolling", type=int, default=5)
    ap.add_argument("--dmotor-min", type=float, default=0.02)
    ap.add_argument("--gap-ms", type=float, default=5000)
    ap.add_argument("--dtheta-min", type=float, default=None)
    # Suggest a,b
    ap.add_argument("--suggest-affine", type=int, default=1)
    args = ap.parse_args()

    ensure_dir(args.out)
    df = pd.read_csv(args.aligned_csv, encoding="utf-8-sig")
    rlj = load_json(args.r_l_json)

    # Load L->θ
    if args.thetaL_json and os.path.isfile(args.thetaL_json):
        thetaL, ycol_used, thetaL_src = load_thetaL_from_json(args.thetaL_json, args.finger)
    elif args.thetaL_summary and os.path.isfile(args.thetaL_summary):
        thetaL, ycol_used, thetaL_src = load_thetaL_from_csv(args.thetaL_summary, args.finger)
    else:
        raise FileNotFoundError("Need --thetaL-json (preferred) or --thetaL-summary.")

    # Ground truth θ column
    has_theta = False
    if args.finger in ("index","middle"):
        if "∠ABC_smooth" in df.columns and "∠BCD_smooth" in df.columns:
            df["θ_total"] = df["∠ABC_smooth"] + df["∠BCD_smooth"]
            gt_col = "θ_total"
            has_theta = (args.finger == "index")    # Middle usually has no reliable θ
        else:
            gt_col = None
    else:
        gt_col = "∠ABD_smooth" if "∠ABD_smooth" in df.columns else None
        has_theta = (gt_col is not None and args.finger=="thumb")

    # -------- Generate ALL time-series composed θ (for comparison & visualization) --------
    L_all, Th_all, Ph_all = [], [], []
    for _, row in df.iterrows():
        s = row.get(args.sensor_col, np.nan)
        if not np.isfinite(s):
            L_all.append(np.nan); Th_all.append(np.nan); Ph_all.append(None); continue
        mid = int(row["MotorID"]) if np.isfinite(row.get("MotorID", np.nan)) else None
        phase = phase_of_row(mid, args.flex_id, args.extend_id)
        ph_cfg = rlj.get("phases", {}).get(phase) or next(iter(rlj.get("phases", {}).values()))
        L_est = estimate_L_from_R(s, ph_cfg)
        th = compose_theta(L_est, thetaL.get(phase) or next(iter(thetaL.values())))
        th = args.affine_a * th + args.affine_b
        L_all.append(L_est); Th_all.append(th); Ph_all.append(phase)

    df_all = df.copy()
    df_all["L_est_cm"] = L_all
    df_all["theta_from_R(compose)"] = Th_all
    df_all["phase_used"] = Ph_all
    df_all.to_csv(os.path.join(args.out, "R_theta_composed_timeseries_ALL.csv"),
                  index=False, encoding="utf-8-sig")

    # --------Active segment filtering + combination + evaluation (within phase)--------
    rows = []
    all_active_pairs = []  

    for phase, mid, mdir in [("Flex", args.flex_id, args.flex_dir),
                             ("Extend", args.extend_id, args.extend_dir)]:
        act = filter_active(df, motor_id=mid, dir_flag=mdir,
                            rolling=args.rolling, dmotor_min=args.dmotor_min,
                            gap_ms=args.gap_ms, dtheta_min=args.dtheta_min,
                            theta_col=(gt_col if has_theta else None))
        act_csv = os.path.join(args.out, f"active_{phase}.csv")
        act.to_csv(act_csv, index=False, encoding="utf-8-sig")

        if act.empty:
            rows.append({"phase": phase, "N": 0})
            continue

        ph_cfg = rlj.get("phases", {}).get(phase) or next(iter(rlj.get("phases", {}).values()))
        coeffs = thetaL.get(phase) or next(iter(thetaL.values()))

        L_list, Th_list = [], []
        for _, r in act.iterrows():
            s = r.get(args.sensor_col, np.nan)
            if not np.isfinite(s):
                L_list.append(np.nan); Th_list.append(np.nan); continue
            L_est = estimate_L_from_R(s, ph_cfg)
            th = compose_theta(L_est, coeffs)
            Th_list.append(args.affine_a * th + args.affine_b)
            L_list.append(L_est)

        act_out = act.copy()
        act_out["L_est_cm"] = L_list
        act_out["theta_from_R(compose)"] = Th_list
        ts_csv = os.path.join(args.out, f"R_theta_composed_timeseries_ACTIVE_{phase}.csv")
        act_out.to_csv(ts_csv, index=False, encoding="utf-8-sig")

        # In-phase evaluation (if there is a true value θ)
        metrics = {"phase": phase, "N": int(np.isfinite(act_out["theta_from_R(compose)"]).sum())}
        if has_theta and gt_col in act_out.columns:
            m = np.isfinite(act_out["theta_from_R(compose)"]) & np.isfinite(act_out[gt_col])
            if m.sum() >= 10:
                y = act_out.loc[m, gt_col].to_numpy()
                yhat = act_out.loc[m, "theta_from_R(compose)"].to_numpy()
                ss_res = float(np.sum((y - yhat)**2))
                ss_tot = float(np.sum((y - np.mean(y))**2))
                rmse_p = float(np.sqrt(ss_res / m.sum()))
                r2_p   = 1 - ss_res/ss_tot if ss_tot>0 else float("nan")
                metrics.update({"RMSE_active": rmse_p, "R2_active": r2_p})
                # Collect the global active segment pool
                all_active_pairs.append((yhat, y))
        rows.append(metrics)

        # Scattered points in the active segment（R_norm vs θ）
        lo = float(ph_cfg["sensor_norm_lo"]); hi = float(ph_cfg["sensor_norm_hi"])
        plt.figure(figsize=(10,6))
        for cyc, g in act_out.groupby("cycle"):
            z = (g[args.sensor_col].to_numpy() - lo) / (hi - lo if hi>lo else 1e-9)
            plt.scatter(z, g["theta_from_R(compose)"], s=12, alpha=0.7, label=f"cycle {int(cyc)}")
        plt.xlabel(f"{args.sensor_col} (normalized)")
        plt.ylabel("θ (deg)")
        plt.title(f"{args.finger.capitalize()} {phase} (ACTIVE): R→θ (composed)")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(args.out, f"R_theta_composed_ACTIVE_scatter_{phase}.png"), dpi=300)
        plt.close()

    # -------- Active segment "Global" RMSE/R² (for title/Suggestion a,b) --------
    rmse_active, r2_active = float("nan"), float("nan")
    if has_theta and len(all_active_pairs) > 0:
        yhat_all = np.concatenate([p[0] for p in all_active_pairs])
        y_all    = np.concatenate([p[1] for p in all_active_pairs])
        m = np.isfinite(y_all) & np.isfinite(yhat_all)
        if m.sum() >= 10:
            ss_res = float(np.sum((y_all[m] - yhat_all[m])**2))
            ss_tot = float(np.sum((y_all[m] - np.mean(y_all[m]))**2))
            rmse_active = float(np.sqrt(ss_res / m.sum()))
            r2_active   = 1 - ss_res/ss_tot if ss_tot>0 else float("nan")

    if has_theta and args.suggest_affine and len(all_active_pairs) > 0:
        a_s, b_s, rmse0, rmse1, r20, r21 = suggest_affine(yhat_all[m], y_all[m])
        print(f"[SUGGEST] a={a_s:.4f}, b={b_s:.4f} | RMSE {rmse0:.2f}->{rmse1:.2f}, R² {r20:.3f}->{r21:.3f}")

    # -------- Draw the ALL sequence diagram --------
    t = (df_all["Time(ms)"].to_numpy()/1000.0) if "Time(ms)" in df_all.columns else np.arange(len(df_all))
    plt.figure(figsize=(12,6))
    plt.plot(t, df_all["theta_from_R(compose)"], label="theta_from_R(compose) ALL")
    if has_theta and (("θ_total" in df_all.columns) or ("∠ABD_smooth" in df_all.columns)):
        gt_all_col = "θ_total" if "θ_total" in df_all.columns else ("∠ABD_smooth" if "∠ABD_smooth" in df_all.columns else None)
        if gt_all_col:
            plt.plot(t, df_all[gt_all_col], label=f"ground truth ({gt_all_col})", alpha=0.6)
    title = f"R->θ composed ({args.finger}) | RMSE={rmse_active:.2f}, R²={r2_active:.3f}"
    plt.title(title)
    plt.xlabel("Time (s)"); plt.ylabel("θ (deg)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "R_theta_composed_timeseries_ALL.png"), dpi=300)
    plt.close()

    # -------- Export the simple summary (phase level) --------
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "R_theta_active_summary.csv"),
                              index=False, encoding="utf-8-sig")

    # -------- Export deployment JSON--------
    export = {
        "finger": args.finger,
        "sensor_col": args.sensor_col,
        "affine": {"a": float(args.affine_a), "b": float(args.affine_b)},
        "R_to_L_model": rlj,
        "L_to_theta_model": {"phase_quadratic": thetaL, "ycol_used": ycol_used, "source": thetaL_src},
        "motor_ids": {"flex_id": int(args.flex_id), "extend_id": int(args.extend_id)},
        "active_filter": {
            "flex_dir": args.flex_dir, "extend_dir": args.extend_dir,
            "rolling": int(args.rolling), "dmotor_min": float(args.dmotor_min),
            "gap_ms": float(args.gap_ms), "dtheta_min": (None if args.dtheta_min is None else float(args.dtheta_min))
        }
    }
    with open(os.path.join(args.out, "R_theta_composed_model_ACTIVE.json"), "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved:\n- {os.path.join(args.out,'R_theta_active_summary.csv')}\n"
          f"- R_theta_composed_timeseries_ALL.csv/.png\n"
          f"- ACTIVE_* CSV/PNG per phase\n"
          f"- R_theta_composed_model_ACTIVE.json")

if __name__ == "__main__":
    main()

