# model_R_theta.py  — Align theta & sensor by time (per MotorID), fit R↔θ, export plots & metrics.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- Config ----------------
THETA_CSV  = r"C:\Users\nicey\Desktop\Soft_glove\output\aligned_data_ind3.csv"
SENSOR_CSV = r"C:\Users\nicey\Desktop\Soft_glove\data\csv\flex_sensor\motor_flex_log_20250809_210522.csv"
OUTPUT_DIR = r"C:\Users\nicey\Desktop\Soft_glove\output\model_r_theta_aligned"

FINGER_MAP = {
    "Thumb":  {"motor_id": 1, "sensor_col": "SensorA0"},
    "Index":  {"motor_id": 2, "sensor_col": "SensorA1"},
    "Middle": {"motor_id": 3, "sensor_col": "SensorA2"},
}

ANGLE_COLS = ["∠ABC", "∠BCD"]
MAX_ALIGN_TOL_MS = 40

os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_name(name: str) -> str:
    return name.replace("∠", "").replace("/", "_").replace("\\", "_").strip()

def load_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")

def save_txt(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def fit_models(R, theta):
    p1 = np.polyfit(R, theta, 1)
    y_lin = np.polyval(p1, R)
    r2_lin = r2_score(theta, y_lin)
    mse_lin = mean_squared_error(theta, y_lin)

    p2 = np.polyfit(R, theta, 2)
    y_quad = np.polyval(p2, R)
    r2_quad = r2_score(theta, y_quad)
    mse_quad = mean_squared_error(theta, y_quad)
    return (p1, r2_lin, mse_lin), (p2, r2_quad, mse_quad)

def plot_fit(R, theta, p1, p2, title, xlab, ylab, save_path):
    order = np.argsort(R)
    R_sorted = R[order]
    y_lin  = np.polyval(p1, R_sorted)
    y_quad = np.polyval(p2, R_sorted)

    plt.figure(figsize=(8,6))
    plt.scatter(R, theta, s=10, alpha=0.6, label="Data")
    plt.plot(R_sorted, y_lin,  color="green", label="Linear fit")
    plt.plot(R_sorted, y_quad, color="red",   label="Quadratic fit")
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.title(title); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()

# -------- Load --------
theta_df  = load_csv(THETA_CSV)
sensor_df = load_csv(SENSOR_CSV)

needed_theta_cols  = {"Time(ms)", "MotorID"} | set(ANGLE_COLS)
needed_sensor_cols = {"Time(ms)", "MotorID", "SensorA0", "SensorA1", "SensorA2"}

missing_theta  = [c for c in needed_theta_cols if c not in theta_df.columns]
missing_sensor = [c for c in needed_sensor_cols if c not in sensor_df.columns]
if missing_theta:
    raise ValueError(f"Angle file missing columns: {missing_theta}")
if missing_sensor:
    raise ValueError(f"Sensor file missing columns: {missing_sensor}")

theta_keep = ["Time(ms)", "MotorID"] + ANGLE_COLS
theta_df = theta_df[theta_keep].copy()
theta_df["AvgAngle"] = theta_df[ANGLE_COLS].mean(axis=1)

sensor_df = sensor_df[["Time(ms)", "MotorID", "SensorA0", "SensorA1", "SensorA2"]].copy()

# sort by MotorID + Time, and use by='MotorID' in merge_asof
theta_df  = theta_df.sort_values(["MotorID", "Time(ms)"]).reset_index(drop=True)
sensor_df = sensor_df.sort_values(["MotorID", "Time(ms)"]).reset_index(drop=True)

aligned = pd.merge_asof(
    left=theta_df,
    right=sensor_df,
    on="Time(ms)",
    by="MotorID",                   
    direction="nearest",
    tolerance=MAX_ALIGN_TOL_MS,
    suffixes=("", "_r")            
)

# Drop rows that cannot be aligned within tolerance
aligned = aligned.dropna(subset=["SensorA0", "SensorA1", "SensorA2"]).reset_index(drop=True)

# Save aligned result
aligned_path = os.path.join(OUTPUT_DIR, "aligned_theta_sensor.csv")
aligned.to_csv(aligned_path, index=False, encoding="utf-8-sig")

if "MotorID" not in aligned.columns:
    raise RuntimeError(f"Aligned table has no MotorID column. Current columns: {list(aligned.columns)}")

# -------- Fit per finger --------
report_lines = []
for finger, info in FINGER_MAP.items():
    motor_id  = info["motor_id"]
    sensor_col = info["sensor_col"]

    sub = aligned[aligned["MotorID"] == motor_id].copy()
    if sub.empty:
        report_lines.append(f"[WARN] {finger}: MotorID={motor_id} has no aligned data.")
        continue

    for theta_col in ANGLE_COLS + ["AvgAngle"]:
        tmp = sub.dropna(subset=[theta_col, sensor_col]).copy()
        if tmp.empty:
            report_lines.append(f"[WARN] {finger} - {theta_col}: no available data.")
            continue

        R     = tmp[sensor_col].to_numpy(float)
        theta = tmp[theta_col].to_numpy(float)

        (p1, r2_lin, mse_lin), (p2, r2_quad, mse_quad) = fit_models(R, theta)

        txt = [
            f"Finger: {finger} | Sensor: {sensor_col} | Theta: {theta_col}",
            f"Linear:    y = {p1[0]:.6f} * x + {p1[1]:.6f}",
            f"Quadratic: y = {p2[0]:.6f} * x^2 + {p2[1]:.6f} * x + {p2[2]:.6f}",
            f"R2 Linear: {r2_lin:.6f} | MSE Linear: {mse_lin:.6f}",
            f"R2 Quad:   {r2_quad:.6f} | MSE Quad:   {mse_quad:.6f}",
        ]
        block = "\n".join(txt)
        print("\n" + block)

        metrics_name = f"{finger}_{safe_name(theta_col)}_metrics.txt"
        save_txt(os.path.join(OUTPUT_DIR, metrics_name), block)

        plot_name = f"{finger}_{safe_name(theta_col)}_fit.png"
        plot_fit(R, theta, p1, p2,
                 title=f"{finger}: {sensor_col} vs {theta_col}",
                 xlab=f"{sensor_col} (ADC)", ylab=f"{theta_col} (deg)",
                 save_path=os.path.join(OUTPUT_DIR, plot_name))

        data_name = f"{finger}_{safe_name(theta_col)}_data.csv"
        tmp[[sensor_col, theta_col]].to_csv(
            os.path.join(OUTPUT_DIR, data_name), index=False, encoding="utf-8-sig"
        )

        report_lines.append(block)

# Summary report
summary_path = os.path.join(OUTPUT_DIR, "summary_report.txt")
save_txt(summary_path, "\n\n".join(report_lines))

print(f"\n[INFO] Done. Results saved in: {OUTPUT_DIR}")
print(f"[INFO] Generated: aligned data CSV, fit plots/data/metrics per finger-angle, and summary_report.txt")