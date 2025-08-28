import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
FLEX_ID = 2
EXTEND_ID = 5
r_cm = 1.0
rolling_window = 5
output_dir = "output/model0"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv("output/index_angles/aligned_data_index.csv")

# Extract and group by cycles 
def extract_cycles(df, motor_id, angle_label, time_gap_threshold=5000):
    sub_df = df[df["MotorID"] == motor_id].copy()
    sub_df = sub_df.dropna(subset=[angle_label])
    sub_df["angle_smooth"] = sub_df["Angle(deg)"].rolling(window=rolling_window, min_periods=1).mean()
    sub_df["L_cm"] = np.radians(sub_df["angle_smooth"]) * r_cm
    sub_df["delta_motor"] = sub_df["angle_smooth"].diff()
    sub_df = sub_df[sub_df["delta_motor"] < 0]  # Clockwise segment
    sub_df = sub_df.reset_index(drop=True)

    # Automatically group cycles (based on time gaps)
    sub_df["time_diff"] = sub_df["Time(ms)"].diff().fillna(0)
    cycle_id = 0
    cycle_ids = []
    for diff in sub_df["time_diff"]:
        if diff > time_gap_threshold:
            cycle_id += 1
        cycle_ids.append(cycle_id)
    sub_df["cycle"] = cycle_ids

    return sub_df

# Fit and plot
def fit_and_plot_cycles(df, angle_label, motor_label):
    L = df['L_cm'].values
    theta = df[angle_label].values

    # fit
    p1 = np.polyfit(L, theta, 1)
    p2 = np.polyfit(L, theta, 2)

    theta_pred_linear = np.polyval(p1, L)
    theta_pred_quad = np.polyval(p2, L)

    r2_lin = r2_score(theta, theta_pred_linear)
    r2_quad = r2_score(theta, theta_pred_quad)
    mse_lin = mean_squared_error(theta, theta_pred_linear)
    mse_quad = mean_squared_error(theta, theta_pred_quad)

    print(f"\n[{motor_label} | {angle_label}]")
    print(f"  Linear Fit:    y = {p1[0]:.4f}x + {p1[1]:.4f}")
    print(f"  Quadratic Fit: y = {p2[0]:.4f}x² + {p2[1]:.4f}x + {p2[2]:.4f}")
    print(f"  R² Linear:     {r2_lin:.4f}")
    print(f"  R² Quadratic:  {r2_quad:.4f}")
    print(f"  MSE Linear:    {mse_lin:.4f}")
    print(f"  MSE Quadratic: {mse_quad:.4f}")

    # Plotting by cycle
    plt.figure(figsize=(10, 6))
    unique_cycles = sorted(df["cycle"].unique())

    colormap = plt.cm.plasma  
    colors = [colormap(i / (len(unique_cycles) - 1)) for i in range(len(unique_cycles))]

    for i, cycle in enumerate(unique_cycles):
        sub = df[df["cycle"] == cycle]
        plt.scatter(sub["L_cm"], sub[angle_label], s=15, alpha=0.8,
                    color=colors[i], label=f"{motor_label}{i+1}")


    L_sorted = np.sort(L)
    plt.plot(L_sorted, np.polyval(p1, L_sorted), color='green', label='Linear Fit')
    plt.plot(L_sorted, np.polyval(p2, L_sorted), color='red', label='Quadratic Fit')

    plt.xlabel('Cable Displacement L (cm)')
    plt.ylabel(f'Joint Angle {angle_label} (deg)')
    plt.title(f'{motor_label} Motor → {angle_label}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save
    prefix = f"{motor_label.lower()}_{angle_label.replace('∠','').lower()}"
    plt.savefig(os.path.join(output_dir, f"{prefix}_cycles.png"), dpi=300)
    plt.close()
    df.to_csv(os.path.join(output_dir, f"{prefix}_cycles.csv"), index=False)

#Main process
for motor_id, motor_label in zip([FLEX_ID, EXTEND_ID], ["Flex", "Extend"]):
    for angle_label in ["∠ABC", "∠BCD"]:
        data = extract_cycles(df, motor_id, angle_label)
        fit_and_plot_cycles(data, angle_label, motor_label)
