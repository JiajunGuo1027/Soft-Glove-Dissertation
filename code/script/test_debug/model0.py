import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
import os

# Parameter Configuration
FLEX_ID = 2
EXTEND_ID = 5
r_cm = 1.0  # Spool radius (cm)
rolling_window = 5   # Smoothing window size
output_dir = "output/model"
os.makedirs(output_dir, exist_ok=True)

#  Load Data
df = pd.read_csv("output/aligned_data.csv")

# Extract and Preprocess Data
def extract_data(df, motor_id, angle_label):
    sub_df = df[df['MotorID'] == motor_id].copy()
    sub_df = sub_df.dropna(subset=[angle_label])
    
    # Smooth angle
    sub_df[angle_label] = sub_df[angle_label].rolling(window=rolling_window, min_periods=1).mean()
    
    # Calculate cable displacement L (unit: cm)
    sub_df['L_cm'] = np.radians(sub_df['Angle(deg)']) * r_cm
    
    return sub_df[['L_cm', angle_label, 'Time(ms)']].dropna()

# Fit and Plot
def fit_and_plot(data, angle_label, motor_label):
    L = data['L_cm'].values
    theta = data[angle_label].values
    time = data['Time(ms)'].values
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(L, theta, c=time, cmap='viridis', s=10, alpha=0.8, label='Data')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time (ms)')

    # Linear fit
    p1 = np.polyfit(L, theta, 1)
    plt.plot(np.sort(L), np.polyval(p1, np.sort(L)), color='green', label='Linear Fit')

    # Quadratic fit
    p2 = np.polyfit(L, theta, 2)
    plt.plot(np.sort(L), np.polyval(p2, np.sort(L)), color='red', label='Quadratic Fit')

    plt.xlabel('Cable Displacement L (cm)')
    plt.ylabel(f'Joint Angle {angle_label} (deg)')
    plt.title(f'{motor_label} Motor → {angle_label}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save image
    save_path = os.path.join(output_dir, f'plot_{motor_label.lower()}_{angle_label.replace("∠","").lower()}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

# Modeling and Plotting for Each Case

for motor_id, motor_label in zip([FLEX_ID, EXTEND_ID], ["Flex", "Extend"]):
    for angle_label in ["∠ABC", "∠BCD"]:
        data = extract_data(df, motor_id, angle_label)
        fit_and_plot(data, angle_label, motor_label)
