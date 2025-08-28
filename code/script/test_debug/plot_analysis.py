# plot_analysis.py: Analyze the data in the aligned_data.csv file obtained by running anagle_detect.py. Visualize the data.

import pandas as pd
import matplotlib.pyplot as plt

# Read alignment data (including Motor angles and ∠ABC, ∠BCD)
df = pd.read_csv("output/aligned_data.csv")

# Only use the main motor
df = df[df["MotorID"] == 5]

plt.figure(figsize=(10, 6))
plt.plot(df["Time(ms)"], df["Angle(deg)"], label="Motor Angle", linewidth=2)
plt.plot(df["Time(ms)"], df["∠ABC"], label="Joint ∠ABC", linewidth=2)
plt.plot(df["Time(ms)"], df["∠BCD"], label="Joint ∠BCD", linewidth=2)

plt.xlabel("Time (ms)")
plt.ylabel("Angle (°)")
plt.title("Motor Angle vs Finger Joint Angles")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/plot_motor_vs_joint.png", dpi=300)
plt.show()
