import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

csv_path = r"D:\Soft_glove\output\control_results\experiment_log_multi.csv"  # <- 改成你的路径
df = pd.read_csv(csv_path)

df.columns = [c.strip() for c in df.columns]

# Compute error
df["Error_deg"] = df["Theta_target_deg"] - df["Theta_meas_deg"]

# Fig A: Measured vs Target (with y=x, linear fit, R^2)
plt.figure(figsize=(8.5, 7))
colors = {"Thumb":"#8c564b", "Index":"#1f77b4", "Middle":"#2ca02c"}
for finger, d in df.groupby("Finger"):
    plt.scatter(d["Theta_target_deg"], d["Theta_meas_deg"], s=8, alpha=0.15, color=colors.get(finger, None), label=finger)
    # Linear regression & R^2
    x = d["Theta_target_deg"].values
    y = d["Theta_meas_deg"].values
    slope, intercept, r, p, stderr = stats.linregress(x, y)
    xline = np.linspace(x.min(), x.max(), 100)
    yfit = slope * xline + intercept
    plt.plot(xline, yfit, color=colors.get(finger, None), linewidth=2,
             label=f"{finger} fit: y={slope:.3f}x+{intercept:.2f}")

#Consistency baseline y=x
xymin = min(df["Theta_target_deg"].min(), df["Theta_meas_deg"].min())
xymax = max(df["Theta_target_deg"].max(), df["Theta_meas_deg"].max())
plt.plot([xymin, xymax], [xymin, xymax], "k--", linewidth=1, label="y=x")

plt.xlabel(r"$\theta_{\rm target}$ (deg)")
plt.ylabel(r"$\theta_{\rm measured}$ (deg)")
plt.title("Measured vs. Target Angles (All Fingers)")
plt.legend(ncol=2, fontsize=9, frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig_meas_vs_target.png", dpi=300)
plt.close()

# Fig B: Error distribution violin + box (with RMSE/MAE/95% overlay)
stats_rows = []
order = ["Thumb","Index","Middle"]
groups = [df[df["Finger"]==f]["Error_deg"].dropna() for f in order]

plt.figure(figsize=(9, 5.5))
vp = plt.violinplot(groups, showmeans=False, showmedians=False, showextrema=False)
# Custom colors & transparency
for i, b in enumerate(vp['bodies']):
    b.set_facecolor(list(colors.values())[i])
    b.set_alpha(0.25)
    b.set_edgecolor('none')

# Overlay boxplot (quartiles + median)
box = plt.boxplot(groups, widths=0.2, vert=True, patch_artist=True, showfliers=False)
for i, patch in enumerate(box['boxes']):
    patch.set(facecolor=list(colors.values())[i], alpha=0.35, edgecolor="k")

# Overlay mean points
for i, g in enumerate(groups, start=1):
    plt.scatter(i, np.mean(g), marker="o", color="k", s=18, zorder=3)

# Compute and print per-finger statistics (RMSE, MAE, P2.5–P97.5)
for f in order:
    e = df.loc[df["Finger"]==f, "Error_deg"].dropna().values
    rmse = np.sqrt(np.mean(e**2))
    mae  = np.mean(np.abs(e))
    p_lo, p_hi = np.percentile(e, [2.5, 97.5])
    stats_rows.append([f, rmse, mae, p_lo, p_hi])

plt.xticks(range(1, len(order)+1), order)
plt.ylabel("Error (deg)  =  $\\theta_{target} - \\theta_{measured}$")
plt.title("Error Distribution per Finger (violin + box)")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig("fig_error_violin.png", dpi=300)
plt.close()

# print the table to terminal
import pandas as pd
tbl = pd.DataFrame(stats_rows, columns=["Finger","RMSE (deg)","MAE (deg)","P2.5 (deg)","P97.5 (deg)"])
print("\n=== Summary (use in results table) ===")
print(tbl.to_string(index=False))
