import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

# ============================================================
# FINAL CROSS-VALIDATED RESULTS (LOCKED)
# ============================================================

horizons = [1, 3, 6, 12, 24]

results = {
    "Linear (Full)": {
        "RMSE": [6.93, 13.02, 17.72, 22.48, 28.74],
        "R2":   [0.717, 0.650, 0.390, -0.141, -0.928]
    },
    "Gradient Boosting": {
        "RMSE": [7.04, 13.28, 16.22, 19.02, 23.07],
        "R2":   [0.710, 0.636, 0.488, 0.184, -0.243]
    },
    "Random Forest": {
        "RMSE": [7.06, 13.67, 16.08, 17.59, 18.39],
        "R2":   [0.708, 0.614, 0.497, 0.301, 0.210]
    },
    "GRU": {
        "RMSE": [6.98, 11.40, 13.93, 16.12, 17.45],
        "R2":   [0.809, 0.493, 0.256, 0.006, -0.165]
    }
}

Path("results").mkdir(exist_ok=True)

# ============================================================
# 1. RMSE vs Horizon
# ============================================================

plt.figure(figsize=(10, 6))

for model, vals in results.items():
    plt.plot(horizons, vals["RMSE"], marker="o", label=model)

plt.axvspan(1, 3, color="green", alpha=0.08, label="Reliable (1–3h)")
plt.axvspan(3, 6, color="yellow", alpha=0.08, label="Caution (6h)")
plt.axvspan(6, 24, color="red", alpha=0.05, label="High Uncertainty (12–24h)")

plt.xlabel("Forecast Horizon (hours)")
plt.ylabel("RMSE (µg/m³)")
plt.title("RMSE vs Forecast Horizon (Cross-Validated)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/rmse_vs_horizon.png", dpi=300)
plt.close()

# ============================================================
# 2. R² vs Horizon
# ============================================================

plt.figure(figsize=(10, 6))

for model, vals in results.items():
    plt.plot(horizons, vals["R2"], marker="o", label=model)

plt.axhline(0, color="black", linestyle="--", lw=1)
plt.axvline(6, color="gray", linestyle=":", lw=1)
plt.text(6.1, 0.05, "Reliability Cliff (~6h)", color="gray")

plt.xlabel("Forecast Horizon (hours)")
plt.ylabel("R² Score")
plt.title("R² vs Forecast Horizon (Cross-Validated)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/r2_vs_horizon.png", dpi=300)
plt.close()

# ============================================================
# 3. Heatmaps
# ============================================================

rmse_df = pd.DataFrame(
    {m: results[m]["RMSE"] for m in results},
    index=horizons
).T

r2_df = pd.DataFrame(
    {m: results[m]["R2"] for m in results},
    index=horizons
).T

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(rmse_df, annot=True, fmt=".2f", cmap="Reds", ax=axes[0])
axes[0].set_title("RMSE Heatmap (Lower = Better)")
axes[0].set_xlabel("Horizon (h)")
axes[0].set_ylabel("Model")

sns.heatmap(r2_df, annot=True, fmt=".3f", cmap="coolwarm", center=0, ax=axes[1])
axes[1].set_title("R² Heatmap (Higher = Better)")
axes[1].set_xlabel("Horizon (h)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig("results/performance_heatmaps.png", dpi=300)
plt.close()

print("✓ Saved honest CV-based comparison visuals")