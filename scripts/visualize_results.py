import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "GRU"
HORIZON_HOURS = 1

INPUT_FILE = "results/predictions.csv"
OUTPUT_FILE = "results/model_results.png"

Path("results").mkdir(exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_cols = {"actual", "predicted"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Optional datetime handling
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

residuals = df["actual"] - df["predicted"]
rmse = np.sqrt((residuals ** 2).mean())
mae = np.abs(residuals).mean()

# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    f"{MODEL_NAME} – {HORIZON_HOURS}h Forecast Diagnostics\n"
    f"RMSE={rmse:.2f}, MAE={mae:.2f}",
    fontsize=14
)

# ------------------------------------------------------------
# 1. Actual vs Predicted
# ------------------------------------------------------------
ax = axes[0, 0]
ax.scatter(df["actual"], df["predicted"], alpha=0.4, s=12)

max_val = max(df["actual"].max(), df["predicted"].max())
ax.plot([0, max_val], [0, max_val], "k--", lw=2)

ax.set_title("Actual vs Predicted")
ax.set_xlabel("Actual PM2.5 (µg/m³)")
ax.set_ylabel("Predicted PM2.5 (µg/m³)")
ax.grid(alpha=0.3)

# ------------------------------------------------------------
# 2. Time Series (first window)
# ------------------------------------------------------------
ax = axes[0, 1]

if "datetime" in df.columns:
    sample = df.iloc[:500]
    ax.plot(sample["datetime"], sample["actual"], label="Actual", alpha=0.8)
    ax.plot(sample["datetime"], sample["predicted"], label="Predicted", alpha=0.8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
else:
    ax.plot(df["actual"].iloc[:500], label="Actual", alpha=0.8)
    ax.plot(df["predicted"].iloc[:500], label="Predicted", alpha=0.8)

ax.set_title("Time Series (Sample Window)")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.legend()
ax.grid(alpha=0.3)

# ------------------------------------------------------------
# 3. Residual Distribution
# ------------------------------------------------------------
ax = axes[1, 0]
ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
ax.axvline(0, color="red", linestyle="--", lw=2)

ax.set_title("Residual Distribution")
ax.set_xlabel("Residual (Actual − Predicted)")
ax.set_ylabel("Frequency")
ax.grid(alpha=0.3)

# ------------------------------------------------------------
# 4. Residuals vs Predicted
# ------------------------------------------------------------
ax = axes[1, 1]
ax.scatter(df["predicted"], residuals, alpha=0.4, s=12)
ax.axhline(0, color="red", linestyle="--", lw=2)

ax.set_title("Residuals vs Predicted")
ax.set_xlabel("Predicted PM2.5 (µg/m³)")
ax.set_ylabel("Residual")
ax.grid(alpha=0.3)

# ============================================================
# SAVE
# ============================================================

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close()

print(f"✓ Saved diagnostics to {OUTPUT_FILE}")