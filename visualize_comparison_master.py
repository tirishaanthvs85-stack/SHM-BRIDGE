"""
visualize_comparison_master.py
Master Benchmark & Visual Comparison: PINN vs Random Forest vs SVM
Bridge Structural Health Monitoring (SHM)
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------------------------------
# LOAD DATA & MODELS
# --------------------------------------------------
DATA_PATH = "dataset/bridge_data.csv"
data = pd.read_csv(DATA_PATH)

x_raw = data[["x", "sensor_measurement"]].values
x_pos = data["x"].values
healthy = data["healthy_deflection"].values
damaged = data["damaged_deflection"].values
sensor  = data["sensor_measurement"].values

# 80/20 train/test split consistent with individual training pipelines
X_tr, X_te, y_tr, y_te, x_tr, x_te = train_test_split(
    x_raw, damaged, x_pos, test_size=0.2, random_state=42
)

# Load Models
pinn_model = tf.keras.models.load_model("models/bridge_pinn.keras", safe_mode=False)
svm_model  = joblib.load("svm_bridge_model.pkl")
rf_model   = joblib.load("rf_bridge_model.pkl")

def get_pinn(x_arr):
    t = tf.convert_to_tensor(x_arr.reshape(-1, 1), dtype=tf.float32)
    pred = pinn_model(t).numpy().flatten()
    if np.max(np.abs(pred)) > 0.1:
        pred = pred / 1000.0
    return pred

# Full-span predictions for continuous curves
pinn_full = get_pinn(x_pos)
svm_full  = svm_model.predict(x_raw)
rf_full   = rf_model.predict(x_raw)

# Test-set predictions for fair quantitative benchmarking
pinn_test = get_pinn(x_te)
svm_test  = svm_model.predict(X_te)
rf_test   = rf_model.predict(X_te)

# Sort along span for spatial plotting
sorted_idx     = x_pos.argsort()
x_sorted       = x_pos[sorted_idx]
healthy_sorted = healthy[sorted_idx]
damaged_sorted = damaged[sorted_idx]
sensor_sorted  = sensor[sorted_idx]
pinn_sorted    = pinn_full[sorted_idx]
svm_sorted     = svm_full[sorted_idx]
rf_sorted      = rf_full[sorted_idx]

# Residual Errors
pinn_err_mm = np.abs(damaged_sorted - pinn_sorted) * 1000.0
svm_err_mm  = np.abs(damaged_sorted - svm_sorted) * 1000.0
rf_err_mm   = np.abs(damaged_sorted - rf_sorted) * 1000.0

# --------------------------------------------------
# COMPUTE QUANTITATIVE METRICS (TEST SET)
# --------------------------------------------------
def calc_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse) * 1000.0 # in mm
    mae  = mean_absolute_error(y_true, y_pred) * 1000.0 # in mm
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-12))) * 100
    max_e = np.max(np.abs(y_true - y_pred)) * 1000.0
    return {"R2": r2, "RMSE_mm": rmse, "MAE_mm": mae, "MAPE": mape, "MaxErr_mm": max_e}

metrics = {
    "SVM (SVR)": calc_metrics(y_te, svm_test),
    "Random Forest": calc_metrics(y_te, rf_test),
    "PINN (Ours)": calc_metrics(y_te, pinn_test),
}

# --------------------------------------------------
# MASTER 4-PANEL BENCHMARK FIGURE
# --------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Palette
c_gt = "#111827"      # Dark slate
c_pinn = "#2563EB"    # Blue
c_rf = "#7C3AED"      # Purple
c_svm = "#F59E0B"     # Amber / Orange
c_healthy = "#10B981" # Emerald Green
c_sensor = "#EF4444"  # Red

# ---------------- SUBPLOT 1: DEFLECTION PROFILES ----------------
ax1 = axes[0, 0]
ax1.plot(x_sorted, healthy_sorted * 1000, color=c_healthy, linestyle=":", linewidth=2, label="Healthy Baseline")
ax1.plot(x_sorted, damaged_sorted * 1000, color=c_gt, linewidth=2.5, label="Ground Truth (Damaged)")
ax1.scatter(x_sorted, sensor_sorted * 1000, color=c_sensor, s=14, alpha=0.5, label="Noisy Sensor Signals", zorder=3)
ax1.plot(x_sorted, svm_sorted * 1000, color=c_svm, linestyle="--", linewidth=1.8, label="SVM (SVR-RBF)")
ax1.plot(x_sorted, rf_sorted * 1000, color=c_rf, linestyle="-.", linewidth=1.8, label="Random Forest")
ax1.plot(x_sorted, pinn_sorted * 1000, color=c_pinn, linestyle="-", linewidth=2.2, label="PINN (Physics-Informed)")

ax1.axvspan(17.5, 29.5, color="#FEE2E2", alpha=0.45, label="Simulated Crack Zone [17.5 - 29.5m]")
ax1.set_title("(a) Bridge Deflection Field vs Ground Truth & Sensors", fontsize=12, fontweight="bold", pad=8)
ax1.set_xlabel("Bridge Span Position x (m)", fontsize=11)
ax1.set_ylabel("Deflection (mm)", fontsize=11)
ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax1.grid(True, linestyle="--", alpha=0.6)

# ---------------- SUBPLOT 2: SPATIAL RESIDUALS & CRACK LOCALIZATION ----------------
ax2 = axes[0, 1]
ax2.plot(x_sorted, svm_err_mm, color=c_svm, linestyle="--", linewidth=1.6, label="SVM Absolute Residual")
ax2.plot(x_sorted, rf_err_mm, color=c_rf, linestyle="-.", linewidth=1.6, label="RF Absolute Residual")
ax2.plot(x_sorted, pinn_err_mm, color=c_pinn, linewidth=2.2, label="PINN Absolute Residual")

ax2.axvspan(17.5, 29.5, color="#FEE2E2", alpha=0.45, label="Crack Defect Zone")
ax2.set_title("(b) Absolute Spatial Residual Error & Crack Localization", fontsize=12, fontweight="bold", pad=8)
ax2.set_xlabel("Bridge Span Position x (m)", fontsize=11)
ax2.set_ylabel("Absolute Error |y_true - y_pred| (mm)", fontsize=11)
ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax2.grid(True, linestyle="--", alpha=0.6)

# ---------------- SUBPLOT 3: ACCURACY METRICS COMPARISON (BAR CHART) ----------------
ax3 = axes[1, 0]
models_list = ["SVM (SVR)", "Random Forest", "PINN (Ours)"]
x_indices = np.arange(len(models_list))
width = 0.26

rmse_vals = [metrics[m]["RMSE_mm"] for m in models_list]
mae_vals  = [metrics[m]["MAE_mm"] for m in models_list]
r2_vals   = [metrics[m]["R2"] * 100.0 for m in models_list]

rects1 = ax3.bar(x_indices - width, rmse_vals, width, label="RMSE (mm)", color="#EF4444", alpha=0.85)
rects2 = ax3.bar(x_indices, mae_vals, width, label="MAE (mm)", color="#3B82F6", alpha=0.85)
rects3 = ax3.bar(x_indices + width, [v - 80.0 for v in r2_vals], width, label="R² Score (%) [Offset -80%]", color="#10B981", alpha=0.85)

ax3.set_title("(c) Quantitative Accuracy on Unseen Test Split (20%)", fontsize=12, fontweight="bold", pad=8)
ax3.set_xticks(x_indices)
ax3.set_xticklabels(models_list, fontsize=10, fontweight="bold")
ax3.set_ylabel("Metric Value", fontsize=11)
ax3.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax3.grid(True, axis="y", linestyle="--", alpha=0.6)

# Add text labels on bars
for bar in rects1:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar in rects2:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
for bar, actual_r2 in zip(rects3, r2_vals):
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{actual_r2:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

# ---------------- SUBPLOT 4: PARITY PLOT (ACTUAL VS PREDICTED) ----------------
ax4 = axes[1, 1]
ax4.scatter(y_te * 1000, svm_test * 1000, color=c_svm, alpha=0.65, s=28, marker="^", label=f"SVM (R²={metrics['SVM (SVR)']['R2']:.3f})")
ax4.scatter(y_te * 1000, rf_test * 1000,  color=c_rf,  alpha=0.65, s=28, marker="s", label=f"RF (R²={metrics['Random Forest']['R2']:.3f})")
ax4.scatter(y_te * 1000, pinn_test * 1000, color=c_pinn, alpha=0.85, s=32, marker="o", label=f"PINN (R²={metrics['PINN (Ours)']['R2']:.3f})")

min_v = min(np.min(y_te * 1000), np.min(svm_test * 1000), np.min(rf_test * 1000), np.min(pinn_test * 1000))
max_v = max(np.max(y_te * 1000), np.max(svm_test * 1000), np.max(rf_test * 1000), np.max(pinn_test * 1000))
ax4.plot([min_v, max_v], [min_v, max_v], color="#1F2937", linestyle="--", linewidth=1.5, label="Perfect Agreement (1:1)")

ax4.set_title("(d) Parity Plot: Actual vs Predicted Deflection (Test Set)", fontsize=12, fontweight="bold", pad=8)
ax4.set_xlabel("Ground Truth Deflection (mm)", fontsize=11)
ax4.set_ylabel("Predicted Deflection (mm)", fontsize=11)
ax4.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax4.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
os.makedirs("results", exist_ok=True)
out_path = "results/model_comparison_master.png"
fig.savefig(out_path, dpi=200)
plt.close(fig)
print(f"Master benchmark plot saved to: {out_path}")
