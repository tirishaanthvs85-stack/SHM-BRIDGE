"""
visualize_comparison.py
Comprehensive Comparison Plot: PINN vs SVM vs Random Forest
Bridge Structural Health Monitoring
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

DATA_PATH = "dataset/bridge_data.csv"
data = pd.read_csv(DATA_PATH)

x_raw   = data[["x", "sensor_measurement"]].values
x_pos   = data["x"].values
healthy = data["healthy_deflection"].values
damaged = data["damaged_deflection"].values
sensor  = data["sensor_measurement"].values


# --------------------------------------------------
# LOAD MODELS & PREDICT
# --------------------------------------------------

# 1. PINN
pinn_model = tf.keras.models.load_model("models/bridge_pinn.keras", safe_mode=False)
x_tensor = tf.convert_to_tensor(x_pos.reshape(-1, 1), dtype=tf.float32)
pinn_pred = pinn_model(x_tensor).numpy().flatten()
# Automatically scale to meters if model predicted in millimeters
if np.max(np.abs(pinn_pred)) > 0.1:
    pinn_pred = pinn_pred / 1000.0

# 2. SVM
svm_model = joblib.load("svm_bridge_model.pkl")
svm_pred = svm_model.predict(x_raw)

# 3. Random Forest
rf_model = joblib.load("rf_bridge_model.pkl")
rf_pred = rf_model.predict(x_raw)


# --------------------------------------------------
# SORT BY POSITION
# --------------------------------------------------

sorted_idx     = x_pos.argsort()
x_sorted       = x_pos[sorted_idx]
healthy_sorted = healthy[sorted_idx]
damaged_sorted = damaged[sorted_idx]
sensor_sorted  = sensor[sorted_idx]
pinn_sorted    = pinn_pred[sorted_idx]
svm_sorted     = svm_pred[sorted_idx]
rf_sorted      = rf_pred[sorted_idx]

os.makedirs("results", exist_ok=True)


# ====================================================
# COMPARISON PLOT 1: DEFLECTION COMPARISON (ALL MODELS)
# ====================================================

fig1, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(x_sorted, damaged_sorted, label="Damaged Deflection (Ground Truth)", color="black", linewidth=2)
ax1.plot(x_sorted, pinn_sorted,    label="PINN Prediction",                   color="tab:blue",   linestyle="--", linewidth=1.5)
ax1.plot(x_sorted, svm_sorted,     label="SVM Prediction",                    color="tab:cyan",   linestyle="-.", linewidth=1.5)
ax1.plot(x_sorted, rf_sorted,      label="Random Forest Prediction",          color="tab:purple", linestyle=":",  linewidth=1.8)
ax1.plot(x_sorted, healthy_sorted, label="Healthy Baseline",                  color="tab:green",  linestyle=":",  alpha=0.7)
ax1.scatter(x_sorted, sensor_sorted, label="Sensors",                         color="tab:red",    s=12, alpha=0.6, zorder=5)

ax1.set_xlabel("Bridge Position (m)")
ax1.set_ylabel("Deflection (m)")
ax1.set_title("Bridge Digital Twin Deflection: PINN vs SVM vs Random Forest")
ax1.legend()
ax1.grid(True)

fig1.tight_layout()
fig1.savefig("results/model_comparison_deflection.png", dpi=150)
plt.close(fig1)
print("\nPlot saved: results/model_comparison_deflection.png")


# ====================================================
# COMPARISON PLOT 2: DAMAGE LOCALIZATION MAP (ALL MODELS)
# ====================================================

fig2, ax2 = plt.subplots(figsize=(12, 5))

pinn_di = np.abs(damaged_sorted - pinn_sorted)
svm_di  = np.abs(damaged_sorted - svm_sorted)
rf_di   = np.abs(damaged_sorted - rf_sorted)

ax2.plot(x_sorted, pinn_di, label="PINN Error |actual - pred|", color="tab:blue",   linewidth=1.5)
ax2.plot(x_sorted, svm_di,  label="SVM Error |actual - pred|",  color="tab:orange", linestyle="--", linewidth=1.5)
ax2.plot(x_sorted, rf_di,   label="RF Error |actual - pred|",   color="tab:purple", linestyle="-.", linewidth=1.5)

ax2.set_xlabel("Bridge Position (m)")
ax2.set_ylabel("Damage Index / Prediction Residual")
ax2.set_title("Crack Localization Residual Comparison: PINN vs SVM vs Random Forest")
ax2.legend()
ax2.grid(True)

fig2.tight_layout()
fig2.savefig("results/model_comparison_residuals.png", dpi=150)
plt.close(fig2)
print("Plot saved: results/model_comparison_residuals.png")
