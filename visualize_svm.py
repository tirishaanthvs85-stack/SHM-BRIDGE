"""
visualize_svm.py
Bridge SHM - SVM Visualization
Produces two plots identical in structure to the PINN visualization:
  Plot 1 : Deflection comparison  (Healthy / Actual Damaged / SVM Prediction / Sensor scatter)
  Plot 2 : Crack Localization Map (|actual - predicted|)
Results are saved to the results/ folder.
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for subprocess / headless
import matplotlib.pyplot as plt


# --------------------------------------------------
# LOAD DATA  (same CSV used by PINN)
# --------------------------------------------------

DATA_PATH = "dataset/bridge_data.csv"

data = pd.read_csv(DATA_PATH)

x_raw   = data[["x", "sensor_measurement"]].values   # 2-feature input (matches training)
x_pos   = data["x"].values                            # 1-D position for plotting
healthy = data["healthy_deflection"].values
damaged = data["damaged_deflection"].values
sensor  = data["sensor_measurement"].values


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load("svm_bridge_model.pkl")


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

svm_prediction = model.predict(x_raw)


# --------------------------------------------------
# SORT BY POSITION (for clean line plots)
# --------------------------------------------------

sorted_idx     = x_pos.argsort()
x_sorted       = x_pos[sorted_idx]
healthy_sorted = healthy[sorted_idx]
damaged_sorted = damaged[sorted_idx]
sensor_sorted  = sensor[sorted_idx]
svm_sorted     = svm_prediction[sorted_idx]


# --------------------------------------------------
# ENSURE RESULTS FOLDER EXISTS
# --------------------------------------------------

os.makedirs("results", exist_ok=True)


# ====================================================
# PLOT 1 : DEFLECTION COMPARISON
# (mirrors PINN visualization.py Plot 1)
# ====================================================

fig1, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(x_sorted, damaged_sorted, label="Damaged Deflection",  color="tab:orange")
ax1.plot(x_sorted, svm_sorted,     label="SVM Prediction",      color="tab:blue",  linestyle="--")
ax1.plot(x_sorted, healthy_sorted, label="Healthy Bridge",       color="tab:green", linestyle=":")
ax1.scatter(x_sorted, sensor_sorted, label="Sensors",           color="tab:red",   s=15, zorder=5)

ax1.set_xlabel("Bridge Position")
ax1.set_ylabel("Deflection")
ax1.set_title("Bridge Digital Twin Damage Localization (SVM)")
ax1.legend()
ax1.grid(True)

fig1.tight_layout()
fig1.savefig("results/svm_deflection_comparison.png", dpi=150)
plt.close(fig1)
print("\nPlot saved: results/svm_deflection_comparison.png")


# ====================================================
# PLOT 2 : CRACK / DAMAGE LOCALIZATION MAP
# (mirrors PINN visualization.py Plot 2)
# ====================================================

damage_index = np.abs(damaged_sorted - svm_sorted)

fig2, ax2 = plt.subplots(figsize=(10, 4))

ax2.plot(x_sorted, damage_index,        color="tab:red", label="Damage Index")
ax2.fill_between(x_sorted, damage_index, alpha=0.25,     color="tab:red")

ax2.set_xlabel("Bridge Position")
ax2.set_ylabel("Damage Index  |actual − predicted|")
ax2.set_title("Crack Localization Map (SVM)")
ax2.legend()
ax2.grid(True)

fig2.tight_layout()
fig2.savefig("results/svm_crack_localization.png", dpi=150)
plt.close(fig2)
print("Plot saved: results/svm_crack_localization.png")