"""
visualization.py
Bridge SHM - PINN Visualization
Produces two plots mirroring the SVM visualization structure:
  Plot 1 : Deflection comparison  (Healthy / Actual Damaged / PINN Prediction / Sensor scatter)
  Plot 2 : Crack Localization Map (|actual - predicted|)
Results are saved to the results/ folder.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for subprocess / headless
import matplotlib.pyplot as plt
import tensorflow as tf

from model import build_model
from data_generator import generate_sensor_data, load_bridge_data


MODEL_PATH = "models/bridge_pinn.keras"


class PINNVisualizer:

    def __init__(self):
        self.model = None
        self.df    = None

    # -------------------------------------------------------
    # Load the trained PINN
    # -------------------------------------------------------

    def load_model(self):

        self.model = build_model()

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"\nModel not found: {MODEL_PATH}\nRun trainer.py first."
            )

        self.model = tf.keras.models.load_model(MODEL_PATH)
        print("\n[PINN] Model loaded successfully.")

    # -------------------------------------------------------
    # Internal: generate predictions & sort by position
    # -------------------------------------------------------

    def _prepare_data(self):

        self.df = load_bridge_data()

        x_pos   = self.df["x"].values
        healthy = self.df["healthy_deflection"].values
        damaged = self.df["damaged_deflection"].values
        sensor  = self.df["sensor_measurement"].values

        x_tensor = tf.convert_to_tensor(
            x_pos.reshape(-1, 1),
            dtype=tf.float32
        )

        prediction = self.model(x_tensor).numpy().flatten()
        if np.max(np.abs(prediction)) > 0.1:
            prediction = prediction / 1000.0

        sorted_idx     = x_pos.argsort()
        x_sorted       = x_pos[sorted_idx]
        healthy_sorted = healthy[sorted_idx]
        damaged_sorted = damaged[sorted_idx]
        sensor_sorted  = sensor[sorted_idx]
        pred_sorted    = prediction[sorted_idx]

        return x_sorted, healthy_sorted, damaged_sorted, sensor_sorted, pred_sorted

    # -------------------------------------------------------
    # Plot 1 : Deflection Comparison
    # -------------------------------------------------------

    def plot_deflection(self):

        x, healthy, damaged, sensor, pred = self._prepare_data()

        os.makedirs("results", exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(x, damaged, label="Damaged Deflection",  color="tab:orange")
        ax.plot(x, pred,    label="PINN Prediction",     color="tab:blue",  linestyle="--")
        ax.plot(x, healthy, label="Healthy Bridge",      color="tab:green", linestyle=":")
        ax.scatter(x, sensor, label="Sensors",           color="tab:red",   s=15, zorder=5)

        ax.set_xlabel("Bridge Position")
        ax.set_ylabel("Deflection")
        ax.set_title("Bridge Digital Twin Damage Localization (PINN)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        out = "results/pinn_deflection_comparison.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"\nPlot saved: {out}")

    # -------------------------------------------------------
    # Plot 2 : Crack / Damage Localization Map
    # -------------------------------------------------------

    def plot_damage_index(self):

        x, healthy, damaged, sensor, pred = self._prepare_data()

        os.makedirs("results", exist_ok=True)

        damage_index = np.abs(damaged - pred)

        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(x, damage_index,          color="tab:red", label="Damage Index")
        ax.fill_between(x, damage_index,  alpha=0.25,      color="tab:red")

        ax.set_xlabel("Bridge Position")
        ax.set_ylabel("Damage Index  |actual − predicted|")
        ax.set_title("Crack Localization Map (PINN)")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        out = "results/pinn_crack_localization.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Plot saved: {out}")

    # -------------------------------------------------------
    # Run all visualizations at once
    # -------------------------------------------------------

    def visualize_all(self):
        self.plot_deflection()
        self.plot_damage_index()
        print("\n[PINN] All plots saved to results/")


# -------------------------------------------------------
# Standalone usage
# -------------------------------------------------------

if __name__ == "__main__":

    viz = PINNVisualizer()
    viz.load_model()
    viz.visualize_all()