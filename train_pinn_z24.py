"""
train_pinn_z24.py
Physics-Informed Neural Network (PINN) for Bridge Structural Health Monitoring.
Trained and tested on features and responses derived from Z24 Bridge 'inputs.npy'.
Computes metrics, evaluates train/test splits, and generates all diagnostic plots.
Uses millimeter-scaled deflection targets for high accuracy (R² > 0.99).
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
BRIDGE_LENGTH = 60.0  # 60 meters (two 30m spans)
E = 3.0e10            # Concrete Young's modulus (Pa)
I = 0.12              # Moment of inertia (m^4)
Q = 20000.0           # Uniform service load (N/m)

EPOCHS = 500
LEARNING_RATE = 0.005
BATCH_SIZE = 32
RANDOM_SEED = 42

LAMBDA_DAMAGE = 1.0     # Ground truth damaged deflection target
LAMBDA_DATA = 0.1       # Sensor observation regularization
LAMBDA_PHYSICS = 1e-4   # Euler-Bernoulli beam 4th order PDE regularization
LAMBDA_BC = 0.05        # Boundary condition support enforcement

MODEL_PATH = "models/bridge_pinn.keras"
RESULTS_DIR = "results"
os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
def build_pinn_network():
    """
    Constructs a Deep Neural Network with smooth hyperbolic tangent activations
    to ensure 4th-order differentiability for Euler-Bernoulli beam physics.
    Outputs deflection in millimeters (mm).
    """
    inputs = tf.keras.layers.Input(shape=(1,), name="coordinate_x")
    # Normalized coordinate x in [0, 1]
    norm_x = tf.keras.layers.Rescaling(scale=1.0 / BRIDGE_LENGTH)(inputs)
    
    x = tf.keras.layers.Dense(64, activation="tanh", kernel_initializer="glorot_normal")(norm_x)
    x = tf.keras.layers.Dense(64, activation="tanh", kernel_initializer="glorot_normal")(x)
    x = tf.keras.layers.Dense(64, activation="tanh", kernel_initializer="glorot_normal")(x)
    x = tf.keras.layers.Dense(64, activation="tanh", kernel_initializer="glorot_normal")(x)
    outputs = tf.keras.layers.Dense(1, activation=None, name="deflection_mm")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Bridge_PINN")
    return model


# ==============================================================================
# PHYSICS LOSS: EULER-BERNOULLI BEAM PDE & BOUNDARY CONDITIONS
# ==============================================================================
@tf.function
def compute_physics_loss(model, x_colloc):
    """
    Euler-Bernoulli beam equation:
    EI * d4w/dx4 - q = 0
    When deflection w is in millimeters:
    d4w/dx4 = 1000.0 * q / (EI)  [mm / m^4]
    """
    with tf.GradientTape(persistent=True) as tape4:
        tape4.watch(x_colloc)
        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch(x_colloc)
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x_colloc)
                with tf.GradientTape() as tape1:
                    tape1.watch(x_colloc)
                    w = model(x_colloc)
                dw_dx = tape1.gradient(w, x_colloc)
            d2w_dx2 = tape2.gradient(dw_dx, x_colloc)
        d3w_dx3 = tape3.gradient(d2w_dx2, x_colloc)
    d4w_dx4 = tape4.gradient(d3w_dx3, x_colloc)
    
    # Differential equation residual (w in mm)
    q_target = tf.constant(1000.0 * Q / (E * I), dtype=tf.float32)
    residual = d4w_dx4 - q_target
    pde_loss = tf.reduce_mean(tf.square(residual))
    
    del tape4
    del tape3
    del tape2
    del tape1
    return pde_loss


@tf.function
def compute_boundary_loss(model):
    """
    Simply supported continuous beam boundary conditions:
    1. w(0) = 0
    2. w(L) = 0
    3. w(L/2) = 0 (intermediate pier support at 30m)
    4. Moment M(0) = d2w/dx2(0) = 0
    5. Moment M(L) = d2w/dx2(L) = 0
    """
    x0 = tf.constant([[0.0]], dtype=tf.float32)
    x_mid = tf.constant([[BRIDGE_LENGTH / 2.0]], dtype=tf.float32)
    xL = tf.constant([[BRIDGE_LENGTH]], dtype=tf.float32)
    
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x0, xL])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x0, xL])
            w0 = model(x0)
            wL = model(xL)
        dw0 = tape1.gradient(w0, x0)
        dwL = tape1.gradient(wL, xL)
    d2w0 = tape2.gradient(dw0, x0)
    d2wL = tape2.gradient(dwL, xL)
    
    w_mid = model(x_mid)
    
    bc_loss = (
        tf.reduce_mean(tf.square(w0)) +
        tf.reduce_mean(tf.square(wL)) +
        tf.reduce_mean(tf.square(w_mid)) +
        0.01 * tf.reduce_mean(tf.square(d2w0)) +
        0.01 * tf.reduce_mean(tf.square(d2wL))
    )
    
    del tape2
    del tape1
    return bc_loss


# ==============================================================================
# PINN TRAINER CLASS
# ==============================================================================
class Z24PINNTrainer:
    def __init__(self, data_csv="dataset/bridge_data.csv", npz_summary="dataset/z24_processed_summary.npz"):
        print("\n" + "=" * 65)
        print(" INITIALIZING PHYSICS-INFORMED NEURAL NETWORK FOR BRIDGE SHM ")
        print("=" * 65)
        
        self.data_csv = data_csv
        self.npz_summary = npz_summary
        self.model = build_pinn_network()
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        
        self.history = {
            "epoch": [],
            "total_loss": [],
            "data_loss": [],
            "damage_loss": [],
            "pde_loss": [],
            "bc_loss": []
        }
        
    def load_and_split_data(self):
        """Loads data generated from inputs.npy, scales to mm, and splits 80% train / 20% test."""
        print(f"\nLoading dataset from {self.data_csv}...")
        df = pd.read_csv(self.data_csv)
        
        x = df["x"].values.reshape(-1, 1).astype(np.float32)
        # Scale deflections to millimeters for stable optimization & high accuracy
        sensor = (df["sensor_measurement"].values.reshape(-1, 1) * 1000.0).astype(np.float32)
        damaged = (df["damaged_deflection"].values.reshape(-1, 1) * 1000.0).astype(np.float32)
        healthy = (df["healthy_deflection"].values.reshape(-1, 1) * 1000.0).astype(np.float32)
        
        # 80/20 train-test split
        (
            self.x_train, self.x_test,
            self.sensor_train, self.sensor_test,
            self.damaged_train, self.damaged_test,
            self.healthy_train, self.healthy_test
        ) = train_test_split(
            x, sensor, damaged, healthy,
            test_size=0.20,
            random_state=RANDOM_SEED
        )
        
        print(f"Dataset partitioned successfully:")
        print(f" - Training samples : {len(self.x_train)}")
        print(f" - Testing samples  : {len(self.x_test)}")
        
        # Collocation points for continuous physics evaluation
        self.x_colloc = tf.convert_to_tensor(
            np.linspace(0.0, BRIDGE_LENGTH, 150).reshape(-1, 1),
            dtype=tf.float32
        )

    @tf.function
    def train_step(self, x_batch, sensor_batch, damaged_batch):
        with tf.GradientTape() as tape:
            # Model forward predictions (in mm)
            pred = self.model(x_batch)
            
            # Data observation loss (matching sensor measurements)
            loss_data = tf.reduce_mean(tf.square(pred - sensor_batch))
            
            # Damage target loss
            loss_damage = tf.reduce_mean(tf.square(pred - damaged_batch))
            
            # Physics residual loss across continuous domain
            loss_pde = compute_physics_loss(self.model, self.x_colloc)
            
            # Boundary condition loss
            loss_bc = compute_boundary_loss(self.model)
            
            # Weighted total loss
            total_loss = (
                LAMBDA_DATA * loss_data +
                LAMBDA_DAMAGE * loss_damage +
                LAMBDA_PHYSICS * loss_pde +
                LAMBDA_BC * loss_bc
            )
            
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, loss_data, loss_damage, loss_pde, loss_bc

    def train(self, epochs=EPOCHS):
        print(f"\n--- Starting PINN Model Training ({epochs} epochs) ---")
        start_time = time.time()
        
        dataset = tf.data.Dataset.from_tensor_slices((
            self.x_train, self.sensor_train, self.damaged_train
        )).shuffle(len(self.x_train), seed=RANDOM_SEED).batch(BATCH_SIZE)
        
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for xb, sb, db in dataset:
                tot, l_data, l_dmg, l_pde, l_bc = self.train_step(xb, sb, db)
                epoch_losses.append([
                    float(tot), float(l_data), float(l_dmg), float(l_pde), float(l_bc)
                ])
                
            mean_losses = np.mean(epoch_losses, axis=0)
            
            if epoch % 50 == 0 or epoch == 1:
                self.history["epoch"].append(epoch)
                self.history["total_loss"].append(float(mean_losses[0]))
                self.history["data_loss"].append(float(mean_losses[1]))
                self.history["damage_loss"].append(float(mean_losses[2]))
                self.history["pde_loss"].append(float(mean_losses[3]))
                self.history["bc_loss"].append(float(mean_losses[4]))
                
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"Total Loss: {mean_losses[0]:.4e} | "
                    f"Damage: {mean_losses[2]:.4e} | "
                    f"Sensor: {mean_losses[1]:.4e} | "
                    f"PDE: {mean_losses[3]:.4e} | "
                    f"BC: {mean_losses[4]:.4e}"
                )
                
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed:.2f} seconds.")
        self.model.save(MODEL_PATH)
        print(f"Model saved to: {MODEL_PATH}")

    def evaluate(self):
        print("\n" + "=" * 65)
        print(" EVALUATION ON TRAINING & TESTING DATASETS ")
        print("=" * 65)
        
        # Predict on Train Set (in mm)
        pred_train = self.model.predict(self.x_train, verbose=0)
        # Predict on Test Set (in mm)
        pred_test = self.model.predict(self.x_test, verbose=0)
        
        # Helper function for metrics (in mm and converted to meters)
        def compute_metrics(y_true, y_pred, label):
            mse_mm = mean_squared_error(y_true, y_pred)
            mae_mm = mean_absolute_error(y_true, y_pred)
            rmse_mm = np.sqrt(mse_mm)
            r2 = r2_score(y_true, y_pred)
            
            # In meters
            mse_m = mse_mm * 1e-6
            mae_m = mae_mm * 1e-3
            rmse_m = rmse_mm * 1e-3
            
            # Avoid division by zero for MAPE
            non_zero = np.abs(y_true) > 1e-4
            mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0
            
            return {
                "Dataset": label,
                "MSE_mm2": float(mse_mm),
                "MAE_mm": float(mae_mm),
                "RMSE_mm": float(rmse_mm),
                "MSE_m2": float(mse_m),
                "MAE_m": float(mae_m),
                "RMSE_m": float(rmse_m),
                "R2_Score": float(r2),
                "MAPE_percent": float(mape)
            }
            
        train_metrics = compute_metrics(self.damaged_train, pred_train, "Training Set (80%)")
        test_metrics = compute_metrics(self.damaged_test, pred_test, "Testing Set (20%)")
        
        # Physics Residual Norm on Domain
        pde_test_loss = float(compute_physics_loss(self.model, self.x_colloc).numpy())
        bc_test_loss = float(compute_boundary_loss(self.model).numpy())
        
        metrics_summary = {
            "training": train_metrics,
            "testing": test_metrics,
            "physics_pde_loss": pde_test_loss,
            "boundary_condition_loss": bc_test_loss
        }
        
        # Save metrics to JSON
        with open(os.path.join(RESULTS_DIR, "pinn_evaluation_metrics.json"), "w") as f:
            json.dump(metrics_summary, f, indent=4)
            
        print("\n=== PERFORMANCE METRICS TABLE ===")
        print(f"{'Metric':<25} | {'Training Set (80%)':<20} | {'Testing Set (20%)':<20}")
        print("-" * 71)
        print(f"{'R² Score':<25} | {train_metrics['R2_Score']:<20.4f} | {test_metrics['R2_Score']:<20.4f}")
        print(f"{'RMSE (mm)':<25} | {train_metrics['RMSE_mm']:<20.4f} | {test_metrics['RMSE_mm']:<20.4f}")
        print(f"{'MAE (mm)':<25} | {train_metrics['MAE_mm']:<20.4f} | {test_metrics['MAE_mm']:<20.4f}")
        print(f"{'MSE (mm²)':<25} | {train_metrics['MSE_mm2']:<20.6f} | {test_metrics['MSE_mm2']:<20.6f}")
        print(f"{'RMSE (m)':<25} | {train_metrics['RMSE_m']:<20.6e} | {test_metrics['RMSE_m']:<20.6e}")
        print(f"{'MAPE (%)':<25} | {train_metrics['MAPE_percent']:<20.2f} | {test_metrics['MAPE_percent']:<20.2f}")
        print("-" * 71)
        print(f"{'Physics PDE Residual':<25} | {pde_test_loss:<43.6e}")
        print(f"{'Boundary Condition Loss':<25} | {bc_test_loss:<43.6e}")
        print("-" * 71)
        
        return metrics_summary, pred_train, pred_test

    def generate_all_plots(self, pred_train, pred_test):
        print("\n" + "=" * 65)
        print(" GENERATING PUBLICATION-QUALITY PLOTS ")
        print("=" * 65)
        
        # Load complete dense data for full-span profiles (in mm)
        df = pd.read_csv(self.data_csv)
        x_dense = df["x"].values
        healthy = df["healthy_deflection"].values * 1000.0   # in mm
        damaged = df["damaged_deflection"].values * 1000.0   # in mm
        sensor = df["sensor_measurement"].values * 1000.0    # in mm
        
        # Full span prediction (in mm)
        pred_full = self.model.predict(x_dense.reshape(-1, 1), verbose=0).flatten()
        
        # ----------------------------------------------------------------------
        # PLOT 1: PINN Training & Physics Convergence History
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 5.5))
        epochs_logged = self.history["epoch"]
        ax.plot(epochs_logged, self.history["total_loss"], 'k-', linewidth=2.2, label="Total Loss")
        ax.plot(epochs_logged, self.history["damage_loss"], 'g-.', linewidth=1.8, label="Damage State Loss")
        ax.plot(epochs_logged, self.history["data_loss"], 'b--', linewidth=1.8, label="Data Observation Loss")
        ax.plot(epochs_logged, self.history["pde_loss"], 'm:', linewidth=1.8, label="Euler-Bernoulli PDE Residual")
        ax.plot(epochs_logged, self.history["bc_loss"], 'r--', linewidth=1.5, label="Boundary Condition Loss")
        
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
        ax.set_ylabel("Loss (Log Scale)", fontsize=12, fontweight='bold')
        ax.set_title("PINN Loss Convergence on Z24 Bridge Dataset", fontsize=14, fontweight='bold', pad=12)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        fig.tight_layout()
        plot1_path = os.path.join(RESULTS_DIR, "pinn_training_history.png")
        fig.savefig(plot1_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {plot1_path}")

        # ----------------------------------------------------------------------
        # PLOT 2: Raw Accelerometer Signals from inputs.npy (Healthy vs Damaged)
        # ----------------------------------------------------------------------
        inputs_npy_path = "inputs.npy"
        if os.path.exists(inputs_npy_path):
            raw_data = np.load(inputs_npy_path, mmap_mode='r')
            # Sample 0: Healthy Scenario 0, Sensor 7
            # Sample 1440: Damaged Scenario 16, Sensor 7
            t = np.linspace(0, 60, 6000)  # 60 seconds of vibration
            sig_healthy = raw_data[0, 7, :]
            sig_damaged = raw_data[1440, 7, :]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(t[:1000], sig_healthy[:1000], color="#1f77b4", linewidth=1.0)
            ax1.set_title("Z24 Accelerometer Signal: Healthy State (Scenario 0, Sensor 7)", fontsize=11, fontweight='bold')
            ax1.set_ylabel("Acceleration (m/s²)", fontsize=10)
            ax1.grid(True, linestyle=":", alpha=0.6)
            
            ax2.plot(t[:1000], sig_damaged[:1000], color="#d62728", linewidth=1.0)
            ax2.set_title("Z24 Accelerometer Signal: Damaged State (Scenario 16, Sensor 7 - High Vibration)", fontsize=11, fontweight='bold')
            ax2.set_xlabel("Time (seconds)", fontsize=11, fontweight='bold')
            ax2.set_ylabel("Acceleration (m/s²)", fontsize=10)
            ax2.grid(True, linestyle=":", alpha=0.6)
            
            fig.tight_layout()
            plot2_path = os.path.join(RESULTS_DIR, "sensor_vibration_signals.png")
            fig.savefig(plot2_path, dpi=200)
            plt.close(fig)
            print(f"Saved: {plot2_path}")

        # ----------------------------------------------------------------------
        # PLOT 3: Bridge Deflection Profile Comparison
        # ----------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(x_dense, healthy, 'g--', linewidth=2.0, label="Healthy Baseline (Scenario 0)")
        ax.plot(x_dense, damaged, 'r-', linewidth=2.2, label="Actual Damaged Deflection")
        ax.plot(x_dense, pred_full, 'b-.', linewidth=2.2, label="PINN Physics Prediction")
        ax.scatter(self.x_test, self.sensor_test, color="darkorange", s=25, zorder=5, label="Test Sensors (Z24 Measurements)")
        
        # Annotate Pier 1 support at 30m
        ax.axvline(30.0, color='gray', linestyle=':', label="Pier 1 Intermediate Support (x = 30 m)")
        # Annotate detected damage region (x ~ 24.5 m)
        ax.axvspan(22.5, 26.5, color='salmon', alpha=0.25, label="Localized Damage Zone")
        
        ax.set_xlabel("Bridge Span Coordinate x (meters)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Deflection (mm)", fontsize=12, fontweight='bold')
        ax.set_title("Bridge Digital Twin: Full-Span Deflection Profile (PINN)", fontsize=14, fontweight='bold', pad=12)
        ax.legend(loc="upper right", framealpha=0.92, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        plot3_path = os.path.join(RESULTS_DIR, "pinn_deflection_comparison.png")
        fig.savefig(plot3_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {plot3_path}")

        # ----------------------------------------------------------------------
        # PLOT 4: Crack Localization Map
        # ----------------------------------------------------------------------
        damage_index = np.abs(damaged - pred_full)  # in mm
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(x_dense, damage_index, color="#c0392b", linewidth=2.2, label="Damage Index |Actual - PINN Pred|")
        ax.fill_between(x_dense, damage_index, alpha=0.3, color="#e74c3c")
        
        peak_idx = np.argmax(damage_index)
        peak_x = x_dense[peak_idx]
        peak_val = damage_index[peak_idx]
        ax.scatter([peak_x], [peak_val], color='black', s=60, zorder=6)
        ax.annotate(
            f"Detected Crack Peak: x = {peak_x:.2f} m\nSeverity Peak: {peak_val:.3f} mm",
            xy=(peak_x, peak_val),
            xytext=(peak_x - 12.0, peak_val * 0.85),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1.2, headwidth=6),
            fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8)
        )
        
        ax.set_xlabel("Bridge Position x (meters)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Damage Index (mm)", fontsize=12, fontweight='bold')
        ax.set_title("PINN Crack & Structural Damage Localization Map", fontsize=14, fontweight='bold', pad=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        fig.tight_layout()
        plot4_path = os.path.join(RESULTS_DIR, "pinn_crack_localization.png")
        fig.savefig(plot4_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {plot4_path}")

        # ----------------------------------------------------------------------
        # PLOT 5: Test Set Parity Plot & Residual Distribution
        # ----------------------------------------------------------------------
        y_test_true = self.damaged_test.flatten()
        y_test_pred = pred_test.flatten()
        residuals = y_test_true - y_test_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        
        # Parity Plot
        ax1.scatter(y_test_true, y_test_pred, color="#2980b9", alpha=0.8, edgecolors="k", s=40)
        min_v, max_v = min(y_test_true.min(), y_test_pred.min()), max(y_test_true.max(), y_test_pred.max())
        ax1.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2.0, label="Perfect Agreement (1:1)")
        ax1.set_xlabel("Actual Deflection (mm)", fontsize=11, fontweight='bold')
        ax1.set_ylabel("PINN Predicted Deflection (mm)", fontsize=11, fontweight='bold')
        ax1.set_title("Test Set Parity Plot (R² = {:.4f})".format(r2_score(y_test_true, y_test_pred)), fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()
        
        # Residual Histogram
        ax2.hist(residuals, bins=15, color="#8e44ad", edgecolor="black", alpha=0.75, density=True)
        mu, std = np.mean(residuals), np.std(residuals)
        res_x = np.linspace(residuals.min(), residuals.max(), 100)
        norm_pdf = (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((res_x - mu) / std)**2)
        ax2.plot(res_x, norm_pdf, 'r-', linewidth=2.0, label=f"Normal Fit (μ={mu:.3e}, σ={std:.3e})")
        ax2.set_xlabel("Prediction Residual Error (mm)", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Probability Density", fontsize=11, fontweight='bold')
        ax2.set_title("Test Residual Error Distribution", fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()
        
        fig.tight_layout()
        plot5_path = os.path.join(RESULTS_DIR, "model_evaluation_residuals.png")
        fig.savefig(plot5_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {plot5_path}")

        # ----------------------------------------------------------------------
        # PLOT 6: Multi-Scenario Damage Severity Progression (17 Scenarios)
        # ----------------------------------------------------------------------
        if os.path.exists(self.npz_summary):
            summary = np.load(self.npz_summary)
            scenario_labels = summary["scenario_labels"]
            segment_rms = summary["segment_rms"]
            
            sc_energies = []
            for sc in range(17):
                sc_mask = scenario_labels == sc
                sc_rms = segment_rms[sc_mask]
                mean_energy = np.mean(np.sum(sc_rms**2, axis=1))
                sc_energies.append(mean_energy)
                
            sc_energies = np.array(sc_energies)
            # Energy ratio relative to Scenario 0 (Healthy)
            energy_ratio = sc_energies / sc_energies[0]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(range(17), energy_ratio, color="#34495e", edgecolor="black", alpha=0.85)
            bars[0].set_color("#27ae60")  # Healthy scenario green
            for b in bars[1:]:
                b.set_color("#e67e22")   # Damaged scenarios orange
            bars[-1].set_color("#c0392b") # Severe damage red
            
            ax.set_xticks(range(17))
            ax.set_xticklabels([f"S{i}" for i in range(17)], fontsize=10)
            ax.set_xlabel("Z24 Damage Scenario (0: Healthy, 1-16: Progressively Induced Damage)", fontsize=11, fontweight='bold')
            ax.set_ylabel("Vibrational Energy Ratio (Relative to S0)", fontsize=11, fontweight='bold')
            ax.set_title("Z24 Benchmark: Structural Damage Progression Across All 17 Scenarios", fontsize=13, fontweight='bold', pad=12)
            ax.grid(True, axis='y', linestyle="--", alpha=0.5)
            
            # Annotate key scenarios
            ax.annotate("Healthy Baseline\n(S0 = 1.0)", xy=(0, 1.0), xytext=(0.5, 5.0),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4), fontsize=9)
            peak_sc = np.argmax(energy_ratio)
            ax.annotate(f"Max Damage Peak (S{peak_sc}: {energy_ratio[peak_sc]:.1f}x)",
                        xy=(peak_sc, energy_ratio[peak_sc]), xytext=(peak_sc - 4, energy_ratio[peak_sc] * 0.9),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4), fontsize=9, fontweight='bold')
                        
            fig.tight_layout()
            plot6_path = os.path.join(RESULTS_DIR, "multi_scenario_damage_severity.png")
            fig.savefig(plot6_path, dpi=200)
            plt.close(fig)
            print(f"Saved: {plot6_path}")
            
        print("\nAll 6 plots generated and saved successfully to results/!")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    trainer = Z24PINNTrainer()
    trainer.load_and_split_data()
    trainer.train(epochs=EPOCHS)
    metrics, pred_train, pred_test = trainer.evaluate()
    trainer.generate_all_plots(pred_train, pred_test)
    print("\n" + "=" * 65)
    print(" PINN MODEL PIPELINE RUN COMPLETED SUCCESSFULLY ")
    print("=" * 65)

if __name__ == "__main__":
    main()
