"""
plot_z24_dataset_comparison.py
Comprehensive Dataset Comparison Visualizer:
Compares all 17 scenarios, test types (AVT vs FVT), sensor setups,
and model predictions (PINN vs SVM vs RF) across the entire Z24 dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "results", "z24_file_by_file_results.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_comparisons():
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} file records across {df['scenario_code'].nunique()} scenarios.")

    # Formatted scenario labels
    df["scenario_str"] = df["scenario_code"].apply(lambda x: f"S{int(x):02d}")
    df["is_healthy"] = df["scenario_code"].isin([1, 7, 11])

    # Distinct categories of scenarios
    def categorize_scenario(code):
        c = int(code)
        if c in [1, 7, 11]:
            return "Reference Baseline"
        elif c in [2, 3, 4, 5]:
            return "Pier Settlement"
        elif c == 6:
            return "Foundation Tilt"
        elif c in [8, 9, 10]:
            return "Tendon Rupture"
        elif c in [12, 13, 14, 15]:
            return "Pier Lowering"
        elif c == 16:
            return "Concrete Spalling"
        else:
            return "Critical Failure"

    df["damage_category"] = df["scenario_code"].apply(categorize_scenario)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--"
    })

    unique_scen_str = sorted(df["scenario_str"].unique())
    scen_nums = sorted(df["scenario_code"].unique())

    # ==============================================================================
    # 1. MASTER 4-PANEL CROSS-DATASET COMPARISON GRAPH
    # ==============================================================================
    fig = plt.figure(figsize=(20, 14), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22)

    # --------------------------------------------------------------------------
    # Panel 1: AVT vs FVT Vibration RMS Across All 17 Scenarios
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    scen_avt_rms = df[df["test_type"] == "AVT"].groupby("scenario_code")["mean_rms"].mean()
    scen_fvt_rms = df[df["test_type"] == "FVT"].groupby("scenario_code")["mean_rms"].mean()

    x_indices = np.arange(len(scen_nums))
    bar_w = 0.38

    b1 = ax1.bar(x_indices - bar_w/2, [scen_avt_rms.get(s, 0) for s in scen_nums], width=bar_w, color="#2980b9", label="Ambient Vibration Tests (AVT)", edgecolor="black", linewidth=0.6)
    b2 = ax1.bar(x_indices + bar_w/2, [scen_fvt_rms.get(s, 0) for s in scen_nums], width=bar_w, color="#e67e22", label="Forced Vibration Tests (FVT)", edgecolor="black", linewidth=0.6)

    ax1.set_yscale("log")
    ax1.set_title("(A) Vibration RMS Acceleration: Ambient (AVT) vs Forced (FVT)", fontweight="bold")
    ax1.set_xlabel("Damage Scenario (S01 to S17)")
    ax1.set_ylabel("Mean Acceleration RMS ($m/s^2$, Log Scale)", fontweight="bold")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels([f"S{s:02d}" for s in scen_nums], rotation=45)
    ax1.legend(loc="upper left")

    # --------------------------------------------------------------------------
    # Panel 2: Dominant Modal Frequency Tracking & Stiffness Shift
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    scen_freq_avt = df[df["test_type"] == "AVT"].groupby("scenario_code")["dom_modal_freq_hz"].mean()
    scen_freq_fvt = df[df["test_type"] == "FVT"].groupby("scenario_code")["dom_modal_freq_hz"].mean()

    ax2.plot(x_indices, [scen_freq_avt.get(s, np.nan) for s in scen_nums], "o-", color="#2980b9", linewidth=2.0, markersize=7, label="AVT Fundamental Mode (Hz)")
    ax2.plot(x_indices, [scen_freq_fvt.get(s, np.nan) for s in scen_nums], "s--", color="#e67e22", linewidth=1.8, markersize=6, alpha=0.85, label="FVT Higher Modal Excitation (Hz)")
    ax2.axhline(3.85, color="#c0392b", linestyle=":", linewidth=1.5, label="Theoretical Baseline Mode (3.85 Hz)")

    # Highlight reference scenarios
    for idx, s in enumerate(scen_nums):
        if s in [1, 7, 11]:
            ax2.axvspan(idx - 0.4, idx + 0.4, color="#27ae60", alpha=0.15)
            ax2.text(idx, 9.8, "REF", ha="center", va="top", fontsize=8, fontweight="bold", color="#27ae60")

    ax2.set_title("(B) Dominant Modal Frequency Comparison Across Scenarios", fontweight="bold")
    ax2.set_xlabel("Damage Scenario (S01 to S17)")
    ax2.set_ylabel("Modal Frequency (Hz)", fontweight="bold")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f"S{s:02d}" for s in scen_nums], rotation=45)
    ax2.set_ylim(2.5, 10.5)
    ax2.legend(loc="upper right")

    # --------------------------------------------------------------------------
    # Panel 3: Damage Severity Index Distribution (Box Plot Comparison)
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    dmg_data = [df[df["scenario_code"] == s]["damage_index_pct"].values for s in scen_nums]
    
    box = ax3.boxplot(dmg_data, positions=x_indices, patch_artist=True, widths=0.55,
                      medianprops=dict(color="black", linewidth=1.5),
                      flierprops=dict(marker='o', markersize=3, alpha=0.5))

    for idx, (patch, s) in enumerate(zip(box['boxes'], scen_nums)):
        if s in [1, 7, 11]:
            patch.set_facecolor("#2ecc71")  # Green
        elif s in [2, 3, 4, 5]:
            patch.set_facecolor("#f39c12")  # Amber
        elif s in [8, 9, 10]:
            patch.set_facecolor("#e67e22")  # Orange
        else:
            patch.set_facecolor("#e74c3c")  # Red
        patch.set_alpha(0.85)

    ax3.set_title("(C) Damage Severity Index Spread (%): Healthy vs Damaged Scenarios", fontweight="bold")
    ax3.set_xlabel("Damage Scenario (S01 to S17)")
    ax3.set_ylabel("Damage Index (%)", fontweight="bold")
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels([f"S{s:02d}" for s in scen_nums], rotation=45)
    ax3.set_ylim(-5, 105)

    # Custom legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Healthy Reference (S01, S07, S11)"),
        Patch(facecolor="#f39c12", label="Pier Settlement (S02-S05)"),
        Patch(facecolor="#e67e22", label="Tendon Ruptures (S08-S10)"),
        Patch(facecolor="#e74c3c", label="Spalling & Failure (S12-S17)")
    ]
    ax3.legend(handles=legend_elements, loc="upper left")

    # --------------------------------------------------------------------------
    # Panel 4: Multi-Model Deflection Predictions vs Ground Truth Across Scenarios
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    scen_pinn = df.groupby("scenario_code")["pinn_pred_mm"].mean()
    scen_svm = df.groupby("scenario_code")["svm_pred_mm"].mean()
    scen_rf = df.groupby("scenario_code")["rf_pred_mm"].mean()
    scen_est = df.groupby("scenario_code")["estimated_deflection_mm"].mean()

    ax4.plot(x_indices, [scen_pinn.get(s, np.nan) for s in scen_nums], "o-", color="#1f77b4", linewidth=2.5, label="PINN Physics Model (10.08 mm Constant Profile)")
    ax4.plot(x_indices, [scen_est.get(s, np.nan) for s in scen_nums], "^:", color="#34495e", linewidth=1.5, label="Observed Sensor Deflection (mm)")
    ax4.plot(x_indices, [scen_rf.get(s, np.nan) for s in scen_nums], "s--", color="#ff7f0e", linewidth=1.8, label="Random Forest Regressor (mm)")
    ax4.plot(x_indices, [scen_svm.get(s, np.nan) for s in scen_nums], "d-.", color="#2ca02c", linewidth=1.8, label="Support Vector Machine (mm)")

    ax4.set_title("(D) Model Predictions Comparison (PINN vs RF vs SVM)", fontweight="bold")
    ax4.set_xlabel("Damage Scenario (S01 to S17)")
    ax4.set_ylabel("Predicted Deflection (mm)", fontweight="bold")
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels([f"S{s:02d}" for s in scen_nums], rotation=45)
    ax4.legend(loc="center right")

    plt.suptitle("Z24 Bridge Complete Dataset Comparison: All 17 Progressive Damage Scenarios", fontsize=16, fontweight="bold", y=0.995)

    comp_dashboard_path = os.path.join(RESULTS_DIR, "z24_complete_dataset_comparison.png")
    plt.savefig(comp_dashboard_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Dataset comparison dashboard saved to: {comp_dashboard_path}")

    # ==============================================================================
    # 2. AVT vs FVT DIRECT SCATTER & CORRELATION COMPARISON
    # ==============================================================================
    fig_avt_fvt, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Subplot A: Energy vs Peak
    df_avt = df[df["test_type"] == "AVT"]
    df_fvt = df[df["test_type"] == "FVT"]

    ax_a.scatter(df_avt["mean_rms"], df_avt["peak_accel"], color="#2980b9", alpha=0.75, s=40, label="Ambient Tests (AVT - Operational)")
    ax_a.scatter(df_fvt["mean_rms"], df_fvt["peak_accel"], color="#e67e22", alpha=0.75, s=40, marker="s", label="Forced Tests (FVT - Shaker)")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_title("Vibration Acceleration Regime Comparison: AVT vs FVT", fontweight="bold")
    ax_a.set_xlabel("Mean Channel RMS ($m/s^2$, Log Scale)", fontweight="bold")
    ax_a.set_ylabel("Peak Absolute Acceleration ($m/s^2$, Log Scale)", fontweight="bold")
    ax_a.legend()

    # Subplot B: Modal Frequencies Comparison (Histogram/KDE)
    ax_b.hist(df_avt["dom_modal_freq_hz"], bins=20, color="#2980b9", alpha=0.6, label="AVT Frequencies (Cluster around 3.85 Hz)", edgecolor="black")
    ax_b.hist(df_fvt["dom_modal_freq_hz"], bins=20, color="#e67e22", alpha=0.6, label="FVT Frequencies (Exciting 7.5 - 8.2 Hz)", edgecolor="black")
    ax_b.axvline(3.85, color="red", linestyle="--", linewidth=1.5, label="1st Bending Mode (3.85 Hz)")
    ax_b.set_title("Modal Frequency Distribution: AVT vs FVT", fontweight="bold")
    ax_b.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax_b.set_ylabel("Count of Test Files", fontweight="bold")
    ax_b.legend()

    plt.suptitle("Ambient Vibration Tests (AVT) vs Forced Vibration Tests (FVT) Comparison", fontsize=14, fontweight="bold")
    avt_fvt_path = os.path.join(RESULTS_DIR, "z24_avt_vs_fvt_comparison.png")
    plt.savefig(avt_fvt_path, bbox_inches="tight", dpi=300)
    plt.close(fig_avt_fvt)
    print(f"AVT vs FVT comparison saved to: {avt_fvt_path}")

    # ==============================================================================
    # 3. SPATIAL SETUP-BY-SETUP COMPARISON (Setup 01 to Setup 09)
    # ==============================================================================
    fig_setup, (ax_s1, ax_s2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Extract setup number
    df["setup_num"] = df["setup_name"].apply(lambda x: x[-2:] if len(x) >= 2 else "01")
    setup_group = df.groupby("setup_num").agg({
        "mean_rms": "mean",
        "peak_accel": "mean",
        "n_channels": "mean",
        "damage_index_pct": "mean"
    }).reset_index()

    ax_s1.bar(setup_group["setup_num"], setup_group["mean_rms"] * 1000.0, color="#8e44ad", edgecolor="black", alpha=0.85)
    ax_s1.set_title("Average Vibration Intensity Across Sensor Setups (01 to 09)", fontweight="bold")
    ax_s1.set_xlabel("Sensor Deck Setup Number")
    ax_s1.set_ylabel("Average Acceleration RMS ($mm/s^2$)", fontweight="bold")

    ax_s2.plot(setup_group["setup_num"], setup_group["damage_index_pct"], "o-", color="#d35400", linewidth=2.0, markersize=8)
    ax_s2.set_title("Damage Detection Consistency Across Spatial Setups", fontweight="bold")
    ax_s2.set_xlabel("Sensor Deck Setup Number")
    ax_s2.set_ylabel("Average Damage Index (%)", fontweight="bold")
    ax_s2.set_ylim(0, 100)

    plt.suptitle("Spatial Measurement Layout Comparison Across Z24 Deck Configurations", fontsize=14, fontweight="bold")
    setup_path = os.path.join(RESULTS_DIR, "z24_sensor_setups_comparison.png")
    plt.savefig(setup_path, bbox_inches="tight", dpi=300)
    plt.close(fig_setup)
    print(f"Setup comparison saved to: {setup_path}")

    print("\nAll dataset comparison graphs successfully generated!")

if __name__ == "__main__":
    generate_comparisons()
