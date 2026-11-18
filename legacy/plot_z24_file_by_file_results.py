"""
plot_z24_file_by_file_results.py
Generates comprehensive, publication-quality graphical visualizations
from the file-by-file Z24 dataset execution results (z24_file_by_file_results.csv).
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


def generate_graphs():
    print(f"Loading file-by-file results from: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Results CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Successfully loaded {len(df)} file records.")

    # Apply modern styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 15,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--"
    })

    file_indices = df["file_index"].values
    scenarios = df["scenario_code"].astype(str).str.zfill(2).values

    # Scenario boundary points (where scenario changes)
    boundary_indices = [1]
    scenario_ticks = []
    scenario_tick_labels = []
    
    unique_scenarios = sorted(df["scenario_code"].unique())
    for sc in unique_scenarios:
        sc_df = df[df["scenario_code"] == sc]
        first_idx = sc_df["file_index"].iloc[0]
        mid_idx = sc_df["file_index"].mean()
        boundary_indices.append(first_idx)
        scenario_ticks.append(mid_idx)
        scenario_tick_labels.append(f"S{int(sc):02d}")

    # ==============================================================================
    # 1. MASTER MULTI-PANEL SHM DASHBOARD (6 Panels)
    # ==============================================================================
    fig = plt.figure(figsize=(20, 15), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.32, wspace=0.22)

    # Palette
    color_pinn = "#1f77b4"
    color_svm = "#2ca02c"
    color_rf = "#ff7f0e"
    color_dmg = "#d62728"
    color_avt = "#007acc"
    color_fvt = "#e67e22"

    # --- PANEL 1: File-by-File Damage Index Progression ---
    ax1 = fig.add_subplot(gs[0, :])
    # Shading alternating scenarios
    for i in range(len(unique_scenarios)):
        sc = unique_scenarios[i]
        sc_df = df[df["scenario_code"] == sc]
        xmin = sc_df["file_index"].min() - 0.5
        xmax = sc_df["file_index"].max() + 0.5
        alpha_val = 0.08 if i % 2 == 0 else 0.02
        ax1.axvspan(xmin, xmax, color="gray", alpha=alpha_val)

    # Plot AVT and FVT separately
    mask_avt = df["test_type"] == "AVT"
    mask_fvt = df["test_type"] == "FVT"

    ax1.plot(df["file_index"], df["damage_index_pct"], color="#990000", linewidth=1.5, alpha=0.7, label="Damage Severity Trend")
    ax1.scatter(df.loc[mask_avt, "file_index"], df.loc[mask_avt, "damage_index_pct"], color=color_avt, s=28, alpha=0.85, label="Ambient Vibration Tests (AVT)", zorder=4)
    ax1.scatter(df.loc[mask_fvt, "file_index"], df.loc[mask_fvt, "damage_index_pct"], color=color_fvt, s=28, alpha=0.85, marker="s", label="Forced Vibration Tests (FVT)", zorder=4)

    # Highlight reference healthy baselines (S01, S07, S11)
    baseline_mask = df["scenario_code"].isin([1, 7, 11])
    ax1.scatter(df.loc[baseline_mask, "file_index"], df.loc[baseline_mask, "damage_index_pct"], color="#27ae60", s=65, edgecolors="black", linewidth=1.2, label="Healthy Reference States (S01, S07, S11)", zorder=5)

    ax1.set_title("(A) File-by-File Structural Health & Damage Severity Index (%) Across All 306 Test Files", fontweight="bold")
    ax1.set_ylabel("Damage Index (%)", fontweight="bold")
    ax1.set_xlim(0, len(df) + 1)
    ax1.set_ylim(-5, 105)
    ax1.set_xticks(scenario_ticks)
    ax1.set_xticklabels(scenario_tick_labels)
    ax1.legend(loc="upper left", ncol=4, frameon=True)

    # --- PANEL 2: Vibration RMS Acceleration (Log Scale) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df["file_index"], df["mean_rms"], color="#34495e", linewidth=1.2, label="Mean Channel RMS")
    ax2.plot(df["file_index"], df["max_rms"], color="#e74c3c", linewidth=1.0, linestyle=":", alpha=0.75, label="Peak Channel RMS")
    ax2.set_yscale("log")
    ax2.set_title("(B) Vibration Acceleration Levels (RMS) Across All Files", fontweight="bold")
    ax2.set_xlabel("Z24 File Index (1 to 306)")
    ax2.set_ylabel("Acceleration RMS ($m/s^2$)", fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.set_xlim(0, len(df) + 1)

    # --- PANEL 3: Dominant Modal Frequency Tracking ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df.loc[mask_avt, "file_index"], df.loc[mask_avt, "dom_modal_freq_hz"], "o-", color=color_avt, markersize=3.5, linewidth=1.0, label="AVT Fundamental Mode (1st Bending ~3.85 Hz)")
    ax3.plot(df.loc[mask_fvt, "file_index"], df.loc[mask_fvt, "dom_modal_freq_hz"], "s-", color=color_fvt, markersize=3.5, linewidth=1.0, alpha=0.7, label="FVT Higher Modal Excitation (~7-8 Hz)")
    ax3.axhline(3.85, color="red", linestyle="--", linewidth=1.2, label="Design Baseline Bending (3.85 Hz)")
    ax3.set_title("(C) Dominant Modal Frequency Tracking File-by-File", fontweight="bold")
    ax3.set_xlabel("Z24 File Index (1 to 306)")
    ax3.set_ylabel("Frequency (Hz)", fontweight="bold")
    ax3.set_ylim(0, 11)
    ax3.legend(loc="lower right")
    ax3.set_xlim(0, len(df) + 1)

    # --- PANEL 4: Multi-Model Deflection Predictions File-by-File ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(df["file_index"], df["pinn_pred_mm"], color=color_pinn, linewidth=2.0, label="PINN Model (Physics-Informed)")
    ax4.plot(df["file_index"], df["rf_pred_mm"], color=color_rf, linewidth=1.2, alpha=0.85, label="Random Forest Model")
    ax4.plot(df["file_index"], df["svm_pred_mm"], color=color_svm, linewidth=1.2, linestyle="--", alpha=0.85, label="Support Vector Machine (SVM)")
    ax4.set_title("(D) File-by-File Multi-Model Deflection Predictions (mm)", fontweight="bold")
    ax4.set_xlabel("Z24 File Index (1 to 306)")
    ax4.set_ylabel("Deflection Prediction (mm)", fontweight="bold")
    ax4.legend(loc="upper right")
    ax4.set_xlim(0, len(df) + 1)

    # --- PANEL 5: Scenario-wise Average Damage Severity Bar Chart ---
    ax5 = fig.add_subplot(gs[2, 1])
    scen_summary = df.groupby("scenario_code").agg({
        "damage_index_pct": "mean",
        "scenario_desc": "first"
    }).reset_index()

    bar_colors = []
    for sc in scen_summary["scenario_code"]:
        if sc in [1, 7, 11]:
            bar_colors.append("#27ae60")  # Healthy Green
        elif sc in [2, 3, 4, 5]:
            bar_colors.append("#f39c12")  # Settlement Amber
        elif sc in [8, 9, 10]:
            bar_colors.append("#e67e22")  # Tendon Rupture Orange
        else:
            bar_colors.append("#c0392b")  # Severe Red

    bars = ax5.bar(
        [f"S{int(s):02d}" for s in scen_summary["scenario_code"]],
        scen_summary["damage_index_pct"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9
    )
    ax5.set_title("(E) Scenario-Aggregated Average Damage Index (%)", fontweight="bold")
    ax5.set_ylabel("Average Damage Severity (%)", fontweight="bold")
    ax5.set_xlabel("Damage Scenario (S01: Healthy -> S17: Critical Failure)")
    ax5.set_ylim(0, 110)

    # Add labels on top of bars
    for bar in bars:
        h = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.suptitle("Z24 Continuous Bridge Structural Health Monitoring: Complete File-by-File Analysis", fontsize=16, fontweight="bold", y=0.995)

    dashboard_path = os.path.join(RESULTS_DIR, "z24_file_by_file_dashboard.png")
    plt.savefig(dashboard_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Master dashboard successfully saved to: {dashboard_path}")

    # ==============================================================================
    # 2. INDIVIDUAL STANDALONE GRAPHS FOR DETAILED INSPECTION
    # ==============================================================================
    # Standalone Graph 1: High-Resolution Damage Index Timeline
    fig1, ax = plt.subplots(figsize=(14, 6), dpi=300)
    for i in range(len(unique_scenarios)):
        sc = unique_scenarios[i]
        sc_df = df[df["scenario_code"] == sc]
        xmin = sc_df["file_index"].min() - 0.5
        xmax = sc_df["file_index"].max() + 0.5
        alpha_val = 0.08 if i % 2 == 0 else 0.02
        ax.axvspan(xmin, xmax, color="gray", alpha=alpha_val)

    ax.plot(df["file_index"], df["damage_index_pct"], color="#b71540", linewidth=1.8, label="Damage Severity Trend")
    ax.scatter(df.loc[mask_avt, "file_index"], df.loc[mask_avt, "damage_index_pct"], color="#0a3d62", s=35, label="Ambient Vibration Tests (AVT)")
    ax.scatter(df.loc[mask_fvt, "file_index"], df.loc[mask_fvt, "damage_index_pct"], color="#e58e26", s=35, marker="s", label="Forced Vibration Tests (FVT)")
    ax.scatter(df.loc[baseline_mask, "file_index"], df.loc[baseline_mask, "damage_index_pct"], color="#38ada9", s=70, edgecolors="black", label="Baseline Reference States")

    ax.set_title("Z24 Structural Health Monitoring: File-by-File Damage Index Progression", fontsize=14, fontweight="bold")
    ax.set_xlabel("Progressive Test File Index (1 to 306)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Damage Severity Index (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(scenario_ticks)
    ax.set_xticklabels(scenario_tick_labels)
    ax.set_xlim(0, len(df) + 1)
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left", frameon=True)
    g1_path = os.path.join(RESULTS_DIR, "z24_damage_index_timeline.png")
    plt.savefig(g1_path, bbox_inches="tight", dpi=300)
    plt.close(fig1)

    # Standalone Graph 2: Multi-Model Deflection Predictions Comparison
    fig2, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.plot(df["file_index"], df["pinn_pred_mm"], color="#1e3799", linewidth=2.5, label="PINN (Physics-Informed Neural Network)")
    ax.plot(df["file_index"], df["rf_pred_mm"], color="#e55039", linewidth=1.5, alpha=0.9, label="Random Forest Model")
    ax.plot(df["file_index"], df["svm_pred_mm"], color="#78e08f", linewidth=1.5, linestyle="--", alpha=0.9, label="Support Vector Machine (SVM)")
    ax.set_title("PINN vs SVM vs Random Forest: File-by-File Predicted Deflection (mm)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Progressive Test File Index (1 to 306)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Deflection Prediction (mm)", fontsize=11, fontweight="bold")
    ax.set_xticks(scenario_ticks)
    ax.set_xticklabels(scenario_tick_labels)
    ax.set_xlim(0, len(df) + 1)
    ax.legend(loc="upper right", frameon=True)
    g2_path = os.path.join(RESULTS_DIR, "z24_model_predictions_comparison.png")
    plt.savefig(g2_path, bbox_inches="tight", dpi=300)
    plt.close(fig2)

    # Standalone Graph 3: Vibration Acceleration Energy Timeline
    fig3, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.plot(df["file_index"], df["mean_rms"], color="#4a69bd", linewidth=1.5, label="Mean Sensor RMS ($m/s^2$)")
    ax.plot(df["file_index"], df["max_rms"], color="#eb2f06", linewidth=1.2, linestyle=":", alpha=0.8, label="Max Sensor RMS ($m/s^2$)")
    ax.set_yscale("log")
    ax.set_title("Z24 Sensor Acceleration Vibration Levels Across All 306 Files", fontsize=14, fontweight="bold")
    ax.set_xlabel("Progressive Test File Index (1 to 306)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Acceleration RMS ($m/s^2$, Log Scale)", fontsize=11, fontweight="bold")
    ax.set_xticks(scenario_ticks)
    ax.set_xticklabels(scenario_tick_labels)
    ax.set_xlim(0, len(df) + 1)
    ax.legend(loc="upper right", frameon=True)
    g3_path = os.path.join(RESULTS_DIR, "z24_vibration_rms_timeline.png")
    plt.savefig(g3_path, bbox_inches="tight", dpi=300)
    plt.close(fig3)

    print("All standalone graphs successfully exported:")
    print(f" - {g1_path}")
    print(f" - {g2_path}")
    print(f" - {g3_path}")


if __name__ == "__main__":
    generate_graphs()
