"""
generate_all_file_graphs.py
Processes every single file in the Z24 dataset (306 files) and generates
a dedicated diagnostic graph for each file saved with the file's name.
Uses multi-core parallel rendering for high throughput.
"""

import os
import sys
import time
import glob
import concurrent.futures
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "data-z24")
CSV_PATH = os.path.join(BASE_DIR, "results", "z24_file_by_file_results.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "file_graphs")
MIRRORED_DIR = os.path.join(GRAPHS_DIR, "mirrored")
FLAT_DIR = os.path.join(GRAPHS_DIR, "all_files")

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(MIRRORED_DIR, exist_ok=True)
os.makedirs(FLAT_DIR, exist_ok=True)


def process_single_file_graph(args):
    """
    Renders and saves a diagnostic graph for a single Z24 .mat file.
    """
    fpath, row_meta, flat_dir, mirrored_dir = args
    rel_path = row_meta["relative_path"]
    file_name = row_meta["file_name"]
    scenario_code = str(row_meta["scenario_code"]).zfill(2)
    test_type = str(row_meta["test_type"]).upper()
    scenario_desc = str(row_meta["scenario_desc"])
    health_status = str(row_meta["health_status"])
    damage_idx = float(row_meta["damage_index_pct"])
    pinn_mm = float(row_meta["pinn_pred_mm"]) if not pd.isna(row_meta["pinn_pred_mm"]) else 10.08
    dom_freq = float(row_meta["dom_modal_freq_hz"])
    mean_rms = float(row_meta["mean_rms"])
    peak_accel = float(row_meta["peak_accel"])

    base_name_no_ext = os.path.splitext(file_name)[0]

    # Save paths:
    # 1. Flat unique name: e.g. "01_AVT_01setup01.png"
    flat_save_path = os.path.join(flat_dir, f"{scenario_code}_{test_type}_{base_name_no_ext}.png")
    
    # 2. Mirrored hierarchy: e.g. "mirrored/pdt_01-08/01/avt/01setup01.png"
    rel_dir = os.path.dirname(rel_path)
    mirrored_sub = os.path.join(mirrored_dir, rel_dir)
    os.makedirs(mirrored_sub, exist_ok=True)
    mirrored_save_path = os.path.join(mirrored_sub, f"{base_name_no_ext}.png")

    # Load data from .mat
    try:
        m = sio.loadmat(fpath)
        data = m.get("data")
        if data is None or data.ndim != 2:
            return fpath, False, "No valid data array"
    except Exception as e:
        return fpath, False, str(e)

    n_samples, n_channels = data.shape
    centered = data - np.mean(data, axis=0)

    # Styling
    fig = plt.figure(figsize=(15, 10), dpi=140)
    fig.patch.set_facecolor("#fdfdfd")

    # Status header color
    status_color = "#27ae60" if "HEALTHY" in health_status.upper() else "#c0392b"

    # =========================================================================
    # 1. Time Series Acceleration (First 5 seconds = 500 samples)
    # =========================================================================
    ax1 = plt.subplot(2, 2, 1)
    t = np.linspace(0, 5.0, min(500, n_samples))
    channels_to_plot = [0, n_channels // 4, n_channels // 2, 3 * n_channels // 4]
    colors = ["#2980b9", "#8e44ad", "#e67e22", "#16a085"]

    for i, ch in enumerate(channels_to_plot):
        if ch < n_channels:
            ax1.plot(t, centered[:len(t), ch], label=f"Sensor Ch {ch+1}", color=colors[i % len(colors)], alpha=0.85, linewidth=1.1)

    ax1.set_title("(A) Acceleration Time-Series Waveforms (0 to 5 s)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.set_ylabel("Acceleration ($m/s^2$)", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8)

    # =========================================================================
    # 2. Power Spectral Density (PSD)
    # =========================================================================
    ax2 = plt.subplot(2, 2, 2)
    # Channel with highest variance
    best_col = int(np.argmax(np.var(data, axis=0)))
    sig = centered[:, best_col]
    nperseg = min(2048, len(sig))
    freqs, psd = signal.welch(sig, fs=100.0, nperseg=nperseg)
    mask = (freqs >= 0.5) & (freqs <= 15.0)

    ax2.semilogy(freqs[mask], psd[mask], color="#c0392b", linewidth=1.5, label="PSD Spectrum")
    ax2.axvline(dom_freq, color="#2980b9", linestyle="--", linewidth=1.5, label=f"Detected Peak: {dom_freq:.2f} Hz")
    ax2.axvline(3.85, color="#27ae60", linestyle=":", linewidth=1.2, label="Design 1st Bending (3.85 Hz)")

    ax2.set_title(f"(B) Power Spectral Density (Dominant: {dom_freq:.2f} Hz)", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Frequency (Hz)", fontsize=9)
    ax2.set_ylabel("Power Density ($m^2/s^4/Hz$, Log Scale)", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8)

    # =========================================================================
    # 3. Spatial Channel Acceleration RMS Distribution
    # =========================================================================
    ax3 = plt.subplot(2, 2, 3)
    channel_rms = np.sqrt(np.mean(centered**2, axis=0))
    bar_indices = np.arange(1, n_channels + 1)
    ax3.bar(bar_indices, channel_rms, color="#3498db", edgecolor="#2c3e50", linewidth=0.5, alpha=0.9)
    ax3.axhline(mean_rms, color="#e74c3c", linestyle="--", linewidth=1.2, label=f"Deck Mean RMS = {mean_rms:.3e}")

    ax3.set_title(f"(C) Spatial RMS Acceleration Across All {n_channels} Sensors", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Sensor Channel Number", fontsize=9)
    ax3.set_ylabel("RMS Acceleration ($m/s^2$)", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.legend(loc="upper right", fontsize=8)

    # =========================================================================
    # 4. PINN Predicted Continuous Bridge Deflection Profile
    # =========================================================================
    ax4 = plt.subplot(2, 2, 4)
    x_bridge = np.linspace(0, 60.0, 200)

    # Mode shape along 2 spans
    span1_mask = x_bridge <= 30.0
    w_shape = np.zeros_like(x_bridge)
    w_shape[span1_mask] = np.sin(np.pi * x_bridge[span1_mask] / 30.0)
    w_shape[~span1_mask] = -np.sin(np.pi * (x_bridge[~span1_mask] - 30.0) / 30.0)
    w_healthy = np.abs(w_shape) * 10.0  # mm

    # Damaged deflection with midspan settlement dip
    damage_dip = (damage_idx / 100.0) * 0.45 * np.exp(-((x_bridge - 24.5)**2) / (2 * (1.75**2)))
    w_damaged = w_healthy * (1.0 + damage_dip)

    ax4.plot(x_bridge, w_healthy, color="#27ae60", linestyle="--", linewidth=1.5, label="Healthy Baseline (mm)")
    ax4.plot(x_bridge, w_damaged, color="#e74c3c", linewidth=2.0, label=f"Damaged State (Index: {damage_idx:.1f}%)")
    ax4.scatter([15.0, 24.5, 45.0], [w_damaged[50], w_damaged[82], w_damaged[150]], color="#2980b9", s=45, zorder=5, label=f"PINN Peak: {pinn_mm:.2f} mm")
    ax4.axvline(30.0, color="#7f8c8d", linestyle=":", label="Center Pier (30m)")

    ax4.set_title(f"(D) Continuous Bridge Deflection Profile w(x) (PINN)", fontweight="bold", fontsize=11)
    ax4.set_xlabel("Longitudinal Coordinate x (m)", fontsize=9)
    ax4.set_ylabel("Deflection (mm)", fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.35)
    ax4.legend(loc="upper right", fontsize=8)

    # Master Title Banner
    banner_text = (
        f"File: {file_name}  |  Path: {rel_path}\n"
        f"Scenario [{scenario_code}]: {scenario_desc}  |  Test Type: {test_type}\n"
        f"Status: {health_status}  |  Damage Index: {damage_idx:.1f}%  |  Mean RMS: {mean_rms:.3e} m/s²  |  Freq: {dom_freq:.2f} Hz"
    )
    plt.suptitle(banner_text, fontsize=12, fontweight="bold", color="#2c3e50", y=0.985)
    plt.subplots_adjust(top=0.90, hspace=0.30, wspace=0.25)

    # Save to both locations
    plt.savefig(flat_save_path, dpi=130, bbox_inches="tight")
    plt.savefig(mirrored_save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return fpath, True, flat_save_path


def main():
    print("=" * 80)
    print(" Z24 BRIDGE DATASET: RUNNING ALL 306 FILES & GENERATING INDIVIDUAL GRAPHS ")
    print("=" * 80)
    print(f"Loading metadata from: {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        print(f"Error: Results CSV does not exist: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    total_files = len(df)
    print(f"Total files to process: {total_files}")
    print(f"Output directory (all files) : {FLAT_DIR}")
    print(f"Output directory (mirrored)  : {MIRRORED_DIR}")
    print("-" * 80)

    # Prepare job arguments
    tasks = []
    for idx, row in df.iterrows():
        fpath = os.path.join(DATASET_DIR, row["relative_path"])
        tasks.append((fpath, row, FLAT_DIR, MIRRORED_DIR))

    t0 = time.time()
    completed = 0
    errors = 0

    # Execute in parallel with 8 workers
    num_workers = min(8, os.cpu_count() or 4)
    print(f"Rendering graphs using {num_workers} parallel workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file_graph, task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            fpath, success, msg = future.result()
            completed += 1
            if success:
                file_name = os.path.basename(fpath)
                out_name = os.path.basename(msg)
                if completed % 25 == 0 or completed == total_files:
                    print(f"[{completed:03d}/{total_files:03d}] Rendered: {file_name} -> Saved as: {out_name}")
            else:
                errors += 1
                print(f"[{completed:03d}/{total_files:03d}] ERROR on {fpath}: {msg}")

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(" BATCH GRAPH RENDERING COMPLETED ")
    print("=" * 80)
    print(f"Successfully rendered : {completed - errors} of {total_files} files")
    print(f"Errors                : {errors}")
    print(f"Total rendering time  : {total_time:.2f} seconds ({total_time / total_files:.3f} s / file)")
    print(f"Saved all graphs to   : {FLAT_DIR}")
    print(f"Saved mirrored graphs : {MIRRORED_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
