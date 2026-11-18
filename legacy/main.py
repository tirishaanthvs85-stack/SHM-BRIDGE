"""
Main Execution File
Bridge Structural Health Monitoring using PINN + SVM
"""

import subprocess
import sys

from train_pinn_z24 import Z24PINNTrainer
from evaluate import Evaluator
from visualization import PINNVisualizer


def main():

    print("\n")
    print("=" * 70)
    print(" BRIDGE STRUCTURAL HEALTH MONITORING USING PINN ")
    print("=" * 70)

    # ------------------------------------------------
    # STEP 0 : PROCESS DATA-Z24 (VIBRATION FEATURE EXTRACTION)
    # ------------------------------------------------

    print("\nSTEP 0 : EXTRACTING FEATURES FROM Z24 BENCHMARK DATASET")
    print("-" * 70)
    subprocess.run(
        [sys.executable, "process_inputs.py"],
        check=True
    )

    # ------------------------------------------------
    # STEP 1 : TRAINING
    # ------------------------------------------------

    print("\nSTEP 1 : TRAINING PINN (Z24 BRIDGE BENCHMARK)")
    print("-" * 70)

    trainer = Z24PINNTrainer()
    trainer.load_and_split_data()
    trainer.train()
    metrics, pred_train, pred_test = trainer.evaluate()
    trainer.generate_all_plots(pred_train, pred_test)

    # ------------------------------------------------
    # STEP 2 : EVALUATION
    # ------------------------------------------------

    print("\nSTEP 2 : EVALUATION")
    print("-" * 70)

    evaluator = Evaluator()

    evaluator.load_model()
    evaluator.load_test_data()
    evaluator.predict()
    evaluator.calculate_metrics()

    # ------------------------------------------------
    # STEP 3 : VISUALIZATION
    # ------------------------------------------------

    print("\nSTEP 3 : VISUALIZATION")
    print("-" * 70)

    visualizer = PINNVisualizer()

    visualizer.load_model()
    visualizer.visualize_all()

    # ------------------------------------------------
    # STEP 4 : SVM MODEL
    # ------------------------------------------------

    print("\nSTEP 4 : SVM MODEL")
    print("-" * 70)

    # 4a  Train SVM
    print("\n[SVM] Training...")
    subprocess.run(
        [sys.executable, "train_svm.py"],
        check=True
    )

    # 4b  Evaluate SVM
    print("\n[SVM] Evaluating...")
    subprocess.run(
        [sys.executable, "evaluate_svm.py"],
        check=True
    )

    # 4c  Visualize SVM (saves plots to results/)
    print("\n[SVM] Generating plots...")
    subprocess.run(
        [sys.executable, "visualize_svm.py"],
        check=True
    )

    # ------------------------------------------------
    # STEP 5 : RANDOM FOREST MODEL
    # ------------------------------------------------

    print("\nSTEP 5 : RANDOM FOREST MODEL")
    print("-" * 70)

    # 5a  Train Random Forest
    print("\n[Random Forest] Training...")
    subprocess.run(
        [sys.executable, "train_rf.py"],
        check=True
    )

    # 5b  Evaluate Random Forest
    print("\n[Random Forest] Evaluating...")
    subprocess.run(
        [sys.executable, "evaluate_rf.py"],
        check=True
    )

    # 5c  Visualize Random Forest (saves plots to results/)
    print("\n[Random Forest] Generating plots...")
    subprocess.run(
        [sys.executable, "visualize_rf.py"],
        check=True
    )

    # ------------------------------------------------
    # STEP 6 : MULTI-MODEL COMPARISON
    # ------------------------------------------------

    print("\nSTEP 6 : MULTI-MODEL COMPARISON (PINN vs SVM vs RF)")
    print("-" * 70)
    subprocess.run(
        [sys.executable, "visualize_comparison.py"],
        check=True
    )
    subprocess.run(
        [sys.executable, "visualize_comparison_master.py"],
        check=True
    )

    print("\n")
    print("=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY ")
    print("=" * 70)


if __name__ == "__main__":
    main()