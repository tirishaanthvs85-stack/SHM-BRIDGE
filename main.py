"""
Main Execution File
Bridge Structural Health Monitoring using PINN
"""

from trainer import Trainer
from evaluate import Evaluator
from visualization import PINNVisualizer


def main():

    print("\n")
    print("=" * 70)
    print(" BRIDGE STRUCTURAL HEALTH MONITORING USING PINN ")
    print("=" * 70)

    # ------------------------------------------------
    # STEP 1 : TRAINING
    # ------------------------------------------------

    print("\nSTEP 1 : TRAINING")
    print("-" * 70)

    trainer = Trainer()
    trainer.train()

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

    print("\n")
    print("=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY ")
    print("=" * 70)


if __name__ == "__main__":
    main()