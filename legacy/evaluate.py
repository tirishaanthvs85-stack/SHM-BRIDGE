"""
evaluate.py
Bridge SHM PINN Evaluation
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from model import build_model
from data_generator import generate_sensor_data

MODEL_PATH = "models/bridge_pinn.keras"


class Evaluator:

    def __init__(self):

        self.model = None

        self.x = None
        self.sensor = None
        self.damage = None
        self.prediction = None

    # --------------------------------------------
    # Load trained model
    # --------------------------------------------

    def load_model(self):

        if not os.path.exists(MODEL_PATH):

            raise FileNotFoundError(
                f"\nModel not found:\n{MODEL_PATH}\n"
                "Run trainer.py first."
            )

        self.model = tf.keras.models.load_model(MODEL_PATH)

        print("\nModel Loaded Successfully")

    # --------------------------------------------
    # Load Test Data
    # --------------------------------------------

    def load_test_data(self):

        self.x, self.sensor, self.damage = generate_sensor_data()

    # --------------------------------------------
    # Prediction
    # --------------------------------------------

    def predict(self):

        self.prediction = self.model.predict(
            self.x,
            verbose=0
        )
        if np.max(np.abs(self.prediction)) > 0.1:
            self.prediction = self.prediction / 1000.0

        print("\nPrediction Completed")

    # --------------------------------------------
    # Metrics
    # --------------------------------------------

    def calculate_metrics(self):

        mse = mean_squared_error(
            self.damage,
            self.prediction
        )

        mae = mean_absolute_error(
            self.damage,
            self.prediction
        )

        r2 = r2_score(
            self.damage,
            self.prediction
        )

        print("\nEvaluation Results")
        print("-" * 40)

        print(f"MSE : {mse:.8f}")
        print(f"MAE : {mae:.8f}")
        print(f"R²  : {r2:.4f}")


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.load_model()
    evaluator.load_test_data()
    evaluator.predict()
    evaluator.calculate_metrics()