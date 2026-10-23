"""
evaluate_rf.py
Bridge SHM - Random Forest Evaluation
Uses the same dataset as the PINN & SVM pipelines.
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# --------------------------------------------------
# LOAD DATA (same CSV used by PINN & SVM)
# --------------------------------------------------

DATA_PATH = "dataset/bridge_data.csv"

data = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# FEATURES (must match what the model was trained on)
# --------------------------------------------------

X = data[
    [
        "x",
        "sensor_measurement"
    ]
].values

y_true = data[
    "damaged_deflection"
].values


# --------------------------------------------------
# LOAD RANDOM FOREST MODEL
# --------------------------------------------------

model = joblib.load(
    "rf_bridge_model.pkl"
)


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

y_pred = model.predict(X)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

mse = mean_squared_error(
    y_true,
    y_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_true,
    y_pred
)

r2 = r2_score(
    y_true,
    y_pred
)


print("\n================================")
print("RANDOM FOREST EVALUATION RESULTS")
print("================================")

print(f"MSE  : {mse:.10f}")
print(f"RMSE : {rmse:.10f}")
print(f"MAE  : {mae:.10f}")
print(f"R²   : {r2:.6f}")
