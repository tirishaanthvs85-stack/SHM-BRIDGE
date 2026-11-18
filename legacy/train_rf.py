"""
train_rf.py
Bridge SHM - Random Forest Training
Uses the same dataset as the PINN & SVM pipelines.
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from rf_model import build_rf_model


# ==================================================
# LOAD DATASET (same CSV used by PINN & SVM)
# ==================================================

DATA_PATH = "dataset/bridge_data.csv"

data = pd.read_csv(DATA_PATH)

print("\nDataset:")
print(data.head())


# ==================================================
# FEATURES
# ==================================================

X = data[
    [
        "x",
        "sensor_measurement"
    ]
].values


# ==================================================
# TARGET
# ==================================================

y = data[
    "damaged_deflection"
].values


# ==================================================
# SPLIT DATA
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# CREATE RANDOM FOREST MODEL
# ==================================================

rf_model = build_rf_model(n_estimators=200, random_state=42)


# ==================================================
# TRAIN
# ==================================================

print("\nTraining Random Forest Regressor...")

rf_model.fit(
    X_train,
    y_train
)

print("Random Forest training completed.")


# ==================================================
# PREDICT ON TEST SET
# ==================================================

y_pred = rf_model.predict(
    X_test
)


# ==================================================
# METRICS
# ==================================================

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test,
    y_pred
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("RANDOM FOREST RESULTS (Test Set)")
print("==============================")

print(f"MSE  : {mse:.10f}")
print(f"RMSE : {rmse:.10f}")
print(f"MAE  : {mae:.10f}")
print(f"R²   : {r2:.6f}")


# ==================================================
# SAVE MODEL
# ==================================================

joblib.dump(
    rf_model,
    "rf_bridge_model.pkl"
)

print("\nRandom Forest model saved to rf_bridge_model.pkl")
