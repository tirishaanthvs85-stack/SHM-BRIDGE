"""
train_svm.py
Bridge SHM - Support Vector Machine Training
Uses the same dataset as the PINN pipeline.
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ==================================================
# LOAD DATASET  (same CSV used by PINN)
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
# CREATE SVM MODEL  (RBF kernel SVR pipeline)
# ==================================================

svm_model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "svr",
        SVR(
            kernel="rbf",
            C=100,
            gamma="scale",
            epsilon=0.001
        )
    )

])


# ==================================================
# TRAIN
# ==================================================

print("\nTraining Support Vector Machine...")

svm_model.fit(
    X_train,
    y_train
)

print("SVM training completed.")


# ==================================================
# PREDICT ON TEST SET
# ==================================================

y_pred = svm_model.predict(
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
print("SVM RESULTS  (Test Set)")
print("==============================")

print(
    f"MSE  : {mse:.10f}"
)

print(
    f"RMSE : {rmse:.10f}"
)

print(
    f"MAE  : {mae:.10f}"
)

print(
    f"R²   : {r2:.6f}"
)


# ==================================================
# SAVE MODEL
# ==================================================

joblib.dump(
    svm_model,
    "svm_bridge_model.pkl"
)


print(
    "\nSVM model saved to svm_bridge_model.pkl"
)