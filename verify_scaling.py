"""
Quick test script to verify scaling targets to millimeters for high R2 (> 0.98)
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv("dataset/bridge_data.csv")
x = df["x"].values.reshape(-1, 1).astype(np.float32)
y = (df["damaged_deflection"].values.reshape(-1, 1) * 1000.0).astype(np.float32) # in mm
s = (df["sensor_measurement"].values.reshape(-1, 1) * 1000.0).astype(np.float32) # in mm

x_train, x_test, y_train, y_test, s_train, s_test = train_test_split(x, y, s, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Lambda(lambda x: x / 60.0), # normalize x to [0, 1]
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(64, activation="tanh"),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss="mse")
model.fit(x_train, y_train, epochs=300, verbose=0, batch_size=32)

pred_train = model.predict(x_train, verbose=0)
pred_test = model.predict(x_test, verbose=0)

r2_tr = r2_score(y_train, pred_train)
r2_te = r2_score(y_test, pred_test)
print(f"Scaled Train R2: {r2_tr:.4f}, Test R2: {r2_te:.4f}")
print(f"Scaled Train RMSE (mm): {np.sqrt(mean_squared_error(y_train, pred_train)):.4f}")
print(f"Scaled Test RMSE (mm): {np.sqrt(mean_squared_error(y_test, pred_test)):.4f}")
