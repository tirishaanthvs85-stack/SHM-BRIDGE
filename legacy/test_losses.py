import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from losses import pinn_loss
from model import build_model

# ----------------------------------------------------
# Build model
# ----------------------------------------------------

model = build_model()

# ----------------------------------------------------
# Test 1: Random data
# ----------------------------------------------------

x = tf.random.normal((10, 1))
sensor = tf.random.normal((10, 1))
damage = tf.random.normal((10, 1))
collocation = tf.random.normal((100, 1))

loss = pinn_loss(
    model,
    x,
    sensor,
    damage,
    collocation
)

print("Random Test Loss:")
print(loss.numpy())

# ----------------------------------------------------
# Test 2: Dummy bridge data
# ----------------------------------------------------

x = np.linspace(0, 10, 200).reshape(-1, 1)

sensor = np.sin(x)

damage = np.sin(x) + 0.05 * np.random.randn(*x.shape)

x = tf.convert_to_tensor(
    x,
    dtype=tf.float32
)

sensor = tf.convert_to_tensor(
    sensor,
    dtype=tf.float32
)

damage = tf.convert_to_tensor(
    damage,
    dtype=tf.float32
)

collocation = tf.random.uniform(
    (500, 1),
    minval=0,
    maxval=10,
    dtype=tf.float32
)

loss = pinn_loss(
    model,
    x,
    sensor,
    damage,
    collocation
)

print("\nDummy Bridge Data Loss:")
print(loss.numpy())

# ----------------------------------------------------
# Plot
# ----------------------------------------------------

plt.figure(figsize=(8,4))

plt.plot(
    x.numpy(),
    sensor.numpy(),
    label="Sensor"
)

plt.plot(
    x.numpy(),
    damage.numpy(),
    label="Damaged"
)

plt.xlabel("Bridge Position")

plt.ylabel("Deflection")

plt.title("Dummy Bridge Dataset")

plt.legend()

plt.grid(True)

plt.show()