import tensorflow as tf
import numpy as np

from model import build_model

model = build_model()

x = np.linspace(0,10,20).reshape(-1,1)

x = tf.convert_to_tensor(
    x,
    dtype=tf.float32
)

y = model(x)

print(y.numpy())