import tensorflow as tf
import numpy as np

from model import build_model
from beam_pde import BeamPDE

model = build_model()

physics = BeamPDE()

x = np.linspace(0,10,100).reshape(-1,1)

x = tf.convert_to_tensor(x,dtype=tf.float32)

r = physics.residual(model,x)

print()

print("Residual Shape")

print(r.shape)

print()

print("Physics Loss")

print(physics.physics_loss(model,x).numpy())

print()

print("Boundary Loss")

print(physics.boundary_loss(model).numpy())