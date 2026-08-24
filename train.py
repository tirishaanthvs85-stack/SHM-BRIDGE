"""
PINN Training
Bridge Damage Localization

Physics:
Euler-Bernoulli Beam Equation
EI d⁴u/dx⁴ = 0
"""

import tensorflow as tf

from model import build_model
from losses import pinn_loss
from data_generator import (
    generate_sensor_data,
    generate_collocation_points
)


class Trainer:

    def __init__(self):

        print("=" * 60)
        print("Initializing Bridge PINN Trainer")
        print("=" * 60)

        # Build Model
        self.model = build_model()

        # Load Dataset
        (
            self.x_train,
            self.y_sensor,
            self.y_damage
        ) = generate_sensor_data()

        # Physics Collocation Points
        self.x_collocation = generate_collocation_points()

        # Convert to TensorFlow tensors
        self.x_train = tf.convert_to_tensor(
            self.x_train,
            dtype=tf.float32
        )

        self.y_sensor = tf.convert_to_tensor(
            self.y_sensor,
            dtype=tf.float32
        )

        self.y_damage = tf.convert_to_tensor(
            self.y_damage,
            dtype=tf.float32
        )

        self.x_collocation = tf.convert_to_tensor(
            self.x_collocation,
            dtype=tf.float32
        )

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001
        )

    # ----------------------------------------------------
    # Training
    # ----------------------------------------------------

    def train(self, epochs=5000):

        for epoch in range(epochs):

            with tf.GradientTape() as tape:

                (
                    total_loss,
                    sensor_loss,
                    physics_loss_value,
                    damage_loss
                ) = pinn_loss(
                    self.model,
                    self.x_train,
                    self.y_sensor,
                    self.y_damage,
                    self.x_collocation
                )

            gradients = tape.gradient(
                total_loss,
                self.model.trainable_variables
            )

            self.optimizer.apply_gradients(
                zip(
                    gradients,
                    self.model.trainable_variables
                )
            )

            if epoch % 500 == 0:

                print(
                    f"Epoch {epoch:5d} | "
                    f"Total = {total_loss.numpy():.8f} | "
                    f"Sensor = {sensor_loss.numpy():.8f} | "
                    f"Physics = {physics_loss_value.numpy():.8f} | "
                    f"Damage = {damage_loss.numpy():.8f}"
                )


# ----------------------------------------------------
# Main
# ----------------------------------------------------

if __name__ == "__main__":

    trainer = Trainer()

    trainer.train()

    trainer.model.save_weights(
        "bridge_pinn_damage.weights.h5"
    )

    print("\nModel Saved Successfully")