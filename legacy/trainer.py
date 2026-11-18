import os
import tensorflow as tf

from config import *
from model import build_model
from losses import pinn_loss
from data_generator import (
    generate_sensor_data,
    generate_collocation_points
)


class Trainer:

    def __init__(self):

        print("=" * 60)
        print("Initializing Bridge SHM PINN")
        print("=" * 60)

        self.model = build_model()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )

        self.history = {
            "total": [],
            "sensor": [],
            "physics": [],
            "damage": []
        }

    @tf.function
    def train_step(
        self,
        x_data,
        sensor_data,
        damaged_data,
        x_physics
    ):

        with tf.GradientTape() as tape:

            (
                total_loss,
                sensor_loss,
                physics_loss,
                damage_loss
            ) = pinn_loss(
                self.model,
                x_data,
                sensor_data,
                damaged_data,
                x_physics
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

        return (
            total_loss,
            sensor_loss,
            physics_loss,
            damage_loss
        )

    def train(self):

        print("\nLoading Bridge Dataset...")

        (
            x_data,
            sensor_data,
            damaged_data
        ) = generate_sensor_data()

        x_physics = generate_collocation_points()

        x_data = tf.convert_to_tensor(
            x_data,
            dtype=tf.float32
        )

        sensor_data = tf.convert_to_tensor(
            sensor_data,
            dtype=tf.float32
        )

        damaged_data = tf.convert_to_tensor(
            damaged_data,
            dtype=tf.float32
        )

        x_physics = tf.convert_to_tensor(
            x_physics,
            dtype=tf.float32
        )

        print("\nTraining Started\n")

        for epoch in range(EPOCHS):

            (
                total_loss,
                sensor_loss,
                physics_loss,
                damage_loss
            ) = self.train_step(
                x_data,
                sensor_data,
                damaged_data,
                x_physics
            )

            self.history["total"].append(
                float(total_loss)
            )

            self.history["sensor"].append(
                float(sensor_loss)
            )

            self.history["physics"].append(
                float(physics_loss)
            )

            self.history["damage"].append(
                float(damage_loss)
            )

            if epoch % 100 == 0:

                print(
                    f"Epoch {epoch:5d} | "
                    f"Total={total_loss.numpy():.6e} | "
                    f"Sensor={sensor_loss.numpy():.6e} | "
                    f"Physics={physics_loss.numpy():.6e} | "
                    f"Damage={damage_loss.numpy():.6e}"
                )

        print("\nTraining Completed")

        self.save_model()

    def save_model(self):

        os.makedirs(
            os.path.dirname(MODEL_PATH),
            exist_ok=True
        )

        self.model.save(MODEL_PATH)

        print("\nModel saved to")

        print(MODEL_PATH)

    def get_model(self):

        return self.model

    def get_history(self):

        return self.history


if __name__ == "__main__":

    trainer = Trainer()

    trainer.train()