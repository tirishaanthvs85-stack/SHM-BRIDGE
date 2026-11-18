"""
L-BFGS Optimizer for TensorFlow PINN Training

Author: PINN Implementation
Purpose:
    Use SciPy L-BFGS-B optimizer with TensorFlow models
"""

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize


class LBFGS:
    """
    L-BFGS optimizer wrapper for TensorFlow models
    """

    def __init__(
        self,
        model,
        loss_function,
        max_iterations=50000,
        tolerance=1e-12
    ):

        self.model = model
        self.loss_function = loss_function

        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.loss_history = []


    # Convert model weights into single vector
    def get_weights(self):

        weights = []

        for variable in self.model.trainable_variables:
            weights.append(tf.reshape(variable, [-1]))

        return tf.concat(weights, axis=0)


    # Assign vector back to model weights
    def set_weights(self, weights):

        index = 0

        for variable in self.model.trainable_variables:

            shape = variable.shape

            size = tf.size(variable).numpy()

            new_values = weights[index:index+size]

            new_values = tf.reshape(
                new_values,
                shape
            )

            variable.assign(new_values)

            index += size



    # Calculate loss and gradients
    def loss_and_gradients(self, weights):

        weights = tf.convert_to_tensor(
            weights,
            dtype=tf.float32
        )

        self.set_weights(weights)


        with tf.GradientTape() as tape:

            loss = self.loss_function()


        gradients = tape.gradient(
            loss,
            self.model.trainable_variables
        )


        gradients_flat = tf.concat(
            [
                tf.reshape(g, [-1])
                for g in gradients
            ],
            axis=0
        )


        loss_value = loss.numpy()

        gradient_value = gradients_flat.numpy()


        self.loss_history.append(loss_value)


        return loss_value, gradient_value



    # Callback during optimization
    def callback(self, weights):

        loss_value, _ = self.loss_and_gradients(weights)

        print(
            "Iteration:",
            len(self.loss_history),
            "Loss:",
            loss_value
        )



    # Train using L-BFGS
    def train(self):

        initial_weights = (
            self.get_weights()
            .numpy()
        )


        result = minimize(

            fun=self.loss_and_gradients,

            x0=initial_weights,

            method="L-BFGS-B",

            jac=True,

            callback=self.callback,

            options={
                "maxiter": self.max_iterations,
                "ftol": self.tolerance,
                "maxls": 50,
                "maxcor": 50
            }

        )


        self.set_weights(
            result.x
        )


        print("\n========== L-BFGS Finished ==========")

        print(
            "Final Loss:",
            result.fun
        )

        print(
            "Iterations:",
            result.nit
        )


        return result