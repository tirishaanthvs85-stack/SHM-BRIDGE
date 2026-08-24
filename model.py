import tensorflow as tf


def build_model():


    model = tf.keras.Sequential([


        tf.keras.layers.Input(
            shape=(1,)
        ),


        tf.keras.layers.Dense(
            64,
            activation="tanh"
        ),


        tf.keras.layers.Dense(
            64,
            activation="tanh"
        ),


        tf.keras.layers.Dense(
            64,
            activation="tanh"
        ),


        tf.keras.layers.Dense(
            1
        )

    ])


    return model