import tensorflow as tf


# -----------------------------------------------------
# Physics Loss (Euler-Bernoulli Beam)
# -----------------------------------------------------

def physics_loss(model, x):

    with tf.GradientTape(persistent=True) as tape4:
        tape4.watch(x)

        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch(x)

            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch(x)

                with tf.GradientTape() as tape1:
                    tape1.watch(x)

                    u = model(x)

                du = tape1.gradient(u, x)

            d2u = tape2.gradient(du, x)

        d3u = tape3.gradient(d2u, x)

    d4u = tape4.gradient(d3u, x)

    physics = tf.reduce_mean(tf.square(d4u))

    del tape1
    del tape2
    del tape3
    del tape4

    return physics


# -----------------------------------------------------
# PINN Total Loss
# -----------------------------------------------------

def pinn_loss(
    model,
    x_data,
    sensor_data,
    damaged_data,
    x_physics
):

    prediction = model(x_data)

    sensor_loss = tf.reduce_mean(
        tf.square(prediction - sensor_data)
    )

    damage_loss = tf.reduce_mean(
        tf.square(prediction - damaged_data)
    )

    p_loss = physics_loss(
        model,
        x_physics
    )

    total_loss = (
        sensor_loss
        + damage_loss
        + 0.01 * p_loss
    )

    return (
        total_loss,
        sensor_loss,
        p_loss,
        damage_loss
    )