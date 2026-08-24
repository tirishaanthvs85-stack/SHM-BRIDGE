import matplotlib.pyplot as plt
import tensorflow as tf


from model import build_model
from data_generator import generate_sensor_data



model=build_model()


model = tf.keras.models.load_model(
    "models/bridge_pinn.keras"
)



x,sensor,damage = generate_sensor_data()



x_tensor=tf.convert_to_tensor(
    x,
    dtype=tf.float32
)



prediction=model(
    x_tensor
).numpy()



plt.figure(figsize=(10,5))


plt.plot(
    x,
    damage,
    label="Damaged Deflection"
)


plt.plot(
    x,
    prediction,
    label="PINN Prediction"
)


plt.scatter(
    x,
    sensor,
    label="Sensors"
)


plt.xlabel(
    "Bridge Position"
)


plt.ylabel(
    "Deflection"
)


plt.title(
    "Bridge Digital Twin Damage Localization"
)


plt.legend()

plt.grid()

plt.show()



# Damage Index

damage_index=abs(
    damage-prediction
)


plt.figure(figsize=(10,4))


plt.plot(
    x,
    damage_index
)


plt.xlabel(
    "Bridge Position"
)


plt.ylabel(
    "Damage Index"
)


plt.title(
    "Crack Localization Map"
)


plt.grid()

plt.show()