"""
Bridge Digital Twin Dataset Loader
PINN Structural Health Monitoring
"""


import pandas as pd
import numpy as np


DATA_PATH = r"C:\Users\tirishaanth\Downloads\archive (5)\bridge_digital_twin_dataset.csv"



def load_bridge_data():

    df = pd.read_csv(DATA_PATH)

    print("="*50)
    print("Dataset Loaded")
    print("="*50)

    print(df.head())

    return df



def generate_sensor_data():

    df = load_bridge_data()


    x = df[
        ["x"]
    ].values


    sensor = df[
        ["sensor_measurement"]
    ].values


    damaged = df[
        ["damaged_deflection"]
    ].values


    return x, sensor, damaged




def generate_collocation_points(
        n_points=1000
):


    x_collocation = np.linspace(
        0,
        1,
        n_points
    )


    x_collocation = x_collocation.reshape(
        -1,1
    )


    return x_collocation