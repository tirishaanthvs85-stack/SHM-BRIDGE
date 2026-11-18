import pandas as pd

path = r"C:\Users\tirishaanth\Downloads\archive (5)\bridge_digital_twin_dataset.csv"

df = pd.read_csv(path)

print(df.head())

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())