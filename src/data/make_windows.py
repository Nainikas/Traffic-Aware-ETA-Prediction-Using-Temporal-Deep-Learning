import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path

INPUT_STEPS = 12
PREDICT_STEP = 1
TARGET_SENSOR = '400001'

# Load data properly from HDF5
with h5py.File("data/pems-bay.h5", "r") as f:
    speed_group = f["speed"]
    raw = speed_group["block0_values"][:]  # (52116, 325)
    sensor_ids = speed_group["block0_items"][:]
    sensor_ids = [str(sid) for sid in sensor_ids]  # Convert from bytes to str

df = pd.DataFrame(raw, columns=sensor_ids)

# Print basic info
print("DataFrame shape:", df.shape)
print("Sensor columns:", df.columns[:5])

# Extract target sensor's series
speed_series = df[TARGET_SENSOR].values

print("Speed sample:", speed_series[:10])
print("Min speed:", np.min(speed_series))
print("Max speed:", np.max(speed_series))

X, y = [], []

for i in tqdm(range(INPUT_STEPS, len(speed_series) - PREDICT_STEP)):
    seq_x = speed_series[i - INPUT_STEPS:i]
    speed = speed_series[i + PREDICT_STEP - 1]

    if speed <= 0 or speed > 100:
        continue

    eta = (1 / speed) * 60  # minutes per 1 mile

    if eta > 30:
        continue

    X.append(seq_x)
    y.append(eta)

X = np.array(X)
y = np.array(y)

np.savez("data/lstm_dataset.npz", X=X, y=y)
print(f"Saved dataset: X shape {X.shape}, y shape {y.shape}")
# Save the dataset to a CSV for easier inspection
dataset_df = pd.DataFrame(X, columns=[f"sensor_{i}" for i in range(INPUT_STEPS)])
dataset_df['eta'] = y
dataset_df.to_csv("data/lstm_dataset.csv", index=False)
print("Dataset saved to data/lstm_dataset.csv")