import pandas as pd

# Load preprocessed dataset
df = pd.read_csv("data/lstm_dataset.csv")

# Assuming sensor IDs are in column names like: sensor_0, sensor_1, ..., sensor_11
# We'll simulate 3 sensors by slicing rows

sensor_speed_map = {
    "400001": df.iloc[0, :-1].tolist(),  # first row as sensor A
    "400017": df.iloc[1, :-1].tolist(),  # second row as sensor B
    "400030": df.iloc[2, :-1].tolist(),  # third row as sensor C
}
