import pandas as pd
from pathlib import Path

# Path to your HDF5 file
data_path = Path("data/pems-bay.h5")

# Load using pandas
df = pd.read_hdf(data_path, key='speed')

# Preview the data
print("DataFrame shape:", df.shape)
print("Columns (sensors):", df.columns[:5])
print(df.head())

# Check for missing values or zero speeds
print("\nAny NaNs?", df.isna().any().any())
print("Basic stats:\n", df.describe())
