import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Load raw file
df = pd.read_hdf("data/pems-bay.h5", key="speed")
print("Original shape:", df.shape)

# Step 1: Replace 0s with NaN (optional: 0 speed may mean sensor offline)
df.replace(0, np.nan, inplace=True)

# Step 2: Interpolate missing values
df.interpolate(method='linear', axis=0, inplace=True)
df.bfill(inplace=True)
df.ffill(inplace=True)

# Step 3: Normalize each sensor (column) to [0, 1]
scaler = MinMaxScaler()
scaled_array = scaler.fit_transform(df)

# Back to DataFrame
df_scaled = pd.DataFrame(
    scaled_array,
    index=df.index,
    columns=df.columns
)

# Step 4: Save cleaned version
cleaned_path = Path("data/cleaned_pems_bay.csv")
df_scaled.to_csv(cleaned_path)
print(f"Cleaned data saved to {cleaned_path}")
