import sys
from pathlib import Path
# Add src/ to path early
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from model.lstm_model import LSTMETAModel

# Dataset class
class ETADataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(data["y"], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

if __name__ == "__main__":
    dataset = ETADataset("data/lstm_dataset.npz")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_ds = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_ds, batch_size=64)

    model = LSTMETAModel()
    model.load_state_dict(torch.load("models/lstm_eta_model.pth"))
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x).squeeze()
            all_preds.extend(preds.numpy())
            all_targets.extend(y.numpy())

    # Convert to arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Before applying the mask
    print("== Diagnostics ==")
    print("Min/Max preds:", np.min(all_preds), np.max(all_preds))
    print("Min/Max targets:", np.min(all_targets), np.max(all_targets))
    print("Any NaNs in preds?", np.isnan(all_preds).any())
    print("Any Infs in preds?", np.isinf(all_preds).any())
    print("Any NaNs in targets?", np.isnan(all_targets).any())
    print("Any Infs in targets?", np.isinf(all_targets).any())
    print("Total predictions:", len(all_preds))

    # Filter invalid values
    mask = np.isfinite(all_preds) & np.isfinite(all_targets)
    all_preds = all_preds[mask]
    all_targets = all_targets[mask]

    # Final check
    if len(all_preds) == 0:
        print("No valid predictions to evaluate.")
        exit()

    # Metrics
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)

    print(f"\nEvaluation Results:")
    print(f"MAE:  {mae:.3f} minutes")
    print(f"RMSE: {rmse:.3f} minutes")

    print(f"Total predictions evaluated: {len(all_preds)}")

    # Save results
    output_path = Path("data/predictions.npz")
    np.savez(output_path, predictions=all_preds, targets=all_targets)
    print(f"Predictions saved to {output_path}")
    print("Evaluation complete.")

    