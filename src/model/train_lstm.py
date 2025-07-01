import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import math
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import Trainer
from model.lstm_model import LSTMETAModel

# ==== Dataset ====
class ETADataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(data["y"], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==== Training Pipeline ====
if __name__ == "__main__":
    wandb.init(
        project="Traffic-Aware-ETA-Prediction-Using-Temporal-Deep-Learning",
        name="lstm-final",
        entity="nainikas-california-state-university-northridge"
    )

    config = {
        "input_steps": 12,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 64,
        "epochs": 10,
        "lr": 0.001
    }
    wandb.config.update(config)

    dataset = ETADataset("data/lstm_dataset.npz")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], num_workers=0)

    model = LSTMETAModel(
        input_dim=1,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        lr=config["lr"]
    )

    wandb.watch(model, log="all")

    trainer = Trainer(max_epochs=config["epochs"])
    trainer.fit(model, train_loader, val_loader)

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_eta_model.pth")

    # === Evaluate ===
    all_preds, all_targets = [], []
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x).squeeze()
            all_preds.extend(preds.numpy())
            all_targets.extend(y.numpy())

    mae = mean_absolute_error(all_targets, all_preds)
    rmse = math.sqrt(mean_squared_error(all_targets, all_preds))

    wandb.log({"MAE": mae, "RMSE": rmse})

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(all_targets[:200], label="True ETA")
    plt.plot(all_preds[:200], label="Predicted ETA", linestyle="--")
    plt.legend()
    plt.title("ETA Prediction vs Ground Truth (First 200 Samples)")
    plt.tight_layout()
    plot_path = "outputs/eta_vs_truth.png"
    plt.savefig(plot_path)

    wandb.log({"Prediction Plot": wandb.Image(plot_path)})

    print(" W&B tracking complete. See your run at:", wandb.run.url)
