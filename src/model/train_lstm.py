import sys
from pathlib import Path

# Add the src/ folder to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model.lstm_model import LSTMETAModel

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import Trainer
from model.lstm_model import LSTMETAModel
import os

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
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=0)

    model = LSTMETAModel()

    trainer = Trainer(max_epochs=10)
    trainer.fit(model, train_loader, val_loader)
    print("Training complete. Model saved to 'lightning_logs' directory.")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_eta_model.pth")
    print("Model weights saved to 'models/lstm_eta_model.pth'.")