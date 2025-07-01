import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTMETAModel(pl.LightningModule):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, dropout=0.0, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last hidden state
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x).squeeze()
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
