import joblib
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.features import load_and_engineer, fit_scaler, apply_scaler, make_sequences
from src.dataset import TransactionSequenceDataset, split_normal_anomaly
from src.model import build_model


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        x_hat = model(x)
        loss = reconstruction_loss(x, x_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_epoch(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat = model(x)
            total_loss += reconstruction_loss(x, x_hat).item()
    return total_loss / len(loader)


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    df = load_and_engineer(config["data"]["csv_path"], config["features"]["numeric_cols"])
    scaler = fit_scaler(df, config["features"]["numeric_cols"])
    df = apply_scaler(df, scaler, config["features"]["numeric_cols"])
    joblib.dump(scaler, config["output"]["scaler_path"])

    seqs, labels = make_sequences(df, "account_id", config["features"]["numeric_cols"], config["data"]["sequence_length"])
    normal, anomaly = split_normal_anomaly(seqs, labels)

    split = int(len(normal) * 0.8)
    train_ds = TransactionSequenceDataset(normal[:split], np.zeros(split, dtype=np.int64))
    val_ds = TransactionSequenceDataset(normal[split:], np.zeros(len(normal) - split, dtype=np.int64))
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"],
                                  weight_decay=config["training"]["weight_decay"])
    stopper = EarlyStopping(patience=config["training"]["patience"])
    best_val = float("inf")

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)
        print(f"Epoch {epoch:3d} | train={train_loss:.4f} | val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), config["output"]["checkpoint_path"])
        if stopper.step(val_loss):
            print("Early stopping.")
            break

    print(f"Best model saved to {config['output']['checkpoint_path']}")


if __name__ == "__main__":
    main()
