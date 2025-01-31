import json
import numpy as np
import torch
import joblib
from pathlib import Path

from src.model import build_model

model = None
scaler = None
threshold = None
_config = None


def load_artifacts(checkpoint_path: str, scaler_path: str, threshold_path: str):
    global model, scaler, threshold, _config
    import yaml
    with open("configs/config.yaml") as f:
        _config = yaml.safe_load(f)
    scaler = joblib.load(scaler_path)
    with open(threshold_path) as f:
        data = json.load(f)
        threshold = data["tau"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(_config["model"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()


def score_sequence(transactions: list) -> dict:
    global model, scaler, threshold, _config
    numeric_cols = _config["features"]["numeric_cols"]
    seq_len = _config["data"]["sequence_length"]
    device = next(model.parameters()).device

    arr = np.array([[t[c] for c in numeric_cols] for t in transactions], dtype=np.float32)
    arr = scaler.transform(arr)
    # Pad or truncate to seq_len
    if len(arr) < seq_len:
        pad = np.zeros((seq_len - len(arr), arr.shape[1]), dtype=np.float32)
        arr = np.vstack([pad, arr])
    else:
        arr = arr[-seq_len:]

    x = torch.tensor(arr[np.newaxis], dtype=torch.float32).to(device)
    with torch.no_grad():
        x_hat = model(x)
        error = float(((x - x_hat) ** 2).mean().item())

    return {"reconstruction_error": error, "is_anomaly": error > threshold, "threshold": threshold}
