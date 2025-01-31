import json
import yaml
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.features import load_and_engineer, apply_scaler, make_sequences
from src.dataset import TransactionSequenceDataset, split_normal_anomaly
from src.model import build_model


def compute_reconstruction_errors(model, loader, device: str) -> np.ndarray:
    model.eval()
    errors = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_hat = model(x)
            mse = ((x - x_hat) ** 2).mean(dim=(1, 2)).cpu().numpy()
            errors.extend(mse)
    return np.array(errors)


def find_threshold(normal_errors: np.ndarray, percentile: int) -> float:
    return float(np.percentile(normal_errors, percentile))


def evaluate_threshold(threshold: float, normal_errors: np.ndarray, anomaly_errors: np.ndarray) -> dict:
    from sklearn.metrics import recall_score, precision_score, f1_score
    all_errors = np.concatenate([normal_errors, anomaly_errors])
    all_labels = np.concatenate([np.zeros(len(normal_errors)), np.ones(len(anomaly_errors))])
    preds = (all_errors > threshold).astype(int)
    return {
        "recall": float(recall_score(all_labels, preds, zero_division=0)),
        "precision": float(precision_score(all_labels, preds, zero_division=0)),
        "f1": float(f1_score(all_labels, preds, zero_division=0)),
        "fpr": float((preds[all_labels == 0]).mean()),
    }


def sweep_thresholds(normal_errors, anomaly_errors, percentiles) -> pd.DataFrame:
    rows = []
    for p in percentiles:
        tau = find_threshold(normal_errors, p)
        metrics = evaluate_threshold(tau, normal_errors, anomaly_errors)
        rows.append({"percentile": p, "threshold": tau, **metrics})
    return pd.DataFrame(rows)


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(config["model"]).to(device)
    model.load_state_dict(torch.load(config["output"]["checkpoint_path"], map_location=device))
    scaler = joblib.load(config["output"]["scaler_path"])

    df = load_and_engineer(config["data"]["csv_path"], config["features"]["numeric_cols"])
    df = apply_scaler(df, scaler, config["features"]["numeric_cols"])
    seqs, labels = make_sequences(df, "account_id", config["features"]["numeric_cols"], config["data"]["sequence_length"])
    normal, anomaly = split_normal_anomaly(seqs, labels)

    split = int(len(normal) * 0.8)
    val_normal = normal[split:]
    val_ds = TransactionSequenceDataset(val_normal, np.zeros(len(val_normal), dtype=np.int64))
    val_loader = DataLoader(val_ds, batch_size=64)
    anomaly_ds = TransactionSequenceDataset(anomaly, np.ones(len(anomaly), dtype=np.int64))
    anomaly_loader = DataLoader(anomaly_ds, batch_size=64)

    normal_errors = compute_reconstruction_errors(model, val_loader, device)
    anomaly_errors = compute_reconstruction_errors(model, anomaly_loader, device)

    percentiles = np.arange(90, 100, 0.5).tolist()
    sweep = sweep_thresholds(normal_errors, anomaly_errors, percentiles)

    # Select lowest threshold with recall >= 0.94
    candidates = sweep[sweep["recall"] >= 0.94]
    if len(candidates) == 0:
        chosen = sweep.iloc[0]
        print("Warning: could not achieve 0.94 recall, using highest recall row")
    else:
        chosen = candidates.iloc[0]

    result = {
        "tau": float(chosen["threshold"]),
        "percentile": float(chosen["percentile"]),
        "recall": float(chosen["recall"]),
        "precision": float(chosen["precision"]),
    }
    with open(config["output"]["threshold_path"], "w") as f:
        json.dump(result, f, indent=2)
    print(f"Threshold: {result}")


if __name__ == "__main__":
    import numpy as np
    main()
