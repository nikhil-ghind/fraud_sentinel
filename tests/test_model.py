import numpy as np
import torch
import pytest
from src.model import build_model, LSTMAutoencoder


CONFIG = {"input_size": 6, "hidden_size": 128, "num_layers": 2, "dropout": 0.2}


def test_forward_shape():
    model = build_model(CONFIG)
    x = torch.randn(32, 30, 6)
    out = model(x)
    assert out.shape == (32, 30, 6)


def test_no_nan():
    model = build_model(CONFIG)
    x = torch.randn(4, 30, 6)
    out = model(x)
    assert not torch.isnan(out).any()


def test_make_sequences():
    from src.features import make_sequences
    import pandas as pd
    rng = np.random.default_rng(0)
    N = 200
    df = pd.DataFrame({
        "account_id": ["A"] * N,
        "timestamp": pd.date_range("2023-01-01", periods=N, freq="1h"),
        "amount": rng.exponential(100, N),
        "balance_before": rng.uniform(1000, 10000, N),
        "balance_after": rng.uniform(1000, 10000, N),
        "transaction_hour": rng.integers(0, 24, N).astype(float),
        "day_of_week": rng.integers(0, 7, N).astype(float),
        "days_since_last_txn": rng.uniform(0, 5, N),
        "is_anomaly": np.zeros(N, dtype=int),
    })
    numeric_cols = ["amount", "balance_before", "balance_after", "transaction_hour", "day_of_week", "days_since_last_txn"]
    seqs, labels = make_sequences(df, "account_id", numeric_cols, seq_len=30)
    assert seqs.shape[1:] == (30, 6)
