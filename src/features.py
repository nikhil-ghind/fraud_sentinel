import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_engineer(csv_path: str, numeric_cols: list) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    df["transaction_hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["days_since_last_txn"] = (
        df.groupby("account_id")["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(86400)
        .fillna(0)
    )
    return df


def fit_scaler(df: pd.DataFrame, numeric_cols: list) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[numeric_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, numeric_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


def make_sequences(df: pd.DataFrame, account_col: str, numeric_cols: list, seq_len: int) -> tuple:
    sequences, labels = [], []
    for _, group in df.groupby(account_col):
        group = group.reset_index(drop=True)
        vals = group[numeric_cols].values
        lbls = group["is_anomaly"].values if "is_anomaly" in group.columns else np.zeros(len(group))
        for i in range(len(group) - seq_len + 1):
            sequences.append(vals[i : i + seq_len])
            labels.append(int(lbls[i : i + seq_len].any()))
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)
