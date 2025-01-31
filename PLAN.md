# Fraud Sentinel

## Project Overview

An LSTM Autoencoder trained on sequences of financial transaction features to model normal transaction behavior. Anomalies are detected by measuring reconstruction error against a learned threshold. The system achieves 94% recall on held-out anomaly sets and is deployed as a FastAPI microservice that accepts transaction sequences and returns anomaly scores and binary flags.

Key goals:
- Train an LSTM Autoencoder on normal transaction sequences.
- Tune the anomaly threshold to achieve >= 94% recall on held-out labeled anomalies.
- Expose a low-latency (<50ms) inference endpoint via FastAPI.
- Support batch scoring and single-record scoring.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep learning | PyTorch 2.2 |
| Model | LSTM Autoencoder (encoder-decoder with LSTM cells) |
| Data processing | Pandas 2.2, NumPy 1.26, scikit-learn 1.4 (StandardScaler) |
| API server | FastAPI 0.110, Uvicorn 0.29 |
| Serialization | Pydantic v2, joblib 1.3 (scaler persistence) |
| Testing | pytest 8.1, httpx 0.27 |
| Environment | Python 3.11 |

---

## Architecture Overview

```
[Raw transaction CSV]
        |
        v
[Feature Engineering (Pandas)] --> normalized sequences of shape (T, F)
        |
        v
[LSTM Encoder] --> hidden state h (latent representation)
        |
        v
[LSTM Decoder] --> reconstructed sequence (T, F)
        |
        v
[Reconstruction Error: MSE per sample]
        |
        v
[Threshold tau (percentile of train errors)] --> anomaly_flag (bool)
        |
        v
[FastAPI /score endpoint] <-- JSON payload: {transactions: [...]}
```

Model architecture:
- **Encoder**: 2-layer LSTM reading the input sequence, output is final hidden state `(h, c)`.
- **Decoder**: 2-layer LSTM initialized with encoder's `(h, c)`, generating T output steps by repeating the encoded state as input at each step (teacher-forcing disabled at inference).
- **Output projection**: `nn.Linear(hidden_dim, num_features)` applied to each decoder step.
- Anomaly score = mean squared error between input sequence and reconstructed sequence.

---

## Phase 1: Project Scaffolding

**Goal**: Set up project layout, virtual environment, and configuration.

### Tasks

1. Create the directory layout:
   ```
   financialAnomalyDetectionLstm/
   ├── src/
   │   ├── __init__.py
   │   ├── features.py
   │   ├── dataset.py
   │   ├── model.py
   │   ├── train.py
   │   ├── threshold.py
   │   └── predict.py
   ├── api/
   │   ├── __init__.py
   │   ├── main.py
   │   ├── schemas.py
   │   └── inference.py
   ├── tests/
   │   ├── test_model.py
   │   └── test_api.py
   ├── data/
   │   └── sample_transactions.csv
   ├── checkpoints/
   ├── configs/
   │   └── config.yaml
   └── requirements.txt
   ```

2. Create `requirements.txt`:
   ```
   torch==2.2.2
   pandas==2.2.1
   numpy==1.26.4
   scikit-learn==1.4.1
   fastapi==0.110.0
   uvicorn[standard]==0.29.0
   pydantic==2.6.4
   pydantic-settings==2.2.1
   joblib==1.3.2
   httpx==0.27.0
   pytest==8.1.1
   pytest-asyncio==0.23.6
   tqdm==4.66.2
   pyyaml==6.0.1
   ```

3. Create `configs/config.yaml`:
   ```yaml
   data:
     csv_path: data/sample_transactions.csv
     sequence_length: 30
     test_size: 0.2
     anomaly_label_column: is_anomaly   # set to null if unsupervised split
   features:
     numeric_cols:
       - amount
       - balance_before
       - balance_after
       - transaction_hour
       - day_of_week
       - days_since_last_txn
   model:
     input_size: 6       # must equal len(numeric_cols)
     hidden_size: 128
     num_layers: 2
     dropout: 0.2
   training:
     epochs: 50
     batch_size: 64
     lr: 1e-3
     weight_decay: 1e-5
     patience: 7         # early stopping
   threshold:
     percentile: 99      # train-set reconstruction error percentile
   output:
     checkpoint_path: checkpoints/best_model.pt
     scaler_path: checkpoints/scaler.joblib
     threshold_path: checkpoints/threshold.json
   ```

4. Install dependencies: `pip install -r requirements.txt`.

---

## Phase 2: Feature Engineering and Dataset

**Goal**: Load raw transaction data, engineer sequence features, and build PyTorch datasets.

### Tasks

1. Create `data/sample_transactions.csv` with columns: `transaction_id`, `account_id`, `timestamp`, `amount`, `balance_before`, `balance_after`, `merchant_category`, `is_anomaly (0/1)`. Include at least 10,000 normal rows and 200 anomaly rows for testing.

2. Create `src/features.py`:
   - `load_and_engineer(csv_path: str, numeric_cols: list[str]) -> pd.DataFrame`:
     - Read CSV, parse `timestamp` as datetime.
     - Sort by `account_id`, `timestamp`.
     - Compute `transaction_hour = timestamp.dt.hour`.
     - Compute `day_of_week = timestamp.dt.dayofweek`.
     - Compute `days_since_last_txn`: per-account diff of timestamps in days, fill first occurrence with 0.
     - Return DataFrame with `numeric_cols + ["account_id", "is_anomaly"]`.
   - `fit_scaler(df: pd.DataFrame, numeric_cols: list[str]) -> StandardScaler`: fit `StandardScaler` on `df[numeric_cols]`, return scaler.
   - `apply_scaler(df: pd.DataFrame, scaler: StandardScaler, numeric_cols: list[str]) -> pd.DataFrame`: apply transform in-place, return df.
   - `make_sequences(df: pd.DataFrame, account_col: str, numeric_cols: list[str], seq_len: int) -> tuple[np.ndarray, np.ndarray]`:
     - For each account, create sliding window sequences of shape `(seq_len, num_features)`.
     - Label: 1 if any transaction in the window has `is_anomaly == 1`, else 0.
     - Return `(sequences: np.ndarray of shape (N, seq_len, num_features), labels: np.ndarray of shape (N,))`.

3. Create `src/dataset.py`:
   - `class TransactionSequenceDataset(torch.utils.data.Dataset)`:
     - `__init__(self, sequences: np.ndarray, labels: np.ndarray)`: stores tensors.
     - `__len__`, `__getitem__`: returns `(sequence_tensor: float32, label_tensor: long)`.
   - `split_normal_anomaly(sequences, labels) -> tuple`: returns `(normal_sequences, anomaly_sequences)` by filtering on label.

4. Test in `tests/test_model.py`:
   - Load sample CSV, run `load_and_engineer`, verify no NaN in numeric_cols.
   - Run `make_sequences` with `seq_len=30`, verify output shape `(N, 30, 6)`.

---

## Phase 3: LSTM Autoencoder Model

**Goal**: Implement the encoder-decoder LSTM architecture.

### Tasks

1. Create `src/model.py`:
   - `class LSTMEncoder(nn.Module)`:
     - `__init__(self, input_size, hidden_size, num_layers, dropout)`: `self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)`.
     - `forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]`: returns `(output, (h_n, c_n))` from LSTM.
   
   - `class LSTMDecoder(nn.Module)`:
     - `__init__(self, hidden_size, output_size, num_layers, dropout)`: `self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)`, `self.out = nn.Linear(hidden_size, output_size)`.
     - `forward(self, h: Tensor, c: Tensor, seq_len: int) -> Tensor`:
       - Create decoder input: repeat `h[-1]` (last layer hidden state) `seq_len` times along sequence dim: `decoder_input = h[-1].unsqueeze(1).repeat(1, seq_len, 1)`.
       - Run LSTM: `out, _ = self.lstm(decoder_input, (h, c))`.
       - Project: `return self.out(out)` → shape `(B, seq_len, output_size)`.
   
   - `class LSTMAutoencoder(nn.Module)`:
     - `__init__(self, input_size, hidden_size, num_layers, dropout)`: instantiates `LSTMEncoder` and `LSTMDecoder`.
     - `forward(self, x: Tensor) -> Tensor`: encode `x`, decode using encoder's `(h_n, c_n)`, return reconstruction.
     - `encode(self, x: Tensor) -> Tensor`: returns `h_n[-1]` (last-layer final hidden state) as the latent vector.
   
   - `def build_model(config: dict) -> LSTMAutoencoder`: constructs from config dict.

2. Add tests in `tests/test_model.py`:
   - Instantiate autoencoder with `input_size=6, hidden_size=128, num_layers=2`.
   - Forward pass with `(32, 30, 6)` input. Assert output shape is `(32, 30, 6)`.
   - Assert no NaN in output.

---

## Phase 4: Training

**Goal**: Train the autoencoder on normal transactions only, with early stopping.

### Tasks

1. Create `src/train.py`:
   - `reconstruction_loss(x: Tensor, x_hat: Tensor) -> Tensor`: returns `F.mse_loss(x_hat, x)`.
   - `train_epoch(model, loader, optimizer, device) -> float`: returns mean MSE loss over all normal batches.
   - `validate_epoch(model, loader, device) -> float`: same but no gradient, no optimizer step.
   - `class EarlyStopping`: tracks best val loss, increments patience counter. `step(val_loss) -> bool` returns `True` if training should stop.
   - `main()`:
     - Load config from `configs/config.yaml`.
     - Run `load_and_engineer`, `fit_scaler`, `apply_scaler`, `make_sequences`.
     - Split into normal train (80%), normal val (20%) — do NOT include anomalies in training. Keep anomaly sequences aside for threshold calibration.
     - Save scaler: `joblib.dump(scaler, config["output"]["scaler_path"])`.
     - Build `TransactionSequenceDataset` and `DataLoader` for train and val (normal only).
     - Instantiate model via `build_model(config["model"])`, move to device.
     - Optimizer: `Adam(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])`.
     - Loop for `epochs`: `train_epoch`, `validate_epoch`, early stopping check, save best checkpoint when val loss improves.
     - Save best checkpoint to `config["output"]["checkpoint_path"]`.

2. Run training:
   ```
   python src/train.py
   ```
   Confirm loss decreases over epochs, best checkpoint is saved.

---

## Phase 5: Threshold Calibration

**Goal**: Set the anomaly detection threshold to maximize recall >= 94% on held-out anomalies.

### Tasks

1. Create `src/threshold.py`:
   - `compute_reconstruction_errors(model: LSTMAutoencoder, loader: DataLoader, device: str) -> np.ndarray`:
     - Run model in eval mode on loader.
     - Compute per-sample MSE: `((x - x_hat) ** 2).mean(dim=(1, 2))`.
     - Return numpy array of errors.
   - `find_threshold(normal_errors: np.ndarray, percentile: int) -> float`:
     - Returns `np.percentile(normal_errors, percentile)`.
   - `evaluate_threshold(threshold: float, normal_errors: np.ndarray, anomaly_errors: np.ndarray) -> dict`:
     - Concatenates errors and creates labels `(0 = normal, 1 = anomaly)`.
     - Predicts `1` where `error > threshold`.
     - Computes and returns `{"recall": float, "precision": float, "f1": float, "fpr": float}` using sklearn.
   - `sweep_thresholds(normal_errors, anomaly_errors, percentiles: range) -> pd.DataFrame`:
     - For each percentile in range, compute threshold and evaluate. Return DataFrame.
   - `main()`:
     - Load model checkpoint and scaler.
     - Load anomaly sequences.
     - Compute train-set normal errors (on val split).
     - Compute anomaly errors.
     - Sweep percentiles 90–99.9 in 0.5 steps.
     - Select lowest threshold with recall >= 0.94.
     - Save chosen threshold to `checkpoints/threshold.json`: `{"tau": float, "percentile": int, "recall": float, "precision": float}`.

2. Run threshold calibration:
   ```
   python src/threshold.py
   ```
   Confirm recall >= 94% at chosen threshold.

---

## Phase 6: FastAPI Inference Service

**Goal**: Deploy the trained model as a REST API for real-time anomaly scoring.

### Tasks

1. Create `api/schemas.py`:
   - `class Transaction(BaseModel)`: fields matching all `numeric_cols` (amount, balance_before, balance_after, transaction_hour, day_of_week, days_since_last_txn), all `float`.
   - `class ScoreRequest(BaseModel)`: `transactions: list[Transaction]` (length 1 to 100).
   - `class ScoreResponse(BaseModel)`: `reconstruction_error: float`, `is_anomaly: bool`, `threshold: float`.
   - `class BatchScoreResponse(BaseModel)`: `results: list[ScoreResponse]`.

2. Create `api/inference.py`:
   - Module-level `model`, `scaler`, `threshold` variables initialized to `None`.
   - `load_artifacts(checkpoint_path, scaler_path, threshold_path)`: loads model, scaler, threshold JSON into module-level variables.
   - `score_sequence(transactions: list[dict]) -> dict`:
     - Convert to numpy array of shape `(1, T, num_features)`.
     - Apply `scaler.transform` on the `(T, num_features)` slice.
     - Pad or truncate to `seq_len = 30`.
     - Convert to tensor, move to device.
     - Run model: `x_hat = model(x)`.
     - Compute MSE: `error = ((x - x_hat) ** 2).mean().item()`.
     - Return `{"reconstruction_error": error, "is_anomaly": error > threshold, "threshold": threshold}`.

3. Create `api/main.py`:
   - Initialize `FastAPI(title="Financial Anomaly Detection API", version="1.0")`.
   - On startup event: call `load_artifacts(...)` with paths from env vars `MODEL_CHECKPOINT`, `SCALER_PATH`, `THRESHOLD_PATH`.
   - `POST /score`: accepts `ScoreRequest`, calls `score_sequence`, returns `ScoreResponse`.
   - `POST /score/batch`: accepts list of `ScoreRequest` (up to 100), calls `score_sequence` for each, returns `BatchScoreResponse`.
   - `GET /health`: returns `{"status": "ok", "model_loaded": bool}`.

4. Create `tests/test_api.py` using `httpx.AsyncClient`:
   - Test `GET /health` returns 200 with `model_loaded: true`.
   - Test `POST /score` with 30 synthetic transactions returns `ScoreResponse` with valid numeric error and boolean flag.
   - Test `POST /score` with fewer than 30 transactions (e.g., 10) — verify it does not crash (padding behavior).

5. Start the server and verify:
   ```
   MODEL_CHECKPOINT=checkpoints/best_model.pt SCALER_PATH=checkpoints/scaler.joblib THRESHOLD_PATH=checkpoints/threshold.json uvicorn api.main:app --port 8001 --reload
   curl -X POST http://localhost:8001/score -H "Content-Type: application/json" -d '{"transactions": [...]}'
   ```

6. Measure inference latency: run 1000 `/score` requests with a Python script using `time.perf_counter`, report mean latency. Assert mean < 50ms.

7. Run the full pytest suite:
   ```
   pytest tests/ -v
   ```
   All tests must pass.
