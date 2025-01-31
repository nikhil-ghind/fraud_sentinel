# Fraud Sentinel

LSTM Autoencoder for financial transaction anomaly detection. Trains on normal transaction sequences, detects anomalies via reconstruction error threshold with ≥94% recall. Deployed as a FastAPI microservice.

## Overview

- **LSTM Autoencoder**: Encoder (2-layer LSTM) → Decoder (2-layer LSTM) → reconstruction MSE
- **Threshold calibration**: Sweep percentiles 90–99.9%, select lowest threshold achieving ≥94% recall
- **FastAPI service**: `/score` (single) and `/score/batch` endpoints with <50ms latency
- **Distribution shift analysis**: Compare train/test reconstruction error distributions

## Tech Stack

Python 3.11 · PyTorch · FastAPI · Pandas · scikit-learn · joblib

## Quickstart

```bash
pip install -r requirements.txt

# Generate synthetic data
python data/generate_sample.py

# Train autoencoder (on normal transactions only)
python -m src.train

# Calibrate anomaly threshold
python -m src.threshold

# Start API
MODEL_CHECKPOINT=checkpoints/best_model.pt \
SCALER_PATH=checkpoints/scaler.joblib \
THRESHOLD_PATH=checkpoints/threshold.json \
uvicorn api.main:app --port 8001

# Score a sequence
curl -X POST http://localhost:8001/score \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'

# Tests
pytest tests/ -v
```

## Architecture

```
Raw transactions (CSV)
    → feature engineering (hour, day_of_week, days_since_last)
    → StandardScaler normalization
    → sliding window sequences (T=30, F=6)
    → LSTM Encoder → (h, c)
    → LSTM Decoder → reconstructed sequence
    → MSE anomaly score
    → threshold τ (99th percentile of train errors) → binary flag
```

## Model

- Encoder: `LSTM(6, 128, 2 layers)` → `(h_n, c_n)`
- Decoder: repeat `h_n[-1]` × T → `LSTM(128, 128, 2 layers)` → `Linear(128, 6)`
- Loss: MSE between input and reconstruction
- Threshold: 99th percentile of training set reconstruction errors (tuned for ≥94% recall)
