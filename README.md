# Fraud Sentinel

LSTM Autoencoder for financial transaction anomaly detection. Trains on normal transaction sequences, detects anomalies via reconstruction error threshold with ≥94% recall. Deployed as a FastAPI microservice.

## Architecture

```mermaid
flowchart TB
    CSV["raw transactions CSV<br/>data/generate_sample.py for synthetic data"] --> FE["load_and_engineer<br/>hour, day_of_week, days_since_last per account"]
    FE --> SC["StandardScaler fit on training rows<br/>saved to checkpoints for serving"]
    SC --> SEQ["make_sequences<br/>sliding window of seq_len steps per account,<br/>label = any fraud inside the window"]

    subgraph train["Training — src/train.py (normal windows only)"]
        SEQ --> SPL["split_normal_anomaly"]
        SPL --> NRM["normal sequences"]
        NRM --> DL["train / val DataLoaders"]
        DL --> AE["LSTMAutoencoder<br/>encoder LSTM to (h_n, c_n),<br/>decoder LSTM repeated seq_len steps + Linear head"]
        AE --> LOSS["MSE reconstruction loss, Adam"]
        LOSS --> ES["EarlyStopping on val loss<br/>best state_dict written to checkpoint_path"]
    end

    subgraph cal["Calibration — src/threshold.py"]
        SPL --> ANM["anomaly sequences"]
        ES --> ERR["compute_reconstruction_errors<br/>per-sequence MSE"]
        NRM --> ERR
        ANM --> ERR
        ERR --> SWEEP["sweep_thresholds<br/>percentiles 90 to 99.5 in 0.5 steps"]
        SWEEP --> PICK["pick the lowest threshold with recall >= 0.94,<br/>else fall back to the first row"]
        PICK --> TJ["checkpoints/threshold.json<br/>tau, percentile, recall, precision"]
    end

    subgraph serve["Serving — api/"]
        LIFE["lifespan: inference.load_artifacts<br/>state_dict + scaler + threshold.json"]
        SCORE["POST /score and POST /score/batch"]
        SCORE --> ISEQ["rebuild features, scale, window"]
        ISEQ --> IREC["forward pass, MSE against the input"]
        IREC --> FLAG["is_fraud = error > tau"]
        HEALTH["GET /health reports model_loaded"]
    end
    ES --> LIFE
    SC --> LIFE
    TJ --> LIFE
```

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

## Evaluation

Anomaly-detection metrics are computed in `src/threshold.py` against a labelled holdout (mix of normal + fraudulent sequences synthesised by `data/generate_sample.py`).

| Metric | What it measures | How to compute |
|--------|------------------|----------------|
| Recall @ chosen threshold | Fraction of true frauds caught | `python -m src.threshold` sweeps percentile thresholds and selects the lowest one with recall >= 0.94 |
| Precision @ chosen threshold | Fraction of alerts that are real frauds | Reported alongside recall in `checkpoints/threshold.json` |
| F1 @ chosen threshold | Harmonic mean of precision/recall | Same artifact |
| False-positive rate (FPR) | Normal sequences flagged as fraud | Logged for each candidate threshold during the sweep |
| PR-AUC | Threshold-free quality of reconstruction-error scores | `sklearn.metrics.average_precision_score(y_true, recon_error)` over the holdout; printed by the calibration script |
| ROC-AUC | Secondary diagnostic | `sklearn.metrics.roc_auc_score` on the same scores |
| Alert precision @ k | Precision when alerting only on the top-k riskiest sequences per day | Sort scored sequences by reconstruction error, take top-k, compute fraction labelled fraud |

Run end-to-end:

```bash
python data/generate_sample.py        # synthetic train + labelled holdout
python -m src.train                   # train autoencoder on normal-only
python -m src.threshold               # writes checkpoints/threshold.json with metrics
```

Distribution-shift sanity check: compare reconstruction-error histograms on train vs. holdout-normal vs. holdout-fraud (same script). Healthy models show clear separation between the normal and fraud distributions.

## Model

- Encoder: `LSTM(6, 128, 2 layers)` → `(h_n, c_n)`
- Decoder: repeat `h_n[-1]` × T → `LSTM(128, 128, 2 layers)` → `Linear(128, 6)`
- Loss: MSE between input and reconstruction
- Threshold: 99th percentile of training set reconstruction errors (tuned for ≥94% recall)
