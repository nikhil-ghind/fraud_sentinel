import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ScoreRequest, ScoreResponse, BatchScoreResponse
from api import inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = os.getenv("MODEL_CHECKPOINT", "checkpoints/best_model.pt")
    scaler = os.getenv("SCALER_PATH", "checkpoints/scaler.joblib")
    thresh = os.getenv("THRESHOLD_PATH", "checkpoints/threshold.json")
    if all(os.path.exists(p) for p in [checkpoint, scaler, thresh]):
        inference.load_artifacts(checkpoint, scaler, thresh)
    yield


app = FastAPI(title="Financial Anomaly Detection API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": inference.model is not None}


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    txns = [t.model_dump() for t in request.transactions]
    result = inference.score_sequence(txns)
    return ScoreResponse(**result)


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(requests: list[ScoreRequest]):
    results = []
    for req in requests:
        txns = [t.model_dump() for t in req.transactions]
        results.append(ScoreResponse(**inference.score_sequence(txns)))
    return BatchScoreResponse(results=results)
