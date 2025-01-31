from pydantic import BaseModel
from typing import List


class Transaction(BaseModel):
    amount: float
    balance_before: float
    balance_after: float
    transaction_hour: float
    day_of_week: float
    days_since_last_txn: float


class ScoreRequest(BaseModel):
    transactions: List[Transaction]

    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {"amount": 100.0, "balance_before": 5000.0, "balance_after": 4900.0,
                     "transaction_hour": 14.0, "day_of_week": 2.0, "days_since_last_txn": 1.5}
                ] * 30
            }
        }


class ScoreResponse(BaseModel):
    reconstruction_error: float
    is_anomaly: bool
    threshold: float


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
