"""Generate synthetic transaction data for testing."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_NORMAL = 10000
N_ANOMALY = 200
N_ACCOUNTS = 200

def gen_transactions(n, account_ids, is_anomaly=False):
    rows = []
    for _ in range(n):
        acc = rng.choice(account_ids)
        ts = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
        bal_before = max(0, rng.normal(5000, 2000))
        if is_anomaly:
            amount = rng.uniform(5000, 20000)  # unusually large
        else:
            amount = max(0, rng.exponential(200))
        bal_after = max(0, bal_before - amount)
        rows.append({
            "transaction_id": f"TXN{rng.integers(0, 1e9):010d}",
            "account_id": acc,
            "timestamp": ts,
            "amount": round(amount, 2),
            "balance_before": round(bal_before, 2),
            "balance_after": round(bal_after, 2),
            "merchant_category": rng.choice(["retail", "food", "travel", "online"]),
            "is_anomaly": int(is_anomaly),
        })
    return rows

accounts = [f"ACC{i:04d}" for i in range(N_ACCOUNTS)]
rows = gen_transactions(N_NORMAL, accounts, False) + gen_transactions(N_ANOMALY, accounts, True)
df = pd.DataFrame(rows).sort_values(["account_id", "timestamp"]).reset_index(drop=True)
df.to_csv("data/sample_transactions.csv", index=False)
print(f"Generated {len(df)} rows ({N_NORMAL} normal, {N_ANOMALY} anomaly)")
