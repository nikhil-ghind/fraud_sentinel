import numpy as np
import torch
from torch.utils.data import Dataset


class TransactionSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def split_normal_anomaly(sequences: np.ndarray, labels: np.ndarray) -> tuple:
    normal = sequences[labels == 0]
    anomaly = sequences[labels == 1]
    return normal, anomaly
