import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

    def forward(self, x: torch.Tensor):
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, h: torch.Tensor, c: torch.Tensor, seq_len: int) -> torch.Tensor:
        decoder_input = h[-1].unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(decoder_input, (h, c))
        return self.out(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(hidden_size, input_size, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, c_n) = self.encoder(x)
        return self.decoder(h_n, c_n, x.size(1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        return h_n[-1]


def build_model(config: dict) -> LSTMAutoencoder:
    return LSTMAutoencoder(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
