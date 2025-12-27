import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings
    so the Transformer can understand sequence order.
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        x: (batch_size, sequence_length, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class TransformerAnomalyDetector(nn.Module):
    """
    Transformer Encoder-based model for
    time series anomaly detection using reconstruction error.
    """
    def __init__(
        self,
        input_dim=1,
        d_model=64,
        nhead=4,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.decoder = nn.Linear(d_model, input_dim)

    def forward(self, x, src_mask=None):
        """
        x: (batch_size, sequence_length, input_dim)
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask=src_mask)
        x = self.decoder(x)
        return x
