from __future__ import annotations

import torch
from torch import nn

from .model import WindowAutoencoder


class USAD(nn.Module):
    def __init__(self, sequence_length: int, channels: int, latent_dim: int = 5,
                 score_weight: float = 0.5) -> None:
        super().__init__()
        self.sequence_length, self.channels = sequence_length, channels
        self.score_weight = score_weight
        width = sequence_length * channels
        h1, h2 = max(1, width // 2), max(1, width // 4)
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(width, h1), nn.ReLU(True),
                                     nn.Linear(h1, h2), nn.ReLU(True),
                                     nn.Linear(h2, latent_dim), nn.ReLU(True))
        def decoder() -> nn.Sequential:
            return nn.Sequential(nn.Linear(latent_dim, h2), nn.ReLU(True),
                                 nn.Linear(h2, h1), nn.ReLU(True),
                                 nn.Linear(h1, width))
        self.decoder1, self.decoder2 = decoder(), decoder()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, channels = x.shape
        if (length, channels) != (self.sequence_length, self.channels):
            raise ValueError("USAD input shape does not match its locked configuration.")
        z = self.encoder(x)
        ae1, ae2 = self.decoder1(z), self.decoder2(z)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        shape = (batch, length, channels)
        return ae1.view(shape), ae2.view(shape), ae2ae1.view(shape)

    def point_loss(self, x: torch.Tensor, stage: str, epoch_index: int = 0) -> torch.Tensor:
        ae1, ae2, ae2ae1 = self(x)
        if stage == "train":
            n = float(epoch_index + 2)
            l1 = (ae1 - x).square().mean(-1) / n
            l1 = l1 + (1.0 - 1.0 / n) * (ae2ae1 - x).square().mean(-1)
            l2 = (ae2 - x).square().mean(-1) / n
            l2 = l2 - (1.0 - 1.0 / n) * (ae2ae1 - x).square().mean(-1)
            return l1 + l2
        return (self.score_weight * (ae1 - x).square().mean(-1)
                + (1.0 - self.score_weight) * (ae2ae1 - x).square().mean(-1))

    def trainable_for_scope(self, scope: str) -> list[nn.Parameter]:
        if scope == "full_model":
            return list(self.parameters())
        if scope == "reconstruction_path":
            return list(self.decoder1.parameters()) + list(self.decoder2.parameters())
        raise ValueError(f"Unsupported USAD update scope: {scope}")


class LSTMED(nn.Module):
    def __init__(self, sequence_length: int, channels: int, hidden_size: int = 5,
                 layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequence_length, self.channels = sequence_length, channels
        lstm_dropout = dropout if layers > 1 else 0.0
        self.layers, self.hidden_size = layers, hidden_size
        self.encoder = nn.LSTM(channels, hidden_size, layers, batch_first=True,
                               dropout=lstm_dropout)
        self.decoder = nn.LSTM(channels, hidden_size, layers, batch_first=True,
                               dropout=lstm_dropout)
        self.output_layer = nn.Linear(hidden_size, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        if (length, channels) != (self.sequence_length, self.channels):
            raise ValueError("LSTMED input shape does not match 12/12/12 protocol.")
        x = x.float()
        zeros = torch.zeros(self.layers, batch, self.hidden_size,
                            device=x.device, dtype=x.dtype)
        _, hidden = self.encoder(x, (zeros, zeros.clone()))
        outputs: list[torch.Tensor | None] = [None] * length
        for index in reversed(range(length)):
            prediction = self.output_layer(hidden[0][-1])
            outputs[index] = prediction
            decoder_input = x[:, index] if self.training else prediction
            _, hidden = self.decoder(decoder_input.unsqueeze(1), hidden)
        return torch.stack(outputs, dim=1)  # type: ignore[arg-type]

    def point_loss(self, x: torch.Tensor, stage: str, epoch_index: int = 0) -> torch.Tensor:
        return (self(x) - x).square().mean(-1)

    def trainable_for_scope(self, scope: str) -> list[nn.Parameter]:
        if scope == "full_model":
            return list(self.parameters())
        if scope == "reconstruction_path":
            return list(self.decoder.parameters()) + list(self.output_layer.parameters())
        raise ValueError(f"Unsupported LSTMED update scope: {scope}")


def build_section6_model(backbone: str, sequence_length: int, channels: int) -> nn.Module:
    if backbone == "win_ae":
        return WindowAutoencoder(sequence_length, channels)
    if backbone == "usad":
        return USAD(sequence_length, channels)
    if backbone == "lstmed":
        return LSTMED(sequence_length, channels)
    raise ValueError(f"Unsupported Section 6 backbone: {backbone}")
