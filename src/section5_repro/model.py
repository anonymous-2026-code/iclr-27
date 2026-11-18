from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size: int, latent_size: int) -> None:
        super().__init__()
        h1 = max(1, input_size // 2)
        h2 = max(1, input_size // 4)
        self.linear1 = nn.Linear(input_size, h1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(h2, latent_size)
        self.relu3 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return self.relu3(self.linear3(x))


class Decoder(nn.Module):
    def __init__(self, output_size: int, latent_size: int) -> None:
        super().__init__()
        h2 = max(1, output_size // 4)
        h1 = max(1, output_size // 2)
        self.linear1 = nn.Linear(latent_size, h2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(h2, h1)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(h1, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return self.linear3(x)


class WindowAutoencoder(nn.Module):
    """Exact AE architecture used by the Section 5 executable snapshot."""

    def __init__(self, sequence_length: int, channels: int) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.channels = int(channels)
        flattened = self.sequence_length * self.channels
        latent = min(64, max(8, flattened // 8))
        self.encoder = Encoder(flattened, latent)
        self.decoder = Decoder(flattened, latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        if (length, channels) != (self.sequence_length, self.channels):
            raise ValueError(
                f"Expected (*,{self.sequence_length},{self.channels}), got {tuple(x.shape)}"
            )
        encoded = self.encoder(x.reshape(batch, length * channels))
        return self.decoder(encoded).reshape(batch, length, channels)


def pointwise_reconstruction_loss(
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reconstruction = model(x)
    loss = torch.mean((reconstruction - x) ** 2, dim=-1)
    return reconstruction, loss

