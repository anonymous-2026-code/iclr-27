from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model: int = 512) -> None:
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.tcn(value).mean(dim=2)


class InterVariableEncoder(nn.Module):
    """Post-norm attention/FFN block used by the E7 SANA port."""

    def __init__(
        self,
        d_model: int = 512,
        heads: int = 8,
        d_ff: int = 512,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, heads, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            value, value, value, need_weights=False
        )
        value = self.norm1(value + attended)
        return self.norm2(
            value + self.ff2(functional.gelu(self.ff1(value)))
        )


class SANA(nn.Module):
    def __init__(
        self,
        variables: int,
        window: int = 12,
        d_model: int = 512,
        heads: int = 8,
        d_ff: int = 512,
    ) -> None:
        super().__init__()
        self.variables = int(variables)
        self.window = int(window)
        self.d_model = int(d_model)
        self.temporal_embedding = nn.ModuleList(
            [TemporalEmbedding(d_model) for _ in range(self.variables)]
        )
        self.encoder = InterVariableEncoder(d_model, heads, d_ff)
        self.projection = nn.ModuleList(
            [nn.Linear(d_model, window) for _ in range(self.variables)]
        )
        self.gating = nn.Parameter(torch.zeros(self.variables))

    def forward_reference(self, value: torch.Tensor) -> torch.Tensor:
        by_variable = value.permute(0, 2, 1)
        embedded = torch.stack(
            [
                module(by_variable[:, index : index + 1])
                for index, module in enumerate(self.temporal_embedding)
            ],
            dim=1,
        )
        encoded = self.encoder(embedded)
        adjustment = torch.stack(
            [
                module(encoded[:, index])
                for index, module in enumerate(self.projection)
            ],
            dim=2,
        )
        return adjustment * torch.tanh(self.gating).view(1, 1, -1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Vectorized evaluation of the variable-specific reference modules."""
        by_variable = value.permute(0, 2, 1)
        convolutions = [module.tcn[0] for module in self.temporal_embedding]
        convolution_weight = torch.cat(
            [module.weight for module in convolutions], dim=0
        )
        convolution_bias = torch.cat(
            [module.bias for module in convolutions], dim=0
        )
        encoded_time = functional.conv1d(
            by_variable,
            convolution_weight,
            convolution_bias,
            padding=1,
            groups=self.variables,
        )
        embedded = functional.relu(encoded_time).reshape(
            value.shape[0],
            self.variables,
            self.d_model,
            value.shape[1],
        ).mean(dim=3)
        encoded = self.encoder(embedded)
        projection_weight = torch.stack(
            [module.weight for module in self.projection], dim=0
        )
        projection_bias = torch.stack(
            [module.bias for module in self.projection], dim=0
        )
        adjustment = (
            torch.einsum("bcd,cld->bcl", encoded, projection_weight)
            + projection_bias.unsqueeze(0)
        ).permute(0, 2, 1)
        return adjustment * torch.tanh(self.gating).view(1, 1, -1)


class SANAPipeline(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        variables: int,
        window: int = 12,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.sana_in = SANA(variables, window=window)
        self.sana_out = SANA(variables, window=window)

    def train(self, mode: bool = True) -> "SANAPipeline":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        adjusted = raw + self.sana_in(raw)
        frozen_reconstruction = self.backbone(adjusted)
        return frozen_reconstruction - self.sana_out(frozen_reconstruction)

    def sana_parameters(self):
        yield from self.sana_in.parameters()
        yield from self.sana_out.parameters()
