from __future__ import annotations

import torch
import torch.nn as nn


class USADCore(nn.Module):
    def __init__(self, seq_len: int, enc_in: int, latent_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.input_dim = seq_len * enc_in
        self.latent_dim = latent_dim
        # Align encoder/decoder widths with win_ae: d -> d/2 -> d/4 -> z, and mirror back.
        self.hidden_dim1 = max(1, self.input_dim // 2)
        self.hidden_dim2 = max(1, self.input_dim // 4)

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, self.hidden_dim1),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim2, latent_dim),
            nn.ReLU(True),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim2),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim2, self.hidden_dim1),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim1, self.input_dim),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim2),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim2, self.hidden_dim1),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim1, self.input_dim),
        )

    def forward(self, x):
        batch_size, seq_len, n_feats = x.shape
        if seq_len != self.seq_len or n_feats != self.enc_in:
            raise ValueError(
                f"USAD expects input shape [B, {self.seq_len}, {self.enc_in}], "
                f"but got [B, {seq_len}, {n_feats}]"
            )

        z = self.encoder(x)
        ae1 = self.decoder1(z)
        ae2 = self.decoder2(z)
        ae2ae1 = self.decoder2(self.encoder(ae1))
        shape = (batch_size, self.seq_len, self.enc_in)
        return ae1.view(*shape), ae2.view(*shape), ae2ae1.view(*shape)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = int(configs.seq_len)
        self.enc_in = int(configs.enc_in)
        self.latent_dim = int(getattr(configs, 'latent_dim', 5))
        self.score_weight = float(getattr(configs, 'usad_score_weight', 0.5))
        self.core = USADCore(
            seq_len=self.seq_len,
            enc_in=self.enc_in,
            latent_dim=self.latent_dim,
        )

    def get_tta_named_trainable_params(self, scope="full_model"):
        scope = str(scope).strip().lower()
        if scope == "full_model":
            return list(self.named_parameters())
        if scope == "reconstruction_path":
            named = []
            named.extend(list(self.core.decoder1.named_parameters(prefix="core.decoder1")))
            named.extend(list(self.core.decoder2.named_parameters(prefix="core.decoder2")))
            return named
        raise ValueError(f"Unsupported tta_scope '{scope}' for USAD.")

    def _resolve_outputs(self, x, outputs=None):
        if outputs is None:
            outputs = self.forward(x)
        if not isinstance(outputs, tuple) or len(outputs) != 3:
            raise ValueError("USAD expects forward outputs as (ae1, ae2, ae2ae1).")
        return outputs

    @staticmethod
    def _reduce_loss(loss_per_point, reduction):
        if reduction == 'none':
            return loss_per_point
        if reduction == 'sum':
            return loss_per_point.sum()
        return loss_per_point.mean()

    def compute_score(self, x, outputs=None, stage='test'):
        ae1, _, ae2ae1 = self._resolve_outputs(x, outputs=outputs)
        weight = self.score_weight
        return weight * ((ae1 - x) ** 2).mean(dim=-1) + (1.0 - weight) * ((ae2ae1 - x) ** 2).mean(dim=-1)

    def compute_tta_loss(self, x, outputs=None, stage='tta', reduction='none'):
        return self._reduce_loss(self.compute_score(x, outputs=outputs, stage=stage), reduction)

    def compute_loss(self, x, outputs=None, stage='train', epoch=None, reduction='mean'):
        ae1, ae2, ae2ae1 = self._resolve_outputs(x, outputs=outputs)
        if stage in {'tta', 'tta_pre', 'tta_post', 'calibration'}:
            return self.compute_tta_loss(x, outputs=(ae1, ae2, ae2ae1), stage=stage, reduction=reduction)

        epoch_index = 0 if epoch is None else int(epoch)
        n = float(epoch_index + 2)
        l1 = (1.0 / n) * ((ae1 - x) ** 2).mean(dim=-1) + (1.0 - 1.0 / n) * ((ae2ae1 - x) ** 2).mean(dim=-1)
        l2 = (1.0 / n) * ((ae2 - x) ** 2).mean(dim=-1) - (1.0 - 1.0 / n) * ((ae2ae1 - x) ** 2).mean(dim=-1)
        return self._reduce_loss(l1 + l2, reduction)

    def anomaly_detection(self, x_enc):
        ae1, _, _ = self.core(x_enc)
        return ae1

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name != "anomaly_detection":
            raise NotImplementedError("USAD currently supports anomaly_detection only.")
        return self.core(x_enc)
