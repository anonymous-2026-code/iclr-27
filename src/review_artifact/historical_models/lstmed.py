import torch
import torch.nn as nn


class LSTMEDModule(nn.Module):
    """
    LSTM encoder-decoder adapted from Merlion's LSTM-ED implementation.
    """

    def __init__(self, n_features, hidden_size, num_layers, dropout, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bias=True,
        )
        self.decoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bias=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def _init_hidden(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        return h, c

    def forward(self, x):
        x_in = x if x.dtype == torch.float32 else x.float()
        _, dec_hidden = self.encoder(x_in, self._init_hidden(x.size(0)))

        output = torch.empty_like(x_in)
        for i in reversed(range(x.size(1))):
            output[:, i, :] = self.output_layer(dec_hidden[0][-1])
            decoder_input = x_in[:, i, :].unsqueeze(1) if self.training else output[:, i, :].unsqueeze(1)
            _, dec_hidden = self.decoder(decoder_input, dec_hidden)
        return output


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.hidden_size = int(getattr(configs, "latent_dim", 5))
        self.num_layers = int(getattr(configs, "e_layers", 1))
        self.dropout = float(getattr(configs, "dropout", 0.0))

        device = "cuda" if torch.cuda.is_available() and getattr(configs, "use_gpu", True) else "cpu"
        self.backbone = LSTMEDModule(
            n_features=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=device,
        )

    def get_tta_named_trainable_params(self, scope="full_model"):
        scope = str(scope).strip().lower()
        if scope == "full_model":
            return list(self.named_parameters())
        if scope == "reconstruction_path":
            named = list(self.backbone.decoder.named_parameters(prefix="backbone.decoder"))
            named.extend(list(self.backbone.output_layer.named_parameters(prefix="backbone.output_layer")))
            return named
        raise ValueError(f"Unsupported tta_scope '{scope}' for LSTMED.")

    def anomaly_detection(self, x_enc):
        bsz, seq_len, n_feats = x_enc.shape
        if seq_len != self.seq_len or n_feats != self.enc_in:
            raise ValueError(
                f"LSTMED expects input shape [B, {self.seq_len}, {self.enc_in}], "
                f"but got [B, {seq_len}, {n_feats}]"
            )
        return self.backbone(x_enc)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        raise NotImplementedError(f"LSTMED currently supports anomaly_detection only, got {self.task_name}")
