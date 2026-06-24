from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerEncoderConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    pooling: str = "mean"  # "mean" or "last"
    seq_len: int = 24


class TemporalEncoderBase(nn.Module):
    """Base class for temporal encoders."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerTemporalEncoder(TemporalEncoderBase):
    """Transformer encoder over time axis.

    Input:  x [B, L, N]
    Output: h [B, d_model]
    """

    def __init__(self, input_dim: int, cfg: TransformerEncoderConfig, use_custom: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(input_dim, cfg.d_model)
        self.use_custom = use_custom
        if use_custom:
            self.encoder = nn.ModuleList(
                [
                    TransformerBlockCustom(
                        cfg.d_model,
                        cfg.nhead,
                        cfg.dim_feedforward,
                        cfg.dropout,
                    )
                    for _ in range(cfg.num_layers)
                ]
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        h = self.embed(x)  # [B, L, d_model]
        if self.use_custom:
            for blk in self.encoder:
                h = blk(h)
        else:
            h = self.encoder(h)
        if self.cfg.pooling == "last":
            return h[:, -1, :]
        return h.mean(dim=1)


class InvertedTransformerTemporalEncoder(TemporalEncoderBase):
    """iTransformer-style encoder.

    Treats variables as tokens and the lookback window as token features.
    Input:  x [B, L, N]
    Output: h [B, d_model]
    """

    def __init__(self, input_dim: int, cfg: TransformerEncoderConfig, use_custom: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.token_embed = nn.Linear(cfg.seq_len, cfg.d_model)
        self.use_custom = use_custom
        if use_custom:
            self.encoder = nn.ModuleList(
                [
                    TransformerBlockCustom(
                        cfg.d_model,
                        cfg.nhead,
                        cfg.dim_feedforward,
                        cfg.dropout,
                    )
                    for _ in range(cfg.num_layers)
                ]
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N] -> [B, N, L]
        h = self.token_embed(x.transpose(1, 2))
        if self.use_custom:
            for blk in self.encoder:
                h = blk(h)
        else:
            h = self.encoder(h)
        if self.cfg.pooling == "last":
            return h[:, -1, :]
        return h.mean(dim=1)


class PatchTSTTemporalEncoder(TemporalEncoderBase):
    """PatchTST-style encoder with patch tokens over time."""

    def __init__(self, input_dim: int, cfg: TransformerEncoderConfig, patch_len: int = 4, stride: int = 4) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_len = patch_len
        self.stride = stride
        self.embed = nn.Linear(patch_len, cfg.d_model)
        self.channel_proj = nn.Linear(input_dim, cfg.d_model)
        self.encoder = nn.ModuleList(
            [TransformerBlockCustom(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        b, l, n = x.shape
        patches = x.transpose(1, 2).unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, N, P, patch]
        h = self.embed(patches).mean(dim=1)  # [B, P, d_model]
        h = h + self.channel_proj(x.mean(dim=1)).unsqueeze(1)
        for blk in self.encoder:
            h = blk(h)
        return h.mean(dim=1)


class InformerTemporalEncoder(TemporalEncoderBase):
    """Informer-style encoder with inter-layer distilling."""

    def __init__(self, input_dim: int, cfg: TransformerEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(input_dim, cfg.d_model)
        self.encoder = nn.ModuleList(
            [TransformerBlockCustom(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.distill = nn.ModuleList(
            [nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, stride=2, padding=1) for _ in range(max(cfg.num_layers - 1, 1))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for i, blk in enumerate(self.encoder):
            h = blk(h)
            if i < len(self.encoder) - 1:
                h = self.distill[i](h.transpose(1, 2)).transpose(1, 2)
        return h.mean(dim=1)


class AutoformerTemporalEncoder(TemporalEncoderBase):
    """Autoformer-style decomposition encoder."""

    def __init__(self, input_dim: int, cfg: TransformerEncoderConfig, kernel_size: int = 5) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(input_dim, cfg.d_model)
        self.trend_proj = nn.Linear(input_dim, cfg.d_model)
        self.encoder = nn.ModuleList(
            [TransformerBlockCustom(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def _decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self._decompose(x)
        h = self.embed(seasonal)
        for blk in self.encoder:
            h = blk(h)
        return h.mean(dim=1) + self.trend_proj(trend.mean(dim=1))


class FEDformerTemporalEncoder(TemporalEncoderBase):
    """FEDformer-style decomposition plus frequency-domain enhancement."""

    def __init__(self, input_dim: int, cfg: TransformerEncoderConfig, kernel_size: int = 5, topk_freq: int = 4) -> None:
        super().__init__()
        self.cfg = cfg
        self.topk_freq = topk_freq
        self.embed = nn.Linear(input_dim, cfg.d_model)
        self.freq_proj = nn.Linear(2 * topk_freq * input_dim, cfg.d_model)
        self.encoder = nn.ModuleList(
            [TransformerBlockCustom(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        h = self.embed(seasonal)
        for blk in self.encoder:
            h = blk(h)
        spec = torch.fft.rfft(seasonal, dim=1)
        k = min(self.topk_freq, spec.shape[1])
        amp = spec[:, :k, :]
        freq_feat = torch.cat([amp.real.reshape(x.size(0), -1), amp.imag.reshape(x.size(0), -1)], dim=-1)
        return h.mean(dim=1) + self.freq_proj(freq_feat)


class PyraformerTemporalEncoder(TemporalEncoderBase):
    """Pyraformer-style multiresolution encoder."""

    def __init__(self, input_dim: int, cfg: TransformerEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(input_dim, cfg.d_model)
        self.encoder = nn.ModuleList(
            [TransformerBlockCustom(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        levels = [x]
        cur = x
        for _ in range(2):
            if cur.shape[1] <= 2:
                break
            cur = F.avg_pool1d(cur.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            levels.append(cur)
        embeds = []
        for lv in levels:
            h = self.embed(lv)
            for blk in self.encoder:
                h = blk(h)
            embeds.append(h.mean(dim=1))
        return torch.stack(embeds, dim=1).mean(dim=1)


class CrossformerTemporalEncoder(TemporalEncoderBase):
    """Crossformer-style cross-time and cross-dimension encoder."""

    def __init__(self, input_dim: int, cfg: TransformerEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.time_embed = nn.Linear(input_dim, cfg.d_model)
        self.dim_embed = nn.Linear(cfg.seq_len, cfg.d_model)
        self.time_encoder = nn.ModuleList(
            [TransformerBlockCustom(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.num_layers)]
        )
        self.dim_encoder = nn.ModuleList(
            [TransformerBlockCustom(cfg.d_model, cfg.nhead, cfg.dim_feedforward, cfg.dropout) for _ in range(cfg.num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_t = self.time_embed(x)
        for blk in self.time_encoder:
            h_t = blk(h_t)
        h_d = self.dim_embed(x.transpose(1, 2))
        for blk in self.dim_encoder:
            h_d = blk(h_d)
        return 0.5 * (h_t.mean(dim=1) + h_d.mean(dim=1))


class VariableAxisExpert(nn.Module):
    """Transformer expert over variable axis.

    Input:  x [B, L, N]
    Output: e [B, L, d_model]
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float, use_custom: bool = True) -> None:
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.use_custom = use_custom
        if use_custom:
            self.encoder = nn.ModuleList(
                [
                    TransformerBlockCustom(d_model, nhead, 4 * d_model, dropout)
                    for _ in range(num_layers)
                ]
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N]
        b, l, n = x.shape
        x_var = x.view(b * l, n, 1)  # [B*L, N, 1]
        h = self.embed(x_var)  # [B*L, N, d_model]
        if self.use_custom:
            for blk in self.encoder:
                h = blk(h)
        else:
            h = self.encoder(h)
        h = h.mean(dim=1)  # pool over variables -> [B*L, d_model]
        return h.view(b, l, -1)


class SimpleMultiheadAttention(nn.Module):
    """Custom MHA that exposes attention weights for LRP."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.d_head = d_model // nhead
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.last_attn: Optional[torch.Tensor] = None
        self.last_v: Optional[torch.Tensor] = None
        self.last_q: Optional[torch.Tensor] = None
        self.last_k: Optional[torch.Tensor] = None
        self.last_attn_scores: Optional[torch.Tensor] = None
        self.capture_grads: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        b, t, _ = x.shape
        qkv = self.qkv(x)  # [B, T, 3*d]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.nhead, self.d_head).transpose(1, 2)  # [B, H, T, d]
        k = k.view(b, t, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.nhead, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn_scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)  # [B, H, T, d]
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        out = self.out(out)

        if self.capture_grads:
            self.last_q = q
            self.last_k = k
            self.last_attn_scores = attn_scores
            self.last_attn = attn
            self.last_v = v
            for tensor in (q, k, v, attn_scores, attn):
                if tensor.requires_grad:
                    tensor.retain_grad()
        else:
            self.last_attn = attn.detach()
            self.last_v = v.detach()
        return out


class TransformerBlockCustom(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.mha = SimpleMultiheadAttention(d_model, nhead, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.capture_grads: bool = False
        self.last_out: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.mha(x)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        if self.capture_grads:
            self.last_out = x
            if x.requires_grad:
                x.retain_grad()
        return x


_TEMPORAL_ENCODER_REGISTRY: Dict[str, type] = {
    "transformer": TransformerTemporalEncoder,
    "itransformer": InvertedTransformerTemporalEncoder,
    "patchtst": PatchTSTTemporalEncoder,
    "informer": InformerTemporalEncoder,
    "autoformer": AutoformerTemporalEncoder,
    "fedformer": FEDformerTemporalEncoder,
    "pyraformer": PyraformerTemporalEncoder,
    "crossformer": CrossformerTemporalEncoder,
}


def build_temporal_encoder(
    name: str, input_dim: int, cfg: TransformerEncoderConfig
) -> TemporalEncoderBase:
    if name not in _TEMPORAL_ENCODER_REGISTRY:
        raise ValueError(f"Unknown temporal encoder: {name}")
    return _TEMPORAL_ENCODER_REGISTRY[name](input_dim, cfg)


def register_temporal_encoder(name: str, cls: type) -> None:
    _TEMPORAL_ENCODER_REGISTRY[name] = cls
