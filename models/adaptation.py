from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class EMAConfig:
    eta: float = 0.01
    K: int = 10


@contextmanager
def temporary_params(model: nn.Module, delta: Dict[str, torch.Tensor]):
    """Temporarily apply parameter deltas for a single forward pass."""

    original = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in delta:
                original[name] = param.data.clone()
                param.data = param.data + delta[name]
    try:
        yield
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in original:
                    param.data.copy_(original[name])


class OnlineAdaptor:
    """Stage I/II/III adaptation coordinator."""

    def __init__(self, model: nn.Module, ema_cfg: EMAConfig) -> None:
        self.model = model
        self.ref_model = deepcopy(model)  # theta_ref
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ema_cfg = ema_cfg
        self._step = 0

    def update_ref_model(self) -> None:
        if self.ema_cfg.K <= 0:
            return
        if self._step % self.ema_cfg.K != 0:
            return
        eta = self.ema_cfg.eta
        with torch.no_grad():
            for p_ref, p_cur in zip(
                self.ref_model.parameters(), self.model.parameters()
            ):
                p_ref.data.mul_(1.0 - eta).add_(p_cur.data, alpha=eta)

    def step(self) -> None:
        self._step += 1
        self.update_ref_model()

    def stage3_predict(
        self,
        x: torch.Tensor,
        delta: Dict[str, torch.Tensor],
        gate_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stage III transient update: apply delta only for current prediction."""

        with temporary_params(self.model, delta):
            return self.model(x, gate_bias=gate_bias)


class AdaptationGenerator(nn.Module):
    """Low-rank adaptation generator for Stage II/III (Eq. 32-33)."""

    def __init__(self, drift_dim: int, rank: int, max_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(drift_dim, rank)
        self.w2_alpha = nn.Linear(rank, max_dim)
        self.w2_beta = nn.Linear(rank, max_dim)
        self.rank = rank
        self.max_dim = max_dim

    def forward(self, drift: torch.Tensor, din: int, dout: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Produce alpha, beta vectors (d_in, d_out)
        h = torch.sigmoid(self.w1(drift))
        alpha_full = torch.sigmoid(self.w2_alpha(h)) + 1.0
        beta_full = torch.sigmoid(self.w2_beta(h)) + 1.0
        alpha = alpha_full[:din]
        beta = beta_full[:dout]
        return alpha, beta


def build_param_delta(
    params: Dict[str, torch.Tensor],
    drift: torch.Tensor,
    generator: AdaptationGenerator,
    mask: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Build parameter deltas with optional masks."""

    deltas: Dict[str, torch.Tensor] = {}
    for name, p in params.items():
        if p.dim() != 2:
            continue
        din, dout = p.shape
        alpha, beta = generator(drift, din, dout)
        scale = alpha.unsqueeze(1) * beta.unsqueeze(0)
        delta = (scale - 1.0) * p
        if mask is not None and name in mask:
            delta = delta * mask[name]
        deltas[name] = delta
    return deltas


def topk_mask_from_tensor(
    tensor: torch.Tensor, k: int
) -> torch.Tensor:
    """Build a binary mask for top-k absolute entries."""

    if k <= 0:
        return torch.zeros_like(tensor)
    flat = tensor.abs().flatten()
    if k >= flat.numel():
        return torch.ones_like(tensor)
    thresh = torch.topk(flat, k, largest=True).values.min()
    return (tensor.abs() >= thresh).to(tensor.dtype)


def threshold_mask_from_tensor(
    tensor: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Binary mask by fixed threshold on absolute values."""

    if threshold <= 0:
        return torch.ones_like(tensor)
    return (tensor.abs() >= threshold).to(tensor.dtype)


def build_param_masks_from_relevance(
    model: nn.Module,
    relevance: Dict[str, torch.Tensor],
    k_per_layer: int,
) -> Dict[str, torch.Tensor]:
    """Create parameter masks per layer based on relevance maps."""

    masks: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.dim() != 2:
            continue
        if name in relevance:
            masks[name] = topk_mask_from_tensor(relevance[name], k_per_layer)
    return masks


def build_param_masks_from_grads(
    model: nn.Module,
    k_per_layer: int,
) -> Dict[str, torch.Tensor]:
    """Build masks from parameter gradients (TopK by |grad|)."""

    masks: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is None or p.dim() != 2:
            continue
        masks[name] = topk_mask_from_tensor(p.grad, k_per_layer)
    return masks


def build_masks_from_attention_heads(
    model: nn.Module,
    head_scores: Dict[int, torch.Tensor],
    topk_heads: int,
) -> Dict[str, torch.Tensor]:
    """Build masks for qkv/out weights based on head relevance."""

    masks: Dict[str, torch.Tensor] = {}
    mha_idx = -1
    for module_name, m in model.named_modules():
        if not isinstance(m, nn.Module):
            continue
        # match SimpleMultiheadAttention by attribute presence
        if hasattr(m, "qkv") and hasattr(m, "out") and hasattr(m, "nhead"):
            mha_idx += 1
            if mha_idx not in head_scores:
                continue
            scores = head_scores[mha_idx]
            if scores.numel() == 0:
                continue
            if topk_heads <= 0:
                continue
            k = min(topk_heads, scores.numel())
            topk = torch.topk(scores, k=k).indices
            d_head = m.d_head
            d_model = m.d_model

            # qkv weights: [3*d_model, d_model]
            qkv_name = f"{module_name}.qkv.weight"
            out_name = f"{module_name}.out.weight"
            for name in (qkv_name, out_name):
                if name not in dict(model.named_parameters()):
                    continue
                p = dict(model.named_parameters())[name]
                mask = torch.zeros_like(p)
                for h in topk:
                    h = int(h.item())
                    # rows corresponding to head h
                    row_start = h * d_head
                    row_end = (h + 1) * d_head
                    if name == qkv_name:
                        # q, k, v blocks
                        for blk in range(3):
                            r0 = blk * d_model + row_start
                            r1 = blk * d_model + row_end
                            mask[r0:r1, :] = 1.0
                    else:
                        # out weight: columns correspond to heads
                        mask[:, row_start:row_end] = 1.0
                masks[name] = mask
    return masks


def build_masks_from_attention_scores(
    model: nn.Module,
    attn_scores: Dict[int, torch.Tensor],
    topk_heads: int,
) -> Dict[str, torch.Tensor]:
    """Build masks from S(A) attention causal scores."""

    # Reduce S(A) to head-level scores and reuse head masking.
    head_scores: Dict[int, torch.Tensor] = {}
    for idx, s in attn_scores.items():
        # s: [B, H, T, T]
        head_scores[idx] = s.abs().mean(dim=(0, 2, 3))
    return build_masks_from_attention_heads(model, head_scores, topk_heads)


def merge_masks(*masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for m in masks:
        for name, mask in m.items():
            if name not in merged:
                merged[name] = mask.clone()
            else:
                merged[name] = torch.maximum(merged[name], mask)
    return merged


def apply_param_delta(
    model: nn.Module,
    deltas: Dict[str, torch.Tensor],
    scale: float = 1.0,
    max_delta: Optional[float] = None,
) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in deltas:
                delta = deltas[name]
                if max_delta is not None:
                    delta = torch.clamp(delta, -max_delta, max_delta)
                p.add_(delta, alpha=scale)
