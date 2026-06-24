from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


STAGE3_RISK_FEATURE_NAMES: tuple[str, ...] = (
    "drift_norm",
    "delta_norm_pre_clamp",
    "delta_norm_post_clamp",
    "clamp_fraction",
    "pred_shift",
    "recon_error",
    "base_abs_mean",
    "candidate_abs_mean",
    "candidate_minus_base_mean",
)


class Stage3RiskFeatureStandardizer:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.fit_sample_count = 0

    def fit(self, features: torch.Tensor) -> "Stage3RiskFeatureStandardizer":
        if features.ndim != 2:
            raise ValueError("risk gate features must be 2-D")
        if features.shape[0] < 2:
            raise ValueError("at least two samples are required to fit risk feature standardizer")
        self.mean = features.detach().float().mean(dim=0)
        self.std = torch.clamp(features.detach().float().std(dim=0, unbiased=False), min=self.eps)
        self.fit_sample_count = int(features.shape[0])
        return self

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Stage3RiskFeatureStandardizer must be fit before transform")
        mean = self.mean.to(device=features.device, dtype=features.dtype)
        std = self.std.to(device=features.device, dtype=features.dtype)
        return (features - mean) / std

    def state_dict(self) -> dict[str, torch.Tensor | int | float]:
        if self.mean is None or self.std is None:
            raise RuntimeError("Stage3RiskFeatureStandardizer must be fit before serialization")
        return {
            "mean": self.mean.detach().clone(),
            "std": self.std.detach().clone(),
            "fit_sample_count": self.fit_sample_count,
            "eps": self.eps,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor | int | float]) -> "Stage3RiskFeatureStandardizer":
        self.mean = torch.as_tensor(state["mean"]).detach().float()
        self.std = torch.as_tensor(state["std"]).detach().float()
        self.fit_sample_count = int(state["fit_sample_count"])
        self.eps = float(state.get("eps", self.eps))
        return self


class Stage3RiskGateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features).squeeze(-1))

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(features)

    def to_config(self) -> dict[str, Any]:
        return {
            "model_type": "mlp",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "feature_names": list(STAGE3_RISK_FEATURE_NAMES),
        }


class Stage3RiskGateLogistic(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(features).squeeze(-1))

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(features)

    def to_config(self) -> dict[str, Any]:
        return {
            "model_type": "logistic",
            "input_dim": self.input_dim,
            "feature_names": list(STAGE3_RISK_FEATURE_NAMES),
        }


class Stage3RiskGateBundle(nn.Module):
    def __init__(
        self,
        gate: nn.Module,
        standardizer: Stage3RiskFeatureStandardizer,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.standardizer = standardizer

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        return self.gate(self.standardizer.transform(raw_features))


@dataclass(frozen=True)
class Stage3RiskGateTrainResult:
    gate: nn.Module
    standardizer: Stage3RiskFeatureStandardizer
    metrics: dict[str, Any]


def build_stage3_risk_features(
    *,
    drift_vec: torch.Tensor,
    delta_summary: dict[str, float],
    pred_shift: float,
    recon_error: float,
    y_base: torch.Tensor,
    y_candidate: torch.Tensor,
) -> torch.Tensor:
    drift_norm = torch.linalg.vector_norm(drift_vec.detach()).reshape(())
    base_abs_mean = y_base.detach().abs().mean()
    candidate_abs_mean = y_candidate.detach().abs().mean()
    candidate_minus_base_mean = (y_candidate.detach() - y_base.detach()).mean()
    values = [
        drift_norm,
        torch.as_tensor(float(delta_summary["delta_norm_pre_clamp"]), device=y_base.device, dtype=y_base.dtype),
        torch.as_tensor(float(delta_summary["delta_norm_post_clamp"]), device=y_base.device, dtype=y_base.dtype),
        torch.as_tensor(float(delta_summary["clamp_fraction"]), device=y_base.device, dtype=y_base.dtype),
        torch.as_tensor(float(pred_shift), device=y_base.device, dtype=y_base.dtype),
        torch.as_tensor(float(recon_error), device=y_base.device, dtype=y_base.dtype),
        base_abs_mean,
        candidate_abs_mean,
        candidate_minus_base_mean,
    ]
    return torch.stack([value.to(device=y_base.device, dtype=y_base.dtype) for value in values]).unsqueeze(0)


def train_stage3_risk_gate(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    hidden_dim: int = 32,
    dropout: float = 0.0,
    epochs: int = 100,
    lr: float = 1e-3,
    positive_weight: float | None = None,
    model_type: str = "mlp",
) -> tuple[nn.Module, Stage3RiskFeatureStandardizer, dict[str, Any]]:
    if features.ndim != 2:
        raise ValueError("risk gate features must be 2-D")
    labels = labels.detach().float().reshape(-1)
    if labels.shape[0] != features.shape[0]:
        raise ValueError("risk gate features and labels must have the same sample count")
    if features.shape[0] < 2:
        raise ValueError("at least two samples are required to train risk gate")

    standardizer = Stage3RiskFeatureStandardizer().fit(features)
    x = standardizer.transform(features.detach().float())
    y = labels.to(device=x.device, dtype=x.dtype)
    if model_type == "mlp":
        gate: nn.Module = Stage3RiskGateMLP(features.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    elif model_type == "logistic":
        gate = Stage3RiskGateLogistic(features.shape[1])
    else:
        raise ValueError(f"unsupported Stage III risk gate model_type: {model_type}")
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    weight = None
    if positive_weight is not None:
        weight = torch.where(y > 0.5, torch.full_like(y, float(positive_weight)), torch.ones_like(y))

    with torch.no_grad():
        initial_loss = F.binary_cross_entropy(gate(x), y, weight=weight).item()
    for _ in range(int(epochs)):
        pred = gate(x)
        loss = F.binary_cross_entropy(pred, y, weight=weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final_pred = gate(x)
        final_loss = F.binary_cross_entropy(final_pred, y, weight=weight).item()
        pred_label = (final_pred >= 0.5).float()
        accuracy = (pred_label == y).float().mean().item()
        positive_mask = y > 0.5
        negative_mask = ~positive_mask
        true_positive = ((pred_label > 0.5) & positive_mask).float().sum().item()
        false_positive = ((pred_label > 0.5) & negative_mask).float().sum().item()
        true_negative = ((pred_label <= 0.5) & negative_mask).float().sum().item()
        false_negative = ((pred_label <= 0.5) & positive_mask).float().sum().item()
        positive_count = float(positive_mask.float().sum().item())
        negative_count = float(negative_mask.float().sum().item())
    return gate, standardizer, {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "train_accuracy": float(accuracy),
        "positive_rate": float(y.mean().item()),
        "predicted_positive_rate": float((pred_label > 0.5).float().mean().item()),
        "positive_recall": float(true_positive / positive_count) if positive_count > 0 else 0.0,
        "negative_recall": float(true_negative / negative_count) if negative_count > 0 else 0.0,
        "false_positive_rate": float(false_positive / negative_count) if negative_count > 0 else 0.0,
        "false_negative_rate": float(false_negative / positive_count) if positive_count > 0 else 0.0,
        "sample_count": float(features.shape[0]),
        "model_type": model_type,
    }


def save_stage3_risk_gate_checkpoint(
    *,
    path: str | Path,
    gate: nn.Module,
    standardizer: Stage3RiskFeatureStandardizer,
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V2 risk gate checkpoint: {path}")
    torch.save(
        {
            "risk_gate_class": gate.__class__.__name__,
            "risk_gate_config": gate.to_config(),
            "risk_gate_state_dict": gate.state_dict(),
            "standardizer_state_dict": standardizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_stage3_risk_gate_checkpoint(path: str | Path) -> tuple[Stage3RiskGateBundle, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    config = checkpoint["risk_gate_config"]
    model_type = config.get("model_type", "mlp")
    if model_type == "mlp":
        gate: nn.Module = Stage3RiskGateMLP(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            dropout=float(config.get("dropout", 0.0)),
        )
    elif model_type == "logistic":
        gate = Stage3RiskGateLogistic(input_dim=int(config["input_dim"]))
    else:
        raise ValueError(f"Unsupported Stage III risk gate checkpoint model_type: {model_type}")
    gate.load_state_dict(checkpoint["risk_gate_state_dict"])
    standardizer = Stage3RiskFeatureStandardizer().load_state_dict(checkpoint["standardizer_state_dict"])
    bundle = Stage3RiskGateBundle(gate, standardizer)
    bundle.eval()
    return bundle, dict(checkpoint.get("metadata", {}))
