from __future__ import annotations

from typing import Any

import torch


def _as_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value.detach().float()
    return None


def _tensor_stat(values: torch.Tensor | None, stat: str) -> float:
    if values is None or values.numel() == 0:
        return float("nan")
    flat = values.detach().float().reshape(-1)
    if stat == "mean":
        return float(flat.mean().cpu().item())
    if stat == "std":
        return float(flat.std(unbiased=False).cpu().item())
    if stat == "max":
        return float(flat.max().cpu().item())
    if stat == "min":
        return float(flat.min().cpu().item())
    raise ValueError(f"unknown tensor stat: {stat}")


def _norm(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == 0:
        return tensor.reshape(1)
    return torch.linalg.vector_norm(tensor.float(), dim=-1)


def _entropy(probabilities: torch.Tensor | None) -> torch.Tensor | None:
    if probabilities is None:
        return None
    probs = probabilities.float().clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=-1)


def _top1_and_margin(probabilities: torch.Tensor | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if probabilities is None or probabilities.ndim < 2:
        return None, None
    top = torch.topk(probabilities.float(), k=min(2, probabilities.shape[-1]), dim=-1).values
    top1 = top[..., 0]
    if top.shape[-1] == 1:
        margin = top1
    else:
        margin = top[..., 0] - top[..., 1]
    return top1, margin


def _attr(output: Any, name: str) -> torch.Tensor | None:
    return _as_tensor(getattr(output, name, None))


def _sigmoid_if_logits(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if bool(torch.all((tensor >= 0.0) & (tensor <= 1.0)).cpu().item()):
        return tensor
    return torch.sigmoid(tensor)


def build_online_gate_dynamic_feature_row(
    output: Any,
    *,
    split: str,
    batch_index: int,
    sample_start: int,
    sample_end: int,
    validation_window_id: str,
    target_name: str = "raw_residual_vector",
) -> dict[str, Any]:
    """Summarize V29 model-forward state for a future online mode gate.

    The function intentionally accepts only the model output and schedule
    metadata. It does not accept ground truth, teacher gains, counterfactual
    outcomes, or oracle labels.
    """

    alpha = _attr(output, "alpha")
    alpha_entropy = _entropy(alpha)
    alpha_top1, alpha_margin = _top1_and_margin(alpha)
    local_env_entropy = _entropy(_attr(output, "expert_local_env_posteriors"))
    release_confidence = _sigmoid_if_logits(_attr(output, "expert_release_confidence"))
    support = _sigmoid_if_logits(_attr(output, "expert_release_support_score"))
    suppression = _sigmoid_if_logits(_attr(output, "expert_release_suppression_score"))
    servable_change_norm = _norm(_attr(output, "pooled_servable_change"))
    release_state_norm = _norm(_attr(output, "pooled_residual_release_state"))
    invariant_correction_norm = _norm(_attr(output, "expert_invariant_correction"))
    correction_gate = _attr(output, "expert_invariant_correction_gate")
    prediction_norm = _norm(_attr(output, "y_hat"))

    sample_count = max(int(sample_end) - int(sample_start), 0)
    if alpha is not None and alpha.ndim > 0:
        sample_count = int(alpha.shape[0])

    return {
        "target_name": target_name,
        "split": split,
        "batch_index": int(batch_index),
        "validation_window_id": str(validation_window_id),
        "feature_dynamic_status": "available_from_model_forward",
        "feature_dynamic_sample_start": int(sample_start),
        "feature_dynamic_sample_end": int(sample_end),
        "feature_dynamic_sample_count": sample_count,
        "feature_dynamic_alpha_entropy_mean": _tensor_stat(alpha_entropy, "mean"),
        "feature_dynamic_alpha_entropy_std": _tensor_stat(alpha_entropy, "std"),
        "feature_dynamic_alpha_top1_mean": _tensor_stat(alpha_top1, "mean"),
        "feature_dynamic_alpha_margin_mean": _tensor_stat(alpha_margin, "mean"),
        "feature_dynamic_local_env_entropy_mean": _tensor_stat(local_env_entropy, "mean"),
        "feature_dynamic_local_env_entropy_std": _tensor_stat(local_env_entropy, "std"),
        "feature_dynamic_release_confidence_mean": _tensor_stat(release_confidence, "mean"),
        "feature_dynamic_release_confidence_std": _tensor_stat(release_confidence, "std"),
        "feature_dynamic_release_support_mean": _tensor_stat(support, "mean"),
        "feature_dynamic_release_suppression_mean": _tensor_stat(suppression, "mean"),
        "feature_dynamic_servable_change_norm_mean": _tensor_stat(servable_change_norm, "mean"),
        "feature_dynamic_servable_change_norm_std": _tensor_stat(servable_change_norm, "std"),
        "feature_dynamic_servable_change_norm_max": _tensor_stat(servable_change_norm, "max"),
        "feature_dynamic_release_state_norm_mean": _tensor_stat(release_state_norm, "mean"),
        "feature_dynamic_release_state_norm_max": _tensor_stat(release_state_norm, "max"),
        "feature_dynamic_invariant_correction_norm_mean": _tensor_stat(invariant_correction_norm, "mean"),
        "feature_dynamic_invariant_correction_gate_mean": _tensor_stat(correction_gate, "mean"),
        "feature_dynamic_prediction_norm_mean": _tensor_stat(prediction_norm, "mean"),
        "feature_dynamic_source": "model_forward_pre_target_update",
    }
