from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F


TEACHER_COUNTERFACTUAL_MODES = {
    "zero",
    "shuffled",
    "conditional_shuffled",
    "conditional_permuted_drift",
    "sign_flipped",
    "random_matched_norm",
    "conditional_random_matched_norm",
    "amplified_2x",
    "amplified_5x",
}


def _safe_gain(base: float, updated: float) -> float:
    if base != base or updated != updated or base == 0.0:
        return float("nan")
    return (base - updated) / base


def compute_teacher_correction_metrics(
    *,
    y_base: torch.Tensor,
    y_true: torch.Tensor,
    correction: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Summarize whether a residual teacher correction points the right way."""

    y_base = y_base.detach().float()
    y_true = y_true.detach().float()
    correction = correction.detach().float()
    if y_base.shape != y_true.shape or y_base.shape != correction.shape:
        raise ValueError("y_base, y_true, and correction must have identical shapes")

    y_corrected = y_base + correction
    residual = y_true - y_base
    residual_flat = residual.reshape(residual.shape[0], -1)
    correction_flat = correction.reshape(correction.shape[0], -1)
    base_mse = float(F.mse_loss(y_base, y_true).item())
    corrected_mse = float(F.mse_loss(y_corrected, y_true).item())
    base_mae = float(F.l1_loss(y_base, y_true).item())
    corrected_mae = float(F.l1_loss(y_corrected, y_true).item())
    residual_norm = torch.linalg.vector_norm(residual_flat, dim=1)
    correction_norm = torch.linalg.vector_norm(correction_flat, dim=1)
    nonzero_correction = correction_norm > eps
    nonzero_residual = residual_norm > eps
    sign_mask = residual_flat.abs() > eps
    if sign_mask.any():
        sign_agreement = torch.eq(torch.sign(residual_flat[sign_mask]), torch.sign(correction_flat[sign_mask]))
        sign_agreement_rate = float(sign_agreement.float().mean().item())
    else:
        sign_agreement_rate = float("nan")
    helpful_element_rate = float(((residual_flat * correction_flat) > 0.0).float().mean().item())
    harmful_element_rate = float(((residual_flat * correction_flat) < 0.0).float().mean().item())
    overcorrection_rate = float((correction_flat.abs() > residual_flat.abs().clamp_min(eps)).float().mean().item())
    if bool(nonzero_residual.any() and nonzero_correction.any()):
        cosine = F.cosine_similarity(residual_flat, correction_flat, dim=1, eps=eps)
        residual_correction_cosine = float(cosine.mean().item())
    else:
        residual_correction_cosine = float("nan")
    norm_ratio = correction_norm / residual_norm.clamp_min(eps)

    return {
        "teacher_base_mse": base_mse,
        "teacher_corrected_mse": corrected_mse,
        "teacher_mse_gain": _safe_gain(base_mse, corrected_mse),
        "teacher_base_mae": base_mae,
        "teacher_corrected_mae": corrected_mae,
        "teacher_mae_gain": _safe_gain(base_mae, corrected_mae),
        "teacher_residual_norm_mean": float(residual_norm.mean().item()),
        "teacher_correction_norm_mean": float(correction_norm.mean().item()),
        "teacher_correction_to_residual_norm_ratio_mean": float(norm_ratio.mean().item()),
        "teacher_correction_nonzero_rate": float(nonzero_correction.float().mean().item()),
        "teacher_sign_agreement_rate": sign_agreement_rate,
        "teacher_helpful_element_rate": helpful_element_rate,
        "teacher_harmful_element_rate": harmful_element_rate,
        "teacher_overcorrection_rate": overcorrection_rate,
        "teacher_residual_correction_cosine": residual_correction_cosine,
        "teacher_candidate_shift_l1": float(F.l1_loss(y_corrected, y_base).item()),
    }


def summarize_teacher_mode_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("teacher diagnostic summary requires at least one mode row")
    by_mode = {str(row["mode"]): row for row in rows}
    if "real" not in by_mode:
        raise ValueError("teacher diagnostic summary requires a real mode row")

    real = by_mode["real"]
    counterfactual_rows = {
        mode: row for mode, row in by_mode.items() if mode in TEACHER_COUNTERFACTUAL_MODES
    }
    best_mode = ""
    best_gain = float("-inf")
    for mode, row in counterfactual_rows.items():
        gain = float(row.get("teacher_mse_gain", float("nan")))
        if gain == gain and gain > best_gain:
            best_gain = gain
            best_mode = mode
    if best_gain == float("-inf"):
        best_gain = float("nan")

    real_gain = float(real.get("teacher_mse_gain", float("nan")))
    real_mae_gain = float(real.get("teacher_mae_gain", float("nan")))
    sign_agreement = float(real.get("teacher_sign_agreement_rate", float("nan")))
    margin = real_gain - best_gain if real_gain == real_gain and best_gain == best_gain else float("nan")
    return {
        "teacher_real_mse_gain": real_gain,
        "teacher_real_mae_gain": real_mae_gain,
        "teacher_real_sign_agreement_rate": sign_agreement,
        "teacher_real_residual_correction_cosine": real.get("teacher_residual_correction_cosine", float("nan")),
        "teacher_best_counterfactual_mode": best_mode,
        "teacher_best_counterfactual_mse_gain": best_gain,
        "teacher_real_vs_best_counterfactual_mse_gain_margin": margin,
        "teacher_real_is_best_mode": best_mode == "" or real_gain > best_gain,
    }
