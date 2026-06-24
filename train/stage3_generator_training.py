from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adaptation_v2 import (
    IdentitySignedSeparableAdaptationGenerator,
    LayerAwareHyperAdapterGenerator,
    LayerAwareHyperLoRAGenerator,
    build_stage3_v2_param_delta,
    summarize_delta,
)
from models.adaptation_v9 import ResidualBasisHyperGeneratorV9
from train.stage3_eval_policy import assert_deterministic_eval_model
from train.stage3_functional import functional_call_with_param_delta


@dataclass
class Stage3GeneratorLossConfig:
    lambda_delta: float = 1e-4
    lambda_zero: float = 1e-2
    lambda_shift: float = 1e-2
    lambda_improvement: float = 0.0
    improvement_margin: float = 0.0
    lambda_intervention_compare: float = 0.0
    intervention_margin: float = 0.0
    lambda_shift_budget: float = 0.0
    shift_budget: float = 0.0
    max_delta: float | None = None
    blend_alpha: float = 1.0
    max_pred_shift: float | None = None
    soft_reject_temperature: float = 0.0
    detach_negative_mse: bool = True
    lambda_teacher_correction: float = 0.0
    lambda_residual_prediction: float = 0.0
    lambda_teacher_cosine: float = 0.0


def freeze_module_parameters(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
        param.grad = None


def _extract_prediction(output: Any) -> torch.Tensor:
    return output.y_hat if hasattr(output, "y_hat") else output


def _delta_norm(delta: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not delta:
        return torch.zeros(())
    return torch.stack([value.pow(2).mean() for value in delta.values()]).mean()


def _clone_detached_params(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in module.named_parameters()}


def _assert_frozen_params_unchanged(module: nn.Module, before: Mapping[str, torch.Tensor]) -> None:
    for name, param in module.named_parameters():
        if not torch.allclose(param.detach(), before[name]):
            raise RuntimeError(f"frozen base parameter changed during Stage III generator training: {name}")


def _apply_generator_residual_correction(
    *,
    generator: nn.Module,
    drift: torch.Tensor,
    y_candidate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not hasattr(generator, "residual_correction"):
        return y_candidate, None
    correction = generator.residual_correction(drift, y_candidate)
    return y_candidate + correction, correction


def stage3_generator_train_step(
    *,
    model: nn.Module,
    generator: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    drift: torch.Tensor,
    loss_cfg: Stage3GeneratorLossConfig,
    target_prefixes: tuple[str, ...] | None = None,
    negative_drifts: Mapping[str, torch.Tensor] | None = None,
    forward_kwargs: dict[str, Any] | None = None,
    target_names: tuple[str, ...] = ("weight",),
    teacher_correction: torch.Tensor | None = None,
) -> dict[str, float]:
    """One differentiable training step for the Stage III V2 generator."""

    assert_deterministic_eval_model(model, context="Stage III V2 generator training")
    freeze_module_parameters(model)
    forward_kwargs = forward_kwargs or {}
    before = _clone_detached_params(model)

    with torch.no_grad():
        base_output = model(x, **forward_kwargs)
        y_base = _extract_prediction(base_output)
        base_mse = F.mse_loss(y_base, y)

    named_params = dict(model.named_parameters())
    delta_pre_clamp = build_stage3_v2_param_delta(
        named_params,
        drift,
        generator,
        target_prefixes=target_prefixes,
        target_names=target_names,
    )
    delta_summary = summarize_delta(delta_pre_clamp, max_delta=loss_cfg.max_delta)
    delta = delta_pre_clamp
    if loss_cfg.max_delta is not None:
        delta = {name: torch.clamp(value, -loss_cfg.max_delta, loss_cfg.max_delta) for name, value in delta.items()}
    adapted_output = functional_call_with_param_delta(model, delta, x, **forward_kwargs)
    y_adapted = _extract_prediction(adapted_output)
    y_adapted, residual_correction = _apply_generator_residual_correction(
        generator=generator,
        drift=drift.to(device=x.device, dtype=x.dtype),
        y_candidate=y_adapted,
    )

    adapted_mse = F.mse_loss(y_adapted, y)
    adapted_mae = F.l1_loss(y_adapted, y)
    delta_reg = _delta_norm(delta).to(device=x.device, dtype=x.dtype)
    zero_delta = build_stage3_v2_param_delta(
        named_params,
        torch.zeros_like(drift),
        generator,
        target_prefixes=target_prefixes,
        target_names=target_names,
    )
    zero_reg = _delta_norm(zero_delta).to(device=x.device, dtype=x.dtype)
    zero_residual_reg = torch.zeros((), device=x.device, dtype=x.dtype)
    if hasattr(generator, "residual_correction"):
        zero_reference = torch.zeros_like(y_base.detach())
        zero_residual = generator.residual_correction(torch.zeros_like(drift), zero_reference)
        zero_residual_reg = zero_residual.pow(2).mean()
    shift_l1 = F.l1_loss(y_adapted, y_base.detach())
    shift_mse = F.mse_loss(y_adapted, y_base.detach())
    accept_weight = torch.ones((), device=x.device, dtype=x.dtype)
    if loss_cfg.max_pred_shift is not None:
        max_pred_shift = torch.as_tensor(loss_cfg.max_pred_shift, device=x.device, dtype=x.dtype)
        if loss_cfg.soft_reject_temperature > 0.0:
            tau = torch.as_tensor(loss_cfg.soft_reject_temperature, device=x.device, dtype=x.dtype)
            accept_weight = torch.sigmoid((max_pred_shift - shift_l1) / tau)
        else:
            accept_weight = (shift_l1 <= max_pred_shift).to(dtype=x.dtype)
    blend = torch.as_tensor(loss_cfg.blend_alpha, device=x.device, dtype=x.dtype) * accept_weight
    y_blended = y_base.detach() + blend * (y_adapted - y_base.detach())
    blended_mse = F.mse_loss(y_blended, y)
    blended_mae = F.l1_loss(y_blended, y)
    improvement_margin = torch.as_tensor(loss_cfg.improvement_margin, device=x.device, dtype=x.dtype)
    improvement_loss = torch.relu(blended_mse - base_mse.detach() + improvement_margin)
    shift_budget = torch.as_tensor(loss_cfg.shift_budget, device=x.device, dtype=x.dtype)
    shift_budget_loss = torch.relu(shift_l1 - shift_budget).pow(2)
    intervention_compare_loss = torch.zeros((), device=x.device, dtype=x.dtype)
    teacher_correction_loss = torch.zeros((), device=x.device, dtype=x.dtype)
    teacher_cosine_loss = torch.zeros((), device=x.device, dtype=x.dtype)
    if teacher_correction is not None:
        teacher_correction = teacher_correction.to(device=x.device, dtype=x.dtype).reshape_as(y_base)
        effective_shift = y_adapted - y_base.detach()
        teacher_correction_loss = F.smooth_l1_loss(effective_shift, teacher_correction)
        effective_flat = effective_shift.flatten()
        teacher_flat = teacher_correction.flatten()
        if torch.linalg.vector_norm(effective_flat) > 0 and torch.linalg.vector_norm(teacher_flat) > 0:
            teacher_cosine_loss = 1.0 - F.cosine_similarity(
                effective_flat.unsqueeze(0),
                teacher_flat.unsqueeze(0),
            ).mean()
    residual_prediction_loss = torch.zeros((), device=x.device, dtype=x.dtype)
    if loss_cfg.lambda_residual_prediction > 0.0:
        residual_prediction_loss = F.smooth_l1_loss(y_adapted - y_base.detach(), y - y_base.detach())
    negative_metrics: dict[str, float] = {}
    for label, negative_drift in (negative_drifts or {}).items():
        negative_delta_pre_clamp = build_stage3_v2_param_delta(
            named_params,
            negative_drift.to(device=x.device, dtype=drift.dtype),
            generator,
            target_prefixes=target_prefixes,
            target_names=target_names,
        )
        negative_delta = negative_delta_pre_clamp
        if loss_cfg.max_delta is not None:
            negative_delta = {
                name: torch.clamp(value, -loss_cfg.max_delta, loss_cfg.max_delta)
                for name, value in negative_delta.items()
            }
        negative_output = functional_call_with_param_delta(model, negative_delta, x, **forward_kwargs)
        y_negative = _extract_prediction(negative_output)
        y_negative, _ = _apply_generator_residual_correction(
            generator=generator,
            drift=negative_drift.to(device=x.device, dtype=drift.dtype),
            y_candidate=y_negative,
        )
        negative_shift_l1 = F.l1_loss(y_negative, y_base.detach())
        negative_accept_weight = torch.ones((), device=x.device, dtype=x.dtype)
        if loss_cfg.max_pred_shift is not None:
            max_pred_shift = torch.as_tensor(loss_cfg.max_pred_shift, device=x.device, dtype=x.dtype)
            if loss_cfg.soft_reject_temperature > 0.0:
                tau = torch.as_tensor(loss_cfg.soft_reject_temperature, device=x.device, dtype=x.dtype)
                negative_accept_weight = torch.sigmoid((max_pred_shift - negative_shift_l1) / tau)
            else:
                negative_accept_weight = (negative_shift_l1 <= max_pred_shift).to(dtype=x.dtype)
        negative_blend = torch.as_tensor(loss_cfg.blend_alpha, device=x.device, dtype=x.dtype) * negative_accept_weight
        y_negative_blended = y_base.detach() + negative_blend * (y_negative - y_base.detach())
        negative_mse = F.mse_loss(y_negative_blended, y)
        negative_reference = negative_mse.detach() if loss_cfg.detach_negative_mse else negative_mse
        intervention_compare_loss = intervention_compare_loss + torch.relu(
            blended_mse - negative_reference + loss_cfg.intervention_margin
        )
        safe_label = label.replace(" ", "_").replace("-", "_")
        negative_metrics[f"negative_{safe_label}_mse"] = float(negative_mse.detach().cpu().item())

    loss = (
        blended_mse
        + loss_cfg.lambda_delta * delta_reg
        + loss_cfg.lambda_zero * (zero_reg + zero_residual_reg)
        + loss_cfg.lambda_shift * shift_l1
        + loss_cfg.lambda_improvement * improvement_loss
        + loss_cfg.lambda_intervention_compare * intervention_compare_loss
        + loss_cfg.lambda_shift_budget * shift_budget_loss
        + loss_cfg.lambda_teacher_correction * teacher_correction_loss
        + loss_cfg.lambda_residual_prediction * residual_prediction_loss
        + loss_cfg.lambda_teacher_cosine * teacher_cosine_loss
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    _assert_frozen_params_unchanged(model, before)

    return {
        "loss": float(loss.detach().cpu().item()),
        "base_mse": float(base_mse.detach().cpu().item()),
        "adapted_mse": float(adapted_mse.detach().cpu().item()),
        "candidate_mse": float(adapted_mse.detach().cpu().item()),
        "blended_mse": float(blended_mse.detach().cpu().item()),
        "base_mae": float(F.l1_loss(y_base, y).detach().cpu().item()),
        "candidate_mae": float(adapted_mae.detach().cpu().item()),
        "blended_mae": float(blended_mae.detach().cpu().item()),
        "delta_norm": float(delta_reg.detach().cpu().item()),
        "delta_norm_pre_clamp": float(delta_summary["delta_norm_pre_clamp"]),
        "delta_norm_post_clamp": float(delta_summary["delta_norm_post_clamp"]),
        "clamp_fraction": float(delta_summary["clamp_fraction"]),
        "zero_delta_norm": float(zero_reg.detach().cpu().item()),
        "zero_residual_correction_norm": float(zero_residual_reg.detach().cpu().item()),
        "candidate_shift": float(shift_l1.detach().cpu().item()),
        "candidate_shift_l1": float(shift_l1.detach().cpu().item()),
        "candidate_shift_mse": float(shift_mse.detach().cpu().item()),
        "improvement_loss": float(improvement_loss.detach().cpu().item()),
        "intervention_compare_loss": float(intervention_compare_loss.detach().cpu().item()),
        "shift_budget_loss": float(shift_budget_loss.detach().cpu().item()),
        "teacher_correction_loss": float(teacher_correction_loss.detach().cpu().item()),
        "residual_prediction_loss": float(residual_prediction_loss.detach().cpu().item()),
        "teacher_cosine_loss": float(teacher_cosine_loss.detach().cpu().item()),
        "residual_correction_norm": float(
            residual_correction.pow(2).mean().detach().cpu().item()
            if residual_correction is not None
            else 0.0
        ),
        **negative_metrics,
    }


def save_stage3_generator_checkpoint(
    *,
    path: str | Path,
    generator: nn.Module,
    metadata: dict[str, Any],
    overwrite: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V2 generator checkpoint: {path}")
    torch.save(
        {
            "generator_class": generator.__class__.__name__,
            "generator_config": generator.to_config() if hasattr(generator, "to_config") else {},
            "generator_state_dict": generator.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_stage3_generator_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    config = checkpoint.get("generator_config", {})
    class_name = checkpoint.get("generator_class") or config.get("class")
    if class_name == "LayerAwareHyperLoRAGenerator":
        generator = LayerAwareHyperLoRAGenerator(
            named_param_shapes={
                name: tuple(shape) for name, shape in config["named_param_shapes"].items()
            },
            drift_dim=int(config["drift_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            rank=int(config["rank"]),
            max_scale=float(config["max_scale"]),
        )
        if model is not None:
            model_shapes = {
                name: tuple(param.shape)
                for name, param in model.named_parameters()
                if name in generator.registered_param_names
            }
            expected = {
                name: tuple(shape) for name, shape in config["named_param_shapes"].items()
            }
            if model_shapes != expected:
                raise ValueError(
                    "loaded Stage III V2 generator target shapes do not match the supplied model"
                )
    elif class_name == "LayerAwareHyperAdapterGenerator":
        generator = LayerAwareHyperAdapterGenerator(
            named_param_shapes={
                name: tuple(shape) for name, shape in config["named_param_shapes"].items()
            },
            drift_dim=int(config["drift_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            rank=int(config["rank"]),
            max_scale=float(config["max_scale"]),
        )
        if model is not None:
            model_shapes = {
                name: tuple(param.shape)
                for name, param in model.named_parameters()
                if name in generator.registered_param_names
            }
            expected = {
                name: tuple(shape) for name, shape in config["named_param_shapes"].items()
            }
            if model_shapes != expected:
                raise ValueError(
                    "loaded Stage III V8 generator target shapes do not match the supplied model"
                )
    elif class_name == "ResidualBasisHyperGeneratorV9":
        generator = ResidualBasisHyperGeneratorV9(
            named_param_shapes={
                name: tuple(shape) for name, shape in config["named_param_shapes"].items()
            },
            drift_dim=int(config["drift_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            rank=int(config["rank"]),
            output_shape=tuple(config["output_shape"]),
            max_scale=float(config["max_scale"]),
            residual_scale=float(config.get("residual_scale", 1.0)),
        )
        if model is not None:
            model_shapes = {
                name: tuple(param.shape)
                for name, param in model.named_parameters()
                if name in generator.registered_param_names
            }
            expected = {
                name: tuple(shape) for name, shape in config["named_param_shapes"].items()
            }
            if model_shapes != expected:
                raise ValueError(
                    "loaded Stage III V9 generator target shapes do not match the supplied model"
                )
    elif class_name == "IdentitySignedSeparableAdaptationGenerator":
        generator = IdentitySignedSeparableAdaptationGenerator(
            drift_dim=int(config["drift_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            max_dim=int(config["max_dim"]),
            max_scale=float(config["max_scale"]),
        )
    else:
        raise ValueError(f"Unsupported Stage III V2 generator checkpoint class: {class_name}")
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    return generator, dict(checkpoint.get("metadata", {}))
