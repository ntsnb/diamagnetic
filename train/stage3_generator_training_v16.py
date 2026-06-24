from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.adaptation_v2 import (
    IdentitySignedSeparableAdaptationGenerator,
    LayerAwareHyperAdapterGenerator,
    LayerAwareHyperLoRAGenerator,
    build_stage3_v2_param_delta,
    summarize_delta,
)
from models.adaptation_v9 import ResidualBasisHyperGeneratorV9
from train.stage3_functional import functional_call_with_param_delta
from train.stage3_generator_training import (
    Stage3GeneratorLossConfig,
    save_stage3_generator_checkpoint,
    stage3_generator_train_step,
)
from train.stage3_risk_gate_v2 import build_stage3_risk_features
from train.stage3_signals import Stage3SignalState, build_stage3_drift_vector


@dataclass(frozen=True)
class Stage3GeneratorCandidateV16:
    epoch: int
    checkpoint_path: Path
    last_train_loss: float | None


def _extract_prediction(output: Any) -> torch.Tensor:
    return output.y_hat if hasattr(output, "y_hat") else output


def _apply_generator_residual_correction(
    *,
    generator: nn.Module,
    drift: torch.Tensor,
    y_candidate: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(generator, "residual_correction"):
        return y_candidate
    return y_candidate + generator.residual_correction(drift, y_candidate)


def _format_amplified_negative_name(factor: float) -> str:
    if float(factor).is_integer():
        return f"amplified_{int(factor)}x"
    safe = str(factor).replace(".", "p").replace("-", "m")
    return f"amplified_{safe}x"


def _build_stage3_generator_negative_drifts(
    *,
    drift_vec: torch.Tensor,
    previous_drift_vec: torch.Tensor | None,
    gen_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    negative_drifts = {
        "zero": torch.zeros_like(drift_vec),
        "sign_flipped": -drift_vec,
    }
    if previous_drift_vec is not None:
        negative_drifts["previous_shuffled"] = previous_drift_vec.to(
            device=drift_vec.device,
            dtype=drift_vec.dtype,
        )
    if bool(gen_cfg.get("include_random_matched_norm_negative", False)):
        drift_norm = torch.linalg.vector_norm(drift_vec)
        if float(drift_norm.detach().cpu().item()) > 0.0:
            random_direction = torch.randn_like(drift_vec)
            random_direction = random_direction / torch.linalg.vector_norm(random_direction).clamp_min(1e-8)
            negative_drifts["random_matched_norm"] = random_direction * drift_norm
    for factor in gen_cfg.get("amplified_negative_factors", []):
        factor = float(factor)
        negative_drifts[_format_amplified_negative_name(factor)] = drift_vec * factor
    return negative_drifts


def _build_risk_gate_drift_candidates(
    *,
    drift_vec: torch.Tensor,
    previous_drift_vec: torch.Tensor | None,
    gen_cfg: dict[str, Any],
    risk_gate_cfg: dict[str, Any],
) -> dict[str, torch.Tensor]:
    candidates = {"real": drift_vec}
    if not bool(risk_gate_cfg.get("include_counterfactual_examples", False)):
        return candidates
    negative_drifts = _build_stage3_generator_negative_drifts(
        drift_vec=drift_vec,
        previous_drift_vec=previous_drift_vec,
        gen_cfg=gen_cfg,
    )
    requested_modes = risk_gate_cfg.get("counterfactual_modes") or [
        "zero",
        "previous_step",
        "sign_flipped",
        "amplified_2x",
        "amplified_5x",
        "random_matched_norm",
    ]
    aliases = {
        "previous_step": "previous_shuffled",
        "previous_drift": "previous_shuffled",
        "shuffled": "previous_shuffled",
    }
    for mode in requested_modes:
        mode = str(mode)
        source_key = aliases.get(mode, mode)
        if source_key in negative_drifts:
            candidates[mode] = negative_drifts[source_key]
    return candidates


def _risk_gate_label_for_mode(
    mode: str,
    *,
    real_accept: float,
    counterfactual_label: float = 0.0,
) -> float:
    if mode == "real":
        return float(real_accept)
    return float(counterfactual_label)


def iter_v16_representation_stream(
    dataset,
    batch_size: int,
    device: torch.device,
    model: nn.Module,
    *,
    max_batches: int | None = None,
):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model.eval()
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = model(x)
        yield x, y, out.pooled_phi_i.detach(), out.pooled_phi_v.detach()


def build_stage3_v16_generator(
    *,
    cfg: dict[str, Any],
    model: nn.Module,
    drift_dim: int,
    output_shape: tuple[int, ...],
) -> nn.Module:
    gen_cfg = cfg.get("stage3_v2", {}).get("generator", {})
    target_prefixes = tuple(cfg["online"]["stage3"].get("target_modules") or ())
    target_names = tuple(gen_cfg.get("target_names") or cfg["online"]["stage3"].get("target_names") or ("weight",))
    generator_type = str(gen_cfg.get("type", "layer_aware_hyperadapter_v16")).lower()
    generator_supports_bias = generator_type in {
        "layer_aware_hyperadapter",
        "layer_aware_hyperadapter_v8",
        "hyperadapter_v8",
        "layer_aware_hyperadapter_v16",
        "hyperadapter_v16",
        "residual_basis_hypergenerator_v9",
        "residual_basis_hypergenerator_v16",
        "residual_basis_hypergenerator",
    }
    named_shapes = {
        name: tuple(param.shape)
        for name, param in model.named_parameters()
        if (param.ndim in {1, 2} if generator_supports_bias else param.ndim == 2)
        and (not target_names or any(name.endswith(target_name) for target_name in target_names))
        and ((not target_prefixes) or any(name.startswith(prefix) for prefix in target_prefixes))
    }
    if not named_shapes:
        raise RuntimeError("Stage III V16 generator found no target parameters")

    hidden_dim = int(gen_cfg.get("hidden_dim", max(32, 2 * drift_dim)))
    rank = int(gen_cfg.get("rank", 4))
    max_scale = float(gen_cfg.get("max_scale", 0.05))
    if generator_type in {
        "residual_basis_hypergenerator_v9",
        "residual_basis_hypergenerator_v16",
        "residual_basis_hypergenerator",
    }:
        return ResidualBasisHyperGeneratorV9(
            named_param_shapes=named_shapes,
            drift_dim=drift_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            output_shape=output_shape,
            max_scale=max_scale,
            residual_scale=float(gen_cfg.get("residual_scale", 1.0)),
        )
    if generator_type in {"layer_aware_hyperlora", "layer_aware_hyperlora_v16", "hyperlora_v16"}:
        return LayerAwareHyperLoRAGenerator(
            named_param_shapes=named_shapes,
            drift_dim=drift_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            max_scale=max_scale,
        )
    if generator_type in {
        "layer_aware_hyperadapter",
        "layer_aware_hyperadapter_v8",
        "hyperadapter_v8",
        "layer_aware_hyperadapter_v16",
        "hyperadapter_v16",
    }:
        return LayerAwareHyperAdapterGenerator(
            named_param_shapes=named_shapes,
            drift_dim=drift_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            max_scale=max_scale,
        )
    if generator_type in {"identity_signed_separable", "identity_signed_separable_v16"}:
        max_dim = max(dim for shape in named_shapes.values() for dim in shape)
        return IdentitySignedSeparableAdaptationGenerator(
            drift_dim=drift_dim,
            hidden_dim=hidden_dim,
            max_dim=max_dim,
            max_scale=max_scale,
        )
    raise ValueError(f"unsupported Stage III V16 generator type: {generator_type}")


def train_stage3_generator_v16(
    *,
    cfg: dict[str, Any],
    model: nn.Module,
    train_ds,
    device: torch.device,
    max_batches: int | None = None,
    candidate_checkpoint_dir: Path | None = None,
    candidate_metadata: dict[str, Any] | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], list[Stage3GeneratorCandidateV16]]:
    gen_cfg = cfg.get("stage3_v2", {}).get("generator", {})
    batch_size = int(cfg.get("stage3_v2", {}).get("online_batch_size", 1))
    if batch_size != 1:
        raise ValueError("Stage III V16 generator training requires stage3_v2.online_batch_size=1")
    sample_y = train_ds[0][1]
    target_prefixes = tuple(cfg["online"]["stage3"].get("target_modules") or ()) or None
    target_names = tuple(gen_cfg.get("target_names") or cfg["online"]["stage3"].get("target_names") or ("weight",))
    generator = build_stage3_v16_generator(
        cfg=cfg,
        model=model,
        drift_dim=int(cfg["invariant"]["repr_dim"]),
        output_shape=tuple(sample_y.shape),
    ).to(device)
    opt = torch.optim.Adam(generator.parameters(), lr=float(gen_cfg.get("lr", 1e-3)))
    loss_cfg = Stage3GeneratorLossConfig(
        lambda_delta=float(gen_cfg.get("lambda_delta", 1e-4)),
        lambda_zero=float(gen_cfg.get("lambda_zero", 1e-2)),
        lambda_shift=float(gen_cfg.get("lambda_shift", 1e-2)),
        lambda_improvement=float(gen_cfg.get("lambda_improvement", 0.0)),
        improvement_margin=float(gen_cfg.get("improvement_margin", 0.0)),
        lambda_intervention_compare=float(gen_cfg.get("lambda_intervention_compare", 0.0)),
        intervention_margin=float(gen_cfg.get("intervention_margin", 0.0)),
        lambda_shift_budget=float(gen_cfg.get("lambda_shift_budget", 0.0)),
        shift_budget=float(gen_cfg.get("shift_budget", cfg["online"]["stage3"].get("max_pred_shift", 0.0))),
        max_delta=cfg["online"]["stage3"].get("max_delta"),
        blend_alpha=float(cfg["online"]["stage3"].get("blend_alpha", 1.0)),
        max_pred_shift=cfg["online"]["stage3"].get("max_pred_shift"),
        soft_reject_temperature=float(gen_cfg.get("soft_reject_temperature", 0.0)),
        detach_negative_mse=bool(gen_cfg.get("detach_negative_mse", True)),
        lambda_teacher_correction=float(gen_cfg.get("lambda_teacher_correction", 0.0)),
        lambda_residual_prediction=float(gen_cfg.get("lambda_residual_prediction", 0.0)),
        lambda_teacher_cosine=float(gen_cfg.get("lambda_teacher_cosine", 0.0)),
    )
    drift_polarity = float(cfg["online"]["stage3"].get("drift_polarity", 1.0))
    rows: list[dict[str, Any]] = []
    candidates: list[Stage3GeneratorCandidateV16] = []
    for epoch in range(int(gen_cfg.get("epochs", 1))):
        state = Stage3SignalState()
        previous_drift_vec: torch.Tensor | None = None
        for step, (x, y, phi_i, phi_v) in enumerate(
            iter_v16_representation_stream(
                train_ds,
                batch_size=batch_size,
                device=device,
                model=model,
                max_batches=max_batches,
            )
        ):
            drift_raw = phi_v - phi_i
            signal_result = build_stage3_drift_vector(
                drift_raw=drift_raw,
                mode=cfg["online"]["stage3"].get("signal_mode", "delta_vector"),
                bottleneck_dim=int(cfg["online"]["stage3"].get("drift_bottleneck", generator.drift_dim)),
                state=state,
                normalize_vector=bool(cfg["online"]["stage3"].get("normalize_vector", True)),
                polarity=drift_polarity,
            )
            drift_vec = signal_result.drift_vec.to(device=device, dtype=x.dtype)
            negative_drifts = _build_stage3_generator_negative_drifts(
                drift_vec=drift_vec,
                previous_drift_vec=previous_drift_vec,
                gen_cfg=gen_cfg,
            )
            metrics = stage3_generator_train_step(
                model=model,
                generator=generator,
                optimizer=opt,
                x=x,
                y=y,
                drift=drift_vec,
                loss_cfg=loss_cfg,
                target_prefixes=target_prefixes,
                target_names=target_names,
                negative_drifts=negative_drifts if loss_cfg.lambda_intervention_compare > 0.0 else None,
            )
            previous_drift_vec = drift_vec.detach().clone()
            rows.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "drift_norm": float(torch.linalg.vector_norm(drift_vec.detach()).cpu().item()),
                    **metrics,
                }
            )
        if candidate_checkpoint_dir is not None:
            candidate_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = candidate_checkpoint_dir / f"generator_v16_epoch_{epoch:03d}.pt"
            last_train_loss = float(rows[-1]["loss"]) if rows else None
            save_stage3_generator_checkpoint(
                path=candidate_path,
                generator=generator,
                metadata={
                    **(candidate_metadata or {}),
                    "protocol_version": "v16_expert_local_dual_stream_probe",
                    "candidate_epoch": epoch,
                    "candidate_last_train_loss": last_train_loss,
                },
            )
            candidates.append(
                Stage3GeneratorCandidateV16(
                    epoch=epoch,
                    checkpoint_path=candidate_path,
                    last_train_loss=last_train_loss,
                )
            )
    generator.eval()
    return generator, rows, candidates


def collect_stage3_v16_risk_gate_examples(
    *,
    cfg: dict[str, Any],
    model: nn.Module,
    generator: nn.Module,
    train_ds,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    model.eval()
    generator.eval()
    batch_size = int(cfg.get("stage3_v2", {}).get("online_batch_size", 1))
    if batch_size != 1:
        raise ValueError("Stage III V16 risk gate training requires stage3_v2.online_batch_size=1")
    target_prefixes = tuple(cfg["online"]["stage3"].get("target_modules") or ()) or None
    target_names = tuple(
        cfg.get("stage3_v2", {}).get("generator", {}).get("target_names")
        or cfg["online"]["stage3"].get("target_names")
        or ("weight",)
    )
    drift_polarity = float(cfg["online"]["stage3"].get("drift_polarity", 1.0))
    gen_cfg = cfg.get("stage3_v2", {}).get("generator", {})
    gate_cfg = cfg.get("stage3_v2", {}).get("risk_gate", {})
    counterfactual_label = float(gate_cfg.get("counterfactual_label", 0.0))
    state = Stage3SignalState()
    previous_drift_vec: torch.Tensor | None = None
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []

    for step, (x, y, phi_i, phi_v) in enumerate(
        iter_v16_representation_stream(
            train_ds,
            batch_size=batch_size,
            device=device,
            model=model,
            max_batches=max_batches,
        )
    ):
        with torch.no_grad():
            base_output = model(x)
            y_base = _extract_prediction(base_output)
            base_mse = F.mse_loss(y_base, y)
            recon_error = torch.mean((base_output.x_recon - x) ** 2).item() if hasattr(base_output, "x_recon") else 0.0
            drift_raw = phi_v - phi_i
            signal_result = build_stage3_drift_vector(
                drift_raw=drift_raw,
                mode=cfg["online"]["stage3"].get("signal_mode", "delta_vector"),
                bottleneck_dim=int(cfg["online"]["stage3"].get("drift_bottleneck", generator.drift_dim)),
                state=state,
                normalize_vector=bool(cfg["online"]["stage3"].get("normalize_vector", True)),
                polarity=drift_polarity,
            )
            drift_vec = signal_result.drift_vec.to(device=device, dtype=x.dtype)
            drift_candidates = _build_risk_gate_drift_candidates(
                drift_vec=drift_vec,
                previous_drift_vec=previous_drift_vec,
                gen_cfg=gen_cfg,
                risk_gate_cfg=gate_cfg,
            )
            step_features: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, float, float, dict[str, float]]] = []
            real_accept_value = 0.0
            for mode, candidate_drift in drift_candidates.items():
                candidate_drift = candidate_drift.to(device=device, dtype=x.dtype)
                delta = build_stage3_v2_param_delta(
                    dict(model.named_parameters()),
                    candidate_drift,
                    generator,
                    target_prefixes=target_prefixes,
                    target_names=target_names,
                )
                delta_summary = summarize_delta(delta, max_delta=float(cfg["online"]["stage3"]["max_delta"]))
                clamped_delta = {
                    name: torch.clamp(
                        value,
                        -float(cfg["online"]["stage3"]["max_delta"]),
                        float(cfg["online"]["stage3"]["max_delta"]),
                    )
                    for name, value in delta.items()
                }
                y_candidate = _extract_prediction(functional_call_with_param_delta(model, clamped_delta, x))
                y_candidate = _apply_generator_residual_correction(
                    generator=generator,
                    drift=candidate_drift,
                    y_candidate=y_candidate,
                )
                pred_shift = F.l1_loss(y_candidate, y_base).item()
                blend = float(cfg["online"]["stage3"].get("blend_alpha", 1.0))
                y_blended = (1.0 - blend) * y_base + blend * y_candidate
                blended_mse = F.mse_loss(y_blended, y)
                mode_accept = float((blended_mse < base_mse).float().item())
                if mode == "real":
                    real_accept_value = mode_accept
                feature = build_stage3_risk_features(
                    drift_vec=candidate_drift,
                    delta_summary=delta_summary,
                    pred_shift=pred_shift,
                    recon_error=recon_error,
                    y_base=y_base,
                    y_candidate=y_candidate,
                ).detach().cpu()
                step_features.append(
                    (
                        mode,
                        candidate_drift.detach().cpu(),
                        feature,
                        blended_mse.detach().cpu(),
                        mode_accept,
                        pred_shift,
                        delta_summary,
                    )
                )
        for mode, candidate_drift_cpu, feature, blended_mse_cpu, mode_accept, pred_shift, delta_summary in step_features:
            label_value = _risk_gate_label_for_mode(
                mode,
                real_accept=real_accept_value,
                counterfactual_label=counterfactual_label,
            )
            label = torch.tensor([label_value], dtype=torch.float32)
            features.append(feature)
            labels.append(label)
            rows.append(
                {
                    "step": step,
                    "mode": mode,
                    "base_mse": float(base_mse.detach().cpu().item()),
                    "blended_mse": float(blended_mse_cpu.item()),
                    "would_accept_by_loss": float(mode_accept),
                    "accept_label": float(label_value),
                    "pred_shift": float(pred_shift),
                    "drift_norm": float(torch.linalg.vector_norm(candidate_drift_cpu).item()),
                    **{key: float(value) for key, value in delta_summary.items()},
                    **{f"feature_{idx}": float(feature.reshape(-1)[idx].item()) for idx in range(feature.shape[1])},
                }
            )
        previous_drift_vec = drift_vec.detach().clone()
    if not features:
        raise ValueError("Stage III V16 risk gate training collected no examples")
    return torch.cat(features, dim=0), torch.cat(labels, dim=0).reshape(-1), rows
