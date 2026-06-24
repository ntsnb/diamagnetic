from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn

from models.adaptation_v2 import (
    IdentitySignedSeparableAdaptationGenerator,
    build_stage3_v2_param_delta,
    summarize_delta,
)
from train.stage3_eval_policy import assert_deterministic_eval_model
from train.stage3_functional import functional_call_with_param_delta
from train.stage3_risk_gate_v2 import build_stage3_risk_features
from train.stage3_signals import Stage3SignalState, build_stage3_drift_vector


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _apply_generator_residual_correction(generator, drift: torch.Tensor, y_candidate: torch.Tensor) -> torch.Tensor:
    if not hasattr(generator, "residual_correction"):
        return y_candidate
    return y_candidate + generator.residual_correction(drift, y_candidate)


def _unpack_batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if len(batch) == 4:
        x, y, phi_i, phi_v = batch
        return x.to(device), y.to(device), phi_i.to(device), phi_v.to(device)
    if len(batch) == 2:
        x, y = batch
        return x.to(device), y.to(device), None, None
    raise ValueError("Stage III V16 stream item must be (x,y) or (x,y,phi_i,phi_v)")


def _extract_v16_representations(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(output, "pooled_phi_i") and hasattr(output, "pooled_phi_v"):
        return output.pooled_phi_i, output.pooled_phi_v
    if hasattr(output, "phi_i") and hasattr(output, "phi_v"):
        return output.phi_i, output.phi_v
    raise ValueError("V16 online adaptation requires model outputs to expose pooled_phi_i/pooled_phi_v or phi_i/phi_v")


def run_online_adaptation_v16(
    *,
    stream: Iterable,
    model,
    generator: nn.Module | None,
    risk_gate=None,
    cfg: dict,
    device: torch.device,
    return_step_rows: bool = False,
    trace_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    online_cfg = cfg["online"] if "online" in cfg else cfg
    stage3_cfg = online_cfg["stage3"]
    assert_deterministic_eval_model(
        model,
        context="Stage III V16 online adaptation",
        allow_mc_eval=bool(stage3_cfg.get("mc_eval", False)),
    )
    causal_online = bool(stage3_cfg.get("causal_online", True))
    allow_untrained = bool(stage3_cfg.get("allow_untrained_generator", False))
    if generator is None and not allow_untrained:
        raise ValueError("Stage III V16 requires a trained generator or allow_untrained_generator=True")
    if generator is None:
        drift_dim = int(stage3_cfg.get("diagnostic_drift_dim", 16))
        generator = IdentitySignedSeparableAdaptationGenerator(
            drift_dim=drift_dim,
            hidden_dim=int(stage3_cfg.get("generator_hidden_dim", max(16, 2 * drift_dim))),
            max_dim=int(stage3_cfg.get("generator_max_dim", 512)),
            max_scale=float(stage3_cfg.get("generator_max_scale", 0.05)),
        ).to(device)

    model.eval()
    generator.eval()
    if risk_gate is not None:
        risk_gate.eval()
    target_names = tuple(stage3_cfg.get("target_names") or ("weight",))
    signal_state = Stage3SignalState()
    base_mses: list[float] = []
    base_maes: list[float] = []
    stage3_mses: list[float] = []
    stage3_maes: list[float] = []
    use_rates: list[float] = []
    recon_errors: list[float] = []
    pred_shifts: list[float] = []
    blends: list[float] = []
    risk_scores: list[float] = []
    delta_norms_pre: list[float] = []
    delta_norms_post: list[float] = []
    clamp_fractions: list[float] = []
    reject_shift_count = 0.0
    reject_risk_count = 0.0
    batch_count = 0.0
    sample_count = 0.0
    step_rows: list[dict[str, Any]] = []

    for batch in stream:
        x, y, phi_i_external, phi_v_external = _unpack_batch(batch, device)
        batch_count += 1.0
        sample_count += float(x.shape[0])
        with torch.no_grad():
            base_output = model(x)
            y_base = base_output.y_hat if hasattr(base_output, "y_hat") else base_output
            if phi_i_external is None or phi_v_external is None:
                phi_i, phi_v = _extract_v16_representations(base_output)
            else:
                phi_i, phi_v = phi_i_external, phi_v_external
            recon_error = (
                float(torch.mean((base_output.x_recon - x) ** 2).detach().cpu().item())
                if hasattr(base_output, "x_recon")
                else 0.0
            )
            base_mse = F.mse_loss(y_base, y).item()
            base_mae = F.l1_loss(y_base, y).item()

        y_final = y_base
        used = 0.0
        pred_shift = 0.0
        blend = 0.0
        risk_score = 1.0
        reject_reason = "accepted"
        delta_summary = {
            "delta_param_count": 0,
            "delta_norm_pre_clamp": 0.0,
            "delta_norm_post_clamp": 0.0,
            "clamp_fraction": 0.0,
        }

        if causal_online and x.shape[0] != 1:
            raise ValueError("Stage III V16 causal online mode requires batch_size=1")
        drift_raw = phi_v - phi_i
        drift_vec = build_stage3_drift_vector(
            drift_raw=drift_raw,
            mode=stage3_cfg.get("signal_mode", "delta_vector"),
            bottleneck_dim=int(stage3_cfg.get("drift_bottleneck", generator.drift_dim)),
            state=signal_state,
            normalize_vector=bool(stage3_cfg.get("normalize_vector", True)),
            polarity=float(stage3_cfg.get("drift_polarity", 1.0)),
        ).drift_vec.to(device=device, dtype=x.dtype)
        delta = build_stage3_v2_param_delta(
            dict(model.named_parameters()),
            drift_vec,
            generator,
            target_prefixes=tuple(stage3_cfg.get("target_modules") or ()) or None,
            target_names=target_names,
        )
        delta_summary = summarize_delta(delta, max_delta=float(stage3_cfg["max_delta"]))
        clamped_delta = {
            name: torch.clamp(value, -float(stage3_cfg["max_delta"]), float(stage3_cfg["max_delta"]))
            for name, value in delta.items()
        }
        adapted_output = functional_call_with_param_delta(model, clamped_delta, x)
        y_candidate = adapted_output.y_hat if hasattr(adapted_output, "y_hat") else adapted_output
        y_candidate = _apply_generator_residual_correction(generator, drift_vec, y_candidate)
        pred_shift = F.l1_loss(y_candidate, y_base).item()
        if pred_shift > float(stage3_cfg["max_pred_shift"]):
            reject_shift_count += 1.0
            reject_reason = "shift"
        else:
            if bool(stage3_cfg.get("risk_gate_enable", False)):
                if risk_gate is None:
                    raise ValueError("risk_gate_enable=True requires a trained risk gate")
                risk_features = build_stage3_risk_features(
                    drift_vec=drift_vec,
                    delta_summary=delta_summary,
                    pred_shift=pred_shift,
                    recon_error=recon_error,
                    y_base=y_base,
                    y_candidate=y_candidate,
                ).to(device=device, dtype=x.dtype)
                with torch.no_grad():
                    risk_score = float(risk_gate(risk_features).detach().reshape(-1)[0].cpu().item())
                if risk_score < float(stage3_cfg.get("risk_gate_threshold", 0.5)):
                    reject_risk_count += 1.0
                    reject_reason = "risk_gate"
                else:
                    blend = float(stage3_cfg.get("blend_alpha", 1.0))
                    y_final = (1.0 - blend) * y_base + blend * y_candidate
                    used = 1.0
            else:
                blend = float(stage3_cfg.get("blend_alpha", 1.0))
                y_final = (1.0 - blend) * y_base + blend * y_candidate
                used = 1.0

        stage3_mse = F.mse_loss(y_final, y).item()
        stage3_mae = F.l1_loss(y_final, y).item()
        if return_step_rows:
            row = dict(trace_context or {})
            row.update(
                {
                    "batch_count": batch_count,
                    "batch_sample_count": float(x.shape[0]),
                    "sample_count": sample_count,
                    "base_mse": float(base_mse),
                    "base_mae": float(base_mae),
                    "final_mse": float(stage3_mse),
                    "final_mae": float(stage3_mae),
                    "used": float(used),
                    "reject_reason": reject_reason,
                    "pred_shift": float(pred_shift),
                    "blend": float(blend),
                    "risk_score": float(risk_score),
                    "recon_error": float(recon_error),
                    "delta_param_count": int(delta_summary["delta_param_count"]),
                    "delta_norm_pre_clamp": float(delta_summary["delta_norm_pre_clamp"]),
                    "delta_norm_post_clamp": float(delta_summary["delta_norm_post_clamp"]),
                    "clamp_fraction": float(delta_summary["clamp_fraction"]),
                }
            )
            step_rows.append(row)

        base_mses.append(float(base_mse))
        base_maes.append(float(base_mae))
        stage3_mses.append(float(stage3_mse))
        stage3_maes.append(float(stage3_mae))
        use_rates.append(used)
        recon_errors.append(float(recon_error))
        pred_shifts.append(float(pred_shift))
        blends.append(float(blend))
        risk_scores.append(float(risk_score))
        delta_norms_pre.append(float(delta_summary["delta_norm_pre_clamp"]))
        delta_norms_post.append(float(delta_summary["delta_norm_post_clamp"]))
        clamp_fractions.append(float(delta_summary["clamp_fraction"]))

    metrics = {
        "online_base_mse": _mean(base_mses),
        "online_base_mae": _mean(base_maes),
        "online_stage3_mse": _mean(stage3_mses),
        "online_stage3_mae": _mean(stage3_maes),
        "online_stage3_use_rate": _mean(use_rates),
        "online_stage3_recon_error": _mean(recon_errors),
        "online_stage3_pred_shift": _mean(pred_shifts),
        "online_stage3_blend": _mean(blends),
        "online_stage3_risk_score": _mean(risk_scores),
        "online_stage3_delta_norm_pre_clamp": _mean(delta_norms_pre),
        "online_stage3_delta_norm_post_clamp": _mean(delta_norms_post),
        "online_stage3_clamp_fraction": _mean(clamp_fractions),
        "online_stage3_reject_shift_count": reject_shift_count,
        "online_stage3_reject_risk_count": reject_risk_count,
        "online_batch_count": batch_count,
        "online_sample_count": sample_count,
    }
    if return_step_rows:
        metrics["step_rows"] = step_rows
    return metrics
