from __future__ import annotations

from typing import Iterable
from copy import deepcopy
from collections import deque

import torch
import torch.nn.functional as F

from models.adaptation import (
    AdaptationGenerator,
    EMAConfig,
    OnlineAdaptor,
    apply_param_delta,
    build_masks_from_attention_heads,
    build_masks_from_attention_scores,
    build_param_delta,
    build_param_masks_from_grads,
    merge_masks,
    topk_mask_from_tensor,
    threshold_mask_from_tensor,
)
from models.lrp import LRPEngine
from train.online_adapt import LabelBuffer, OnlineLoop, StabilityConfig
from utils.logging import log_csv, log_dict
from utils.tb_logger import get_tb_writer


def _filter_named_tensors(named_tensors: dict[str, torch.Tensor], prefixes: list[str]) -> dict[str, torch.Tensor]:
    if not prefixes:
        return named_tensors
    return {
        name: tensor
        for name, tensor in named_tensors.items()
        if any(name.startswith(prefix) for prefix in prefixes)
    }


def _snapshot_module_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _restore_module_state(module, snapshot: dict) -> None:
    module.load_state_dict(snapshot)


def _snapshot_named_parameters(model, names: set[str]) -> dict[str, torch.Tensor]:
    out = {}
    for name, param in model.named_parameters():
        if name in names:
            out[name] = param.detach().clone()
    return out


def _restore_named_parameters(model, snapshot: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in snapshot:
                param.copy_(snapshot[name])


def run_online_adaptation(
    stream: Iterable,
    model,
    m_eta_optimizer,
    cfg,
    device: torch.device,
):
    """Skeleton for online Stage I/II/III adaptation."""

    online_cfg = cfg["online"] if "online" in cfg else cfg
    adaptor = OnlineAdaptor(
        model, EMAConfig(online_cfg["stage2"]["ema_eta"], online_cfg["stage2"]["ema_K"])
    )
    base_model = deepcopy(model).to(device)
    # keep all models in eval mode for fair comparison
    model.eval()
    base_model.eval()
    feedback_delay_steps = int(online_cfg.get("feedback_delay_steps", cfg["data"]["lookback"]))
    label_buffer = LabelBuffer(feedback_delay_steps) if feedback_delay_steps > 0 else None
    stability = OnlineLoop(
        StabilityConfig(online_cfg["stability"]["gamma"], online_cfg["stability"]["window_size"])
    )
    lrp = LRPEngine()
    stage1_cfg = online_cfg.get("stage1", {})
    n_experts = len(model.experts)
    gate_bias = torch.nn.Parameter(torch.zeros(n_experts, device=device))
    if m_eta_optimizer is None:
        stage1_lr = stage1_cfg.get("lr", cfg["train"]["lr"])
        optimizer_name = stage1_cfg.get("optimizer", "adagrad").lower()
        if optimizer_name == "adam":
            m_eta_optimizer = torch.optim.Adam([gate_bias], lr=stage1_lr)
        else:
            m_eta_optimizer = torch.optim.Adagrad([gate_bias], lr=stage1_lr)
    tb = get_tb_writer("logs/tensorboard")

    # Generator for Stage II/III (max_dim should cover max in/out of linear layers)
    gen = AdaptationGenerator(
        drift_dim=online_cfg["stage2"]["lrp_bottleneck"],
        rank=online_cfg["stage2"]["lrp_bottleneck"],
        max_dim=512,
    ).to(device)
    stage1_lr = stage1_cfg.get("lr", cfg["train"]["lr"])
    stage1_decay = stage1_cfg.get("lr_decay", 1.0)
    stage1_grad_clip = stage1_cfg.get("grad_clip_norm")
    stage1_max_loss_increase = stage1_cfg.get("max_loss_increase", 0.0)
    stage1_rollback = stage1_cfg.get("rollback_on_worse", True)
    stage1_bias_clip = stage1_cfg.get("bias_clip", 0.35)
    stage1_alpha_kl = stage1_cfg.get("alpha_kl_weight", 0.05)
    stage1_bias_l2 = stage1_cfg.get("bias_l2_weight", 0.001)
    stage1_schedule = stage1_cfg.get("schedule", "invsqrt")
    stage1_trigger_loss = stage1_cfg.get("min_trigger_loss", 0.0)
    stage1_update_interval = max(1, int(stage1_cfg.get("update_interval", 1)))
    stage2_decay = online_cfg["stage2"].get("lr_decay", 1.0)
    stage2_rollback = online_cfg["stage2"].get("rollback_on_worse", True)
    stage2_max_loss_increase = online_cfg["stage2"].get("max_loss_increase", 0.0)
    stage2_prefixes = online_cfg["stage2"].get("target_modules", ["experts", "output_head"])
    stage2_prefixes_no_stage1 = online_cfg["stage2"].get("target_modules_no_stage1", stage2_prefixes)
    stage2_accept_buffer_size = max(1, int(online_cfg["stage2"].get("accept_buffer_size", 1)))
    stage2_use_accept_buffer = bool(online_cfg["stage2"].get("use_accept_buffer", True))
    stage2_use_ema_reference = bool(online_cfg["stage2"].get("use_ema_reference", True))
    stage2_warmup_steps = int(online_cfg["stage2"].get("warmup_steps", online_cfg["stability"]["window_size"]))
    stage2_cooldown_steps = max(0, int(online_cfg["stage2"].get("cooldown_steps", 0)))
    stage2_gain_ema_decay = float(online_cfg["stage2"].get("gain_ema_decay", 0.9))
    stage2_min_gain_ema = float(online_cfg["stage2"].get("min_gain_ema", -1e9))
    stage2_reject_cooldown_threshold = float(
        online_cfg["stage2"].get("reject_cooldown_threshold", -1e9)
    )
    stage2_min_omega_density = float(online_cfg["stage2"].get("min_omega_density", 0.0))
    stage2_min_gap = online_cfg["stage2"].get("min_instability_gap", 0.0)
    stage2_relevance_method = online_cfg["stage2"].get("relevance_method", "rrp")
    stage2_use_sparse_mask = bool(online_cfg["stage2"].get("use_sparse_mask", True))
    stage2_skip_if_stage3_gain_above = float(
        online_cfg["stage2"].get("skip_if_stage3_gain_above", 1e9)
    )
    stage2_min_drift_abs = online_cfg["stage2"].get("min_drift_abs", 0.0)
    stage2_gate_by_recon = online_cfg["stage2"].get("gate_by_recon", False)
    stage2_min_recon_error = online_cfg["stage2"].get("min_recon_error", 0.0)
    stage2_with_stage1_scale = online_cfg["stage2"].get("with_stage1_scale", 1.0)
    stage2_eval_with_stage3 = online_cfg["stage2"].get("eval_with_stage3", True)
    stage2_min_improve_abs = online_cfg["stage2"].get("min_improve_abs", 0.0)
    stage2_with_stage1_min_improve_abs = online_cfg["stage2"].get(
        "with_stage1_min_improve_abs", stage2_min_improve_abs
    )
    stage2_no_stage1_lr_scale = online_cfg["stage2"].get("no_stage1_lr_scale", online_cfg["stage2"]["lr_scale"])
    stage2_no_stage1_min_drift_abs = online_cfg["stage2"].get("no_stage1_min_drift_abs", stage2_min_drift_abs)
    stage2_no_stage1_min_recon_error = online_cfg["stage2"].get("no_stage1_min_recon_error", stage2_min_recon_error)
    stage2_no_stage1_min_improve_abs = online_cfg["stage2"].get("no_stage1_min_improve_abs", stage2_min_improve_abs)
    stage2_no_stage1_update_interval = max(
        1, int(online_cfg["stage2"].get("no_stage1_update_interval", online_cfg["stage2"]["update_interval"]))
    )
    stage3_decay = online_cfg["stage3"].get("lr_decay", 1.0)
    stage3_prefixes = online_cfg["stage3"].get("target_modules", ["experts", "output_head"])
    stage3_blend_alpha = online_cfg["stage3"].get("blend_alpha", 1.0)
    stage3_blend_alpha_min = online_cfg["stage3"].get("blend_alpha_min", stage3_blend_alpha)
    stage3_blend_alpha_max = online_cfg["stage3"].get("blend_alpha_max", stage3_blend_alpha)
    stage3_with_stage2_blend_scale = online_cfg["stage3"].get("with_stage2_blend_scale", 1.0)
    stage3_with_stage1_stage2_blend_scale = online_cfg["stage3"].get(
        "with_stage1_stage2_blend_scale", stage3_with_stage2_blend_scale
    )
    compare_log_path = online_cfg.get("compare_log_path", "logs/online_compare.csv")
    stage1_log_path = online_cfg.get("stage1_log_path", "logs/online_stage1.csv")
    stage2_log_path = online_cfg.get("stage2_log_path", "logs/online_stage2.csv")
    base_losses = []
    stage2_losses = []
    stage3_losses = []
    base_maes = []
    stage2_maes = []
    stage3_maes = []
    accept_buffer = deque(maxlen=stage2_accept_buffer_size)
    stage2_gain_ema = 0.0
    stage2_controller_ready = False
    stage2_cooldown_until = -1

    def compute_stage3_prediction(
        x_cur: torch.Tensor,
        outputs_cur,
        phi_i_cur: torch.Tensor | None,
        phi_v_cur: torch.Tensor | None,
        step_idx: int,
    ):
        stage3_used = False
        stage3_recon_error = torch.mean((outputs_cur.x_recon - x_cur) ** 2).item()
        stage3_pred_shift = 0.0
        stage3_blend = 0.0
        if not (online_cfg["stage3"]["enable"] and online_cfg["stage3"]["apply_transient"]):
            return outputs_cur.y_hat, stage3_used, stage3_recon_error, stage3_pred_shift, stage3_blend

        if online_cfg["stage3"].get("use_invariant_delta", True) and phi_i_cur is not None and phi_v_cur is not None:
            d_v = phi_v_cur - phi_i_cur
        else:
            d_v = outputs_cur.z

        should_try_stage3 = True
        if online_cfg["stage3"].get("gate_by_recon", True):
            should_try_stage3 = stage3_recon_error >= online_cfg["stage3"]["min_recon_error"]
        if not should_try_stage3:
            return outputs_cur.y_hat, stage3_used, stage3_recon_error, stage3_pred_shift, stage3_blend

        drift_scalar = d_v.mean()
        drift_vec = (drift_scalar / (d_v.abs().mean() + 1e-6)).repeat(
            online_cfg["stage2"]["lrp_bottleneck"]
        ).to(device)
        delta = build_param_delta(dict(model.named_parameters()), drift_vec, gen)
        delta = _filter_named_tensors(delta, stage3_prefixes)
        delta = {
            k: v * online_cfg["stage3"]["lr_scale"] * (stage3_decay ** step_idx) for k, v in delta.items()
        }
        if not delta:
            return outputs_cur.y_hat, stage3_used, stage3_recon_error, stage3_pred_shift, stage3_blend

        y_hat_candidate = adaptor.stage3_predict(
            x_cur,
            {
                k: torch.clamp(v, -online_cfg["stage3"]["max_delta"], online_cfg["stage3"]["max_delta"])
                for k, v in delta.items()
            },
            gate_bias=gate_bias if stage1_cfg.get("enable", True) else None,
        )
        with torch.no_grad():
            stage3_pred_shift = torch.mean(torch.abs(y_hat_candidate.y_hat - outputs_cur.y_hat)).item()
        if stage3_pred_shift > online_cfg["stage3"]["max_pred_shift"]:
            return outputs_cur.y_hat, stage3_used, stage3_recon_error, stage3_pred_shift, stage3_blend

        recon_floor = max(online_cfg["stage3"]["min_recon_error"], 1e-6)
        recon_ratio = max(0.0, stage3_recon_error - recon_floor) / recon_floor
        recon_ratio = min(recon_ratio, 1.0)
        shift_ratio = min(1.0, stage3_pred_shift / max(online_cfg["stage3"]["max_pred_shift"], 1e-6))
        stage3_blend = stage3_blend_alpha * (1.0 + 0.5 * recon_ratio) * (1.0 - 0.35 * shift_ratio)
        if online_cfg["stage2"]["enable"]:
            stage3_blend *= stage3_with_stage2_blend_scale
        if online_cfg["stage2"]["enable"] and stage1_cfg.get("enable", True):
            stage3_blend *= stage3_with_stage1_stage2_blend_scale
        stage3_blend = max(stage3_blend_alpha_min, min(stage3_blend, stage3_blend_alpha_max))
        blended = (1.0 - stage3_blend) * outputs_cur.y_hat + stage3_blend * y_hat_candidate.y_hat
        stage3_used = True
        return blended, stage3_used, stage3_recon_error, stage3_pred_shift, stage3_blend

    def evaluate_deployed_loss(buffer_items) -> float:
        losses = []
        with torch.no_grad():
            for x_buf, y_buf, phi_i_buf, phi_v_buf in buffer_items:
                outputs_buf = model(
                    x_buf,
                    gate_bias=gate_bias if stage1_cfg.get("enable", True) else None,
                )
                y_hat_buf, _, _, _, _ = compute_stage3_prediction(
                    x_buf, outputs_buf, phi_i_buf, phi_v_buf, step
                )
                y_hat_val = y_hat_buf.y_hat if hasattr(y_hat_buf, "y_hat") else y_hat_buf
                losses.append(torch.mean((y_hat_val - y_buf) ** 2).item())
        return float(sum(losses) / len(losses)) if losses else float("inf")

    for step, batch in enumerate(stream):
        # stream item can be (x, y) or (x, y, phi_I, phi_V)
        if len(batch) == 2:
            x_t, y_t = batch
            phi_I, phi_V = None, None
        else:
            x_t, y_t, phi_I, phi_V = batch

        x_t = x_t.to(device)
        y_t = y_t.to(device)
        if phi_I is not None:
            phi_I = phi_I.to(device)
        if phi_V is not None:
            phi_V = phi_V.to(device)

        # ensure consistent eval mode for fair comparison
        model.eval()
        base_model.eval()
        outputs = model(x_t, gate_bias=gate_bias if stage1_cfg.get("enable", True) else None)
        with torch.no_grad():
            base_out = base_model(x_t)
            base_loss = torch.mean((base_out.y_hat - y_t) ** 2).item()
            stage2_loss = torch.mean((outputs.y_hat - y_t) ** 2).item()
            base_mae = torch.mean(torch.abs(base_out.y_hat - y_t)).item()
            stage2_mae = torch.mean(torch.abs(outputs.y_hat - y_t)).item()
            diff = torch.max(torch.abs(base_out.y_hat - outputs.y_hat)).item()
        if label_buffer is not None:
            label_buffer.push(outputs.z, x_t, outputs.alpha)

        y_hat, stage3_used, stage3_recon_error, stage3_pred_shift, stage3_blend = compute_stage3_prediction(
            x_t, outputs, phi_I, phi_V, step
        )
        with torch.no_grad():
            y_hat_val = y_hat.y_hat if hasattr(y_hat, "y_hat") else y_hat
            adapted_loss = torch.mean((y_hat_val - y_t) ** 2).item()
            adapted_mae = torch.mean(torch.abs(y_hat_val - y_t)).item()
            stage3_gain = stage2_loss - adapted_loss
        accept_buffer.append(
            (
                x_t.detach().clone(),
                y_t.detach().clone(),
                None if phi_I is None else phi_I.detach().clone(),
                None if phi_V is None else phi_V.detach().clone(),
            )
        )

        # delayed label update
        if label_buffer is None:
            delayed = (outputs.z.detach(), x_t.detach(), outputs.alpha.detach(), y_t)
        else:
            delayed = label_buffer.pop_with_label(y_t)
        if delayed is not None and stage1_cfg.get("enable", True):
            z_delayed, x_delayed, alpha_ref, y_delayed = delayed
            gate_snapshot = gate_bias.detach().clone()
            y_hat_delayed = model(x_delayed, gate_bias=gate_bias)
            pred_loss = torch.mean((y_hat_delayed.y_hat - y_delayed) ** 2)
            pre_loss = pred_loss.detach().item()
            if pre_loss >= stage1_trigger_loss and step % stage1_update_interval == 0:
                kl_term = F.kl_div(
                    torch.log(y_hat_delayed.alpha.clamp_min(1e-8)),
                    alpha_ref,
                    reduction="batchmean",
                )
                bias_reg = gate_bias.pow(2).mean()
                loss = pred_loss + stage1_alpha_kl * kl_term + stage1_bias_l2 * bias_reg

                if stage1_schedule == "invsqrt":
                    stage1_step_lr = stage1_lr / ((step + 1) ** 0.5)
                else:
                    stage1_step_lr = stage1_lr * (stage1_decay ** step)
                for pg in m_eta_optimizer.param_groups:
                    pg["lr"] = stage1_step_lr

                m_eta_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if stage1_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_([gate_bias], stage1_grad_clip)
                m_eta_optimizer.step()
                with torch.no_grad():
                    gate_bias.clamp_(-stage1_bias_clip, stage1_bias_clip)

                if stage1_rollback:
                    with torch.no_grad():
                        post_pred = model(x_delayed, gate_bias=gate_bias)
                        post_loss = torch.mean((post_pred.y_hat - y_delayed) ** 2).item()
                    max_allowed = pre_loss * (1.0 + stage1_max_loss_increase)
                    accepted = float(torch.isfinite(torch.tensor(post_loss)) and post_loss <= max_allowed)
                    if not accepted:
                        with torch.no_grad():
                            gate_bias.copy_(gate_snapshot)
                        post_loss = pre_loss
                    log_csv(
                        stage1_log_path,
                        {
                            "step": step,
                            "pre_loss": pre_loss,
                            "post_loss": post_loss,
                            "accepted": accepted,
                            "gate_bias_norm": gate_bias.norm().item(),
                            "gate_bias_max": gate_bias.abs().max().item(),
                            "lr": stage1_step_lr,
                            "kl_term": kl_term.item(),
                        },
                    )

        # Stage II: stability + drift (placeholder)
        online_forward_kwargs = {"gate_bias": gate_bias} if stage1_cfg.get("enable", True) else None
        r_map = lrp.compute_relevance_map(
            model,
            x_t,
            method=stage2_relevance_method,
            forward_kwargs=online_forward_kwargs,
        )
        score = stability.update_stability(r_map)
        active_stage2_prefixes = (
            stage2_prefixes if stage1_cfg.get("enable", True) else stage2_prefixes_no_stage1
        )
        active_update_interval = (
            online_cfg["stage2"]["update_interval"]
            if stage1_cfg.get("enable", True)
            else stage2_no_stage1_update_interval
        )
        active_min_recon_error = (
            stage2_min_recon_error if stage1_cfg.get("enable", True) else stage2_no_stage1_min_recon_error
        )
        active_min_drift_abs = (
            stage2_min_drift_abs if stage1_cfg.get("enable", True) else stage2_no_stage1_min_drift_abs
        )
        if (
            online_cfg["stage2"]["enable"]
            and step >= stage2_warmup_steps
            and step >= stage2_cooldown_until
            and score < online_cfg["stability"]["gamma"]
            and (online_cfg["stability"]["gamma"] - score) >= stage2_min_gap
            and ((not stage2_controller_ready) or (stage2_gain_ema >= stage2_min_gain_ema))
            and ((not stage2_gate_by_recon) or (stage3_recon_error >= active_min_recon_error))
            and (stage3_gain < stage2_skip_if_stage3_gain_above)
            and step % active_update_interval == 0
        ):
            ref_model = adaptor.ref_model if stage2_use_ema_reference else base_model
            drift = lrp.compute_drift(
                x_t,
                model,
                ref_model,
                method=stage2_relevance_method,
                forward_kwargs=online_forward_kwargs,
            )
            drift = drift / (drift.abs().mean() + 1e-6)
            drift_scalar = drift.mean()
            drift_vec = drift_scalar.repeat(online_cfg["stage2"]["lrp_bottleneck"]).to(device)
            if stage2_use_sparse_mask:
                if online_cfg["stage2"].get("mask_threshold") is not None:
                    omega_mask = threshold_mask_from_tensor(
                        drift, online_cfg["stage2"]["mask_threshold"]
                    )
                else:
                    omega_mask = topk_mask_from_tensor(drift, online_cfg["stage2"]["topk_omega"])
                omega_density = omega_mask.float().mean().item()
                x_masked = x_t * omega_mask

                model.zero_grad(set_to_none=True)
                y_masked = model(x_masked, gate_bias=gate_bias if stage1_cfg.get("enable", True) else None)
                y_masked.y_hat.sum().backward()
                grad_mask = build_param_masks_from_grads(model, online_cfg["stage2"]["topk_layer"])

                if stage2_relevance_method == "rrp":
                    head_scores = lrp.compute_attention_head_scores(model, x_t, forward_kwargs=online_forward_kwargs)
                    attn_scores = lrp.compute_attention_causal_scores(model, x_t, forward_kwargs=online_forward_kwargs)
                    head_mask = build_masks_from_attention_heads(
                        model, head_scores, online_cfg["stage2"]["topk_layer"]
                    )
                    attn_mask = build_masks_from_attention_scores(
                        model, attn_scores, online_cfg["stage2"]["topk_layer"]
                    )
                    mask = merge_masks(grad_mask, head_mask, attn_mask)
                else:
                    head_scores = {}
                    attn_scores = {}
                    mask = grad_mask
            else:
                omega_density = 1.0
                head_scores = {}
                attn_scores = {}
                mask = {
                    name: torch.ones_like(param)
                    for name, param in model.named_parameters()
                    if param.dim() == 2
                }
            mask = _filter_named_tensors(mask, active_stage2_prefixes)
            deltas = build_param_delta(dict(model.named_parameters()), drift_vec, gen, mask=mask)
            deltas = _filter_named_tensors(deltas, active_stage2_prefixes)
            instability = max(0.0, online_cfg["stability"]["gamma"] - score)
            base_stage2_scale = (
                online_cfg["stage2"]["lr_scale"] if stage1_cfg.get("enable", True) else stage2_no_stage1_lr_scale
            )
            dynamic_stage2_scale = base_stage2_scale * (stage2_decay ** step) * (1.0 + instability)
            if stage1_cfg.get("enable", True):
                dynamic_stage2_scale *= stage2_with_stage1_scale
            if deltas and abs(drift_scalar.item()) >= active_min_drift_abs and omega_density >= stage2_min_omega_density:
                stage2_snapshot = _snapshot_named_parameters(model, set(deltas.keys()))
                apply_param_delta(
                    model,
                    deltas,
                    scale=dynamic_stage2_scale,
                    max_delta=online_cfg["stage3"]["max_delta"],
                )
                accepted = 1.0
                if stage2_rollback:
                    reference_loss = (
                        evaluate_deployed_loss(accept_buffer)
                        if (
                            stage2_use_accept_buffer
                            and stage2_eval_with_stage3
                            and online_cfg["stage3"]["enable"]
                        )
                        else stage2_loss
                    )
                    with torch.no_grad():
                        post_outputs = model(
                            x_t, gate_bias=gate_bias if stage1_cfg.get("enable", True) else None
                        )
                        post_stage2_loss = torch.mean((post_outputs.y_hat - y_t) ** 2).item()
                        post_eval_loss = (
                            evaluate_deployed_loss(accept_buffer)
                            if (
                                stage2_use_accept_buffer
                                and stage2_eval_with_stage3
                                and online_cfg["stage3"]["enable"]
                            )
                            else post_stage2_loss
                        )
                    raw_gain = reference_loss - post_eval_loss
                    stage2_gain_ema = (
                        stage2_gain_ema_decay * stage2_gain_ema + (1.0 - stage2_gain_ema_decay) * raw_gain
                    )
                    stage2_controller_ready = True
                    required_improve = (
                        stage2_with_stage1_min_improve_abs if stage1_cfg.get("enable", True) else stage2_no_stage1_min_improve_abs
                    )
                    max_allowed = reference_loss * (1.0 + stage2_max_loss_increase) - required_improve
                    if (not torch.isfinite(torch.tensor(post_eval_loss))) or post_eval_loss > max_allowed:
                        _restore_named_parameters(model, stage2_snapshot)
                        post_stage2_loss = stage2_loss
                        post_eval_loss = reference_loss
                        accepted = 0.0
                        if raw_gain <= stage2_reject_cooldown_threshold:
                            stage2_cooldown_until = step + stage2_cooldown_steps
                else:
                    post_stage2_loss = float("nan")
                    post_eval_loss = float("nan")
                    raw_gain = 0.0
            else:
                accepted = 0.0
                post_stage2_loss = stage2_loss
                post_eval_loss = adapted_loss
                raw_gain = 0.0

            # mask density stats
            densities = []
            for name, m in mask.items():
                densities.append(m.float().mean().item())
            mask_density = sum(densities) / max(len(densities), 1)
            log_dict(
                "stage2",
                {
                    "drift": drift_scalar.item(),
                    "score": score,
                    "stable": float(score >= online_cfg["stability"]["gamma"]),
                    "mask_layers": float(len(mask)),
                    "mask_density": mask_density,
                    "head_scores": float(sum([v.mean().item() for v in head_scores.values()]) if head_scores else 0.0),
                    "attn_scores": float(sum([v.mean().item() for v in attn_scores.values()]) if attn_scores else 0.0),
                    "omega_density": omega_density,
                    "stage3_gain": stage3_gain,
                    "stage2_scale": dynamic_stage2_scale,
                    "accepted": accepted,
                    "gain_ema": stage2_gain_ema,
                    "cooldown_until": float(stage2_cooldown_until),
                },
            )
            log_csv(
                stage2_log_path,
                {
                    "step": step,
                    "drift": drift_scalar.item(),
                    "score": score,
                    "stable": float(score >= online_cfg["stability"]["gamma"]),
                    "mask_layers": float(len(mask)),
                    "mask_density": mask_density,
                    "omega_density": omega_density,
                    "stage3_gain": stage3_gain,
                    "stage2_scale": dynamic_stage2_scale,
                    "pre_loss": stage2_loss,
                    "post_loss": post_stage2_loss,
                    "post_eval_loss": post_eval_loss,
                    "accepted": accepted,
                    "raw_gain": raw_gain,
                    "gain_ema": stage2_gain_ema,
                    "cooldown_until": stage2_cooldown_until,
                },
            )
        log_csv(
            compare_log_path,
            {
                "step": step,
                "base_mse": base_loss,
                "stage2_mse": stage2_loss,
                "stage3_mse": adapted_loss,
                "base_mae": base_mae,
                "stage2_mae": stage2_mae,
                "stage3_mae": adapted_mae,
                "target_t": y_t[:, 0].mean().item() if y_t.dim() >= 2 else y_t.mean().item(),
                "pred_base_t": base_out.y_hat[:, 0].mean().item() if base_out.y_hat.dim() >= 2 else base_out.y_hat.mean().item(),
                "pred_stage2_t": outputs.y_hat[:, 0].mean().item() if outputs.y_hat.dim() >= 2 else outputs.y_hat.mean().item(),
                "pred_stage3_t": y_hat_val[:, 0].mean().item() if y_hat_val.dim() >= 2 else y_hat_val.mean().item(),
                "stage3_used": float(stage3_used),
                "base_vs_stage2_maxdiff": diff,
                "gate_bias_norm": gate_bias.norm().item(),
                "stage3_recon_error": stage3_recon_error,
                "stage3_pred_shift": stage3_pred_shift,
                "stage3_blend": stage3_blend,
            },
        )
        if step % 50 == 0:
            log_dict(
                "online_compare",
                {
                    "base_mse": base_loss,
                    "stage2_mse": stage2_loss,
                    "stage3_mse": adapted_loss,
                    "base_mae": base_mae,
                    "stage2_mae": stage2_mae,
                    "stage3_mae": adapted_mae,
                    "stage3": float(stage3_used),
                    "stage3_recon_error": stage3_recon_error,
                    "stage3_pred_shift": stage3_pred_shift,
                    "stage3_blend": stage3_blend,
                },
            )
        if tb is not None:
            tb.add_scalar("online/base_mse", base_loss, step)
            tb.add_scalar("online/stage2_mse", stage2_loss, step)
            tb.add_scalar("online/stage3_mse", adapted_loss, step)
            tb.add_scalar("online/stage3_recon_error", stage3_recon_error, step)
            tb.add_scalar("online/stage3_pred_shift", stage3_pred_shift, step)
            tb.add_scalar("online/stage3_blend", stage3_blend, step)

        base_losses.append(base_loss)
        stage2_losses.append(stage2_loss)
        stage3_losses.append(adapted_loss)
        base_maes.append(base_mae)
        stage2_maes.append(stage2_mae)
        stage3_maes.append(adapted_mae)

        if stage2_use_ema_reference:
            adaptor.step()

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "online_base_mse": _mean(base_losses),
        "online_base_mae": _mean(base_maes),
        "online_stage2_mse": _mean(stage2_losses),
        "online_stage2_mae": _mean(stage2_maes),
        "online_stage3_mse": _mean(stage3_losses),
        "online_stage3_mae": _mean(stage3_maes),
    }
