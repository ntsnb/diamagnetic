from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.stage3_residual_teacher_v12b import TrainOnlyChronologicalOOFResidualTeacherV12b


class TrainOnlyChronologicalOOFResidualTeacherV12c(TrainOnlyChronologicalOOFResidualTeacherV12b):
    """V12c residual teacher with separate exploratory and claim gates.

    V12b selected one alpha path and could collapse all diagnostics to zero
    when the strict worst-block gate abstained. V12c keeps three policies:

    - ``ungated``: best OOF alpha, regardless of stability gates.
    - ``relaxed``: alpha allowed when enough chronological blocks pass.
    - ``strict``: relaxed gate plus the V12b worst-block safety checks.

    The class still fits on train only and uses purged forward OOF evidence.
    It is intended for teacher-only diagnosis before any generator training.
    """

    protocol_version = "v12c_multi_policy_oof_residual_teacher"
    _POLICIES = ("ungated", "relaxed", "strict")

    def __init__(
        self,
        *,
        selected_policy: str = "relaxed",
        claim_policy: str = "strict",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if selected_policy not in self._POLICIES:
            raise ValueError(f"unsupported selected_policy: {selected_policy}")
        if claim_policy not in self._POLICIES:
            raise ValueError(f"unsupported claim_policy: {claim_policy}")
        self.selected_policy = selected_policy
        self.claim_policy = claim_policy
        self.alpha_by_policy: dict[str, torch.Tensor] = {
            name: torch.zeros(self.output_dim, dtype=torch.float32) for name in self._POLICIES
        }
        self.policy_summaries: dict[str, dict[str, float | int | str]] = {}

    def alpha_policy_names(self) -> tuple[str, ...]:
        return self._POLICIES

    def _empty_policy_alphas(self, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {name: torch.zeros(self.output_dim, device=device, dtype=dtype) for name in self._POLICIES}

    def _summarize_policy(
        self,
        *,
        policy: str,
        ridge: float,
        residual: torch.Tensor,
        prediction: torch.Tensor,
        alpha: torch.Tensor,
        output_pass_count: int,
        output_count: int,
        worst_mse_values: list[float],
        worst_mae_values: list[float],
        worst_sign_values: list[float],
        total_block_pass: int,
        total_blocks: int,
        best_alpha: torch.Tensor,
    ) -> dict[str, float | int | str]:
        selected_prediction = prediction * alpha.unsqueeze(0)
        best_prediction = prediction * best_alpha.unsqueeze(0)
        base_mse = torch.mean(residual.pow(2)).clamp_min(self.eps)
        selected_mse = torch.mean((residual - selected_prediction).pow(2))
        best_mse = torch.mean((residual - best_prediction).pow(2))
        base_mae = torch.mean(residual.abs()).clamp_min(self.eps)
        selected_mae = torch.mean((residual - selected_prediction).abs())
        sign_mask = residual.abs() > self.eps
        if sign_mask.any() and selected_prediction.abs().max().item() > self.eps:
            sign = torch.eq(torch.sign(residual[sign_mask]), torch.sign(selected_prediction[sign_mask]))
            sign_score = float(sign.float().mean().item())
        else:
            sign_score = 0.0
        return {
            "policy": policy,
            "ridge": float(ridge),
            "oof_base_mse": float(base_mse.item()),
            "oof_selected_mse": float(selected_mse.item()),
            "oof_best_mse": float(best_mse.item()),
            "oof_mse_gain": float(((base_mse - selected_mse) / base_mse).item()),
            "oof_best_mse_gain": float(((base_mse - best_mse) / base_mse).item()),
            "oof_base_mae": float(base_mae.item()),
            "oof_selected_mae": float(selected_mae.item()),
            "oof_mae_gain": float(((base_mae - selected_mae) / base_mae).item()),
            "oof_sign_agreement": sign_score,
            "output_pass_count": int(output_pass_count),
            "output_count": int(output_count),
            "selected_alpha_abs_mean": float(alpha.abs().mean().item()),
            "selected_alpha_nonzero_fraction": float((alpha.abs() > self.eps).float().mean().item()),
            "best_alpha_abs_mean": float(best_alpha.abs().mean().item()),
            "worst_block_mse_gain": float(min(worst_mse_values) if worst_mse_values else 0.0),
            "worst_block_mae_gain": float(min(worst_mae_values) if worst_mae_values else 0.0),
            "worst_block_sign_agreement": float(min(worst_sign_values) if worst_sign_values else 0.0),
            "block_pass_count": int(total_block_pass),
            "effective_block_count": int(total_blocks),
        }

    def _select_alpha_policies_for_oof(
        self,
        *,
        residual: torch.Tensor,
        prediction: torch.Tensor,
        fold_ids: torch.Tensor,
        ridge: float,
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float | int | str]], list[dict[str, int | float | str]]]:
        alpha_grid = self.alpha_grid.to(device=residual.device, dtype=residual.dtype)
        policies = self._empty_policy_alphas(device=residual.device, dtype=residual.dtype)
        best = torch.zeros(self.output_dim, device=residual.device, dtype=residual.dtype)
        pass_counts = {"ungated": self.output_dim, "relaxed": 0, "strict": 0}
        policy_worst_mse: dict[str, list[float]] = {name: [] for name in self._POLICIES}
        policy_worst_mae: dict[str, list[float]] = {name: [] for name in self._POLICIES}
        policy_worst_sign: dict[str, list[float]] = {name: [] for name in self._POLICIES}
        policy_block_pass = {name: 0 for name in self._POLICIES}
        policy_blocks = {name: 0 for name in self._POLICIES}
        all_rows: list[dict[str, int | float | str]] = []

        for output_idx in range(self.output_dim):
            residual_j = residual[:, output_idx]
            pred_j = prediction[:, output_idx]
            best_score = -float("inf")
            best_alpha = 0.0
            best_metrics = (0.0, 0.0, 0.0, 0, 0)
            for alpha_t in alpha_grid:
                alpha = float(alpha_t.item())
                worst_mse, worst_mae, worst_sign, pass_count, block_count, rows = self._block_metrics_by_fold(
                    residual_j=residual_j,
                    pred_j=pred_j,
                    fold_ids=fold_ids,
                    output_idx=output_idx,
                    alpha=alpha,
                    ridge=ridge,
                )
                for row in rows:
                    row = dict(row)
                    row["candidate_policy_family"] = "all"
                    all_rows.append(row)
                pred_alpha = pred_j * alpha
                mean_mse_gain, mean_mae_gain, mean_sign = self._metric_values(
                    residual_j.unsqueeze(1),
                    pred_alpha.unsqueeze(1),
                )
                score = mean_mse_gain + self.calibration_mae_weight * mean_mae_gain + 0.005 * mean_sign
                if score > best_score + self.eps or (
                    abs(score - best_score) <= self.eps and abs(alpha) < abs(best_alpha)
                ):
                    best_score = score
                    best_alpha = alpha
                    best_metrics = (worst_mse, worst_mae, worst_sign, pass_count, block_count)

            best[output_idx] = float(best_alpha)
            policies["ungated"][output_idx] = float(best_alpha)
            best_prediction = pred_j * float(best_alpha)
            output_gain, _, _ = self._metric_values(residual_j.unsqueeze(1), best_prediction.unsqueeze(1))
            worst_mse, worst_mae, worst_sign, pass_count, block_count = best_metrics
            required_passes = int(torch.ceil(torch.tensor(self.min_block_pass_fraction * block_count)).item())
            relaxed_passed = block_count > 0 and pass_count >= required_passes and output_gain >= self.min_output_gain
            strict_passed = (
                relaxed_passed
                and worst_mse >= self.min_block_mse_gain
                and worst_mae >= self.min_block_mae_gain
                and worst_sign >= self.min_block_sign_agreement
            )
            if relaxed_passed:
                policies["relaxed"][output_idx] = float(best_alpha)
                pass_counts["relaxed"] += 1
            if strict_passed:
                policies["strict"][output_idx] = float(best_alpha)
                pass_counts["strict"] += 1

            for name in self._POLICIES:
                if name == "ungated" or (name == "relaxed" and relaxed_passed) or (name == "strict" and strict_passed):
                    policy_worst_mse[name].append(worst_mse)
                    policy_worst_mae[name].append(worst_mae)
                    policy_worst_sign[name].append(worst_sign)
                    policy_block_pass[name] += pass_count
                    policy_blocks[name] += block_count

        summaries = {
            name: self._summarize_policy(
                policy=name,
                ridge=ridge,
                residual=residual,
                prediction=prediction,
                alpha=alpha,
                output_pass_count=pass_counts[name],
                output_count=self.output_dim,
                worst_mse_values=policy_worst_mse[name],
                worst_mae_values=policy_worst_mae[name],
                worst_sign_values=policy_worst_sign[name],
                total_block_pass=policy_block_pass[name],
                total_blocks=policy_blocks[name],
                best_alpha=best,
            )
            for name, alpha in policies.items()
        }
        return policies, summaries, all_rows

    def fit(
        self,
        *,
        drift_train: torch.Tensor,
        y_base_train: torch.Tensor,
        y_train: torch.Tensor,
        split: str = "train",
        sample_ids: torch.Tensor | None = None,
    ) -> "TrainOnlyChronologicalOOFResidualTeacherV12c":
        if split != "train":
            raise ValueError("TrainOnlyChronologicalOOFResidualTeacherV12c can only be fit on the train split")
        drift = self._as_2d_drift(drift_train)
        y_base = y_base_train.detach().float().reshape(drift.shape[0], -1)
        y = y_train.detach().float().reshape(drift.shape[0], -1)
        if y.shape != y_base.shape:
            raise ValueError("y_base_train and y_train must have the same flattened shape")
        if y.shape[1] != self.output_dim:
            raise ValueError(f"teacher output dim {y.shape[1]} does not match expected {self.output_dim}")
        if drift.shape[0] != y.shape[0]:
            raise ValueError("drift_train and y_train must share sample count")

        features = self._feature_map(drift)
        residual = y - y_base
        best_summaries: dict[str, dict[str, float | int | str]] | None = None
        best_policies: dict[str, torch.Tensor] | None = None
        best_rows: list[dict[str, int | float | str]] = []
        best_score = -float("inf")
        best_covered = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
        best_fold_ids = torch.full((features.shape[0],), -1, dtype=torch.long, device=features.device)
        best_oof_prediction = torch.zeros_like(residual)

        self._ridge_rows = []
        for ridge in self.ridge_grid.tolist():
            oof_pred, covered, fold_ids = self._oof_prediction_for_ridge(features, residual, float(ridge))
            if not covered.any():
                continue
            residual_oof = residual[covered]
            pred_oof = oof_pred[covered]
            fold_oof = fold_ids[covered]
            policies, summaries, rows = self._select_alpha_policies_for_oof(
                residual=residual_oof,
                prediction=pred_oof,
                fold_ids=fold_oof,
                ridge=float(ridge),
            )
            for policy, summary in summaries.items():
                row = dict(summary)
                row["policy"] = policy
                self._ridge_rows.append(row)
            selected_summary = summaries[self.selected_policy]
            score = (
                float(selected_summary["oof_mse_gain"])
                + self.calibration_mae_weight * float(selected_summary["oof_mae_gain"])
                + 0.01 * float(selected_summary["oof_sign_agreement"])
                + 0.05 * float(selected_summary["selected_alpha_nonzero_fraction"])
            )
            if score > best_score + self.eps:
                best_score = score
                best_summaries = summaries
                best_policies = policies
                best_rows = rows
                best_covered = covered
                best_fold_ids = fold_ids
                best_oof_prediction = oof_pred

        if best_summaries is None or best_policies is None:
            self.selected_alpha_source = "fallback_no_oof_folds"
            self.alpha_by_policy = {
                name: torch.zeros(self.output_dim, dtype=torch.float32) for name in self._POLICIES
            }
            self.policy_summaries = {}
            self.alpha_by_output = torch.zeros(self.output_dim, dtype=torch.float32)
            self.selected_ridge = float(self.ridge_grid[0].item())
            self._alpha_block_rows = []
        else:
            self.alpha_by_policy = {name: value.detach().float().cpu() for name, value in best_policies.items()}
            self.policy_summaries = {name: dict(value) for name, value in best_summaries.items()}
            self.alpha_by_output = self.alpha_by_policy[self.selected_policy].detach().float().cpu()
            self.selected_ridge = float(best_summaries[self.selected_policy]["ridge"])
            any_nonzero = any(bool((alpha.abs() > self.eps).any()) for alpha in self.alpha_by_policy.values())
            self.selected_alpha_source = "v12c_multi_policy_oof" if any_nonzero else "fallback_oof_gate"
            self._alpha_block_rows = best_rows

        solution = self._solve_weight(features, residual, self.selected_ridge)
        alpha_device = self.alpha_by_output.to(device=features.device, dtype=features.dtype)
        correction = (features @ solution) * alpha_device.unsqueeze(0)
        before_mse = torch.mean(residual.pow(2)).clamp_min(self.eps)
        after_mse = torch.mean((residual - correction).pow(2))
        centered_total = torch.mean((residual - residual.mean(dim=0, keepdim=True)).pow(2)).clamp_min(self.eps)

        self.correction_weight = solution.detach().float()
        self.fit_split = "train"
        self.fit_sample_count = int(drift.shape[0])
        self.oof_covered_sample_count = int(best_covered.sum().item())
        self.oof_fold_count = len([v for v in torch.unique(best_fold_ids).detach().cpu().tolist() if int(v) >= 0])
        if sample_ids is None:
            self.train_sample_ids = torch.arange(self.fit_sample_count, dtype=torch.long)
        else:
            sample_ids = sample_ids.detach().cpu().long().flatten()
            if sample_ids.numel() != self.fit_sample_count:
                raise ValueError("sample_ids must match fit sample count")
            self.train_sample_ids = sample_ids
        self.teacher_residual_mse_before = float(before_mse.item())
        self.teacher_residual_mse_after = float(after_mse.item())
        self.teacher_residual_mse_gain = float(((before_mse - after_mse) / before_mse).item())
        self.teacher_residual_r2 = float(torch.clamp(1.0 - after_mse / centered_total, min=-1.0, max=1.0).item())

        selected_summary = self.policy_summaries.get(self.selected_policy)
        if selected_summary is not None and best_covered.any():
            self.oof_base_mse = float(selected_summary["oof_base_mse"])
            self.oof_selected_mse = float(selected_summary["oof_selected_mse"])
            self.oof_mse_gain = float(selected_summary["oof_mse_gain"])
            self.oof_base_mae = float(selected_summary["oof_base_mae"])
            self.oof_selected_mae = float(selected_summary["oof_selected_mae"])
            self.oof_mae_gain = float(selected_summary["oof_mae_gain"])
            self.oof_sign_agreement = float(selected_summary["oof_sign_agreement"])
            self.output_pass_count = int(selected_summary["output_pass_count"])
            self.output_count = int(selected_summary["output_count"])
            self.best_alpha_abs_mean = float(selected_summary["best_alpha_abs_mean"])
            self.selected_alpha_abs_mean = float(selected_summary["selected_alpha_abs_mean"])
            self.selected_alpha_nonzero_fraction = float(selected_summary["selected_alpha_nonzero_fraction"])
            self.worst_block_mse_gain = float(selected_summary["worst_block_mse_gain"])
            self.worst_block_mae_gain = float(selected_summary["worst_block_mae_gain"])
            self.worst_block_sign_agreement = float(selected_summary["worst_block_sign_agreement"])
            self.block_pass_count = int(selected_summary["block_pass_count"])
            self.effective_block_count = int(selected_summary["effective_block_count"])
            selected_oof = best_oof_prediction[best_covered] * alpha_device.unsqueeze(0)
            self.oof_correction_cosine = self._mean_cosine(residual[best_covered], selected_oof)
            self._calibrate_reliability_gate(
                drift=drift[best_covered],
                residual=residual[best_covered],
                correction=selected_oof,
            )
        else:
            self.oof_base_mse = float("nan")
            self.oof_selected_mse = float("nan")
            self.oof_mse_gain = 0.0
            self.oof_base_mae = float("nan")
            self.oof_selected_mae = float("nan")
            self.oof_mae_gain = 0.0
            self.oof_sign_agreement = 0.0
            self.oof_correction_cosine = float("nan")
            self.output_pass_count = 0
            self.block_pass_count = 0
            self.effective_block_count = 0
            self.reliability_gate_active = False
            self.reliability_gate_reason = "no_oof_prediction"
        return self

    def predict_correction(self, drift: torch.Tensor, *, policy: str | None = None) -> torch.Tensor:
        if self.correction_weight is None:
            raise RuntimeError("TrainOnlyChronologicalOOFResidualTeacherV12c must be fit before prediction")
        policy_name = policy or self.selected_policy
        if policy_name not in self._POLICIES:
            raise ValueError(f"unsupported V12c policy: {policy_name}")
        drift_2d = self._as_2d_drift(drift)
        features = self._feature_map(drift_2d)
        weight = self.correction_weight.to(device=features.device, dtype=features.dtype)
        alpha = self.alpha_by_policy[policy_name].to(device=features.device, dtype=features.dtype)
        correction = (features @ weight) * alpha.unsqueeze(0)
        if policy_name == self.selected_policy:
            correction = correction * self._reliability_gate(drift_2d, correction).unsqueeze(1)
        return correction.reshape(features.shape[0], *self.output_shape)

    def diagnostic_summary(self) -> dict[str, int | float | str]:
        summary = super().diagnostic_summary()
        summary["teacher_method"] = self.protocol_version
        summary["teacher_selected_policy"] = self.selected_policy
        summary["teacher_claim_policy"] = self.claim_policy
        for policy, policy_summary in self.policy_summaries.items():
            prefix = f"teacher_policy_{policy}"
            for key, value in policy_summary.items():
                if key == "policy":
                    continue
                summary[f"{prefix}_{key}"] = value
                if key.startswith("selected_alpha_"):
                    summary[f"{prefix}_{key.replace('selected_alpha_', 'alpha_', 1)}"] = value
        return summary

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["protocol_version"] = self.protocol_version
        state["selected_policy"] = self.selected_policy
        state["claim_policy"] = self.claim_policy
        state["alpha_by_policy"] = {
            name: alpha.detach().clone() for name, alpha in self.alpha_by_policy.items()
        }
        state["policy_summaries"] = {name: dict(summary) for name, summary in self.policy_summaries.items()}
        return state

    def load_state_dict(self, state: dict[str, Any]) -> "TrainOnlyChronologicalOOFResidualTeacherV12c":
        super().load_state_dict(state)
        self.selected_policy = str(state.get("selected_policy", self.selected_policy))
        self.claim_policy = str(state.get("claim_policy", self.claim_policy))
        if self.selected_policy not in self._POLICIES:
            raise ValueError(f"loaded unsupported selected_policy: {self.selected_policy}")
        if self.claim_policy not in self._POLICIES:
            raise ValueError(f"loaded unsupported claim_policy: {self.claim_policy}")
        loaded_alphas = dict(state.get("alpha_by_policy", {}))
        self.alpha_by_policy = {}
        for policy in self._POLICIES:
            value = loaded_alphas.get(policy)
            if value is None:
                value = self.alpha_by_output if policy == self.selected_policy else torch.zeros(self.output_dim)
            alpha = torch.as_tensor(value).detach().float().flatten()
            if alpha.numel() != self.output_dim:
                raise ValueError(f"loaded V12c alpha for policy {policy} has incompatible shape")
            self.alpha_by_policy[policy] = alpha
        self.alpha_by_output = self.alpha_by_policy[self.selected_policy].detach().clone()
        self.policy_summaries = {
            str(name): dict(summary) for name, summary in dict(state.get("policy_summaries", {})).items()
        }
        return self


def save_stage3_v12c_teacher_checkpoint(
    *,
    path: str | Path,
    teacher: TrainOnlyChronologicalOOFResidualTeacherV12c,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refuse to overwrite existing Stage III V12c residual teacher checkpoint: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = dict(metadata or {})
    payload_metadata.setdefault("protocol_version", TrainOnlyChronologicalOOFResidualTeacherV12c.protocol_version)
    payload_metadata.setdefault("class_name", TrainOnlyChronologicalOOFResidualTeacherV12c.__name__)
    signal_builder = getattr(teacher, "stage3_signal_builder_v12c", None)
    if signal_builder is not None and hasattr(signal_builder, "state_dict"):
        payload_metadata.setdefault("stage3_signal_builder_v12c_state", signal_builder.state_dict())
    torch.save(
        {
            "class_name": TrainOnlyChronologicalOOFResidualTeacherV12c.__name__,
            "protocol_version": TrainOnlyChronologicalOOFResidualTeacherV12c.protocol_version,
            "metadata": payload_metadata,
            "state": teacher.state_dict(),
        },
        target,
    )


def load_stage3_v12c_teacher_checkpoint(
    path: str | Path,
    *,
    expected_drift_dim: int | None = None,
    expected_output_shape: tuple[int, ...] | None = None,
) -> tuple[TrainOnlyChronologicalOOFResidualTeacherV12c, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    state = checkpoint.get("state", checkpoint)
    teacher = TrainOnlyChronologicalOOFResidualTeacherV12c(
        drift_dim=int(state["drift_dim"]),
        output_shape=tuple(int(dim) for dim in state["output_shape"]),
        ridge_grid=torch.as_tensor(state.get("ridge_grid", [1e-3])).flatten().tolist(),
        alpha_grid=torch.as_tensor(state.get("alpha_grid", [0.0])).flatten().tolist(),
        eps=float(state.get("eps", 1e-8)),
        fold_count=int(state.get("fold_count", 5)),
        min_prefix_samples=int(state.get("min_prefix_samples", 32)),
        purge_gap=int(state.get("purge_gap", 0)),
        min_block_mse_gain=float(state.get("min_block_mse_gain", 0.0)),
        min_block_mae_gain=float(state.get("min_block_mae_gain", 0.0)),
        min_block_sign_agreement=float(state.get("min_block_sign_agreement", 0.5)),
        min_block_pass_fraction=float(state.get("min_block_pass_fraction", 0.6)),
        min_output_gain=float(state.get("min_output_gain", 0.0)),
        calibration_mae_weight=float(state.get("calibration_mae_weight", 0.0)),
        allow_negative_alpha=bool(state.get("allow_negative_alpha", True)),
        reliability_gate_enabled=bool(state.get("reliability_gate_enabled", True)),
        reliability_gate_quantile=float(state.get("reliability_gate_quantile", 0.95)),
        min_reliable_fraction=float(state.get("min_reliable_fraction", 0.1)),
        disable_if_unreliable=bool(state.get("disable_if_unreliable", False)),
        selected_policy=str(state.get("selected_policy", "relaxed")),
        claim_policy=str(state.get("claim_policy", "strict")),
    ).load_state_dict(state)
    if expected_drift_dim is not None and teacher.drift_dim != int(expected_drift_dim):
        raise ValueError("loaded V12c residual teacher drift_dim does not match expected_drift_dim")
    if expected_output_shape is not None and teacher.output_shape != tuple(expected_output_shape):
        raise ValueError("loaded V12c residual teacher output_shape does not match expected_output_shape")
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.setdefault("protocol_version", TrainOnlyChronologicalOOFResidualTeacherV12c.protocol_version)
    metadata.setdefault("class_name", TrainOnlyChronologicalOOFResidualTeacherV12c.__name__)
    return teacher, metadata
