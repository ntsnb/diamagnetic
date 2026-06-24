from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.loaders import build_dataloaders_from_series
from data.normalization import StandardScaler
from data.readers import load_tabular_timeseries
from data.readers_stage3_v2 import load_split_safe_tabular_timeseries
from models.aligned_representation_v2 import (
    TrainSetDeltaStandardizer,
    TrainSetResidualDriftCanonicalizer,
    TrainSetTemporalResidualCCAAligner,
    TrainSetWhitenedProcrustesTemporalResidualCCAAligner,
    temporal_residual_alignment_diagnostics,
)
from models.stage3_alignment_v7 import TrainSetChronologicalResidualizedReducedRankCCAAligner
from models.stage3_alignment_v9 import TrainSetTeacherResidualizedReducedRankCCAAligner
from models.stage3_alignment_v10 import TrainSetCrossFittedTeacherResidualProjectionCCAAligner
from models.stage3_alignment_v11 import TrainSetCrossFittedResidualTransferPLSAlignerV11
from models.stage3_alignment_v12 import TrainOnlyOutcomeAlignedTransportAlignerV12
from models.adaptation_v2 import (
    LayerAwareHyperAdapterGenerator,
    LayerAwareHyperLoRAGenerator,
    build_stage3_v2_param_delta,
    summarize_delta,
)
from models.adaptation_v9 import ResidualBasisHyperGeneratorV9
from models.adaptation_v10 import ResidualBasisHyperGeneratorV10
from models.factor_mechanism_v2 import FactorMechanismV2Config, build_factor_mechanism_v2
from models.stage3_residual_teacher_v9 import (
    TrainOnlyResidualTeacherV9,
    save_stage3_v9_teacher_checkpoint,
)
from models.stage3_residual_teacher_v10 import (
    TrainOnlyResidualTeacherV10,
    save_stage3_v10_teacher_checkpoint,
)
from models.stage3_residual_teacher_v11 import (
    TrainOnlyRobustBlockResidualTeacherV11,
    save_stage3_v11_teacher_checkpoint,
)
from models.stage3_residual_teacher_v12 import (
    TrainOnlyOutcomeAlignedBlockResidualTeacherV12,
    save_stage3_v12_teacher_checkpoint,
)
from train.offline_aligned_representation_v2 import (
    AlignedRepresentationTrainConfig,
    build_aligned_representation_model,
    evaluate_aligned_representation,
    train_aligned_representation_epoch,
)
from train.offline_factor import evaluate_factor, train_offline_factor
from train.online_loop_stage3_v2 import run_online_adaptation_v2
from train.run_experiment import build_offline_scheduler
from train.run_load_dataset_benchmark import DATASETS, target_dim_for_dataset
from train.stage3_functional import functional_call_with_param_delta
from train.stage3_eval_policy import assert_deterministic_eval_model
from train.stage3_generator_training_v10 import (
    Stage3GeneratorLossConfig,
    load_stage3_generator_checkpoint,
    save_stage3_generator_checkpoint,
    stage3_generator_train_step,
)
from train.stage3_generator_selection_v2 import (
    Stage3GeneratorCandidate,
    build_generator_selection_row,
    select_best_generator_candidate,
    score_served_policy_candidate,
)
from train.stage3_risk_gate_v2 import (
    Stage3RiskGateBundle,
    build_stage3_risk_features,
    save_stage3_risk_gate_checkpoint,
    train_stage3_risk_gate,
)
from train.stage3_sensitivity_gate_v2 import (
    COUNTERFACTUAL_MODES,
    evaluate_rolling_stage3_sensitivity_gates,
    evaluate_stage3_sensitivity_gate,
)
from train.stage3_sensitivity_v2 import build_drift_interventions
from train.stage3_signals import Stage3SignalState, build_stage3_drift_vector
from train.stage3_v9_teacher_diagnostics import (
    compute_teacher_correction_metrics,
    summarize_teacher_mode_rows,
)
from train.stage3_v2_data import build_causal_dataloaders_from_series
from train.stage3_v12_artifacts import load_stage3_v12_artifact_bundle
from utils.config import load_yaml
from utils.seed import set_seed


class LimitedBatches:
    def __init__(self, dataloader, max_batches: int | None) -> None:
        self.dataloader = dataloader
        self.max_batches = max_batches
        self.dataset = getattr(dataloader, "dataset", None)

    def __iter__(self):
        if self.max_batches is None:
            return iter(self.dataloader)
        return islice(iter(self.dataloader), self.max_batches)


def _limit_batches(dataloader, max_batches: int | None):
    return dataloader if max_batches is None else LimitedBatches(dataloader, max_batches)


def _ordered_loader(dataset, batch_size: int, drop_last: bool = False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)


def _contiguous_dataset_windows(dataset, window_count: int) -> list[Subset]:
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    n = len(dataset)
    if n == 0:
        raise ValueError("cannot split an empty dataset into validation windows")
    window_count = min(window_count, n)
    base = n // window_count
    remainder = n % window_count
    windows = []
    start = 0
    for window_id in range(window_count):
        size = base + (1 if window_id < remainder else 0)
        stop = start + size
        windows.append(Subset(dataset, range(start, stop)))
        start = stop
    return windows


def _subset_bounds(dataset) -> tuple[int, int]:
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return 0, len(dataset)
    indices = list(indices)
    if not indices:
        return 0, 0
    return int(indices[0]), int(indices[-1]) + 1


def _intervention_is_online_causal(mode: str) -> bool:
    return mode in {"real", "zero", "sign_flipped"} or mode.startswith("amplified_")


def _selection_modes_for_validation(cfg: dict[str, Any], modes: list[str]) -> list[str]:
    selection_cfg = cfg.get("stage3_v2", {}).get("generator", {}).get("checkpoint_selection", {})
    gate_cfg = cfg.get("stage3_v2", {}).get("claim_gate", {})
    explicit_modes = selection_cfg.get("selection_modes")
    base_modes = [str(mode) for mode in (explicit_modes or modes)]
    include_offline = bool(
        selection_cfg.get(
            "include_offline_diagnostics_in_selection",
            gate_cfg.get("include_offline_diagnostics_in_selection", False),
        )
    )
    selected = base_modes if include_offline else [mode for mode in base_modes if _intervention_is_online_causal(mode)]
    if "real" not in selected and "real" in modes:
        selected = ["real", *selected]
    if not selected:
        raise ValueError("Stage III validation selection requires at least one online-causal mode")
    return selected


def _as_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _raise_if_final_generator_selection_abstained(selected_row: dict[str, Any], *, stage: str) -> None:
    if not _as_bool_flag(selected_row.get("generator_selection_abstained", False)):
        return
    reason = selected_row.get("generator_selection_abstain_reason", "unspecified")
    raise RuntimeError(
        f"{stage} generator selection abstained ({reason}); "
        "no final Stage III served artifact will be saved from a candidate that failed validation."
    )


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_status_short() -> str:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unavailable: {exc}"
    return proc.stdout.strip()


def _write_resolved_config_snapshot(*, cfg: dict[str, Any], out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "resolved_config.json"
    payload = json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
    path.write_bytes(payload)
    digest = _sha256_bytes(payload)
    return {
        "resolved_config_path": str(path),
        "resolved_config_sha256": digest,
        "resolved_config_sha256_from_disk": _sha256_file(path),
    }


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    existing_rows: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for attempt in range(3):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows + rows)
            return
        except FileNotFoundError:
            if attempt == 2:
                raise
            time.sleep(0.2)


def _prepare_fresh_run_dirs(*, out_dir: Path, ckpt_dir: Path, overwrite: bool = False) -> None:
    existing = [path for path in (out_dir, ckpt_dir) if path.exists()]
    if existing and not overwrite:
        joined = "; ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refuse to overwrite existing Stage III run directories: {joined}. "
            "Use a new run_id/out-root/checkpoint-root for audit-safe reruns."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)


def _sensitivity_gate_row(
    *,
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    run_id: str,
    dataset_label: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    gate_cfg = cfg.get("stage3_v2", {}).get("claim_gate", {})
    gate_row = evaluate_stage3_sensitivity_gate(
        rows,
        min_real_mse_gain=float(gate_cfg.get("min_real_mse_gain", 0.0)),
        min_real_mae_gain=(
            float(gate_cfg["min_real_mae_gain"]) if "min_real_mae_gain" in gate_cfg else None
        ),
        counterfactual_margin=float(gate_cfg.get("counterfactual_margin", 0.0)),
        max_real_reject_rate=(
            float(gate_cfg["max_real_reject_rate"]) if "max_real_reject_rate" in gate_cfg else None
        ),
        require_nonzero_delta=bool(gate_cfg.get("require_nonzero_delta", True)),
        require_prediction_shift=bool(gate_cfg.get("require_prediction_shift", True)),
    )
    return {
        "run_id": run_id,
        "dataset": dataset_label,
        "seed": seed,
        "split": split,
        **gate_row,
    }


def _dataset_by_name(name: str) -> dict[str, Any]:
    for dataset in DATASETS:
        if dataset["name"] == name:
            return dataset
    raise ValueError(f"unknown public dataset: {name}")


def _extract_prediction(output: Any) -> torch.Tensor:
    return output.y_hat if hasattr(output, "y_hat") else output


def _apply_generator_residual_correction(generator, drift: torch.Tensor, y_candidate: torch.Tensor) -> torch.Tensor:
    if not hasattr(generator, "residual_correction"):
        return y_candidate
    return y_candidate + generator.residual_correction(drift, y_candidate)


def _apply_dataset_override(cfg: dict[str, Any], dataset_name: str | None) -> dict[str, Any]:
    if dataset_name is None:
        return cfg
    ds = _dataset_by_name(dataset_name)
    cfg = json.loads(json.dumps(cfg))
    cfg["data"]["path"] = ds["path"]
    cfg["data"]["timestamp_col"] = ds["timestamp_col"]
    cfg["data"]["csv_encoding"] = ds.get("encoding")
    cfg["data"]["target_dim"] = target_dim_for_dataset(ds)
    cfg["data"]["target_col"] = ds["target_col"]
    cfg["data"]["missing_sentinels"] = ds.get("missing_sentinels", [])
    cfg["data"]["impute_method"] = ds.get("impute_method", "drop")
    return cfg


def _stage3_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("online", {}).get("stage3", {})


def _stage3_drift_source(cfg: dict[str, Any]) -> str:
    return str(_stage3_cfg(cfg).get("drift_source", cfg.get("stage3_v2", {}).get("drift_source", "phi_gap")))


def _stage3_drift_polarity(cfg: dict[str, Any]) -> float:
    return float(_stage3_cfg(cfg).get("drift_polarity", cfg.get("stage3_v2", {}).get("drift_polarity", 1.0)))


def _stage3_calibrate_delta(cfg: dict[str, Any]) -> bool:
    return bool(_stage3_cfg(cfg).get("calibrate_delta", cfg.get("stage3_v2", {}).get("calibrate_delta", False)))


def _load_series_and_features(cfg: dict[str, Any]) -> tuple[torch.Tensor, list[str] | None, int]:
    if cfg["data"]["source"] == "synthetic":
        series = torch.randn(500, int(cfg["data"].get("synthetic_dim", 8)))
        feature_cols = None
        cfg["_audit_preprocessing_info"] = {
            "split_safe_imputation": True,
            "imputation_method": "synthetic",
            "imputation_uses_future": False,
        }
    else:
        feature_cols = cfg["data"]["feature_cols"] or None
        if bool(cfg.get("stage3_v2", {}).get("split_safe_imputation", False)):
            series, df, preprocessing_info = load_split_safe_tabular_timeseries(
                path=cfg["data"]["path"],
                timestamp_col=cfg["data"]["timestamp_col"],
                feature_cols=feature_cols,
                encoding=cfg["data"].get("csv_encoding") or None,
                missing_sentinels=cfg["data"].get("missing_sentinels"),
                impute_method=cfg.get("stage3_v2", {}).get(
                    "split_safe_impute_method",
                    "causal_ffill_train_median",
                ),
                train_ratio=float(cfg["data"]["train_ratio"]),
                val_ratio=float(cfg["data"]["val_ratio"]),
            )
        else:
            impute_method = cfg["data"].get("impute_method", "drop")
            series, df = load_tabular_timeseries(
                path=cfg["data"]["path"],
                timestamp_col=cfg["data"]["timestamp_col"],
                feature_cols=feature_cols,
                encoding=cfg["data"].get("csv_encoding") or None,
                missing_sentinels=cfg["data"].get("missing_sentinels"),
                impute_method=impute_method,
            )
            preprocessing_info = {
                "split_safe_imputation": False,
                "imputation_method": impute_method,
                "imputation_uses_future": impute_method in {"time_interpolate", "ffill_bfill"},
                "imputation_uses_bfill": impute_method in {"time_interpolate", "ffill_bfill"},
                "imputation_uses_cross_split_interpolation": impute_method == "time_interpolate",
            }
        cfg["_audit_preprocessing_info"] = preprocessing_info
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != cfg["data"]["timestamp_col"]]

    target_dim = cfg["data"].get("target_dim")
    target_col = cfg["data"].get("target_col")
    if target_col:
        if feature_cols is None or target_col not in feature_cols:
            raise ValueError(f"target_col '{target_col}' not found in feature columns")
        target_dim = feature_cols.index(target_col)
    if target_dim is None:
        raise ValueError("data.target_dim or data.target_col must be provided")
    return series, feature_cols, int(target_dim)


def _build_dataloaders(cfg: dict[str, Any], series: torch.Tensor, target_dim: int):
    if cfg["data"].get("normalize", True):
        split_idx = int(series.shape[0] * cfg["data"]["train_ratio"])
        scaler = StandardScaler.fit(series[:split_idx])
        series = scaler.transform(series)
    if cfg.get("stage3_v2", {}).get("causal_raw_split", True):
        return build_causal_dataloaders_from_series(
            series=series,
            lookback=cfg["data"]["lookback"],
            horizon=cfg["data"]["horizon"],
            target_dim=target_dim,
            batch_size=cfg["train"]["batch_size"],
            train_ratio=cfg["data"]["train_ratio"],
            val_ratio=cfg["data"]["val_ratio"],
            split_mode=cfg["data"].get("split_mode", "chronological"),
            test_ratio=cfg["data"].get("online_test_ratio"),
            shift_pool_ratio=cfg["data"].get("shift_pool_ratio"),
            drop_last=True,
            return_datasets=True,
            return_split_info=True,
        )
    return build_dataloaders_from_series(
        series=series,
        lookback=cfg["data"]["lookback"],
        horizon=cfg["data"]["horizon"],
        target_dim=target_dim,
        batch_size=cfg["train"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        split_mode=cfg["data"].get("split_mode", "chronological"),
        test_ratio=cfg["data"].get("online_test_ratio"),
        shift_pool_ratio=cfg["data"].get("shift_pool_ratio"),
        drop_last=True,
        return_datasets=True,
        return_split_info=True,
    )


def _build_model(cfg: dict[str, Any], series_dim: int, target_dim: int):
    mcfg = FactorMechanismV2Config(
        input_dim=series_dim,
        seq_len=cfg["data"]["lookback"],
        out_dim=cfg["data"]["horizon"],
        d_model=cfg["model"]["d_model"],
        n_experts=cfg["model"]["n_experts"],
        latent_dim=cfg["model"]["latent_dim"],
        temporal_encoder=cfg["model"]["temporal_encoder"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        use_revin=cfg["model"].get("use_revin", False),
        target_index=target_dim,
        use_vae=cfg["model"].get("use_vae", True),
        vae_eval_mode=cfg["model"].get("vae_eval_mode", "mean"),
    )
    return build_factor_mechanism_v2(mcfg)


def _train_base_model(cfg: dict[str, Any], model, train_dl, val_dl, device: torch.device) -> list[dict[str, Any]]:
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    sched = build_offline_scheduler(cfg, opt)
    rows = []
    for epoch in range(int(cfg["train"]["epochs"])):
        train_offline_factor(
            dataloader=train_dl,
            model=model,
            opt=opt,
            beta_vae=cfg["train"]["beta_vae"],
            lambda_disc=cfg["train"]["lambda_disc"],
            device=device,
            use_lrp=cfg["train"].get("use_lrp", False),
            lrp_method=cfg["train"].get("lrp_method", "rrp"),
            save_lrp=False,
            pred_loss_type=cfg["train"].get("pred_loss_type", "mse"),
            pred_mae_weight=cfg["train"].get("pred_mae_weight", 0.0),
            pred_huber_delta=cfg["train"].get("pred_huber_delta", 1.0),
        )
        val_mse, val_mae = evaluate_factor(val_dl, model, device)
        rows.append({"epoch": epoch, "val_mse": val_mse, "val_mae": val_mae})
        if sched is not None:
            sched.step()
    return rows


def _train_aligned_model(cfg: dict[str, Any], train_dl, val_dl, series_dim: int, device: torch.device):
    inv_cfg = cfg["invariant"]
    model = build_aligned_representation_model(
        input_dim=cfg["data"]["lookback"] * series_dim,
        repr_dim=int(inv_cfg["repr_dim"]),
        hidden_dim=int(inv_cfg.get("hidden_dim", 2 * int(inv_cfg["repr_dim"]))),
        out_dim=cfg["data"]["horizon"],
        dropout=float(inv_cfg.get("dropout", 0.0)),
        zero_init_variant=bool(inv_cfg.get("zero_init_variant", False)),
        zero_init_variant_regressor=bool(inv_cfg.get("zero_init_variant_regressor", False)),
        use_projected_residual=bool(inv_cfg.get("use_projected_residual", False)),
        stop_gradient_projection_anchor=bool(inv_cfg.get("stop_gradient_projection_anchor", True)),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    train_cfg = AlignedRepresentationTrainConfig(
        lambda_forecast=float(inv_cfg.get("lambda_forecast", 1.0)),
        lambda_residual=float(inv_cfg.get("lambda_residual", 0.5)),
        lambda_orth=float(inv_cfg.get("lambda_orth", 0.1)),
        lambda_smooth=float(inv_cfg.get("lambda_smooth", 0.2)),
        lambda_norm=float(inv_cfg.get("lambda_norm", 0.05)),
        lambda_hsic=float(inv_cfg.get("lambda_hsic", 0.0)),
        lambda_inv_forecast=float(inv_cfg.get("lambda_inv_forecast", 0.0)),
        lambda_barlow=float(inv_cfg.get("lambda_barlow", 0.0)),
        lambda_temporal_ranking=float(inv_cfg.get("lambda_temporal_ranking", 0.0)),
        temporal_ranking_margin=float(inv_cfg.get("temporal_ranking_margin", 0.01)),
        lambda_deployed_temporal_ranking=float(inv_cfg.get("lambda_deployed_temporal_ranking", 0.0)),
        deployed_temporal_ranking_margin=float(inv_cfg.get("deployed_temporal_ranking_margin", 0.01)),
        lambda_temporal_contrastive=float(inv_cfg.get("lambda_temporal_contrastive", 0.0)),
        temporal_contrastive_temperature=float(inv_cfg.get("temporal_contrastive_temperature", 1.0)),
        lambda_vicreg_variance=float(inv_cfg.get("lambda_vicreg_variance", 0.0)),
        lambda_vicreg_covariance=float(inv_cfg.get("lambda_vicreg_covariance", 0.0)),
        lambda_phi_vicreg_variance=float(inv_cfg.get("lambda_phi_vicreg_variance", 0.0)),
        lambda_phi_vicreg_covariance=float(inv_cfg.get("lambda_phi_vicreg_covariance", 0.0)),
        vicreg_variance_target=float(inv_cfg.get("vicreg_variance_target", 1.0)),
    )
    rows = []
    for epoch in range(int(inv_cfg.get("epochs", 1))):
        train_metrics = train_aligned_representation_epoch(train_dl, model, opt, train_cfg, device)
        val_metrics = evaluate_aligned_representation(val_dl, model, device)
        rows.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **val_metrics})
    model.eval()
    return model, rows


def _aligned_stream(
    dataset,
    batch_size: int,
    device: torch.device,
    aligned_model,
    drift_override: torch.Tensor | None = None,
    max_batches: int | None = None,
    drift_source: str = "phi_gap",
    delta_standardizer: TrainSetDeltaStandardizer | None = None,
    residual_canonicalizer: (
        TrainSetResidualDriftCanonicalizer
        | TrainSetTemporalResidualCCAAligner
        | TrainSetWhitenedProcrustesTemporalResidualCCAAligner
        | TrainSetChronologicalResidualizedReducedRankCCAAligner
        | None
    ) = None,
):
    if drift_source not in {"phi_gap", "delta_phi_v"}:
        raise ValueError(f"unknown Stage III V2 drift_source: {drift_source}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    cursor = 0
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            rep = aligned_model(x).representation
        if drift_override is None:
            if drift_source == "phi_gap":
                yield x, y, rep.phi_i, rep.phi_v
            else:
                if not hasattr(rep, "delta_phi_v"):
                    raise ValueError("drift_source='delta_phi_v' requires representation.delta_phi_v")
                batch_drift = rep.delta_phi_v
                if residual_canonicalizer is not None:
                    if hasattr(residual_canonicalizer, "transform_phi_pair"):
                        batch_drift = residual_canonicalizer.transform_phi_pair(rep.phi_i, rep.phi_v)
                    else:
                        batch_drift = residual_canonicalizer.transform_delta(batch_drift)
                if delta_standardizer is not None:
                    batch_drift = delta_standardizer.transform(batch_drift)
                yield x, y, torch.zeros_like(batch_drift), batch_drift
        else:
            batch_drift = drift_override[cursor : cursor + x.shape[0]].to(device)
            cursor += x.shape[0]
            yield x, y, torch.zeros_like(batch_drift), batch_drift


def _collect_raw_drift(
    dataset,
    batch_size: int,
    device: torch.device,
    aligned_model,
    max_batches: int | None = None,
    drift_source: str = "phi_gap",
    delta_standardizer: TrainSetDeltaStandardizer | None = None,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | None = None,
) -> torch.Tensor:
    parts = []
    for _, _, phi_i, phi_v in _aligned_stream(
        dataset,
        batch_size,
        device,
        aligned_model,
        max_batches=max_batches,
        drift_source=drift_source,
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
    ):
        parts.append((phi_v - phi_i).detach().cpu())
    return torch.cat(parts, dim=0)


def _collect_generator_input_drift(
    dataset,
    batch_size: int,
    device: torch.device,
    aligned_model,
    max_batches: int | None = None,
    drift_source: str = "phi_gap",
    delta_standardizer: TrainSetDeltaStandardizer | None = None,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | TrainSetTemporalResidualCCAAligner | None = None,
    signal_mode: str = "delta_vector",
    normalize_vector: bool = True,
    drift_polarity: float = 1.0,
) -> torch.Tensor:
    parts = []
    state = Stage3SignalState()
    for _, _, phi_i, phi_v in _aligned_stream(
        dataset,
        batch_size,
        device,
        aligned_model,
        max_batches=max_batches,
        drift_source=drift_source,
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
    ):
        signal_result = build_stage3_drift_vector(
            drift_raw=phi_v - phi_i,
            mode=signal_mode,
            bottleneck_dim=int((phi_v - phi_i).shape[-1]),
            state=state,
            normalize_vector=normalize_vector,
            polarity=drift_polarity,
        )
        parts.append(signal_result.drift_vec.detach().cpu().reshape(1, -1))
    if not parts:
        raise ValueError("collected no Stage III generator-input drift samples")
    return torch.cat(parts, dim=0)


def _fit_delta_standardizer(
    cfg: dict[str, Any],
    train_ds,
    batch_size: int,
    device: torch.device,
    aligned_model,
    max_batches: int | None = None,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | TrainSetTemporalResidualCCAAligner | None = None,
) -> TrainSetDeltaStandardizer | None:
    if not _stage3_calibrate_delta(cfg):
        return None
    if _stage3_drift_source(cfg) != "delta_phi_v":
        raise ValueError("calibrate_delta=True requires drift_source='delta_phi_v'")
    raw_delta = _collect_raw_drift(
        train_ds,
        batch_size,
        device,
        aligned_model,
        max_batches=max_batches,
        drift_source="delta_phi_v",
        delta_standardizer=None,
        residual_canonicalizer=residual_canonicalizer,
    )
    return TrainSetDeltaStandardizer(
        eps=float(_stage3_cfg(cfg).get("calibration_eps", 1e-6)),
    ).fit(raw_delta)


def _fit_v9_residual_teacher(
    *,
    cfg: dict[str, Any],
    train_ds,
    batch_size: int,
    device: torch.device,
    aligned_model,
    model,
    max_batches: int | None = None,
    delta_standardizer: TrainSetDeltaStandardizer | None = None,
    residual_canonicalizer=None,
) -> tuple[
    TrainOnlyResidualTeacherV9
    | TrainOnlyResidualTeacherV10
    | TrainOnlyRobustBlockResidualTeacherV11
    | TrainOnlyOutcomeAlignedBlockResidualTeacherV12
    | None,
    list[dict[str, Any]],
]:
    teacher_cfg = cfg.get("stage3_v2", {}).get("residual_teacher", {})
    if not bool(teacher_cfg.get("enabled", False)):
        return None, []
    if batch_size != 1:
        raise ValueError("Stage III V9/V10/V11/V12 residual teacher fitting requires online_batch_size=1")

    drift_source = _stage3_drift_source(cfg)
    drift_polarity = _stage3_drift_polarity(cfg)
    signal_state = Stage3SignalState()
    drift_parts: list[torch.Tensor] = []
    y_base_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    sample_ids: list[int] = []
    model.eval()
    for step, (x, y, phi_i, phi_v) in enumerate(
        _aligned_stream(
            train_ds,
            batch_size,
            device,
            aligned_model,
            max_batches=max_batches,
            drift_source=drift_source,
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
        )
    ):
        with torch.no_grad():
            y_base = _extract_prediction(model(x))
            signal_result = build_stage3_drift_vector(
                drift_raw=phi_v - phi_i,
                mode=cfg["online"]["stage3"].get("signal_mode", "delta_vector"),
                bottleneck_dim=int((phi_v - phi_i).shape[-1]),
                state=signal_state,
                normalize_vector=bool(cfg["online"]["stage3"].get("normalize_vector", True)),
                polarity=drift_polarity,
            )
        drift_parts.append(signal_result.drift_vec.detach().cpu().reshape(1, -1))
        y_base_parts.append(y_base.detach().cpu())
        y_parts.append(y.detach().cpu())
        sample_ids.append(step)
    if not drift_parts:
        raise ValueError("Stage III V9/V10/V11/V12 residual teacher collected no train samples")

    drift_train = torch.cat(drift_parts, dim=0)
    y_base_train = torch.cat(y_base_parts, dim=0)
    y_train = torch.cat(y_parts, dim=0)
    teacher_version = str(
        teacher_cfg.get(
            "version",
            cfg.get("stage3_v2", {}).get("protocol_version", ""),
        )
    ).lower()
    if teacher_version in {
        "v12",
        "v12_outcome_aligned_block_residual_teacher",
        "v12_outcome_aligned_teacher_gate",
    }:
        teacher = TrainOnlyOutcomeAlignedBlockResidualTeacherV12(
            drift_dim=drift_train.shape[1],
            output_shape=tuple(y_train.shape[1:]),
            ridge=float(teacher_cfg.get("ridge", 1e-3)),
            inner_fit_fraction=float(teacher_cfg.get("inner_fit_fraction", 0.75)),
            shrinkage_grid=teacher_cfg.get("shrinkage_grid"),
            min_calibration_gain=float(teacher_cfg.get("min_calibration_gain", 0.0)),
            fallback_alpha=float(teacher_cfg.get("fallback_alpha", 0.0)),
            alpha_mode=str(teacher_cfg.get("alpha_mode", "block_scalar")),
            allow_negative_alpha=bool(teacher_cfg.get("allow_negative_alpha", False)),
            calibration_mae_weight=float(teacher_cfg.get("calibration_mae_weight", 0.0)),
            calibration_block_count=int(teacher_cfg.get("calibration_block_count", 5)),
            min_block_mse_gain=float(teacher_cfg.get("min_block_mse_gain", 0.0)),
            min_block_mae_gain=float(teacher_cfg.get("min_block_mae_gain", 0.0)),
            min_block_sign_agreement=float(teacher_cfg.get("min_block_sign_agreement", 0.5)),
            reliability_gate_enabled=bool(teacher_cfg.get("reliability_gate_enabled", True)),
            reliability_gate_quantile=float(teacher_cfg.get("reliability_gate_quantile", 0.9)),
            min_reliable_fraction=float(teacher_cfg.get("min_reliable_fraction", 0.1)),
            disable_if_unreliable=bool(teacher_cfg.get("disable_if_unreliable", True)),
        ).fit(
            drift_train=drift_train,
            y_base_train=y_base_train,
            y_train=y_train,
            split="train",
            sample_ids=torch.tensor(sample_ids, dtype=torch.long),
        )
    elif teacher_version in {"v11", "v11_block_calibrated_residual_teacher", "v11_block_calibrated_teacher_gate"}:
        teacher = TrainOnlyRobustBlockResidualTeacherV11(
            drift_dim=drift_train.shape[1],
            output_shape=tuple(y_train.shape[1:]),
            ridge=float(teacher_cfg.get("ridge", 1e-3)),
            inner_fit_fraction=float(teacher_cfg.get("inner_fit_fraction", 0.75)),
            shrinkage_grid=teacher_cfg.get("shrinkage_grid"),
            min_calibration_gain=float(teacher_cfg.get("min_calibration_gain", 0.0)),
            fallback_alpha=float(teacher_cfg.get("fallback_alpha", 0.0)),
            alpha_mode=str(teacher_cfg.get("alpha_mode", "block_scalar")),
            allow_negative_alpha=bool(teacher_cfg.get("allow_negative_alpha", False)),
            calibration_mae_weight=float(teacher_cfg.get("calibration_mae_weight", 0.25)),
            calibration_block_count=int(teacher_cfg.get("calibration_block_count", 5)),
            min_block_mse_gain=float(teacher_cfg.get("min_block_mse_gain", 0.0)),
            min_block_mae_gain=float(teacher_cfg.get("min_block_mae_gain", 0.0)),
            min_block_sign_agreement=float(teacher_cfg.get("min_block_sign_agreement", 0.52)),
            reliability_gate_enabled=bool(teacher_cfg.get("reliability_gate_enabled", True)),
            reliability_gate_quantile=float(teacher_cfg.get("reliability_gate_quantile", 0.9)),
            min_reliable_fraction=float(teacher_cfg.get("min_reliable_fraction", 0.1)),
            disable_if_unreliable=bool(teacher_cfg.get("disable_if_unreliable", True)),
        ).fit(
            drift_train=drift_train,
            y_base_train=y_base_train,
            y_train=y_train,
            split="train",
            sample_ids=torch.tensor(sample_ids, dtype=torch.long),
        )
    elif teacher_version in {"v10", "v10_crossfit_teacher_projection_served_policy_gate"}:
        teacher = TrainOnlyResidualTeacherV10(
            drift_dim=drift_train.shape[1],
            output_shape=tuple(y_train.shape[1:]),
            ridge=float(teacher_cfg.get("ridge", 1e-3)),
            inner_fit_fraction=float(teacher_cfg.get("inner_fit_fraction", 0.75)),
            shrinkage_grid=teacher_cfg.get("shrinkage_grid"),
            min_calibration_gain=float(teacher_cfg.get("min_calibration_gain", 0.0)),
            fallback_alpha=float(teacher_cfg.get("fallback_alpha", 0.0)),
            alpha_mode=str(teacher_cfg.get("alpha_mode", "scalar_nonnegative")),
            allow_negative_alpha=bool(teacher_cfg.get("allow_negative_alpha", False)),
            calibration_mae_weight=float(teacher_cfg.get("calibration_mae_weight", 0.0)),
        ).fit(
            drift_train=drift_train,
            y_base_train=y_base_train,
            y_train=y_train,
            split="train",
            sample_ids=torch.tensor(sample_ids, dtype=torch.long),
        )
    elif teacher_version in {"v9", "v9_residual_teacher_rrcca_served_policy_gate"}:
        teacher = TrainOnlyResidualTeacherV9(
            drift_dim=drift_train.shape[1],
            output_shape=tuple(y_train.shape[1:]),
            ridge=float(teacher_cfg.get("ridge", 1e-3)),
        ).fit(
            drift_train=drift_train,
            y_base_train=y_base_train,
            y_train=y_train,
            split="train",
            sample_ids=torch.tensor(sample_ids, dtype=torch.long),
        )
    else:
        raise ValueError(f"unknown Stage III residual_teacher.version: {teacher_version!r}")
    rows = [{"split": "train", **teacher.diagnostic_summary()}]
    return teacher, rows


def _stage3_canonicalize_delta(cfg: dict[str, Any]) -> bool:
    return bool(_stage3_cfg(cfg).get("canonicalize_delta", cfg.get("stage3_v2", {}).get("canonicalize_delta", False)))


def _fit_residual_canonicalizer(
    cfg: dict[str, Any],
    train_ds,
    batch_size: int,
    device: torch.device,
    aligned_model,
    max_batches: int | None = None,
) -> (
    TrainSetResidualDriftCanonicalizer
    | TrainSetTemporalResidualCCAAligner
    | TrainSetWhitenedProcrustesTemporalResidualCCAAligner
    | TrainSetChronologicalResidualizedReducedRankCCAAligner
    | None
):
    if not _stage3_canonicalize_delta(cfg):
        return None
    if _stage3_drift_source(cfg) != "delta_phi_v":
        raise ValueError("canonicalize_delta=True requires drift_source='delta_phi_v'")
    delta_parts: list[torch.Tensor] = []
    phi_i_parts: list[torch.Tensor] = []
    phi_v_parts: list[torch.Tensor] = []
    y_inv_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = aligned_model(x)
        rep = out.representation
        delta_parts.append(rep.delta_phi_v.detach().cpu())
        if hasattr(rep, "phi_i") and hasattr(rep, "phi_v"):
            phi_i_parts.append(rep.phi_i.detach().cpu())
            phi_v_parts.append(rep.phi_v.detach().cpu())
        y_inv_parts.append(out.y_inv.detach().cpu())
        y_parts.append(y.detach().cpu())
    if not delta_parts:
        raise ValueError("residual drift canonicalizer collected no train samples")
    method = str(_stage3_cfg(cfg).get("canonicalizer_method", cfg.get("stage3_v2", {}).get("canonicalizer_method", "pls_v2")))
    ridge = float(_stage3_cfg(cfg).get("canonicalizer_ridge", cfg.get("stage3_v2", {}).get("canonicalizer_ridge", 1e-3)))
    eps = float(_stage3_cfg(cfg).get("canonicalizer_eps", cfg.get("stage3_v2", {}).get("canonicalizer_eps", 1e-6)))
    if method in {"temporal_residual_cca", "temporal_residual_cca_v3", "cca_v3"}:
        canonicalizer = TrainSetTemporalResidualCCAAligner(
            ridge=ridge,
            eps=eps,
            scale_by_correlation=bool(
                _stage3_cfg(cfg).get(
                    "canonicalizer_scale_by_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_scale_by_correlation", False),
                )
            ),
        )
    elif method in {
        "whitened_procrustes_temporal_residual_cca_v6",
        "procrustes_temporal_residual_cca_v6",
        "v6_whitened_procrustes_cca",
    }:
        if len(phi_i_parts) != len(delta_parts) or len(phi_v_parts) != len(delta_parts):
            raise ValueError("V6 phi-space canonicalizer requires representation.phi_i and representation.phi_v")
        return TrainSetWhitenedProcrustesTemporalResidualCCAAligner(
            ridge=ridge,
            eps=eps,
            scale_by_correlation=bool(
                _stage3_cfg(cfg).get(
                    "canonicalizer_scale_by_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_scale_by_correlation", False),
                )
            ),
            orthogonalize_delta=bool(
                _stage3_cfg(cfg).get(
                    "canonicalizer_orthogonalize_delta",
                    cfg.get("stage3_v2", {}).get("canonicalizer_orthogonalize_delta", True),
                )
            ),
        ).fit(
            phi_i_train=torch.cat(phi_i_parts, dim=0),
            phi_v_train=torch.cat(phi_v_parts, dim=0),
            y_inv_train=torch.cat(y_inv_parts, dim=0),
            y_train=torch.cat(y_parts, dim=0),
        )
    elif method in {"residualized_rrcca_v7", "chronological_residualized_rrcca_v7", "v7_residualized_rrcca"}:
        if len(phi_i_parts) != len(delta_parts) or len(phi_v_parts) != len(delta_parts):
            raise ValueError("V7 residualized RRCCA canonicalizer requires representation.phi_i and representation.phi_v")
        stage3_cfg = _stage3_cfg(cfg)
        max_rank_value = stage3_cfg.get("canonicalizer_max_rank", cfg.get("stage3_v2", {}).get("canonicalizer_max_rank"))
        max_rank = None if max_rank_value is None else int(max_rank_value)
        return TrainSetChronologicalResidualizedReducedRankCCAAligner(
            ridge=ridge,
            eps=eps,
            inner_fit_fraction=float(
                stage3_cfg.get(
                    "canonicalizer_inner_fit_fraction",
                    cfg.get("stage3_v2", {}).get("canonicalizer_inner_fit_fraction", 0.75),
                )
            ),
            max_rank=max_rank,
            min_fit_correlation=float(
                stage3_cfg.get(
                    "canonicalizer_min_fit_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_fit_correlation", 0.05),
                )
            ),
            min_calibration_correlation=float(
                stage3_cfg.get(
                    "canonicalizer_min_calibration_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_calibration_correlation", 0.05),
                )
            ),
            min_calibration_r2=float(
                stage3_cfg.get(
                    "canonicalizer_min_calibration_r2",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_calibration_r2", 0.0025),
                )
            ),
            bootstrap_samples=int(
                stage3_cfg.get(
                    "canonicalizer_bootstrap_samples",
                    cfg.get("stage3_v2", {}).get("canonicalizer_bootstrap_samples", 64),
                )
            ),
            min_bootstrap_sign_stability=float(
                stage3_cfg.get(
                    "canonicalizer_min_bootstrap_sign_stability",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_bootstrap_sign_stability", 0.6),
                )
            ),
            bootstrap_seed=int(
                stage3_cfg.get(
                    "canonicalizer_bootstrap_seed",
                    cfg.get("stage3_v2", {}).get("canonicalizer_bootstrap_seed", 1729),
                )
            ),
            scale_by_correlation=bool(
                stage3_cfg.get(
                    "canonicalizer_scale_by_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_scale_by_correlation", False),
                )
            ),
        ).fit(
            phi_i_train=torch.cat(phi_i_parts, dim=0),
            phi_v_train=torch.cat(phi_v_parts, dim=0),
            y_inv_train=torch.cat(y_inv_parts, dim=0),
            y_train=torch.cat(y_parts, dim=0),
        )
    elif method in {"teacher_residual_rrcca_v9", "v9_teacher_residual_rrcca"}:
        if len(phi_i_parts) != len(delta_parts) or len(phi_v_parts) != len(delta_parts):
            raise ValueError("V9 teacher residual RRCCA canonicalizer requires representation.phi_i and representation.phi_v")
        stage3_cfg = _stage3_cfg(cfg)
        max_rank_value = stage3_cfg.get("canonicalizer_max_rank", cfg.get("stage3_v2", {}).get("canonicalizer_max_rank"))
        max_rank = None if max_rank_value is None else int(max_rank_value)
        return TrainSetTeacherResidualizedReducedRankCCAAligner(
            ridge=ridge,
            eps=eps,
            inner_fit_fraction=float(
                stage3_cfg.get(
                    "canonicalizer_inner_fit_fraction",
                    cfg.get("stage3_v2", {}).get("canonicalizer_inner_fit_fraction", 0.75),
                )
            ),
            max_rank=max_rank,
            min_fit_correlation=float(
                stage3_cfg.get(
                    "canonicalizer_min_fit_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_fit_correlation", 0.05),
                )
            ),
            min_calibration_correlation=float(
                stage3_cfg.get(
                    "canonicalizer_min_calibration_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_calibration_correlation", 0.05),
                )
            ),
            min_calibration_r2=float(
                stage3_cfg.get(
                    "canonicalizer_min_calibration_r2",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_calibration_r2", 0.0025),
                )
            ),
            bootstrap_samples=int(
                stage3_cfg.get(
                    "canonicalizer_bootstrap_samples",
                    cfg.get("stage3_v2", {}).get("canonicalizer_bootstrap_samples", 64),
                )
            ),
            min_bootstrap_sign_stability=float(
                stage3_cfg.get(
                    "canonicalizer_min_bootstrap_sign_stability",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_bootstrap_sign_stability", 0.6),
                )
            ),
            bootstrap_seed=int(
                stage3_cfg.get(
                    "canonicalizer_bootstrap_seed",
                    cfg.get("stage3_v2", {}).get("canonicalizer_bootstrap_seed", 1729),
                )
            ),
            scale_by_correlation=bool(
                stage3_cfg.get(
                    "canonicalizer_scale_by_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_scale_by_correlation", False),
                )
            ),
        ).fit(
            phi_i_train=torch.cat(phi_i_parts, dim=0),
            phi_v_train=torch.cat(phi_v_parts, dim=0),
            y_base_train=torch.cat(y_inv_parts, dim=0),
            y_train=torch.cat(y_parts, dim=0),
        )
    elif method in {
        "outcome_aligned_transport_v12",
        "v12_outcome_aligned_transport",
        "shared_anchor_residual_transfer_v12",
    }:
        if len(phi_i_parts) != len(delta_parts) or len(phi_v_parts) != len(delta_parts):
            raise ValueError("V12 outcome-aligned transport canonicalizer requires representation.phi_i and representation.phi_v")
        stage3_cfg = _stage3_cfg(cfg)
        max_rank_value = stage3_cfg.get("canonicalizer_max_rank", cfg.get("stage3_v2", {}).get("canonicalizer_max_rank"))
        max_rank = None if max_rank_value is None else int(max_rank_value)
        anchor_rank_value = stage3_cfg.get("canonicalizer_anchor_rank", cfg.get("stage3_v2", {}).get("canonicalizer_anchor_rank"))
        anchor_rank = None if anchor_rank_value is None else int(anchor_rank_value)
        return TrainOnlyOutcomeAlignedTransportAlignerV12(
            ridge=ridge,
            eps=eps,
            anchor_rank=anchor_rank,
            max_rank=max_rank,
            inner_fit_fraction=float(
                stage3_cfg.get(
                    "canonicalizer_inner_fit_fraction",
                    cfg.get("stage3_v2", {}).get("canonicalizer_inner_fit_fraction", 0.75),
                )
            ),
            min_fit_correlation=float(
                stage3_cfg.get(
                    "canonicalizer_min_fit_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_fit_correlation", 0.05),
                )
            ),
            min_calibration_r2=float(
                stage3_cfg.get(
                    "canonicalizer_min_calibration_r2",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_calibration_r2", 0.0025),
                )
            ),
            bootstrap_samples=int(
                stage3_cfg.get(
                    "canonicalizer_bootstrap_samples",
                    cfg.get("stage3_v2", {}).get("canonicalizer_bootstrap_samples", 64),
                )
            ),
            min_bootstrap_sign_stability=float(
                stage3_cfg.get(
                    "canonicalizer_min_bootstrap_sign_stability",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_bootstrap_sign_stability", 0.6),
                )
            ),
            bootstrap_seed=int(
                stage3_cfg.get(
                    "canonicalizer_bootstrap_seed",
                    cfg.get("stage3_v2", {}).get("canonicalizer_bootstrap_seed", 1729),
                )
            ),
            leakage_fold_count=int(
                stage3_cfg.get(
                    "canonicalizer_leakage_fold_count",
                    cfg.get("stage3_v2", {}).get("canonicalizer_leakage_fold_count", 5),
                )
            ),
            max_leakage_rank=int(
                stage3_cfg.get(
                    "canonicalizer_max_leakage_rank",
                    cfg.get("stage3_v2", {}).get("canonicalizer_max_leakage_rank", 16),
                )
            ),
            min_prefix_samples=int(
                stage3_cfg.get(
                    "canonicalizer_min_prefix_samples",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_prefix_samples", 16),
                )
            ),
            min_retained_variance_fraction=float(
                stage3_cfg.get(
                    "canonicalizer_min_retained_variance_fraction",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_retained_variance_fraction", 0.05),
                )
            ),
            max_invariant_predictability_r2=float(
                stage3_cfg.get(
                    "canonicalizer_max_invariant_predictability_r2",
                    cfg.get("stage3_v2", {}).get("canonicalizer_max_invariant_predictability_r2", 0.98),
                )
            ),
            min_residual_predictability_r2=float(
                stage3_cfg.get(
                    "canonicalizer_min_residual_predictability_r2",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_residual_predictability_r2", -1.0),
                )
            ),
            leakage_penalty=float(
                stage3_cfg.get(
                    "canonicalizer_leakage_penalty",
                    cfg.get("stage3_v2", {}).get("canonicalizer_leakage_penalty", 0.1),
                )
            ),
            leakage_removal_reward=float(
                stage3_cfg.get(
                    "canonicalizer_leakage_removal_reward",
                    cfg.get("stage3_v2", {}).get("canonicalizer_leakage_removal_reward", 0.01),
                )
            ),
            transfer_gain_reward=float(
                stage3_cfg.get(
                    "canonicalizer_transfer_gain_reward",
                    cfg.get("stage3_v2", {}).get("canonicalizer_transfer_gain_reward", 1.0),
                )
            ),
            rank_penalty=float(
                stage3_cfg.get(
                    "canonicalizer_rank_penalty",
                    cfg.get("stage3_v2", {}).get("canonicalizer_rank_penalty", 0.001),
                )
            ),
            direct_block_count=int(
                stage3_cfg.get(
                    "canonicalizer_direct_block_count",
                    cfg.get("stage3_v2", {}).get("canonicalizer_direct_block_count", 5),
                )
            ),
            min_block_transfer_r2=float(
                stage3_cfg.get(
                    "canonicalizer_min_block_transfer_r2",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_block_transfer_r2", 0.0),
                )
            ),
            min_block_sign_agreement=float(
                stage3_cfg.get(
                    "canonicalizer_min_block_sign_agreement",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_block_sign_agreement", 0.55),
                )
            ),
            reliability_gate_enabled=bool(
                stage3_cfg.get(
                    "canonicalizer_reliability_gate_enabled",
                    cfg.get("stage3_v2", {}).get("canonicalizer_reliability_gate_enabled", True),
                )
            ),
            reliability_quantile=float(
                stage3_cfg.get(
                    "canonicalizer_reliability_quantile",
                    cfg.get("stage3_v2", {}).get("canonicalizer_reliability_quantile", 0.95),
                )
            ),
            min_reliable_fraction=float(
                stage3_cfg.get(
                    "canonicalizer_min_reliable_fraction",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_reliable_fraction", 0.1),
                )
            ),
        ).fit(
            phi_i_train=torch.cat(phi_i_parts, dim=0),
            phi_v_train=torch.cat(phi_v_parts, dim=0),
            y_base_train=torch.cat(y_inv_parts, dim=0),
            y_train=torch.cat(y_parts, dim=0),
            split="train",
        )
    elif method in {
        "cross_fitted_teacher_projection_rrcca_v10",
        "teacher_residual_rrcca_v10",
        "v10_cross_fitted_projection_rrcca",
        "cross_fitted_residual_transfer_pls_v11",
        "teacher_residual_pls_v11",
        "v11_residual_transfer_pls",
    }:
        if len(phi_i_parts) != len(delta_parts) or len(phi_v_parts) != len(delta_parts):
            raise ValueError("V10/V11 cross-fitted projection canonicalizer requires representation.phi_i and representation.phi_v")
        stage3_cfg = _stage3_cfg(cfg)
        max_rank_value = stage3_cfg.get("canonicalizer_max_rank", cfg.get("stage3_v2", {}).get("canonicalizer_max_rank"))
        max_rank = None if max_rank_value is None else int(max_rank_value)
        is_v11_canonicalizer = "v11" in method
        aligner_cls = (
            TrainSetCrossFittedResidualTransferPLSAlignerV11
            if is_v11_canonicalizer
            else TrainSetCrossFittedTeacherResidualProjectionCCAAligner
        )
        v11_kwargs = {}
        if is_v11_canonicalizer:
            v11_kwargs = {
                "direct_block_count": int(
                    stage3_cfg.get(
                        "canonicalizer_direct_block_count",
                        cfg.get("stage3_v2", {}).get("canonicalizer_direct_block_count", 5),
                    )
                ),
                "min_direct_calibration_correlation": float(
                    stage3_cfg.get(
                        "canonicalizer_min_direct_calibration_correlation",
                        cfg.get("stage3_v2", {}).get("canonicalizer_min_direct_calibration_correlation", 0.03),
                    )
                ),
                "min_block_transfer_r2": float(
                    stage3_cfg.get(
                        "canonicalizer_min_block_transfer_r2",
                        cfg.get("stage3_v2", {}).get("canonicalizer_min_block_transfer_r2", 0.0),
                    )
                ),
            }
        return aligner_cls(
            ridge=ridge,
            eps=eps,
            inner_fit_fraction=float(
                stage3_cfg.get(
                    "canonicalizer_inner_fit_fraction",
                    cfg.get("stage3_v2", {}).get("canonicalizer_inner_fit_fraction", 0.75),
                )
            ),
            max_rank=max_rank,
            min_fit_correlation=float(
                stage3_cfg.get(
                    "canonicalizer_min_fit_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_fit_correlation", 0.05),
                )
            ),
            min_calibration_correlation=float(
                stage3_cfg.get(
                    "canonicalizer_min_calibration_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_calibration_correlation", 0.05),
                )
            ),
            min_calibration_r2=float(
                stage3_cfg.get(
                    "canonicalizer_min_calibration_r2",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_calibration_r2", 0.0025),
                )
            ),
            bootstrap_samples=int(
                stage3_cfg.get(
                    "canonicalizer_bootstrap_samples",
                    cfg.get("stage3_v2", {}).get("canonicalizer_bootstrap_samples", 64),
                )
            ),
            min_bootstrap_sign_stability=float(
                stage3_cfg.get(
                    "canonicalizer_min_bootstrap_sign_stability",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_bootstrap_sign_stability", 0.6),
                )
            ),
            bootstrap_seed=int(
                stage3_cfg.get(
                    "canonicalizer_bootstrap_seed",
                    cfg.get("stage3_v2", {}).get("canonicalizer_bootstrap_seed", 1729),
                )
            ),
            scale_by_correlation=bool(
                stage3_cfg.get(
                    "canonicalizer_scale_by_correlation",
                    cfg.get("stage3_v2", {}).get("canonicalizer_scale_by_correlation", False),
                )
            ),
            leakage_fold_count=int(
                stage3_cfg.get(
                    "canonicalizer_leakage_fold_count",
                    cfg.get("stage3_v2", {}).get("canonicalizer_leakage_fold_count", 5),
                )
            ),
            max_leakage_rank=int(
                stage3_cfg.get(
                    "canonicalizer_max_leakage_rank",
                    cfg.get("stage3_v2", {}).get("canonicalizer_max_leakage_rank", 16),
                )
            ),
            min_prefix_samples=int(
                stage3_cfg.get(
                    "canonicalizer_min_prefix_samples",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_prefix_samples", 16),
                )
            ),
            min_retained_variance_fraction=float(
                stage3_cfg.get(
                    "canonicalizer_min_retained_variance_fraction",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_retained_variance_fraction", 0.2),
                )
            ),
            max_invariant_predictability_r2=float(
                stage3_cfg.get(
                    "canonicalizer_max_invariant_predictability_r2",
                    cfg.get("stage3_v2", {}).get("canonicalizer_max_invariant_predictability_r2", 0.98),
                )
            ),
            min_teacher_residual_r2=float(
                stage3_cfg.get(
                    "canonicalizer_min_teacher_residual_r2",
                    cfg.get("stage3_v2", {}).get("canonicalizer_min_teacher_residual_r2", -1.0),
                )
            ),
            leakage_penalty=float(
                stage3_cfg.get(
                    "canonicalizer_leakage_penalty",
                    cfg.get("stage3_v2", {}).get("canonicalizer_leakage_penalty", 0.1),
                )
            ),
            leakage_removal_reward=float(
                stage3_cfg.get(
                    "canonicalizer_leakage_removal_reward",
                    cfg.get("stage3_v2", {}).get("canonicalizer_leakage_removal_reward", 0.01),
                )
            ),
            collapse_penalty=float(
                stage3_cfg.get(
                    "canonicalizer_collapse_penalty",
                    cfg.get("stage3_v2", {}).get("canonicalizer_collapse_penalty", 0.5),
                )
            ),
            rank_penalty=float(
                stage3_cfg.get(
                    "canonicalizer_rank_penalty",
                    cfg.get("stage3_v2", {}).get("canonicalizer_rank_penalty", 0.001),
                )
            ),
            **v11_kwargs,
        ).fit(
            phi_i_train=torch.cat(phi_i_parts, dim=0),
            phi_v_train=torch.cat(phi_v_parts, dim=0),
            y_base_train=torch.cat(y_inv_parts, dim=0),
            y_train=torch.cat(y_parts, dim=0),
        )
    elif method in {"pls_v2", "cross_covariance_v2", "residual_drift_canonicalizer"}:
        canonicalizer = TrainSetResidualDriftCanonicalizer(
            ridge=ridge,
            eps=eps,
        )
    else:
        raise ValueError(f"unsupported Stage III residual canonicalizer method: {method}")
    return canonicalizer.fit(
        delta_phi_v_train=torch.cat(delta_parts, dim=0),
        y_inv_train=torch.cat(y_inv_parts, dim=0),
        y_train=torch.cat(y_parts, dim=0),
    )


def _alignment_diagnostic_rows(
    *,
    dataset,
    split: str,
    batch_size: int,
    device: torch.device,
    aligned_model,
    max_batches: int | None,
    residual_canonicalizer: (
        TrainSetResidualDriftCanonicalizer
        | TrainSetTemporalResidualCCAAligner
        | TrainSetWhitenedProcrustesTemporalResidualCCAAligner
        | TrainSetChronologicalResidualizedReducedRankCCAAligner
        | None
    ),
) -> list[dict[str, Any]]:
    delta_parts: list[torch.Tensor] = []
    phi_i_parts: list[torch.Tensor] = []
    phi_v_parts: list[torch.Tensor] = []
    y_inv_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            out = aligned_model(x)
        rep = out.representation
        delta_parts.append(rep.delta_phi_v.detach().cpu())
        if hasattr(rep, "phi_i") and hasattr(rep, "phi_v"):
            phi_i_parts.append(rep.phi_i.detach().cpu())
            phi_v_parts.append(rep.phi_v.detach().cpu())
        y_inv_parts.append(out.y_inv.detach().cpu())
        y_parts.append(y.detach().cpu())
    if not delta_parts:
        return []

    delta = torch.cat(delta_parts, dim=0)
    phi_i = torch.cat(phi_i_parts, dim=0) if phi_i_parts else None
    phi_v = torch.cat(phi_v_parts, dim=0) if phi_v_parts else None
    y_inv = torch.cat(y_inv_parts, dim=0)
    y = torch.cat(y_parts, dim=0)
    rows = [
        {
            "split": split,
            "alignment_stage": "pre_canonical",
            **{
                key.replace("pre_", ""): value
                for key, value in temporal_residual_alignment_diagnostics(
                    delta_phi_v=delta,
                    y_inv=y_inv,
                    y=y,
                    prefix="pre",
                ).items()
            },
        }
    ]
    if residual_canonicalizer is not None:
        if hasattr(residual_canonicalizer, "transform_phi_pair") and phi_i is not None and phi_v is not None:
            post_delta = residual_canonicalizer.transform_phi_pair(phi_i, phi_v)
            post = temporal_residual_alignment_diagnostics(
                delta_phi_v=post_delta,
                y_inv=y_inv,
                y=y,
                prefix="post",
            )
            applied_via_phi_pair = True
        else:
            post = temporal_residual_alignment_diagnostics(
                delta_phi_v=delta,
                y_inv=y_inv,
                y=y,
                aligner=residual_canonicalizer,
                prefix="post",
            )
            applied_via_phi_pair = False
        row = {
            "split": split,
            "alignment_stage": "post_canonical",
            "canonicalizer_applied_via_phi_pair": applied_via_phi_pair,
            **{key.replace("post_", ""): value for key, value in post.items()},
        }
        correlations = getattr(residual_canonicalizer, "canonical_correlations", None)
        if correlations is not None:
            corr = correlations.detach().cpu().reshape(-1)
            row.update(
                {
                    "cca_corr_top1": float(corr[0].item()) if corr.numel() else 0.0,
                    "cca_corr_top3_mean": float(corr[: min(3, corr.numel())].mean().item()) if corr.numel() else 0.0,
                    "cca_corr_nonzero_count": float(torch.sum(corr > 1e-6).item()),
                }
            )
        if hasattr(residual_canonicalizer, "diagnostic_summary"):
            summary = dict(residual_canonicalizer.diagnostic_summary())
            if "fit_sample_count" in summary:
                summary["aligner_fit_sample_count"] = summary.pop("fit_sample_count")
            if "transition_sample_count" in summary:
                summary["aligner_transition_sample_count"] = summary.pop("transition_sample_count")
            row.update(summary)
        rows.append(row)
    return rows


def _format_amplified_negative_name(factor: float) -> str:
    factor = float(factor)
    if factor.is_integer():
        return f"amplified_{int(factor)}x"
    return f"amplified_{str(factor).replace('.', 'p')}x"


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
    if bool(gen_cfg.get("include_random_matched_norm_negative", True)):
        random_direction = torch.randn_like(drift_vec)
        random_direction = torch.nn.functional.normalize(random_direction.flatten(), dim=0).reshape_as(drift_vec)
        negative_drifts["random_matched_norm"] = random_direction * torch.linalg.vector_norm(drift_vec)
    for factor in gen_cfg.get("amplified_negative_factors", ()) or ():
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


def _train_generator(
    cfg: dict[str, Any],
    base_model,
    aligned_model,
    train_ds,
    device: torch.device,
    max_batches: int | None = None,
    delta_standardizer: TrainSetDeltaStandardizer | None = None,
    candidate_checkpoint_dir: Path | None = None,
    candidate_metadata: dict[str, Any] | None = None,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | None = None,
    residual_teacher: TrainOnlyResidualTeacherV9
    | TrainOnlyResidualTeacherV10
    | TrainOnlyRobustBlockResidualTeacherV11
    | TrainOnlyOutcomeAlignedBlockResidualTeacherV12
    | None = None,
):
    gen_cfg = cfg["stage3_v2"]["generator"]
    repr_dim = int(cfg["invariant"]["repr_dim"])
    drift_source = _stage3_drift_source(cfg)
    drift_polarity = _stage3_drift_polarity(cfg)
    target_prefixes = tuple(cfg["online"]["stage3"].get("target_modules") or ())
    target_names = tuple(gen_cfg.get("target_names") or cfg["online"]["stage3"].get("target_names") or ("weight",))
    generator_type = str(gen_cfg.get("type", "layer_aware_hyperlora")).lower()
    generator_supports_bias = generator_type in {
        "layer_aware_hyperadapter",
        "layer_aware_hyperadapter_v8",
        "hyperadapter_v8",
        "residual_basis_hypergenerator_v9",
        "residual_basis_hypergenerator",
        "residual_basis_hypergenerator_v10",
    }
    named_shapes = {
        name: tuple(param.shape)
        for name, param in base_model.named_parameters()
        if (param.ndim in {1, 2} if generator_supports_bias else param.ndim == 2)
        and (not target_names or any(name.endswith(target_name) for target_name in target_names))
        and ((not target_prefixes) or any(name.startswith(prefix) for prefix in target_prefixes))
    }
    if not named_shapes:
        raise RuntimeError("Stage III V2 generator found no target parameters")
    hidden_dim = int(gen_cfg.get("hidden_dim", max(32, 2 * repr_dim)))
    rank = int(gen_cfg.get("rank", 4))
    max_scale = float(gen_cfg.get("max_scale", 0.05))
    if generator_type in {"residual_basis_hypergenerator_v10"}:
        sample_y = train_ds[0][1]
        output_shape = tuple(gen_cfg.get("output_shape") or tuple(sample_y.shape))
        generator = ResidualBasisHyperGeneratorV10(
            named_param_shapes=named_shapes,
            drift_dim=repr_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            output_shape=output_shape,
            max_scale=max_scale,
            residual_scale=float(gen_cfg.get("residual_scale", 1.0)),
        ).to(device)
    elif generator_type in {"residual_basis_hypergenerator_v9", "residual_basis_hypergenerator"}:
        sample_y = train_ds[0][1]
        output_shape = tuple(gen_cfg.get("output_shape") or tuple(sample_y.shape))
        generator = ResidualBasisHyperGeneratorV9(
            named_param_shapes=named_shapes,
            drift_dim=repr_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            output_shape=output_shape,
            max_scale=max_scale,
            residual_scale=float(gen_cfg.get("residual_scale", 1.0)),
        ).to(device)
    else:
        generator_cls = LayerAwareHyperAdapterGenerator if generator_supports_bias else LayerAwareHyperLoRAGenerator
        generator = generator_cls(
            named_param_shapes=named_shapes,
            drift_dim=repr_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            max_scale=max_scale,
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
    rows = []
    candidates: list[Stage3GeneratorCandidate] = []
    previous_drift_vec: torch.Tensor | None = None
    for epoch in range(int(gen_cfg.get("epochs", 1))):
        state = Stage3SignalState()
        previous_drift_vec = None
        online_batch_size = int(cfg.get("stage3_v2", {}).get("online_batch_size", 1))
        for step, (x, y, phi_i, phi_v) in enumerate(
            _aligned_stream(
                train_ds,
                online_batch_size,
                device,
                aligned_model,
                drift_source=drift_source,
                delta_standardizer=delta_standardizer,
                residual_canonicalizer=residual_canonicalizer,
            )
        ):
            if max_batches is not None and step >= max_batches:
                break
            drift_raw = phi_v - phi_i
            signal_result = build_stage3_drift_vector(
                drift_raw=drift_raw,
                mode=cfg["online"]["stage3"].get("signal_mode", "delta_vector"),
                bottleneck_dim=repr_dim,
                state=state,
                normalize_vector=bool(cfg["online"]["stage3"].get("normalize_vector", True)),
                polarity=drift_polarity,
            )
            drift_vec = signal_result.drift_vec
            negative_drifts = _build_stage3_generator_negative_drifts(
                drift_vec=drift_vec,
                previous_drift_vec=previous_drift_vec,
                gen_cfg=gen_cfg,
            )
            teacher_correction = None
            if residual_teacher is not None:
                teacher_correction = residual_teacher.predict_correction(
                    drift_vec.detach().cpu().reshape(1, -1)
                ).to(device=device, dtype=x.dtype)
            metrics = stage3_generator_train_step(
                model=base_model,
                generator=generator,
                optimizer=opt,
                x=x,
                y=y,
                drift=drift_vec,
                loss_cfg=loss_cfg,
                target_prefixes=tuple(cfg["online"]["stage3"].get("target_modules") or ()) or None,
                target_names=target_names,
                negative_drifts=negative_drifts if loss_cfg.lambda_intervention_compare > 0.0 else None,
                teacher_correction=teacher_correction,
            )
            previous_drift_vec = drift_vec.detach().clone()
            rows.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "drift_source": drift_source,
                    "drift_polarity": drift_polarity,
                    "drift_norm": float(torch.linalg.vector_norm(drift_vec.detach()).cpu().item()),
                    **metrics,
                }
            )
        if candidate_checkpoint_dir is not None:
            candidate_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = candidate_checkpoint_dir / f"generator_epoch_{epoch:03d}.pt"
            last_train_loss = float(rows[-1]["loss"]) if rows else None
            save_stage3_generator_checkpoint(
                path=candidate_path,
                generator=generator,
                metadata={
                    **(candidate_metadata or {}),
                    "candidate_epoch": epoch,
                    "candidate_last_train_loss": last_train_loss,
                },
            )
            candidates.append(
                Stage3GeneratorCandidate(
                    epoch=epoch,
                    checkpoint_path=candidate_path,
                    last_train_loss=last_train_loss,
                )
            )
    generator.eval()
    return generator, rows, candidates


def _collect_risk_gate_examples(
    *,
    cfg: dict[str, Any],
    model,
    generator,
    aligned_model,
    train_ds,
    device: torch.device,
    max_batches: int | None = None,
    delta_standardizer: TrainSetDeltaStandardizer | None = None,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    model.eval()
    generator.eval()
    batch_size = int(cfg.get("stage3_v2", {}).get("online_batch_size", 1))
    target_prefixes = tuple(cfg["online"]["stage3"].get("target_modules") or ()) or None
    target_names = tuple(
        cfg.get("stage3_v2", {}).get("generator", {}).get("target_names")
        or cfg["online"]["stage3"].get("target_names")
        or ("weight",)
    )
    drift_source = _stage3_drift_source(cfg)
    drift_polarity = _stage3_drift_polarity(cfg)
    gen_cfg = cfg.get("stage3_v2", {}).get("generator", {})
    gate_cfg = cfg.get("stage3_v2", {}).get("risk_gate", {})
    counterfactual_label = float(gate_cfg.get("counterfactual_label", 0.0))
    state = Stage3SignalState()
    previous_drift_vec: torch.Tensor | None = None
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for step, (x, y, phi_i, phi_v) in enumerate(
        _aligned_stream(
            train_ds,
            batch_size,
            device,
            aligned_model,
            drift_source=drift_source,
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
        )
    ):
        if max_batches is not None and step >= max_batches:
            break
        if x.shape[0] != 1:
            raise ValueError("risk gate training requires online_batch_size=1")
        with torch.no_grad():
            base_output = model(x)
            y_base = _extract_prediction(base_output)
            base_mse = torch.nn.functional.mse_loss(y_base, y)
            recon_error = (
                torch.mean((base_output.x_recon - x) ** 2).item()
                if hasattr(base_output, "x_recon")
                else 0.0
            )
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
                y_candidate = _apply_generator_residual_correction(generator, candidate_drift, y_candidate)
                pred_shift = torch.nn.functional.l1_loss(y_candidate, y_base).item()
                blend = float(cfg["online"]["stage3"].get("blend_alpha", 1.0))
                y_blended = (1.0 - blend) * y_base + blend * y_candidate
                blended_mse = torch.nn.functional.mse_loss(y_blended, y)
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
                step_features.append((mode, candidate_drift.detach().cpu(), feature, blended_mse.detach().cpu(), mode_accept, pred_shift, delta_summary))
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
        raise ValueError("risk gate training collected no examples")
    return torch.cat(features, dim=0), torch.cat(labels, dim=0).reshape(-1), rows


def _train_risk_gate(
    *,
    cfg: dict[str, Any],
    model,
    generator,
    aligned_model,
    train_ds,
    device: torch.device,
    max_batches: int | None,
    delta_standardizer: TrainSetDeltaStandardizer | None,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | None,
) -> tuple[Stage3RiskGateBundle, list[dict[str, Any]], dict[str, Any]]:
    gate_cfg = cfg.get("stage3_v2", {}).get("risk_gate", {})
    features, labels, rows = _collect_risk_gate_examples(
        cfg=cfg,
        model=model,
        generator=generator,
        aligned_model=aligned_model,
        train_ds=train_ds,
        device=device,
        max_batches=max_batches,
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
    )
    gate, standardizer, metrics = train_stage3_risk_gate(
        features,
        labels,
        hidden_dim=int(gate_cfg.get("hidden_dim", 32)),
        dropout=float(gate_cfg.get("dropout", 0.0)),
        epochs=int(gate_cfg.get("epochs", 100)),
        lr=float(gate_cfg.get("lr", 1e-3)),
        positive_weight=(float(gate_cfg["positive_weight"]) if "positive_weight" in gate_cfg else None),
        model_type=str(gate_cfg.get("model_type", "mlp")),
    )
    return Stage3RiskGateBundle(gate, standardizer).to(device), rows, metrics


def _run_sensitivity_rows(
    *,
    cfg: dict[str, Any],
    eval_split: str,
    eval_ds,
    dataset_label: str,
    run_id: str,
    seed: int,
    device: torch.device,
    aligned_model,
    model,
    generator,
    risk_gate,
    generator_ckpt: Path,
    train_mse: float,
    train_mae: float,
    offline_eval_mse: float,
    offline_eval_mae: float,
    modes: list[str],
    max_eval_batches: int | None,
    delta_standardizer: TrainSetDeltaStandardizer | None,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | None,
    step_rows_out: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    batch_size = int(cfg.get("stage3_v2", {}).get("online_batch_size", 1))
    gate_role = "selection" if eval_split.startswith("validation") else "report_only"
    intervention_space = str(cfg.get("stage3_v2", {}).get("counterfactual_intervention_space", "generator_input"))
    if intervention_space == "generator_input":
        real_drift = _collect_generator_input_drift(
            eval_ds,
            batch_size,
            device,
            aligned_model,
            max_batches=max_eval_batches,
            drift_source=_stage3_drift_source(cfg),
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
            signal_mode=cfg["online"]["stage3"].get("signal_mode", "delta_vector"),
            normalize_vector=bool(cfg["online"]["stage3"].get("normalize_vector", True)),
            drift_polarity=_stage3_drift_polarity(cfg),
        )
    elif intervention_space == "raw_drift":
        real_drift = _collect_raw_drift(
            eval_ds,
            batch_size,
            device,
            aligned_model,
            max_batches=max_eval_batches,
            drift_source=_stage3_drift_source(cfg),
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
        )
    else:
        raise ValueError(f"unsupported counterfactual intervention space: {intervention_space}")
    interventions = build_drift_interventions(real_drift, seed=seed)
    metric_rows = []
    for mode in modes:
        run_cfg = cfg
        if mode == "real":
            stream = _aligned_stream(
                eval_ds,
                batch_size,
                device,
                aligned_model,
                max_batches=max_eval_batches,
                drift_source=_stage3_drift_source(cfg),
                delta_standardizer=delta_standardizer,
                residual_canonicalizer=residual_canonicalizer,
            )
        else:
            if intervention_space == "generator_input":
                run_cfg = copy.deepcopy(cfg)
                run_cfg["online"]["stage3"]["signal_mode"] = "level_vector"
                run_cfg["online"]["stage3"]["normalize_vector"] = False
                run_cfg["online"]["stage3"]["drift_polarity"] = 1.0
            stream = _aligned_stream(
                eval_ds,
                batch_size,
                device,
                aligned_model,
                drift_override=interventions[mode],
                max_batches=max_eval_batches,
            )
        trace_context = {
            "run_id": run_id,
            "dataset": dataset_label,
            "seed": seed,
            "split": eval_split,
            "gate_role": gate_role,
            "mode": mode,
            "intervention_online_causal": _intervention_is_online_causal(mode),
            "intervention_protocol": (
                "streaming_causal" if _intervention_is_online_causal(mode) else "offline_diagnostic"
            ),
            "counterfactual_intervention_space": intervention_space,
            "generator_checkpoint": str(generator_ckpt),
            "online_batch_size": batch_size,
        }
        metrics = run_online_adaptation_v2(
            stream=stream,
            model=model,
            generator=generator,
            risk_gate=risk_gate,
            cfg=run_cfg,
            device=device,
            return_step_rows=step_rows_out is not None,
            trace_context=trace_context,
        )
        step_rows = metrics.pop("step_rows", [])
        if step_rows_out is not None:
            step_rows_out.extend(step_rows)
        metric_rows.append(
            {
                "run_id": run_id,
                "dataset": dataset_label,
                "seed": seed,
                "split": eval_split,
                "gate_role": gate_role,
                "mode": mode,
                "intervention_online_causal": _intervention_is_online_causal(mode),
                "intervention_protocol": (
                    "streaming_causal" if _intervention_is_online_causal(mode) else "offline_diagnostic"
                ),
                "offline_train_mse": train_mse,
                "offline_train_mae": train_mae,
                "offline_eval_mse": offline_eval_mse,
                "offline_eval_mae": offline_eval_mae,
                "generator_checkpoint": str(generator_ckpt),
                "online_batch_size": batch_size,
                "causal_online": bool(cfg["online"]["stage3"].get("causal_online", True)),
                "require_aligned_embeddings": bool(cfg["online"]["stage3"].get("require_aligned_embeddings", False)),
                "max_delta": cfg["online"]["stage3"].get("max_delta"),
                "max_pred_shift": cfg["online"]["stage3"].get("max_pred_shift"),
                "blend_alpha": cfg["online"]["stage3"].get("blend_alpha"),
                "risk_gate_enable": bool(cfg["online"]["stage3"].get("risk_gate_enable", False)),
                "risk_gate_threshold": cfg["online"]["stage3"].get("risk_gate_threshold"),
                "target_modules": ";".join(cfg["online"]["stage3"].get("target_modules") or []),
                "intervention_seed": seed,
                "intervention_source_split": eval_split,
                "intervention_uses_full_split": max_eval_batches is None,
                "counterfactual_intervention_space": intervention_space,
                **metrics,
            }
        )
    return metric_rows


def _run_v9_teacher_only_rows(
    *,
    cfg: dict[str, Any],
    eval_split: str,
    eval_ds,
    dataset_label: str,
    run_id: str,
    seed: int,
    device: torch.device,
    aligned_model,
    model,
    residual_teacher: TrainOnlyResidualTeacherV9
    | TrainOnlyResidualTeacherV10
    | TrainOnlyRobustBlockResidualTeacherV11
    | TrainOnlyOutcomeAlignedBlockResidualTeacherV12,
    modes: list[str],
    max_eval_batches: int | None,
    delta_standardizer: TrainSetDeltaStandardizer | None,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | None,
    step_rows_out: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    batch_size = int(cfg.get("stage3_v2", {}).get("online_batch_size", 1))
    intervention_space = str(cfg.get("stage3_v2", {}).get("counterfactual_intervention_space", "generator_input"))
    if intervention_space != "generator_input":
        raise ValueError("V9 teacher-only diagnostics require generator_input intervention space")
    real_drift = _collect_generator_input_drift(
        eval_ds,
        batch_size,
        device,
        aligned_model,
        max_batches=max_eval_batches,
        drift_source=_stage3_drift_source(cfg),
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
        signal_mode=cfg["online"]["stage3"].get("signal_mode", "delta_vector"),
        normalize_vector=bool(cfg["online"]["stage3"].get("normalize_vector", True)),
        drift_polarity=_stage3_drift_polarity(cfg),
    )
    interventions = build_drift_interventions(real_drift, seed=seed)
    model.eval()
    metric_rows: list[dict[str, Any]] = []
    for mode in modes:
        run_cfg = cfg
        if mode == "real":
            stream = _aligned_stream(
                eval_ds,
                batch_size,
                device,
                aligned_model,
                max_batches=max_eval_batches,
                drift_source=_stage3_drift_source(cfg),
                delta_standardizer=delta_standardizer,
                residual_canonicalizer=residual_canonicalizer,
            )
        else:
            run_cfg = copy.deepcopy(cfg)
            run_cfg["online"]["stage3"]["signal_mode"] = "level_vector"
            run_cfg["online"]["stage3"]["normalize_vector"] = False
            run_cfg["online"]["stage3"]["drift_polarity"] = 1.0
            stream = _aligned_stream(
                eval_ds,
                batch_size,
                device,
                aligned_model,
                drift_override=interventions[mode],
                max_batches=max_eval_batches,
            )

        signal_state = Stage3SignalState()
        y_base_parts: list[torch.Tensor] = []
        y_parts: list[torch.Tensor] = []
        correction_parts: list[torch.Tensor] = []
        drift_norms: list[float] = []
        for step_index, (x, y, phi_i, phi_v) in enumerate(stream):
            with torch.no_grad():
                y_base = _extract_prediction(model(x))
                drift_raw = phi_v - phi_i
                signal_result = build_stage3_drift_vector(
                    drift_raw=drift_raw,
                    mode=run_cfg["online"]["stage3"].get("signal_mode", "delta_vector"),
                    bottleneck_dim=int(drift_raw.shape[-1]),
                    state=signal_state,
                    normalize_vector=bool(run_cfg["online"]["stage3"].get("normalize_vector", True)),
                    polarity=float(run_cfg["online"]["stage3"].get("drift_polarity", 1.0)),
                )
                drift_vec = signal_result.drift_vec.to(device=device, dtype=x.dtype)
                correction = residual_teacher.predict_correction(drift_vec).to(device=device, dtype=y_base.dtype)
            y_base_parts.append(y_base.detach().cpu())
            y_parts.append(y.detach().cpu())
            correction_parts.append(correction.detach().cpu())
            drift_norm = float(torch.linalg.vector_norm(drift_vec.detach().flatten()).cpu().item())
            drift_norms.append(drift_norm)
            if step_rows_out is not None:
                step_metrics = compute_teacher_correction_metrics(
                    y_base=y_base.detach().cpu(),
                    y_true=y.detach().cpu(),
                    correction=correction.detach().cpu(),
                )
                step_row = {
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": seed,
                    "split": eval_split,
                    "mode": mode,
                    "step_index": step_index,
                    "intervention_online_causal": _intervention_is_online_causal(mode),
                    "intervention_protocol": (
                        "streaming_causal" if _intervention_is_online_causal(mode) else "offline_diagnostic"
                    ),
                    "counterfactual_intervention_space": intervention_space,
                    "teacher_drift_norm": drift_norm,
                    **step_metrics,
                }
                step_rows_out.append(step_row)

        if not y_base_parts:
            raise ValueError(f"V9 teacher-only diagnostics collected no samples for split={eval_split} mode={mode}")
        y_base_all = torch.cat(y_base_parts, dim=0)
        y_all = torch.cat(y_parts, dim=0)
        correction_all = torch.cat(correction_parts, dim=0)
        metrics = compute_teacher_correction_metrics(
            y_base=y_base_all,
            y_true=y_all,
            correction=correction_all,
        )
        metric_rows.append(
            {
                "run_id": run_id,
                "dataset": dataset_label,
                "seed": seed,
                "split": eval_split,
                "mode": mode,
                "intervention_online_causal": _intervention_is_online_causal(mode),
                "intervention_protocol": (
                    "streaming_causal" if _intervention_is_online_causal(mode) else "offline_diagnostic"
                ),
                "counterfactual_intervention_space": intervention_space,
                "teacher_fit_split": residual_teacher.fit_split,
                "teacher_fit_sample_count": residual_teacher.fit_sample_count,
                "teacher_train_residual_r2": residual_teacher.teacher_residual_r2,
                "teacher_sample_count": int(y_base_all.shape[0]),
                "teacher_drift_norm_mean": float(sum(drift_norms) / len(drift_norms)),
                "teacher_drift_norm_max": float(max(drift_norms)),
                **metrics,
            }
        )
    return metric_rows


def _teacher_diagnostic_aggregate(
    *,
    cfg: dict[str, Any],
    run_id: str,
    dataset_label: str,
    seed: int,
    split: str,
    window_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gate_cfg = cfg.get("stage3_v2", {}).get("claim_gate", {})
    teacher_cfg = cfg.get("stage3_v2", {}).get("residual_teacher", {})
    min_mse_gain = float(teacher_cfg.get("diagnostic_min_real_mse_gain", gate_cfg.get("min_real_mse_gain", 0.0)))
    min_mae_gain = float(teacher_cfg.get("diagnostic_min_real_mae_gain", gate_cfg.get("min_real_mae_gain", 0.0)))
    min_margin = float(teacher_cfg.get("diagnostic_counterfactual_margin", gate_cfg.get("counterfactual_margin", 0.0)))
    min_sign_agreement = float(teacher_cfg.get("diagnostic_min_sign_agreement", 0.5))
    if not window_rows:
        raise ValueError("V9 teacher diagnostic aggregate requires window rows")
    worst_mse_gain = min(float(row["teacher_real_mse_gain"]) for row in window_rows)
    worst_mae_gain = min(float(row["teacher_real_mae_gain"]) for row in window_rows)
    worst_margin = min(float(row["teacher_real_vs_best_counterfactual_mse_gain_margin"]) for row in window_rows)
    worst_sign = min(float(row["teacher_real_sign_agreement_rate"]) for row in window_rows)
    failed_rows = []
    failure_reasons: list[str] = []
    for row in window_rows:
        row_reasons = []
        if float(row["teacher_real_mse_gain"]) < min_mse_gain:
            row_reasons.append("teacher_real_mse_gain")
        if float(row["teacher_real_mae_gain"]) < min_mae_gain:
            row_reasons.append("teacher_real_mae_gain")
        if float(row["teacher_real_vs_best_counterfactual_mse_gain_margin"]) <= min_margin:
            row_reasons.append("teacher_counterfactual")
        sign_value = float(row["teacher_real_sign_agreement_rate"])
        if sign_value == sign_value and sign_value < min_sign_agreement:
            row_reasons.append("teacher_sign_agreement")
        if row_reasons:
            failed_rows.append(row)
            for reason in row_reasons:
                if reason not in failure_reasons:
                    failure_reasons.append(reason)
    pass_count = len(window_rows) - len(failed_rows)
    return {
        "run_id": run_id,
        "dataset": dataset_label,
        "seed": seed,
        "split": split,
        "teacher_diagnostic_gate_pass": not failure_reasons,
        "teacher_diagnostic_failure_reasons": ";".join(failure_reasons),
        "teacher_window_count": len(window_rows),
        "teacher_window_pass_count": pass_count,
        "teacher_window_pass_rate": pass_count / len(window_rows),
        "teacher_failed_windows": ";".join(str(row.get("validation_window_id", row.get("split", ""))) for row in failed_rows),
        "teacher_worst_real_mse_gain": worst_mse_gain,
        "teacher_worst_real_mae_gain": worst_mae_gain,
        "teacher_worst_real_vs_best_counterfactual_mse_gain_margin": worst_margin,
        "teacher_worst_sign_agreement_rate": worst_sign,
        "teacher_min_real_mse_gain_required": min_mse_gain,
        "teacher_min_real_mae_gain_required": min_mae_gain,
        "teacher_counterfactual_margin_required": min_margin,
        "teacher_min_sign_agreement_required": min_sign_agreement,
    }


def _run_v9_teacher_rolling_diagnostics(
    *,
    cfg: dict[str, Any],
    val_ds,
    dataset_label: str,
    run_id: str,
    seed: int,
    device: torch.device,
    aligned_model,
    model,
    residual_teacher: TrainOnlyResidualTeacherV9
    | TrainOnlyResidualTeacherV10
    | TrainOnlyRobustBlockResidualTeacherV11
    | TrainOnlyOutcomeAlignedBlockResidualTeacherV12,
    modes: list[str],
    max_eval_batches: int | None,
    delta_standardizer: TrainSetDeltaStandardizer | None,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | None,
    window_count: int,
    step_rows_out: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    windows = _contiguous_dataset_windows(val_ds, window_count)
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for window_id, window_ds in enumerate(windows):
        start_idx, end_idx = _subset_bounds(window_ds)
        rows = _run_v9_teacher_only_rows(
            cfg=cfg,
            eval_split=f"validation_window_{window_id}",
            eval_ds=window_ds,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=seed,
            device=device,
            aligned_model=aligned_model,
            model=model,
            residual_teacher=residual_teacher,
            modes=modes,
            max_eval_batches=max_eval_batches,
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
            step_rows_out=step_rows_out,
        )
        for row in rows:
            row.update(
                {
                    "validation_window_id": window_id,
                    "validation_window_start_sample": start_idx,
                    "validation_window_end_sample": end_idx,
                    "validation_window_count": len(windows),
                }
            )
        summary = summarize_teacher_mode_rows(rows)
        summary_rows.append(
            {
                "run_id": run_id,
                "dataset": dataset_label,
                "seed": seed,
                "split": f"validation_window_{window_id}",
                "validation_window_id": window_id,
                "validation_window_start_sample": start_idx,
                "validation_window_end_sample": end_idx,
                "validation_window_count": len(windows),
                **summary,
            }
        )
        all_rows.extend(rows)
    aggregate = _teacher_diagnostic_aggregate(
        cfg=cfg,
        run_id=run_id,
        dataset_label=dataset_label,
        seed=seed,
        split="validation_rolling",
        window_rows=summary_rows,
    )
    return all_rows, summary_rows, aggregate


def _run_rolling_validation(
    *,
    cfg: dict[str, Any],
    val_ds,
    dataset_label: str,
    run_id: str,
    seed: int,
    device: torch.device,
    aligned_model,
    model,
    generator,
    risk_gate,
    generator_ckpt: Path,
    train_mse: float,
    train_mae: float,
    modes: list[str],
    max_eval_batches: int | None,
    delta_standardizer: TrainSetDeltaStandardizer | None,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | None,
    window_count: int,
    step_rows_out: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    windows = _contiguous_dataset_windows(val_ds, window_count)
    all_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for window_id, window_ds in enumerate(windows):
        start_idx, end_idx = _subset_bounds(window_ds)
        eval_iter = _limit_batches(
            DataLoader(window_ds, batch_size=cfg["train"]["batch_size"]),
            max_eval_batches,
        )
        window_mse, window_mae = evaluate_factor(eval_iter, model, device)
        rows = _run_sensitivity_rows(
            cfg=cfg,
            eval_split=f"validation_window_{window_id}",
            eval_ds=window_ds,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=seed,
            device=device,
            aligned_model=aligned_model,
            model=model,
            generator=generator,
            risk_gate=risk_gate,
            generator_ckpt=generator_ckpt,
            train_mse=train_mse,
            train_mae=train_mae,
            offline_eval_mse=window_mse,
            offline_eval_mae=window_mae,
            modes=modes,
            max_eval_batches=max_eval_batches,
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
            step_rows_out=step_rows_out,
        )
        for row in rows:
            row.update(
                {
                    "validation_window_id": window_id,
                    "validation_window_start_sample": start_idx,
                    "validation_window_end_sample": end_idx,
                    "validation_window_count": len(windows),
                }
            )
        gate_row = _sensitivity_gate_row(
            cfg=cfg,
            rows=rows,
            run_id=run_id,
            dataset_label=dataset_label,
            seed=seed,
            split=f"validation_window_{window_id}",
        )
        gate_row.update(
            {
                "validation_window_id": window_id,
                "validation_window_start_sample": start_idx,
                "validation_window_end_sample": end_idx,
                "validation_window_count": len(windows),
            }
        )
        all_rows.extend(rows)
        gate_rows.append(gate_row)

    gate_cfg = cfg.get("stage3_v2", {}).get("claim_gate", {})
    aggregate = {
        "run_id": run_id,
        "dataset": dataset_label,
        "seed": seed,
        "split": "validation_rolling",
        **evaluate_rolling_stage3_sensitivity_gates(
            gate_rows,
            min_window_pass_rate=float(gate_cfg.get("rolling_min_window_pass_rate", 1.0)),
        ),
    }
    return all_rows, gate_rows, aggregate


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_gain(row: dict[str, Any], *, base_key: str, stage3_key: str) -> float:
    base = _safe_float(row.get(base_key))
    stage3 = _safe_float(row.get(stage3_key))
    if base != base or stage3 != stage3 or base == 0.0:
        return float("nan")
    return (base - stage3) / base


def _validation_window_key(row: dict[str, Any]) -> str:
    if "validation_window_id" in row:
        return str(row["validation_window_id"])
    split = str(row.get("split", ""))
    if split.startswith("validation_window_"):
        return split.rsplit("_", 1)[-1]
    return split


def _validation_window_sort_key(key: str) -> tuple[int, int | str]:
    try:
        return (0, int(key))
    except ValueError:
        return (1, key)


def _candidate_window_diagnostic_rows(
    *,
    candidate: Stage3GeneratorCandidate,
    metric_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    selection_stage: str,
    risk_gate_threshold: Any,
) -> list[dict[str, Any]]:
    metric_rows_by_window: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        metric_rows_by_window.setdefault(_validation_window_key(row), []).append(row)

    gate_rows_by_window = {_validation_window_key(row): row for row in gate_rows}
    diagnostic_rows: list[dict[str, Any]] = []
    for window_key in sorted(metric_rows_by_window, key=_validation_window_sort_key):
        rows = metric_rows_by_window[window_key]
        by_mode = {str(row.get("mode", "")): row for row in rows}
        real_row = by_mode.get("real", {})
        gate_row = gate_rows_by_window.get(window_key, {})
        real_mse_gain = _metric_gain(real_row, base_key="online_base_mse", stage3_key="online_stage3_mse")
        real_mae_gain = _metric_gain(real_row, base_key="online_base_mae", stage3_key="online_stage3_mae")
        counterfactual_gain_by_mode = {
            mode: _metric_gain(row, base_key="online_base_mse", stage3_key="online_stage3_mse")
            for mode, row in by_mode.items()
            if mode in COUNTERFACTUAL_MODES
        }
        best_counterfactual_mode = ""
        best_counterfactual_gain = float("nan")
        finite_counterfactuals = {
            mode: gain for mode, gain in counterfactual_gain_by_mode.items() if gain == gain
        }
        if finite_counterfactuals:
            best_counterfactual_mode = max(finite_counterfactuals, key=finite_counterfactuals.get)
            best_counterfactual_gain = finite_counterfactuals[best_counterfactual_mode]
        margin = (
            real_mse_gain - best_counterfactual_gain
            if real_mse_gain == real_mse_gain and best_counterfactual_gain == best_counterfactual_gain
            else float("nan")
        )

        for row in rows:
            mode = str(row.get("mode", ""))
            diagnostic_rows.append(
                {
                    "candidate_epoch": candidate.epoch,
                    "candidate_checkpoint": str(candidate.checkpoint_path),
                    "candidate_last_train_loss": candidate.last_train_loss,
                    "generator_selection_stage": selection_stage,
                    "risk_gate_threshold": risk_gate_threshold,
                    "validation_window_id": window_key,
                    "validation_window_start_sample": row.get("validation_window_start_sample"),
                    "validation_window_end_sample": row.get("validation_window_end_sample"),
                    "validation_window_count": row.get("validation_window_count"),
                    "split": row.get("split"),
                    "claim_gate_pass": gate_row.get("claim_gate_pass"),
                    "failure_reasons": gate_row.get("failure_reasons"),
                    "real_reject_rate": gate_row.get("real_reject_rate"),
                    "mode": mode,
                    "mode_is_counterfactual": mode in COUNTERFACTUAL_MODES,
                    "mode_is_best_counterfactual": mode == best_counterfactual_mode,
                    "mode_mse_gain": _metric_gain(
                        row,
                        base_key="online_base_mse",
                        stage3_key="online_stage3_mse",
                    ),
                    "mode_mae_gain": _metric_gain(
                        row,
                        base_key="online_base_mae",
                        stage3_key="online_stage3_mae",
                    ),
                    "mode_online_base_mse": row.get("online_base_mse"),
                    "mode_online_stage3_mse": row.get("online_stage3_mse"),
                    "mode_online_base_mae": row.get("online_base_mae"),
                    "mode_online_stage3_mae": row.get("online_stage3_mae"),
                    "mode_use_rate": row.get("online_stage3_use_rate"),
                    "mode_delta_norm_pre_clamp": row.get("online_stage3_delta_norm_pre_clamp"),
                    "mode_pred_shift": row.get("online_stage3_pred_shift"),
                    "mode_reject_shift_count": row.get("online_stage3_reject_shift_count"),
                    "mode_reject_risk_count": row.get("online_stage3_reject_risk_count"),
                    "mode_reject_missing_drift_count": row.get("online_stage3_reject_missing_drift_count"),
                    "real_mse_gain": real_mse_gain,
                    "real_mae_gain": real_mae_gain,
                    "best_counterfactual_mode": best_counterfactual_mode,
                    "best_counterfactual_mse_gain": best_counterfactual_gain,
                    "real_vs_best_counterfactual_mse_gain_margin": margin,
                }
            )
    return diagnostic_rows


def _select_risk_gate_threshold(
    *,
    cfg: dict[str, Any],
    val_ds,
    dataset_label: str,
    run_id: str,
    seed: int,
    device: torch.device,
    aligned_model,
    model,
    generator,
    risk_gate,
    generator_ckpt: Path,
    train_mse: float,
    train_mae: float,
    offline_eval_mse: float,
    offline_eval_mae: float,
    modes: list[str],
    max_eval_batches: int | None,
    delta_standardizer: TrainSetDeltaStandardizer | None,
    validation_window_count: int,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | None,
) -> tuple[float, list[dict[str, Any]]]:
    gate_cfg = cfg.get("stage3_v2", {}).get("risk_gate", {})
    thresholds = gate_cfg.get("threshold_grid")
    if not thresholds:
        current = float(cfg["online"]["stage3"].get("risk_gate_threshold", 0.5))
        return current, []

    selection_modes = _selection_modes_for_validation(cfg, modes)
    best_threshold = float(thresholds[0])
    best_score = float("-inf")
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold = float(threshold)
        cfg["online"]["stage3"]["risk_gate_threshold"] = threshold
        if validation_window_count > 1:
            _, _, aggregate = _run_rolling_validation(
                cfg=cfg,
                val_ds=val_ds,
                dataset_label=dataset_label,
                run_id=run_id,
                seed=seed,
                device=device,
                aligned_model=aligned_model,
                model=model,
                generator=generator,
                risk_gate=risk_gate,
                generator_ckpt=generator_ckpt,
                train_mse=train_mse,
                train_mae=train_mae,
                modes=selection_modes,
                max_eval_batches=max_eval_batches,
                delta_standardizer=delta_standardizer,
                residual_canonicalizer=residual_canonicalizer,
                window_count=validation_window_count,
            )
            score = score_served_policy_candidate(aggregate)
            row = {
                "threshold": threshold,
                "selection_score": score,
                "selection_modes": ";".join(selection_modes),
                "offline_diagnostics_excluded_from_selection": selection_modes != modes,
                **aggregate,
            }
        else:
            metric_rows = _run_sensitivity_rows(
                cfg=cfg,
                eval_split="validation_threshold_selection",
                eval_ds=val_ds,
                dataset_label=dataset_label,
                run_id=run_id,
                seed=seed,
                device=device,
                aligned_model=aligned_model,
                model=model,
                generator=generator,
                risk_gate=risk_gate,
                generator_ckpt=generator_ckpt,
                train_mse=train_mse,
                train_mae=train_mae,
                offline_eval_mse=offline_eval_mse,
                offline_eval_mae=offline_eval_mae,
                modes=selection_modes,
                max_eval_batches=max_eval_batches,
                delta_standardizer=delta_standardizer,
                residual_canonicalizer=residual_canonicalizer,
            )
            gate_row = _sensitivity_gate_row(
                cfg=cfg,
                rows=metric_rows,
                run_id=run_id,
                dataset_label=dataset_label,
                seed=seed,
                split="validation_threshold_selection",
            )
            pass_bonus = 100.0 if gate_row["claim_gate_pass"] else 0.0
            score = (
                pass_bonus
                + float(gate_row["real_mse_gain"])
                + float(gate_row["real_mae_gain"])
                + float(gate_row["real_vs_best_counterfactual_mse_gain_margin"])
            )
            row = {
                "threshold": threshold,
                "selection_score": score,
                "selection_modes": ";".join(selection_modes),
                "offline_diagnostics_excluded_from_selection": selection_modes != modes,
                **gate_row,
            }
        rows.append(row)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    cfg["online"]["stage3"]["risk_gate_threshold"] = best_threshold
    return best_threshold, rows


def _select_generator_checkpoint_on_rolling_validation(
    *,
    cfg: dict[str, Any],
    candidates: list[Stage3GeneratorCandidate],
    val_ds,
    dataset_label: str,
    run_id: str,
    seed: int,
    device: torch.device,
    aligned_model,
    model,
    train_mse: float,
    train_mae: float,
    modes: list[str],
    max_eval_batches: int | None,
    delta_standardizer: TrainSetDeltaStandardizer | None,
    validation_window_count: int,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | None,
) -> tuple[Stage3GeneratorCandidate, list[dict[str, Any]]]:
    if not candidates:
        raise ValueError("generator checkpoint selection requires at least one candidate checkpoint")
    selection_cfg = cfg.get("stage3_v2", {}).get("generator", {}).get("checkpoint_selection", {})
    selection_modes = _selection_modes_for_validation(cfg, modes)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        generator, _ = load_stage3_generator_checkpoint(candidate.checkpoint_path, model=model)
        generator = generator.to(device)
        generator.eval()
        eval_cfg = json.loads(json.dumps(cfg))
        eval_cfg["online"]["stage3"]["risk_gate_enable"] = False
        _, _, aggregate = _run_rolling_validation(
            cfg=eval_cfg,
            val_ds=val_ds,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=seed,
            device=device,
            aligned_model=aligned_model,
            model=model,
            generator=generator,
            risk_gate=None,
            generator_ckpt=candidate.checkpoint_path,
            train_mse=train_mse,
            train_mae=train_mae,
            modes=selection_modes,
            max_eval_batches=max_eval_batches,
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
            window_count=validation_window_count,
        )
        row = build_generator_selection_row(candidate, aggregate)
        row["generator_selection_stage"] = "pre_risk_gate"
        row["selection_modes"] = ";".join(selection_modes)
        row["offline_diagnostics_excluded_from_selection"] = selection_modes != modes
        rows.append(row)
    selected_row = select_best_generator_candidate(
        rows,
        abstain_if_no_pass=bool(selection_cfg.get("abstain_if_no_pass", False)),
    )
    selected_path = Path(str(selected_row["candidate_checkpoint"]))
    selected = next(candidate for candidate in candidates if candidate.checkpoint_path == selected_path)
    return selected, rows


def _select_generator_checkpoint_on_served_policy_validation(
    *,
    cfg: dict[str, Any],
    candidates: list[Stage3GeneratorCandidate],
    val_ds,
    train_ds=None,
    dataset_label: str,
    run_id: str,
    seed: int,
    device: torch.device,
    aligned_model,
    model,
    train_mse: float,
    train_mae: float,
    offline_eval_mse: float = 0.0,
    offline_eval_mae: float = 0.0,
    modes: list[str],
    max_eval_batches: int | None,
    risk_gate_train_max_batches: int | None = None,
    delta_standardizer: TrainSetDeltaStandardizer | None,
    validation_window_count: int,
    residual_canonicalizer: TrainSetResidualDriftCanonicalizer | None,
    risk_gate,
    selection_artifacts: dict[str, Any] | None = None,
) -> tuple[Stage3GeneratorCandidate, list[dict[str, Any]]]:
    if not candidates:
        raise ValueError("served-policy generator checkpoint selection requires at least one candidate checkpoint")
    if risk_gate is None:
        raise ValueError("served-policy generator checkpoint selection requires a trained risk gate")
    selection_cfg = cfg.get("stage3_v2", {}).get("generator", {}).get("checkpoint_selection", {})
    retrain_gate_per_candidate = bool(selection_cfg.get("retrain_risk_gate_per_candidate", False))
    calibrate_threshold_per_candidate = bool(selection_cfg.get("calibrate_threshold_per_candidate", False))
    if retrain_gate_per_candidate and train_ds is None:
        raise ValueError("candidate-specific served-policy selection requires train_ds")
    selection_modes = _selection_modes_for_validation(cfg, modes)
    rows: list[dict[str, Any]] = []
    candidate_artifacts: dict[str, dict[str, Any]] = {}
    candidate_window_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        generator, _ = load_stage3_generator_checkpoint(candidate.checkpoint_path, model=model)
        generator = generator.to(device)
        generator.eval()
        eval_cfg = json.loads(json.dumps(cfg))
        eval_cfg["online"]["stage3"]["risk_gate_enable"] = True
        candidate_risk_gate = risk_gate
        candidate_risk_gate_rows: list[dict[str, Any]] = []
        candidate_risk_gate_metrics: dict[str, Any] = {}
        candidate_threshold_rows: list[dict[str, Any]] = []
        candidate_threshold = eval_cfg["online"]["stage3"].get("risk_gate_threshold")
        if retrain_gate_per_candidate:
            candidate_risk_gate, candidate_risk_gate_rows, candidate_risk_gate_metrics = _train_risk_gate(
                cfg=eval_cfg,
                model=model,
                generator=generator,
                aligned_model=aligned_model,
                train_ds=train_ds,
                device=device,
                max_batches=risk_gate_train_max_batches,
                delta_standardizer=delta_standardizer,
                residual_canonicalizer=residual_canonicalizer,
            )
        if calibrate_threshold_per_candidate and bool(
            eval_cfg.get("stage3_v2", {}).get("risk_gate", {}).get("calibrate_threshold", True)
        ):
            candidate_threshold, candidate_threshold_rows = _select_risk_gate_threshold(
                cfg=eval_cfg,
                val_ds=val_ds,
                dataset_label=dataset_label,
                run_id=run_id,
                seed=seed,
                device=device,
                aligned_model=aligned_model,
                model=model,
                generator=generator,
                risk_gate=candidate_risk_gate,
                generator_ckpt=candidate.checkpoint_path,
                train_mse=train_mse,
                train_mae=train_mae,
                offline_eval_mse=offline_eval_mse,
                offline_eval_mae=offline_eval_mae,
                modes=modes,
                max_eval_batches=max_eval_batches,
                delta_standardizer=delta_standardizer,
                validation_window_count=validation_window_count,
                residual_canonicalizer=residual_canonicalizer,
            )
        metric_rows, gate_rows, aggregate = _run_rolling_validation(
            cfg=eval_cfg,
            val_ds=val_ds,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=seed,
            device=device,
            aligned_model=aligned_model,
            model=model,
            generator=generator,
            risk_gate=candidate_risk_gate,
            generator_ckpt=candidate.checkpoint_path,
            train_mse=train_mse,
            train_mae=train_mae,
            modes=selection_modes,
            max_eval_batches=max_eval_batches,
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
            window_count=validation_window_count,
        )
        candidate_rows = _candidate_window_diagnostic_rows(
            candidate=candidate,
            metric_rows=metric_rows,
            gate_rows=gate_rows,
            selection_stage="served_policy",
            risk_gate_threshold=eval_cfg["online"]["stage3"].get("risk_gate_threshold"),
        )
        candidate_window_rows.extend(candidate_rows)
        row = build_generator_selection_row(candidate, aggregate)
        row["generator_selection_stage"] = "served_policy"
        row["risk_gate_threshold"] = eval_cfg["online"]["stage3"].get("risk_gate_threshold")
        row["candidate_risk_gate_retrained"] = retrain_gate_per_candidate
        row["candidate_risk_gate_train_rows"] = len(candidate_risk_gate_rows)
        row["candidate_risk_gate_threshold_calibrated"] = bool(candidate_threshold_rows)
        row["candidate_risk_gate_threshold_selection_rows"] = len(candidate_threshold_rows)
        row["candidate_risk_gate_selected_threshold"] = candidate_threshold
        row["selection_modes"] = ";".join(selection_modes)
        row["offline_diagnostics_excluded_from_selection"] = selection_modes != modes
        for key, value in candidate_risk_gate_metrics.items():
            row[f"candidate_risk_gate_{key}"] = value
        rows.append(row)
        candidate_artifacts[str(candidate.checkpoint_path)] = {
            "risk_gate": candidate_risk_gate,
            "risk_gate_rows": candidate_risk_gate_rows,
            "risk_gate_metrics": candidate_risk_gate_metrics,
            "risk_gate_threshold_rows": candidate_threshold_rows,
            "selected_risk_gate_threshold": candidate_threshold,
            "selection_modes": selection_modes,
            "candidate_window_rows": candidate_rows,
        }
    selected_row = select_best_generator_candidate(
        rows,
        abstain_if_no_pass=bool(selection_cfg.get("abstain_if_no_pass", False)),
    )
    selected_path = Path(str(selected_row["candidate_checkpoint"]))
    selected = next(candidate for candidate in candidates if candidate.checkpoint_path == selected_path)
    if selection_artifacts is not None:
        selection_artifacts.clear()
        selection_artifacts.update(candidate_artifacts.get(str(selected_path), {}))
        selection_artifacts["selected_row"] = selected_row
        selection_artifacts["abstained"] = selected_row.get("generator_selection_abstained", False)
        selection_artifacts["all_candidate_window_rows"] = candidate_window_rows
    return selected, rows


def run_stage3_v12_audit(args: argparse.Namespace) -> Path:
    cfg = _apply_dataset_override(load_yaml(args.config), args.dataset)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.invariant_epochs is not None:
        cfg["invariant"]["epochs"] = args.invariant_epochs
    if args.generator_epochs is not None:
        cfg["stage3_v2"]["generator"]["epochs"] = args.generator_epochs
    if args.drift_source is not None:
        cfg.setdefault("online", {}).setdefault("stage3", {})["drift_source"] = args.drift_source
    if args.drift_polarity is not None:
        cfg.setdefault("online", {}).setdefault("stage3", {})["drift_polarity"] = args.drift_polarity
    if getattr(args, "signal_mode", None) is not None:
        cfg.setdefault("online", {}).setdefault("stage3", {})["signal_mode"] = args.signal_mode
    if getattr(args, "normalize_vector", None) is not None:
        cfg.setdefault("online", {}).setdefault("stage3", {})["normalize_vector"] = args.normalize_vector
    cfg["seed"] = args.seed
    set_seed(args.seed)

    run_id = args.run_id or f"stage3_v12_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_label = args.dataset or Path(cfg["data"]["path"]).stem
    out_dir = Path(args.out_root) / run_id / dataset_label / f"seed_{args.seed}"
    ckpt_dir = Path(args.checkpoint_root) / dataset_label / run_id / f"seed_{args.seed}"
    _prepare_fresh_run_dirs(out_dir=out_dir, ckpt_dir=ckpt_dir, overwrite=bool(getattr(args, "overwrite", False)))
    resolved_config_snapshot = _write_resolved_config_snapshot(cfg=cfg, out_dir=out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    series, _, target_dim = _load_series_and_features(cfg)
    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds, split_info = _build_dataloaders(cfg, series, target_dim)
    train_iter = _limit_batches(train_dl, args.max_train_batches)
    val_iter = _limit_batches(val_dl, args.max_eval_batches)
    test_iter = _limit_batches(test_dl, args.max_eval_batches)
    ordered_train_iter = _limit_batches(
        _ordered_loader(train_ds, cfg["train"]["batch_size"], drop_last=True),
        args.max_train_batches,
    )
    model = _build_model(cfg, series.shape[1], target_dim).to(device)
    assert_deterministic_eval_model(model, context="Stage III V12 audit runner")

    offline_rows = _train_base_model(cfg, model, train_iter, val_iter, device)
    train_eval_iter = _limit_batches(
        DataLoader(train_ds, batch_size=cfg["train"]["batch_size"]),
        args.max_eval_batches,
    )
    train_mse, train_mae = evaluate_factor(train_eval_iter, model, device)
    val_mse, val_mae = evaluate_factor(val_iter, model, device)
    test_mse, test_mae = evaluate_factor(test_iter, model, device)
    torch.save(model.state_dict(), ckpt_dir / "base_model.pt")
    _append_csv(out_dir / "offline_metrics.csv", offline_rows)

    aligned_model, aligned_rows = _train_aligned_model(cfg, ordered_train_iter, val_iter, series.shape[1], device)
    torch.save(aligned_model.state_dict(), ckpt_dir / "aligned_representation.pt")
    _append_csv(out_dir / "aligned_representation_metrics.csv", aligned_rows)
    residual_canonicalizer = _fit_residual_canonicalizer(
        cfg,
        train_ds,
        int(cfg.get("stage3_v2", {}).get("online_batch_size", 1)),
        device,
        aligned_model,
        max_batches=args.max_generator_batches,
    )
    residual_canonicalizer_ckpt = None
    if residual_canonicalizer is not None:
        residual_canonicalizer_ckpt = ckpt_dir / "residual_drift_canonicalizer.pt"
        torch.save(residual_canonicalizer.state_dict(), residual_canonicalizer_ckpt)
    alignment_rows = []
    for split_name, split_ds in (("train", train_ds), ("validation", val_ds), ("test", test_ds)):
        alignment_rows.extend(
            {
                "run_id": run_id,
                "dataset": dataset_label,
                "seed": args.seed,
                **row,
            }
            for row in _alignment_diagnostic_rows(
                dataset=split_ds,
                split=split_name,
                batch_size=int(cfg.get("stage3_v2", {}).get("online_batch_size", 1)),
                device=device,
                aligned_model=aligned_model,
                max_batches=args.max_generator_batches if split_name == "train" else args.max_eval_batches,
                residual_canonicalizer=residual_canonicalizer,
            )
        )
    _append_csv(out_dir / "alignment_diagnostics.csv", alignment_rows)
    delta_standardizer = _fit_delta_standardizer(
        cfg,
        train_ds,
        int(cfg.get("stage3_v2", {}).get("online_batch_size", 1)),
        device,
        aligned_model,
        max_batches=args.max_generator_batches,
        residual_canonicalizer=residual_canonicalizer,
    )
    delta_standardizer_ckpt = None
    if delta_standardizer is not None:
        delta_standardizer_ckpt = ckpt_dir / "delta_standardizer.pt"
        torch.save(delta_standardizer.state_dict(), delta_standardizer_ckpt)

    residual_teacher, residual_teacher_rows = _fit_v9_residual_teacher(
        cfg=cfg,
        train_ds=train_ds,
        batch_size=int(cfg.get("stage3_v2", {}).get("online_batch_size", 1)),
        device=device,
        aligned_model=aligned_model,
        model=model,
        max_batches=args.max_generator_batches,
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
    )
    residual_teacher_ckpt = None
    if residual_teacher is not None:
        if isinstance(residual_teacher, TrainOnlyOutcomeAlignedBlockResidualTeacherV12):
            residual_teacher_ckpt = ckpt_dir / "residual_teacher_v12.pt"
            save_stage3_v12_teacher_checkpoint(
                path=residual_teacher_ckpt,
                teacher=residual_teacher,
                metadata={"run_id": run_id, "dataset": dataset_label, "seed": args.seed},
            )
            _append_csv(out_dir / "residual_teacher_v12_diagnostics.csv", residual_teacher_rows)
        elif isinstance(residual_teacher, TrainOnlyRobustBlockResidualTeacherV11):
            residual_teacher_ckpt = ckpt_dir / "residual_teacher_v11.pt"
            save_stage3_v11_teacher_checkpoint(
                path=residual_teacher_ckpt,
                teacher=residual_teacher,
                metadata={"run_id": run_id, "dataset": dataset_label, "seed": args.seed},
            )
            _append_csv(out_dir / "residual_teacher_v11_diagnostics.csv", residual_teacher_rows)
        elif isinstance(residual_teacher, TrainOnlyResidualTeacherV10):
            residual_teacher_ckpt = ckpt_dir / "residual_teacher_v10.pt"
            save_stage3_v10_teacher_checkpoint(
                path=residual_teacher_ckpt,
                teacher=residual_teacher,
                metadata={"run_id": run_id, "dataset": dataset_label, "seed": args.seed},
            )
            _append_csv(out_dir / "residual_teacher_v10_diagnostics.csv", residual_teacher_rows)
        else:
            residual_teacher_ckpt = ckpt_dir / "residual_teacher_v9.pt"
            save_stage3_v9_teacher_checkpoint(
                path=residual_teacher_ckpt,
                teacher=residual_teacher,
                metadata={"run_id": run_id, "dataset": dataset_label, "seed": args.seed},
            )
            _append_csv(out_dir / "residual_teacher_v9_diagnostics.csv", residual_teacher_rows)

    modes = args.modes or cfg["stage3_v2"].get("sensitivity_modes") or [
        "real",
        "zero",
        "shuffled",
        "sign_flipped",
        "amplified_2x",
        "random_matched_norm",
    ]
    validation_window_count = int(
        getattr(args, "validation_windows", None) or cfg.get("stage3_v2", {}).get("validation_windows", 1)
    )
    write_step_metrics = bool(cfg.get("stage3_v2", {}).get("write_step_metrics", False))
    teacher_cfg = cfg.get("stage3_v2", {}).get("residual_teacher", {})
    teacher_only_diagnostics_enabled = residual_teacher is not None and bool(
        teacher_cfg.get("run_teacher_only_diagnostics", True)
    )
    teacher_only_validation_summary: dict[str, Any] | None = None
    teacher_only_test_summary: dict[str, Any] | None = None
    teacher_only_rolling_aggregate: dict[str, Any] | None = None
    if teacher_only_diagnostics_enabled and residual_teacher is not None:
        teacher_validation_step_rows: list[dict[str, Any]] | None = [] if write_step_metrics else None
        teacher_validation_rows = _run_v9_teacher_only_rows(
            cfg=cfg,
            eval_split="validation",
            eval_ds=val_ds,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=args.seed,
            device=device,
            aligned_model=aligned_model,
            model=model,
            residual_teacher=residual_teacher,
            modes=modes,
            max_eval_batches=args.max_eval_batches,
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
            step_rows_out=teacher_validation_step_rows,
        )
        teacher_only_validation_summary = {
            "run_id": run_id,
            "dataset": dataset_label,
            "seed": args.seed,
            "split": "validation",
            **summarize_teacher_mode_rows(teacher_validation_rows),
        }
        _append_csv(out_dir / "teacher_only_validation_summary.csv", teacher_validation_rows)
        _append_csv(out_dir / "teacher_only_validation_diagnostic_summary.csv", [teacher_only_validation_summary])
        if teacher_validation_step_rows is not None:
            _append_csv(out_dir / "teacher_only_validation_step_metrics.csv", teacher_validation_step_rows)

        if validation_window_count > 1:
            teacher_rolling_step_rows: list[dict[str, Any]] | None = [] if write_step_metrics else None
            teacher_rolling_rows, teacher_rolling_window_rows, teacher_only_rolling_aggregate = (
                _run_v9_teacher_rolling_diagnostics(
                    cfg=cfg,
                    val_ds=val_ds,
                    dataset_label=dataset_label,
                    run_id=run_id,
                    seed=args.seed,
                    device=device,
                    aligned_model=aligned_model,
                    model=model,
                    residual_teacher=residual_teacher,
                    modes=modes,
                    max_eval_batches=args.max_eval_batches,
                    delta_standardizer=delta_standardizer,
                    residual_canonicalizer=residual_canonicalizer,
                    window_count=validation_window_count,
                    step_rows_out=teacher_rolling_step_rows,
                )
            )
            _append_csv(out_dir / "teacher_only_rolling_validation_summary.csv", teacher_rolling_rows)
            _append_csv(
                out_dir / "teacher_only_rolling_validation_window_summary.csv",
                teacher_rolling_window_rows,
            )
            _append_csv(
                out_dir / "teacher_only_rolling_validation_aggregate.csv",
                [teacher_only_rolling_aggregate],
            )
            if teacher_rolling_step_rows is not None:
                _append_csv(out_dir / "teacher_only_rolling_validation_step_metrics.csv", teacher_rolling_step_rows)

        teacher_test_step_rows: list[dict[str, Any]] | None = [] if write_step_metrics else None
        teacher_test_rows = _run_v9_teacher_only_rows(
            cfg=cfg,
            eval_split="test",
            eval_ds=test_ds,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=args.seed,
            device=device,
            aligned_model=aligned_model,
            model=model,
            residual_teacher=residual_teacher,
            modes=modes,
            max_eval_batches=args.max_eval_batches,
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
            step_rows_out=teacher_test_step_rows,
        )
        teacher_only_test_summary = {
            "run_id": run_id,
            "dataset": dataset_label,
            "seed": args.seed,
            "split": "test",
            **summarize_teacher_mode_rows(teacher_test_rows),
        }
        _append_csv(out_dir / "teacher_only_test_summary.csv", teacher_test_rows)
        _append_csv(out_dir / "teacher_only_test_diagnostic_summary.csv", [teacher_only_test_summary])
        if teacher_test_step_rows is not None:
            _append_csv(out_dir / "teacher_only_test_step_metrics.csv", teacher_test_step_rows)

        if bool(teacher_cfg.get("abort_if_teacher_diagnostic_fails", False)):
            if teacher_only_rolling_aggregate is None:
                raise RuntimeError("abort_if_teacher_diagnostic_fails=True requires rolling teacher diagnostics")
            if not bool(teacher_only_rolling_aggregate.get("teacher_diagnostic_gate_pass", False)):
                abort_payload = {
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": args.seed,
                    "reason": "teacher_only_diagnostic_gate_failed",
                    "teacher_only_rolling_validation_aggregate": teacher_only_rolling_aggregate,
                    "next_step": "Do not train a Stage III generator against a harmful or unsupported teacher.",
                }
                (out_dir / "teacher_only_abort.json").write_text(
                    json.dumps(abort_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                return out_dir

        if bool(getattr(args, "teacher_only", False)):
            if teacher_only_rolling_aggregate is None:
                raise RuntimeError("--teacher-only requires rolling teacher diagnostics")
            complete_payload = {
                "run_id": run_id,
                "dataset": dataset_label,
                "seed": args.seed,
                "reason": "teacher_only_requested",
                "teacher_only_rolling_validation_aggregate": teacher_only_rolling_aggregate,
                "teacher_only_test_summary": teacher_only_test_summary,
                "next_step": (
                    "Generator training remains gated until the teacher-only "
                    "matrix is reviewed across datasets and seeds."
                ),
            }
            (out_dir / "teacher_only_complete.json").write_text(
                json.dumps(complete_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return out_dir

    if args.teacher_only:
        blocked_payload = {
            "run_id": run_id,
            "dataset": dataset_label,
            "seed": args.seed,
            "reason": "teacher_only_requested_without_completed_teacher_diagnostics",
            "message": (
                "--teacher-only is a hard V12 safety guard. Generator training is disabled unless "
                "teacher-only diagnostics explicitly complete and return before this point."
            ),
        }
        (out_dir / "teacher_only_blocked_generator.json").write_text(
            json.dumps(blocked_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise RuntimeError(blocked_payload["message"])

    candidate_checkpoint_dir = ckpt_dir / "generator_candidates"
    generator, generator_rows, generator_candidates = _train_generator(
        cfg,
        model,
        aligned_model,
        train_ds,
        device,
        max_batches=args.max_generator_batches,
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
        residual_teacher=residual_teacher,
        candidate_checkpoint_dir=candidate_checkpoint_dir,
        candidate_metadata={"run_id": run_id, "dataset": dataset_label, "seed": args.seed},
    )
    generator_selection_rows: list[dict[str, Any]] = []
    selected_generator_candidate: Stage3GeneratorCandidate | None = None
    selection_cfg = cfg.get("stage3_v2", {}).get("generator", {}).get("checkpoint_selection", {})
    if bool(selection_cfg.get("enabled", True)):
        selected_generator_candidate, generator_selection_rows = _select_generator_checkpoint_on_rolling_validation(
            cfg=cfg,
            candidates=generator_candidates,
            val_ds=val_ds,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=args.seed,
            device=device,
            aligned_model=aligned_model,
            model=model,
            train_mse=train_mse,
            train_mae=train_mae,
            modes=modes,
            max_eval_batches=args.max_eval_batches,
            delta_standardizer=delta_standardizer,
            validation_window_count=validation_window_count,
            residual_canonicalizer=residual_canonicalizer,
        )
        _append_csv(out_dir / "generator_checkpoint_selection.csv", generator_selection_rows)
        generator, _ = load_stage3_generator_checkpoint(selected_generator_candidate.checkpoint_path, model=model)
        generator = generator.to(device)
    generator_ckpt = ckpt_dir / "generator.pt"
    selected_row = (
        select_best_generator_candidate(
            generator_selection_rows,
            abstain_if_no_pass=bool(selection_cfg.get("abstain_if_no_pass", False)),
        )
        if generator_selection_rows
        else {
            "candidate_epoch": generator_candidates[-1].epoch if generator_candidates else "",
            "generator_selection_score": "",
            "rolling_claim_gate_pass": "",
            "rolling_failure_reasons": "",
            "generator_selection_abstained": False,
            "generator_selection_abstain_reason": "",
        }
    )
    save_stage3_generator_checkpoint(
        path=generator_ckpt,
        generator=generator,
        metadata={
            "run_id": run_id,
            "dataset": dataset_label,
            "seed": args.seed,
            "selected_by": "rolling_validation" if generator_selection_rows else "final_epoch",
            "selected_candidate_checkpoint": (
                str(selected_generator_candidate.checkpoint_path) if selected_generator_candidate is not None else ""
            ),
            "selected_candidate_epoch": selected_row.get("candidate_epoch"),
            "generator_selection_score": selected_row.get("generator_selection_score"),
            "generator_selection_pass": selected_row.get("rolling_claim_gate_pass"),
            "generator_selection_failure_reasons": selected_row.get("rolling_failure_reasons"),
            "generator_selection_abstained": selected_row.get("generator_selection_abstained", False),
            "generator_selection_abstain_reason": selected_row.get("generator_selection_abstain_reason", ""),
        },
    )
    _append_csv(out_dir / "generator_train_metrics.csv", generator_rows)

    risk_gate = None
    risk_gate_ckpt = None
    risk_gate_metrics: dict[str, Any] = {}
    if bool(cfg["online"]["stage3"].get("risk_gate_enable", False)):
        risk_gate, risk_gate_rows, risk_gate_metrics = _train_risk_gate(
            cfg=cfg,
            model=model,
            generator=generator,
            aligned_model=aligned_model,
            train_ds=train_ds,
            device=device,
            max_batches=args.max_generator_batches,
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
        )
        risk_gate_ckpt = ckpt_dir / "risk_gate.pt"
        save_stage3_risk_gate_checkpoint(
            path=risk_gate_ckpt,
            gate=risk_gate.gate,
            standardizer=risk_gate.standardizer,
            metadata={"run_id": run_id, "dataset": dataset_label, "seed": args.seed},
        )
        _append_csv(out_dir / "risk_gate_train_examples.csv", risk_gate_rows)
        _append_csv(out_dir / "risk_gate_train_metrics.csv", [{"run_id": run_id, **risk_gate_metrics}])

    artifact_bundle = load_stage3_v12_artifact_bundle(
        generator_checkpoint=generator_ckpt,
        model=model,
        risk_gate_checkpoint=risk_gate_ckpt,
        delta_standardizer_checkpoint=delta_standardizer_ckpt,
        residual_canonicalizer_checkpoint=residual_canonicalizer_ckpt,
        residual_teacher_checkpoint=residual_teacher_ckpt,
    )
    generator = artifact_bundle.generator.to(device)
    risk_gate = artifact_bundle.risk_gate.to(device) if artifact_bundle.risk_gate is not None else None
    delta_standardizer = artifact_bundle.delta_standardizer
    residual_canonicalizer = artifact_bundle.residual_canonicalizer

    risk_gate_threshold_rows: list[dict[str, Any]] = []
    selected_risk_gate_threshold = cfg["online"]["stage3"].get("risk_gate_threshold")
    if risk_gate is not None and bool(cfg.get("stage3_v2", {}).get("risk_gate", {}).get("calibrate_threshold", True)):
        selected_risk_gate_threshold, risk_gate_threshold_rows = _select_risk_gate_threshold(
            cfg=cfg,
            val_ds=val_ds,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=args.seed,
            device=device,
            aligned_model=aligned_model,
            model=model,
            generator=generator,
            risk_gate=risk_gate,
            generator_ckpt=generator_ckpt,
            train_mse=train_mse,
            train_mae=train_mae,
            offline_eval_mse=val_mse,
            offline_eval_mae=val_mae,
            modes=modes,
            max_eval_batches=args.max_eval_batches,
            delta_standardizer=delta_standardizer,
            validation_window_count=validation_window_count,
            residual_canonicalizer=residual_canonicalizer,
        )
        _append_csv(out_dir / "risk_gate_threshold_selection.csv", risk_gate_threshold_rows)

    served_generator_selection_rows: list[dict[str, Any]] = []
    served_selected_row: dict[str, Any] = {}
    served_selection_artifacts: dict[str, Any] = {}
    if risk_gate is not None and bool(selection_cfg.get("served_policy_reselect_after_risk_gate", False)):
        selected_generator_candidate, served_generator_selection_rows = (
            _select_generator_checkpoint_on_served_policy_validation(
                cfg=cfg,
                candidates=generator_candidates,
                val_ds=val_ds,
                train_ds=train_ds,
                dataset_label=dataset_label,
                run_id=run_id,
                seed=args.seed,
                device=device,
                aligned_model=aligned_model,
                model=model,
                train_mse=train_mse,
                train_mae=train_mae,
                offline_eval_mse=val_mse,
                offline_eval_mae=val_mae,
                modes=modes,
                max_eval_batches=args.max_eval_batches,
                risk_gate_train_max_batches=args.max_generator_batches,
                delta_standardizer=delta_standardizer,
                validation_window_count=validation_window_count,
                residual_canonicalizer=residual_canonicalizer,
                risk_gate=risk_gate,
                selection_artifacts=served_selection_artifacts,
            )
        )
        _append_csv(out_dir / "generator_checkpoint_selection_served_policy.csv", served_generator_selection_rows)
        _append_csv(
            out_dir / "served_policy_candidate_window_breakdown.csv",
            served_selection_artifacts.get("all_candidate_window_rows", []),
        )
        _append_csv(
            out_dir / "served_policy_selected_candidate_window_breakdown.csv",
            served_selection_artifacts.get("candidate_window_rows", []),
        )
        served_selected_row = select_best_generator_candidate(
            served_generator_selection_rows,
            abstain_if_no_pass=bool(selection_cfg.get("abstain_if_no_pass", False)),
        )
        selected_row = served_selected_row
        _raise_if_final_generator_selection_abstained(selected_row, stage="served-policy")
        generator, _ = load_stage3_generator_checkpoint(selected_generator_candidate.checkpoint_path, model=model)
        generator = generator.to(device)
        generator_ckpt = ckpt_dir / "generator_served_policy.pt"
        if "selected_risk_gate_threshold" in served_selection_artifacts:
            selected_risk_gate_threshold = served_selection_artifacts["selected_risk_gate_threshold"]
            cfg["online"]["stage3"]["risk_gate_threshold"] = selected_risk_gate_threshold
        save_stage3_generator_checkpoint(
            path=generator_ckpt,
            generator=generator,
            metadata={
                "run_id": run_id,
                "dataset": dataset_label,
                "seed": args.seed,
                "selected_by": "served_policy_rolling_validation",
                "selected_candidate_checkpoint": str(selected_generator_candidate.checkpoint_path),
                "selected_candidate_epoch": selected_row.get("candidate_epoch"),
                "generator_selection_score": selected_row.get("generator_selection_score"),
                "generator_selection_pass": selected_row.get("rolling_claim_gate_pass"),
                "generator_selection_failure_reasons": selected_row.get("rolling_failure_reasons"),
                "generator_selection_abstained": selected_row.get("generator_selection_abstained", False),
                "generator_selection_abstain_reason": selected_row.get("generator_selection_abstain_reason", ""),
                "risk_gate_threshold_at_selection": selected_risk_gate_threshold,
            },
        )
        if bool(selection_cfg.get("retrain_risk_gate_per_candidate", False)):
            risk_gate = served_selection_artifacts.get("risk_gate", risk_gate)
            risk_gate_rows = served_selection_artifacts.get("risk_gate_rows", [])
            risk_gate_metrics = served_selection_artifacts.get("risk_gate_metrics", {})
            risk_gate_ckpt = ckpt_dir / "risk_gate_served_policy.pt"
            save_stage3_risk_gate_checkpoint(
                path=risk_gate_ckpt,
                gate=risk_gate.gate,
                standardizer=risk_gate.standardizer,
                metadata={
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": args.seed,
                    "selected_generator_checkpoint": str(generator_ckpt),
                    "selected_generator_epoch": selected_row.get("candidate_epoch"),
                    "selected_by": "candidate_specific_served_policy_validation",
                    "risk_gate_threshold_at_selection": selected_risk_gate_threshold,
                },
            )
            if risk_gate_rows:
                _append_csv(out_dir / "risk_gate_train_examples_served_policy.csv", risk_gate_rows)
            _append_csv(out_dir / "risk_gate_train_metrics_served_policy.csv", [{"run_id": run_id, **risk_gate_metrics}])
            if served_selection_artifacts.get("risk_gate_threshold_rows"):
                _append_csv(
                    out_dir / "risk_gate_threshold_selection_served_policy.csv",
                    served_selection_artifacts["risk_gate_threshold_rows"],
                )
        elif bool(selection_cfg.get("retrain_risk_gate_after_served_reselection", False)):
            risk_gate, risk_gate_rows, risk_gate_metrics = _train_risk_gate(
                cfg=cfg,
                model=model,
                generator=generator,
                aligned_model=aligned_model,
                train_ds=train_ds,
                device=device,
                max_batches=args.max_generator_batches,
                delta_standardizer=delta_standardizer,
                residual_canonicalizer=residual_canonicalizer,
            )
            risk_gate_ckpt = ckpt_dir / "risk_gate_served_policy.pt"
            save_stage3_risk_gate_checkpoint(
                path=risk_gate_ckpt,
                gate=risk_gate.gate,
                standardizer=risk_gate.standardizer,
                metadata={
                    "run_id": run_id,
                    "dataset": dataset_label,
                    "seed": args.seed,
                    "selected_generator_checkpoint": str(generator_ckpt),
                    "selected_generator_epoch": selected_row.get("candidate_epoch"),
                },
            )
            _append_csv(out_dir / "risk_gate_train_examples_served_policy.csv", risk_gate_rows)
            _append_csv(out_dir / "risk_gate_train_metrics_served_policy.csv", [{"run_id": run_id, **risk_gate_metrics}])

        artifact_bundle = load_stage3_v12_artifact_bundle(
            generator_checkpoint=generator_ckpt,
            model=model,
        risk_gate_checkpoint=risk_gate_ckpt,
        delta_standardizer_checkpoint=delta_standardizer_ckpt,
        residual_canonicalizer_checkpoint=residual_canonicalizer_ckpt,
        residual_teacher_checkpoint=residual_teacher_ckpt,
    )
        generator = artifact_bundle.generator.to(device)
        risk_gate = artifact_bundle.risk_gate.to(device) if artifact_bundle.risk_gate is not None else None
        delta_standardizer = artifact_bundle.delta_standardizer
        residual_canonicalizer = artifact_bundle.residual_canonicalizer

        if (
            risk_gate is not None
            and bool(selection_cfg.get("recalibrate_threshold_after_served_reselection", True))
            and not bool(selection_cfg.get("calibrate_threshold_per_candidate", False))
        ):
            selected_risk_gate_threshold, served_threshold_rows = _select_risk_gate_threshold(
                cfg=cfg,
                val_ds=val_ds,
                dataset_label=dataset_label,
                run_id=run_id,
                seed=args.seed,
                device=device,
                aligned_model=aligned_model,
                model=model,
                generator=generator,
                risk_gate=risk_gate,
                generator_ckpt=generator_ckpt,
                train_mse=train_mse,
                train_mae=train_mae,
                offline_eval_mse=val_mse,
                offline_eval_mae=val_mae,
                modes=modes,
                max_eval_batches=args.max_eval_batches,
                delta_standardizer=delta_standardizer,
                validation_window_count=validation_window_count,
                residual_canonicalizer=residual_canonicalizer,
            )
            _append_csv(out_dir / "risk_gate_threshold_selection_served_policy.csv", served_threshold_rows)
    validation_step_rows: list[dict[str, Any]] | None = [] if write_step_metrics else None
    validation_rows = _run_sensitivity_rows(
        cfg=cfg,
        eval_split="validation",
        eval_ds=val_ds,
        dataset_label=dataset_label,
        run_id=run_id,
        seed=args.seed,
        device=device,
        aligned_model=aligned_model,
        model=model,
        generator=generator,
        risk_gate=risk_gate,
        generator_ckpt=generator_ckpt,
        train_mse=train_mse,
        train_mae=train_mae,
        offline_eval_mse=val_mse,
        offline_eval_mae=val_mae,
        modes=modes,
        max_eval_batches=args.max_eval_batches,
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
        step_rows_out=validation_step_rows,
    )
    validation_gate_row = _sensitivity_gate_row(
        cfg=cfg,
        rows=validation_rows,
        run_id=run_id,
        dataset_label=dataset_label,
        seed=args.seed,
        split="validation",
    )
    _append_csv(out_dir / "validation_sensitivity_summary.csv", validation_rows)
    _append_csv(out_dir / "validation_sensitivity_claim_gate.csv", [validation_gate_row])
    if validation_step_rows is not None:
        _append_csv(out_dir / "validation_step_metrics.csv", validation_step_rows)
    rolling_validation_gate_row: dict[str, Any] | None = None
    rolling_validation_step_rows: list[dict[str, Any]] | None = [] if write_step_metrics else None
    if validation_window_count > 1:
        rolling_rows, rolling_gate_rows, rolling_validation_gate_row = _run_rolling_validation(
            cfg=cfg,
            val_ds=val_ds,
            dataset_label=dataset_label,
            run_id=run_id,
            seed=args.seed,
            device=device,
            aligned_model=aligned_model,
            model=model,
            generator=generator,
            risk_gate=risk_gate,
            generator_ckpt=generator_ckpt,
            train_mse=train_mse,
            train_mae=train_mae,
            modes=modes,
            max_eval_batches=args.max_eval_batches,
            delta_standardizer=delta_standardizer,
            residual_canonicalizer=residual_canonicalizer,
            window_count=validation_window_count,
            step_rows_out=rolling_validation_step_rows,
        )
        _append_csv(out_dir / "rolling_validation_sensitivity_summary.csv", rolling_rows)
        _append_csv(out_dir / "rolling_validation_claim_gate.csv", rolling_gate_rows)
        _append_csv(out_dir / "rolling_validation_claim_gate_aggregate.csv", [rolling_validation_gate_row])
        if rolling_validation_step_rows is not None:
            _append_csv(out_dir / "rolling_validation_step_metrics.csv", rolling_validation_step_rows)

    test_step_rows: list[dict[str, Any]] | None = [] if write_step_metrics else None
    metric_rows = _run_sensitivity_rows(
        cfg=cfg,
        eval_split="test",
        eval_ds=test_ds,
        dataset_label=dataset_label,
        run_id=run_id,
        seed=args.seed,
        device=device,
        aligned_model=aligned_model,
        model=model,
        generator=generator,
        risk_gate=risk_gate,
        generator_ckpt=generator_ckpt,
        train_mse=train_mse,
        train_mae=train_mae,
        offline_eval_mse=test_mse,
        offline_eval_mae=test_mae,
        modes=modes,
        max_eval_batches=args.max_eval_batches,
        delta_standardizer=delta_standardizer,
        residual_canonicalizer=residual_canonicalizer,
        step_rows_out=test_step_rows,
    )
    _append_csv(out_dir / "sensitivity_summary.csv", metric_rows)
    gate_row = _sensitivity_gate_row(
        cfg=cfg,
        rows=metric_rows,
        run_id=run_id,
        dataset_label=dataset_label,
        seed=args.seed,
        split="test",
    )
    _append_csv(out_dir / "sensitivity_claim_gate.csv", [gate_row])
    if test_step_rows is not None:
        _append_csv(out_dir / "step_metrics.csv", test_step_rows)

    manifest = {
        "run_id": run_id,
        "dataset": dataset_label,
        "seed": args.seed,
        "config": args.config,
        "config_sha256": _sha256_file(args.config),
        **resolved_config_snapshot,
        "data_path": cfg["data"]["path"],
        "data_sha256": _sha256_file(cfg["data"]["path"]) if cfg["data"]["source"] != "synthetic" else "synthetic",
        "output_dir": str(out_dir),
        "checkpoint_dir": str(ckpt_dir),
        "base_model_checkpoint_sha256": _sha256_file(ckpt_dir / "base_model.pt"),
        "aligned_representation_checkpoint_sha256": _sha256_file(ckpt_dir / "aligned_representation.pt"),
        "generator_checkpoint_sha256": _sha256_file(generator_ckpt),
        "risk_gate_checkpoint_sha256": _sha256_file(risk_gate_ckpt) if risk_gate_ckpt is not None else "",
        "residual_teacher_checkpoint_sha256": (
            _sha256_file(residual_teacher_ckpt) if residual_teacher_ckpt is not None else ""
        ),
        "delta_standardizer_checkpoint_sha256": (
            _sha256_file(ckpt_dir / "delta_standardizer.pt") if delta_standardizer is not None else ""
        ),
        "residual_canonicalizer_checkpoint_sha256": (
            _sha256_file(ckpt_dir / "residual_drift_canonicalizer.pt")
            if residual_canonicalizer is not None
            else ""
        ),
        "split_info": split_info,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_status_short": _git_status_short(),
        "stage3_drift_source": _stage3_drift_source(cfg),
        "stage3_drift_polarity": _stage3_drift_polarity(cfg),
        "stage3_protocol_version": cfg.get("stage3_v2", {}).get("protocol_version", cfg["online"]["stage3"].get("version", "v2")),
        "stage3_signal_mode": cfg["online"]["stage3"].get("signal_mode", "delta_vector"),
        "stage3_normalize_vector": bool(cfg["online"]["stage3"].get("normalize_vector", True)),
        "stage3_counterfactual_intervention_space": cfg.get("stage3_v2", {}).get("counterfactual_intervention_space", "generator_input"),
        "stage3_delta_calibrated": delta_standardizer is not None,
        "stage3_delta_calibration_fit_sample_count": (
            delta_standardizer.fit_sample_count if delta_standardizer is not None else 0
        ),
        "stage3_residual_drift_canonicalized": residual_canonicalizer is not None,
        "stage3_residual_canonicalizer_method": cfg["online"]["stage3"].get("canonicalizer_method", cfg.get("stage3_v2", {}).get("canonicalizer_method", "pls_v2")),
        "stage3_residual_canonicalizer_fit_sample_count": (
            residual_canonicalizer.fit_sample_count if residual_canonicalizer is not None else 0
        ),
        "stage3_residual_canonicalizer_transition_sample_count": (
            residual_canonicalizer.transition_sample_count if residual_canonicalizer is not None else 0
        ),
        "stage3_residual_teacher_enabled": residual_teacher is not None,
        "stage3_residual_teacher_fit_sample_count": (
            residual_teacher.fit_sample_count if residual_teacher is not None else 0
        ),
        "stage3_residual_teacher_train_r2": (
            residual_teacher.teacher_residual_r2 if residual_teacher is not None else ""
        ),
        "stage3_teacher_only_diagnostics_enabled": teacher_only_diagnostics_enabled,
        "stage3_teacher_only_validation_summary": teacher_only_validation_summary or {},
        "stage3_teacher_only_rolling_validation_aggregate": teacher_only_rolling_aggregate or {},
        "stage3_teacher_only_test_summary": teacher_only_test_summary or {},
        "stage3_artifacts_reloaded_for_eval": True,
        "stage3_generator_checkpoint_selection_enabled": bool(selection_cfg.get("enabled", True)),
        "stage3_generator_checkpoint_selection_stage": (
            "served_policy" if served_generator_selection_rows else ("pre_risk_gate" if generator_selection_rows else "none")
        ),
        "stage3_generator_selection_modes": ";".join(_selection_modes_for_validation(cfg, modes)),
        "stage3_generator_selection_excludes_offline_diagnostics": _selection_modes_for_validation(cfg, modes) != modes,
        "stage3_generator_retrain_risk_gate_per_candidate": bool(
            selection_cfg.get("retrain_risk_gate_per_candidate", False)
        ),
        "stage3_generator_calibrate_threshold_per_candidate": bool(
            selection_cfg.get("calibrate_threshold_per_candidate", False)
        ),
        "stage3_generator_candidate_count": len(generator_candidates),
        "stage3_generator_selected_candidate_checkpoint": (
            str(selected_generator_candidate.checkpoint_path) if selected_generator_candidate is not None else ""
        ),
        "stage3_generator_selected_candidate_epoch": selected_row.get("candidate_epoch"),
        "stage3_generator_selection_score": selected_row.get("generator_selection_score"),
        "stage3_generator_selection_pass": selected_row.get("rolling_claim_gate_pass"),
        "stage3_generator_selection_failure_reasons": selected_row.get("rolling_failure_reasons"),
        "stage3_generator_selection_abstained": selected_row.get("generator_selection_abstained", False),
        "stage3_generator_selection_abstain_reason": selected_row.get("generator_selection_abstain_reason", ""),
        "stage3_generator_served_policy_selection_enabled": bool(
            selection_cfg.get("served_policy_reselect_after_risk_gate", False)
        ),
        "stage3_generator_served_policy_selection_rows": len(served_generator_selection_rows),
        "stage3_generator_served_policy_selected_candidate_epoch": served_selected_row.get("candidate_epoch", ""),
        "stage3_generator_served_policy_selection_score": served_selected_row.get("generator_selection_score", ""),
        "stage3_generator_served_policy_selection_pass": served_selected_row.get("rolling_claim_gate_pass", ""),
        "stage3_generator_served_policy_selection_abstained": served_selected_row.get(
            "generator_selection_abstained", ""
        ),
        "stage3_generator_final_checkpoint": str(generator_ckpt),
        "stage3_risk_gate_enabled": risk_gate is not None,
        "stage3_risk_gate_threshold": selected_risk_gate_threshold,
        "stage3_risk_gate_threshold_selection_rows": len(risk_gate_threshold_rows),
        "stage3_risk_gate_served_policy_threshold_selection_rows": len(
            served_selection_artifacts.get("risk_gate_threshold_rows", [])
        ),
        "stage3_risk_gate_train_metrics": risk_gate_metrics,
        "stage3_step_metrics_enabled": write_step_metrics,
        "stage3_validation_step_metric_rows": len(validation_step_rows or []),
        "stage3_rolling_validation_step_metric_rows": len(rolling_validation_step_rows or []),
        "stage3_test_step_metric_rows": len(test_step_rows or []),
        "stage3_claim_gate_pass": gate_row["claim_gate_pass"],
        "stage3_claim_gate_failure_reasons": gate_row["failure_reasons"],
        "stage3_test_gate_role": "report_only",
        "stage3_validation_claim_gate_pass": validation_gate_row["claim_gate_pass"],
        "stage3_validation_claim_gate_failure_reasons": validation_gate_row["failure_reasons"],
        "stage3_validation_gate_role": "selection",
        "stage3_rolling_validation_enabled": validation_window_count > 1,
        "stage3_validation_window_count": validation_window_count,
        "stage3_rolling_validation_claim_gate_pass": (
            rolling_validation_gate_row["rolling_claim_gate_pass"] if rolling_validation_gate_row is not None else None
        ),
        "stage3_rolling_validation_claim_gate_failure_reasons": (
            rolling_validation_gate_row["rolling_failure_reasons"] if rolling_validation_gate_row is not None else ""
        ),
        "stage3_rolling_validation_pass_rate": (
            rolling_validation_gate_row["validation_window_pass_rate"] if rolling_validation_gate_row is not None else None
        ),
        "preprocessing_info": cfg.get("_audit_preprocessing_info", {}),
        "max_train_batches": args.max_train_batches,
        "max_generator_batches": args.max_generator_batches,
        "max_eval_batches": args.max_eval_batches,
        "runner_stream_materialized": False,
        "test_gate_report_only": True,
        "counterfactual_protocol_note": (
            "real/zero/sign_flipped/amplified modes are streaming-causal diagnostics; "
            "shuffled/random_matched_norm use full split drift and are offline diagnostics"
        ),
        "preprocessing_leakage_note": (
            "split-safe V2 preprocessing is active"
            if cfg.get("_audit_preprocessing_info", {}).get("split_safe_imputation")
            else "load_tabular_timeseries applies missing-value imputation before raw split; "
            "time_interpolate/ffill/bfill is not split-safe"
        ),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated Stage III V12 sensitivity audit.")
    parser.add_argument("--config", default="configs/stage3_v12_outcome_aligned_teacher_gate.yaml")
    parser.add_argument("--dataset", default=None, help="Optional public dataset name: panama/tetouan/spain_energy_weather")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--invariant-epochs", type=int, default=None)
    parser.add_argument("--generator-epochs", type=int, default=None)
    parser.add_argument("--out-root", default="logs/s3v12m")
    parser.add_argument("--checkpoint-root", default="ckpt/s3v12m")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--modes", nargs="*", default=None)
    parser.add_argument("--drift-source", choices=["phi_gap", "delta_phi_v"], default=None)
    parser.add_argument("--drift-polarity", type=float, default=None)
    parser.add_argument("--signal-mode", choices=["level_scalar", "delta_scalar", "level_vector", "delta_vector"], default=None)
    parser.add_argument("--normalize-vector", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--validation-windows", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-generator-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--teacher-only",
        action="store_true",
        help="Stop after residual-teacher diagnostics even if the teacher gate passes.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing V12 run directory.")
    return parser.parse_args()


def main() -> None:
    out_dir = run_stage3_v12_audit(parse_args())
    print(f"[stage3_v12] wrote audit outputs to {out_dir}")


if __name__ == "__main__":
    main()
