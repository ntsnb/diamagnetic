from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from models.adaptation_v2 import LayerAwareHyperAdapterGenerator, LayerAwareHyperLoRAGenerator
from models.aligned_representation_v2 import (
    TrainSetTemporalResidualCCAAligner,
    TrainSetWhitenedProcrustesTemporalResidualCCAAligner,
)
from models.stage3_alignment_v7 import TrainSetChronologicalResidualizedReducedRankCCAAligner
from train.online_loop_stage3_v2 import run_online_adaptation_v2
import pandas as pd

from train.run_stage3_v2_sensitivity_audit import (
    _alignment_diagnostic_rows,
    _aligned_stream,
    _build_stage3_generator_negative_drifts,
    _build_risk_gate_drift_candidates,
    _contiguous_dataset_windows,
    _candidate_window_diagnostic_rows,
    _fit_residual_canonicalizer,
    _collect_generator_input_drift,
    _load_series_and_features,
    _ordered_loader,
    _prepare_fresh_run_dirs,
    _raise_if_final_generator_selection_abstained,
    _risk_gate_label_for_mode,
    _run_sensitivity_rows,
    _select_generator_checkpoint_on_served_policy_validation,
    _selection_modes_for_validation,
    _write_resolved_config_snapshot,
)
from train.run_stage3_v2_sensitivity_audit import _sensitivity_gate_row
from train.run_stage3_v2_sensitivity_audit import run_stage3_v2_audit
from train.stage3_generator_selection_v2 import Stage3GeneratorCandidate
from train.stage3_generator_training import (
    load_stage3_generator_checkpoint,
    save_stage3_generator_checkpoint,
)
from train.stage3_v2_data import build_causal_dataloaders_from_series


def test_causal_dataloaders_split_raw_timeline_before_windowing():
    series = torch.arange(80, dtype=torch.float32).view(80, 1)
    lookback = 5
    horizon = 3

    result = build_causal_dataloaders_from_series(
        series=series,
        lookback=lookback,
        horizon=horizon,
        target_dim=0,
        batch_size=4,
        train_ratio=0.5,
        val_ratio=0.25,
        drop_last=False,
        split_mode="chronological",
        return_datasets=True,
        return_split_info=True,
    )
    _, _, _, train_ds, val_ds, test_ds, split_info = result

    assert split_info["train_target_end"] <= split_info["val_target_start"]
    assert split_info["val_target_end"] <= split_info["test_target_start"]
    assert train_ds.dataset.y[-1, -1].item() == split_info["train_target_end"] - 1
    assert val_ds.dataset.y[0, 0].item() == split_info["val_target_start"]
    assert test_ds.dataset.y[0, 0].item() == split_info["test_target_start"]


def test_v2_online_loop_rejects_batched_drift_when_causal_online_required():
    model = nn.Linear(2, 1, bias=False)
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes={"weight": tuple(model.weight.shape)},
        drift_dim=3,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )
    x = torch.randn(2, 2)
    y = torch.randn(2, 1)
    phi_i = torch.zeros(2, 3)
    phi_v = torch.ones(2, 3)

    with pytest.raises(ValueError, match="batch_size=1"):
        run_online_adaptation_v2(
            stream=[(x, y, phi_i, phi_v)],
            model=model,
            generator=gen,
            cfg={
                "online": {
                    "stage3": {
                        "enable": True,
                        "apply_transient": True,
                        "require_aligned_embeddings": True,
                        "causal_online": True,
                        "signal_mode": "level_vector",
                        "normalize_vector": False,
                        "target_modules": [],
                        "max_delta": 0.1,
                        "max_pred_shift": 10.0,
                        "blend_alpha": 1.0,
                    }
                }
            },
            device=torch.device("cpu"),
        )


def test_v2_online_loop_requires_embeddings_when_config_demands_alignment():
    model = nn.Linear(2, 1, bias=False)
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes={"weight": tuple(model.weight.shape)},
        drift_dim=3,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )

    with pytest.raises(ValueError, match="aligned embeddings"):
        run_online_adaptation_v2(
            stream=[(torch.randn(1, 2), torch.randn(1, 1))],
            model=model,
            generator=gen,
            cfg={
                "online": {
                    "stage3": {
                        "enable": True,
                        "apply_transient": True,
                        "require_aligned_embeddings": True,
                        "causal_online": True,
                        "signal_mode": "level_vector",
                        "normalize_vector": False,
                        "target_modules": [],
                        "max_delta": 0.1,
                        "max_pred_shift": 10.0,
                        "blend_alpha": 1.0,
                    }
                }
            },
            device=torch.device("cpu"),
        )


def test_aligned_stream_and_ordered_loader_preserve_batch_size_one_for_online_audit():
    dataset = torch.utils.data.TensorDataset(torch.randn(5, 2, 1), torch.randn(5, 1))
    loader = _ordered_loader(dataset, batch_size=2)

    assert [batch[0].shape[0] for batch in loader] == [2, 2, 1]

    class DummyAligned(nn.Module):
        def forward(self, x):
            class Out:
                pass

            class Rep:
                pass

            out = Out()
            out.representation = Rep()
            out.representation.phi_i = torch.zeros(x.shape[0], 3)
            out.representation.phi_v = torch.ones(x.shape[0], 3)
            return out

    stream_batches = list(
        _aligned_stream(
            dataset,
            batch_size=1,
            device=torch.device("cpu"),
            aligned_model=DummyAligned(),
            max_batches=3,
        )
    )

    assert [batch[0].shape[0] for batch in stream_batches] == [1, 1, 1]


def test_contiguous_dataset_windows_preserve_validation_order_and_cover_all_samples():
    dataset = torch.utils.data.TensorDataset(torch.arange(10).view(10, 1), torch.arange(10).view(10, 1))

    windows = _contiguous_dataset_windows(dataset, window_count=3)

    assert [len(window) for window in windows] == [4, 3, 3]
    covered = []
    for window in windows:
        covered.extend(int(window[i][0].item()) for i in range(len(window)))
    assert covered == list(range(10))


def test_stage3_v2_runner_can_use_split_safe_preprocessing(tmp_path: Path):
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=8, freq="h"),
            "load": [0.0, 1.0, 2.0, 3.0, -9999.0, 100.0, 101.0, 102.0],
        }
    ).to_csv(path, index=False)
    cfg = {
        "data": {
            "source": "file",
            "path": str(path),
            "timestamp_col": "date",
            "feature_cols": ["load"],
            "csv_encoding": None,
            "missing_sentinels": [-9999.0],
            "impute_method": "time_interpolate",
            "target_col": "load",
            "train_ratio": 0.625,
            "val_ratio": 0.125,
        },
        "stage3_v2": {
            "split_safe_imputation": True,
            "split_safe_impute_method": "causal_ffill_train_median",
        },
    }

    series, feature_cols, target_dim = _load_series_and_features(cfg)

    assert feature_cols == ["load"]
    assert target_dim == 0
    assert series[4, 0].item() == 3.0
    assert cfg["_audit_preprocessing_info"]["split_safe_imputation"] is True
    assert cfg["_audit_preprocessing_info"]["imputation_uses_future"] is False


def test_stage3_v2_runner_refuses_existing_run_directory_without_overwrite(tmp_path: Path):
    out_dir = tmp_path / "logs" / "run" / "panama" / "seed_42"
    ckpt_dir = tmp_path / "ckpt" / "panama" / "run" / "seed_42"
    out_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)
    sentinel = out_dir / "run_manifest.json"
    sentinel.write_text('{"existing": true}', encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refuse to overwrite"):
        _prepare_fresh_run_dirs(out_dir=out_dir, ckpt_dir=ckpt_dir, overwrite=False)

    assert sentinel.read_text(encoding="utf-8") == '{"existing": true}'


def test_stage3_v2_runner_allows_new_run_directories(tmp_path: Path):
    out_dir = tmp_path / "logs" / "new_run" / "panama" / "seed_42"
    ckpt_dir = tmp_path / "ckpt" / "panama" / "new_run" / "seed_42"

    _prepare_fresh_run_dirs(out_dir=out_dir, ckpt_dir=ckpt_dir, overwrite=False)

    assert out_dir.exists()
    assert ckpt_dir.exists()


def test_resolved_config_snapshot_writes_effective_config_and_hash(tmp_path: Path):
    cfg = {
        "seed": 42,
        "data": {"path": "dataset/weather.csv", "target_col": "OT"},
        "online": {"stage3": {"signal_mode": "delta_vector", "normalize_vector": True}},
    }

    snapshot = _write_resolved_config_snapshot(cfg=cfg, out_dir=tmp_path)

    assert snapshot["resolved_config_path"].endswith("resolved_config.json")
    assert len(snapshot["resolved_config_sha256"]) == 64
    text = (tmp_path / "resolved_config.json").read_text(encoding="utf-8")
    assert '"signal_mode": "delta_vector"' in text
    assert snapshot["resolved_config_sha256"] == snapshot["resolved_config_sha256_from_disk"]


def test_aligned_stream_can_emit_delta_phi_v_as_direct_drift_source():
    dataset = torch.utils.data.TensorDataset(torch.randn(2, 2, 1), torch.randn(2, 1))

    class DummyAligned(nn.Module):
        def forward(self, x):
            class Out:
                pass

            class Rep:
                pass

            out = Out()
            out.representation = Rep()
            out.representation.phi_i = torch.full((x.shape[0], 3), 10.0)
            out.representation.delta_phi_v = torch.tensor([[1.0, -2.0, 3.0]]).repeat(x.shape[0], 1)
            out.representation.phi_v = out.representation.phi_i + out.representation.delta_phi_v
            return out

    batch = next(
        _aligned_stream(
            dataset,
            batch_size=1,
            device=torch.device("cpu"),
            aligned_model=DummyAligned(),
            drift_source="delta_phi_v",
        )
    )

    _, _, phi_i, phi_v = batch
    assert torch.allclose(phi_i, torch.zeros_like(phi_i))
    assert torch.allclose(phi_v, torch.tensor([[1.0, -2.0, 3.0]]))


def test_generator_input_intervention_space_preserves_amplified_magnitude_after_normalization():
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[[1.0]], [[3.0]], [[6.0]]]),
        torch.zeros(3, 1),
    )

    class DummyAligned(nn.Module):
        def forward(self, x):
            class Out:
                pass

            class Rep:
                pass

            out = Out()
            out.representation = Rep()
            out.representation.delta_phi_v = torch.cat([x.reshape(x.shape[0], 1), 2.0 * x.reshape(x.shape[0], 1)], dim=1)
            out.representation.phi_i = torch.zeros_like(out.representation.delta_phi_v)
            out.representation.phi_v = out.representation.delta_phi_v
            return out

    generator_inputs = _collect_generator_input_drift(
        dataset,
        batch_size=1,
        device=torch.device("cpu"),
        aligned_model=DummyAligned(),
        max_batches=None,
        drift_source="delta_phi_v",
        signal_mode="level_vector",
        normalize_vector=True,
        drift_polarity=1.0,
    )

    assert torch.allclose(generator_inputs[0], generator_inputs[1])
    assert torch.allclose(2.0 * generator_inputs[0], 2.0 * generator_inputs[1])


def test_stage3_generator_negative_drifts_can_include_amplified_hard_negatives():
    drift_vec = torch.tensor([1.0, -2.0, 0.5])

    negative_drifts = _build_stage3_generator_negative_drifts(
        drift_vec=drift_vec,
        previous_drift_vec=None,
        gen_cfg={
            "include_random_matched_norm_negative": False,
            "amplified_negative_factors": [2.0, 5.0],
        },
    )

    assert set(negative_drifts) == {"zero", "sign_flipped", "amplified_2x", "amplified_5x"}
    assert torch.allclose(negative_drifts["zero"], torch.zeros_like(drift_vec))
    assert torch.allclose(negative_drifts["sign_flipped"], -drift_vec)
    assert torch.allclose(negative_drifts["amplified_2x"], 2.0 * drift_vec)
    assert torch.allclose(negative_drifts["amplified_5x"], 5.0 * drift_vec)


def test_risk_gate_counterfactual_drift_candidates_are_labeled_as_rejects():
    drift_vec = torch.tensor([1.0, -2.0, 0.5])
    previous_drift_vec = torch.tensor([-0.5, 1.0, 0.25])

    candidates = _build_risk_gate_drift_candidates(
        drift_vec=drift_vec,
        previous_drift_vec=previous_drift_vec,
        gen_cfg={
            "include_random_matched_norm_negative": False,
            "amplified_negative_factors": [2.0],
        },
        risk_gate_cfg={
            "include_counterfactual_examples": True,
            "counterfactual_modes": ["zero", "previous_step", "sign_flipped", "amplified_2x"],
            "counterfactual_label": 0.0,
        },
    )

    assert list(candidates) == ["real", "zero", "previous_step", "sign_flipped", "amplified_2x"]
    assert torch.allclose(candidates["real"], drift_vec)
    assert torch.allclose(candidates["previous_step"], previous_drift_vec)
    assert _risk_gate_label_for_mode("real", real_accept=1.0, counterfactual_label=0.0) == 1.0
    assert _risk_gate_label_for_mode("amplified_2x", real_accept=1.0, counterfactual_label=0.0) == 0.0


def test_selection_modes_exclude_offline_diagnostics_by_default():
    modes = ["real", "zero", "shuffled", "sign_flipped", "amplified_2x", "random_matched_norm"]

    selected = _selection_modes_for_validation({"stage3_v2": {}}, modes)

    assert selected == ["real", "zero", "sign_flipped", "amplified_2x"]


def test_candidate_window_diagnostic_rows_report_mode_and_counterfactual_context(tmp_path: Path):
    candidate = Stage3GeneratorCandidate(
        epoch=2,
        checkpoint_path=tmp_path / "generator_epoch_002.pt",
        last_train_loss=0.123,
    )
    metric_rows = [
        {
            "split": "validation_window_0",
            "validation_window_id": 0,
            "mode": "real",
            "online_base_mse": 10.0,
            "online_stage3_mse": 8.0,
            "online_base_mae": 5.0,
            "online_stage3_mae": 4.0,
            "online_stage3_use_rate": 0.7,
        },
        {
            "split": "validation_window_0",
            "validation_window_id": 0,
            "mode": "zero",
            "online_base_mse": 10.0,
            "online_stage3_mse": 9.0,
            "online_base_mae": 5.0,
            "online_stage3_mae": 4.5,
            "online_stage3_use_rate": 0.1,
        },
        {
            "split": "validation_window_0",
            "validation_window_id": 0,
            "mode": "amplified_2x",
            "online_base_mse": 10.0,
            "online_stage3_mse": 7.0,
            "online_base_mae": 5.0,
            "online_stage3_mae": 3.8,
            "online_stage3_use_rate": 0.9,
        },
    ]
    gate_rows = [
        {
            "split": "validation_window_0",
            "validation_window_id": 0,
            "claim_gate_pass": False,
            "failure_reasons": "counterfactual",
            "real_reject_rate": 0.2,
            "real_mse_gain": 0.2,
            "real_mae_gain": 0.2,
        }
    ]

    rows = _candidate_window_diagnostic_rows(
        candidate=candidate,
        metric_rows=metric_rows,
        gate_rows=gate_rows,
        selection_stage="served_policy",
        risk_gate_threshold=0.05,
    )

    assert len(rows) == 3
    real_row = next(row for row in rows if row["mode"] == "real")
    amplified_row = next(row for row in rows if row["mode"] == "amplified_2x")
    assert real_row["candidate_epoch"] == 2
    assert real_row["claim_gate_pass"] is False
    assert real_row["failure_reasons"] == "counterfactual"
    assert real_row["mode_is_counterfactual"] is False
    assert real_row["real_mse_gain"] == pytest.approx(0.2)
    assert real_row["best_counterfactual_mode"] == "amplified_2x"
    assert real_row["best_counterfactual_mse_gain"] == pytest.approx(0.3)
    assert real_row["real_vs_best_counterfactual_mse_gain_margin"] == pytest.approx(-0.1)
    assert amplified_row["mode_is_counterfactual"] is True
    assert amplified_row["mode_is_best_counterfactual"] is True
    assert amplified_row["mode_mse_gain"] == pytest.approx(0.3)


def test_final_generator_selection_abstention_blocks_served_artifact():
    with pytest.raises(RuntimeError, match="served-policy generator selection abstained"):
        _raise_if_final_generator_selection_abstained(
            {
                "generator_selection_abstained": True,
                "generator_selection_abstain_reason": "no_candidate_passed_claim_gate",
            },
            stage="served-policy",
        )


def test_served_policy_generator_selection_replays_candidates_with_risk_gate(monkeypatch, tmp_path: Path):
    calls: list[dict] = []
    sentinel_gate = object()
    candidates = [
        Stage3GeneratorCandidate(epoch=0, checkpoint_path=tmp_path / "generator_epoch_000.pt", last_train_loss=0.2),
        Stage3GeneratorCandidate(epoch=1, checkpoint_path=tmp_path / "generator_epoch_001.pt", last_train_loss=0.1),
    ]

    class DummyGenerator(nn.Module):
        pass

    def fake_load_stage3_generator_checkpoint(path, model):
        return DummyGenerator(), {}

    def fake_run_rolling_validation(**kwargs):
        calls.append(kwargs)
        candidate_name = Path(kwargs["generator_ckpt"]).name
        margin = 0.001 if candidate_name.endswith("000.pt") else 0.004
        aggregate = {
            "run_id": kwargs["run_id"],
            "dataset": kwargs["dataset_label"],
            "seed": kwargs["seed"],
            "split": "validation_rolling",
            "rolling_claim_gate_pass": candidate_name.endswith("001.pt"),
            "rolling_failure_reasons": "" if candidate_name.endswith("001.pt") else "counterfactual",
            "validation_window_count": 2,
            "validation_window_pass_count": 2 if candidate_name.endswith("001.pt") else 1,
            "validation_window_pass_rate": 1.0 if candidate_name.endswith("001.pt") else 0.5,
            "failed_validation_windows": "" if candidate_name.endswith("001.pt") else "0",
            "worst_window_real_mse_gain": 0.01,
            "worst_window_real_mae_gain": 0.01,
            "best_window_counterfactual_mse_gain": 0.002,
            "worst_window_real_vs_best_counterfactual_mse_gain_margin": margin,
            "max_window_real_reject_rate": 0.05,
        }
        return [], [], aggregate

    monkeypatch.setattr(
        "train.run_stage3_v2_sensitivity_audit.load_stage3_generator_checkpoint",
        fake_load_stage3_generator_checkpoint,
    )
    monkeypatch.setattr(
        "train.run_stage3_v2_sensitivity_audit._run_rolling_validation",
        fake_run_rolling_validation,
    )

    cfg = {
        "online": {"stage3": {"risk_gate_enable": True, "risk_gate_threshold": 0.05}},
        "stage3_v2": {"generator": {"checkpoint_selection": {"abstain_if_no_pass": True}}},
    }
    selected, rows = _select_generator_checkpoint_on_served_policy_validation(
        cfg=cfg,
        candidates=candidates,
        val_ds=object(),
        dataset_label="panama",
        run_id="served_policy_test",
        seed=42,
        device=torch.device("cpu"),
        aligned_model=object(),
        model=nn.Linear(1, 1),
        train_mse=1.0,
        train_mae=1.0,
        modes=["real", "amplified_2x"],
        max_eval_batches=2,
        delta_standardizer=None,
        validation_window_count=2,
        residual_canonicalizer=None,
        risk_gate=sentinel_gate,
    )

    assert selected.epoch == 1
    assert len(rows) == 2
    assert {row["generator_selection_stage"] for row in rows} == {"served_policy"}
    assert all(call["risk_gate"] is sentinel_gate for call in calls)
    assert all(call["cfg"]["online"]["stage3"]["risk_gate_enable"] is True for call in calls)
    assert all(call["cfg"]["online"]["stage3"]["risk_gate_threshold"] == 0.05 for call in calls)
    assert all(call["modes"] == ["real", "amplified_2x"] for call in calls)


def test_served_policy_generator_selection_can_train_candidate_specific_policy(monkeypatch, tmp_path: Path):
    calls: list[dict] = []
    trained_gates: list[object] = []
    threshold_rows_by_gate: dict[int, list[dict]] = {}
    candidates = [
        Stage3GeneratorCandidate(epoch=0, checkpoint_path=tmp_path / "generator_epoch_000.pt", last_train_loss=0.2),
        Stage3GeneratorCandidate(epoch=1, checkpoint_path=tmp_path / "generator_epoch_001.pt", last_train_loss=0.1),
    ]

    class DummyGenerator(nn.Module):
        pass

    def fake_load_stage3_generator_checkpoint(path, model):
        return DummyGenerator(), {}

    def fake_train_risk_gate(**kwargs):
        gate = object()
        trained_gates.append(gate)
        rows = [{"candidate_checkpoint": str(kwargs.get("generator"))}]
        metrics = {"candidate_gate_index": len(trained_gates) - 1}
        return gate, rows, metrics

    def fake_select_threshold(**kwargs):
        threshold = 0.1 if kwargs["generator_ckpt"].name.endswith("000.pt") else 0.2
        rows = [{"threshold": threshold, "candidate_checkpoint": str(kwargs["generator_ckpt"])}]
        threshold_rows_by_gate[id(kwargs["risk_gate"])] = rows
        kwargs["cfg"]["online"]["stage3"]["risk_gate_threshold"] = threshold
        return threshold, rows

    def fake_run_rolling_validation(**kwargs):
        calls.append(kwargs)
        candidate_name = Path(kwargs["generator_ckpt"]).name
        candidate_passes = candidate_name.endswith("001.pt")
        metric_rows = []
        gate_rows = []
        for window_id in range(2):
            for mode in kwargs["modes"]:
                metric_rows.append(
                    {
                        "split": f"validation_window_{window_id}",
                        "validation_window_id": window_id,
                        "mode": mode,
                        "online_base_mse": 10.0,
                        "online_stage3_mse": 8.0 if mode == "real" else 9.0,
                        "online_base_mae": 5.0,
                        "online_stage3_mae": 4.0 if mode == "real" else 4.5,
                        "online_stage3_use_rate": 0.5,
                    }
                )
            gate_rows.append(
                {
                    "split": f"validation_window_{window_id}",
                    "validation_window_id": window_id,
                    "claim_gate_pass": candidate_passes,
                    "failure_reasons": "" if candidate_passes else "counterfactual",
                    "real_reject_rate": 0.05,
                }
            )
        aggregate = {
            "run_id": kwargs["run_id"],
            "dataset": kwargs["dataset_label"],
            "seed": kwargs["seed"],
            "split": "validation_rolling",
            "rolling_claim_gate_pass": candidate_passes,
            "rolling_failure_reasons": "" if candidate_passes else "counterfactual",
            "validation_window_count": 2,
            "validation_window_pass_count": 2 if candidate_passes else 1,
            "validation_window_pass_rate": 1.0 if candidate_passes else 0.5,
            "failed_validation_windows": "" if candidate_passes else "0",
            "worst_window_real_mse_gain": 0.01,
            "worst_window_real_mae_gain": 0.01,
            "best_window_counterfactual_mse_gain": 0.002,
            "worst_window_real_vs_best_counterfactual_mse_gain_margin": 0.004,
            "max_window_real_reject_rate": 0.05,
        }
        return metric_rows, gate_rows, aggregate

    monkeypatch.setattr(
        "train.run_stage3_v2_sensitivity_audit.load_stage3_generator_checkpoint",
        fake_load_stage3_generator_checkpoint,
    )
    monkeypatch.setattr("train.run_stage3_v2_sensitivity_audit._train_risk_gate", fake_train_risk_gate)
    monkeypatch.setattr("train.run_stage3_v2_sensitivity_audit._select_risk_gate_threshold", fake_select_threshold)
    monkeypatch.setattr(
        "train.run_stage3_v2_sensitivity_audit._run_rolling_validation",
        fake_run_rolling_validation,
    )

    artifacts: dict[str, object] = {}
    cfg = {
        "online": {"stage3": {"risk_gate_enable": True, "risk_gate_threshold": 0.05}},
        "stage3_v2": {
            "generator": {
                "checkpoint_selection": {
                    "abstain_if_no_pass": True,
                    "retrain_risk_gate_per_candidate": True,
                    "calibrate_threshold_per_candidate": True,
                }
            },
            "risk_gate": {"calibrate_threshold": True},
        },
    }
    selected, rows = _select_generator_checkpoint_on_served_policy_validation(
        cfg=cfg,
        candidates=candidates,
        val_ds=object(),
        train_ds=object(),
        dataset_label="panama",
        run_id="served_policy_test",
        seed=42,
        device=torch.device("cpu"),
        aligned_model=object(),
        model=nn.Linear(1, 1),
        train_mse=1.0,
        train_mae=1.0,
        offline_eval_mse=0.7,
        offline_eval_mae=0.6,
        modes=["real", "shuffled", "amplified_2x", "random_matched_norm"],
        max_eval_batches=2,
        risk_gate_train_max_batches=3,
        delta_standardizer=None,
        validation_window_count=2,
        residual_canonicalizer=None,
        risk_gate=object(),
        selection_artifacts=artifacts,
    )

    assert selected.epoch == 1
    assert len(rows) == 2
    assert len(trained_gates) == 2
    assert artifacts["risk_gate"] is trained_gates[1]
    assert artifacts["selected_risk_gate_threshold"] == 0.2
    assert artifacts["risk_gate_threshold_rows"] == threshold_rows_by_gate[id(trained_gates[1])]
    assert len(artifacts["all_candidate_window_rows"]) == 8
    assert len(artifacts["candidate_window_rows"]) == 4
    assert {row["candidate_epoch"] for row in artifacts["candidate_window_rows"]} == {1}
    assert {row["risk_gate_threshold"] for row in artifacts["candidate_window_rows"]} == {0.2}
    assert all(call["risk_gate"] in trained_gates for call in calls)
    assert all(call["modes"] == ["real", "amplified_2x"] for call in calls)


def test_aligned_stream_rejects_unknown_drift_source():
    dataset = torch.utils.data.TensorDataset(torch.randn(2, 2, 1), torch.randn(2, 1))

    class DummyAligned(nn.Module):
        def forward(self, x):
            raise AssertionError("invalid drift_source should fail before model forward")

    with pytest.raises(ValueError, match="drift_source"):
        next(
            _aligned_stream(
                dataset,
                batch_size=1,
                device=torch.device("cpu"),
                aligned_model=DummyAligned(),
                drift_source="not_a_source",
            )
        )


def test_fit_residual_canonicalizer_can_select_temporal_residual_cca_v3():
    dataset = torch.utils.data.TensorDataset(torch.randn(10, 2, 1), torch.randn(10, 1))

    class DummyAligned(nn.Module):
        def forward(self, x):
            class Out:
                pass

            class Rep:
                pass

            flat = x.reshape(x.shape[0], -1)
            out = Out()
            out.representation = Rep()
            out.representation.delta_phi_v = torch.cat([flat, flat[:, :1] * 0.5], dim=1)
            out.y_inv = torch.zeros(x.shape[0], 1, device=x.device)
            return out

    canonicalizer = _fit_residual_canonicalizer(
        cfg={
            "stage3_v2": {},
            "online": {
                "stage3": {
                    "canonicalize_delta": True,
                    "canonicalizer_method": "temporal_residual_cca_v3",
                    "drift_source": "delta_phi_v",
                }
            },
        },
        train_ds=dataset,
        batch_size=1,
        device=torch.device("cpu"),
        aligned_model=DummyAligned(),
    )

    assert isinstance(canonicalizer, TrainSetTemporalResidualCCAAligner)
    assert canonicalizer.fit_sample_count == 10


def test_fit_residual_canonicalizer_can_select_whitened_procrustes_v6():
    dataset = torch.utils.data.TensorDataset(torch.randn(12, 2, 1), torch.randn(12, 1))

    class DummyAligned(nn.Module):
        def forward(self, x):
            class Out:
                pass

            class Rep:
                pass

            flat = x.reshape(x.shape[0], -1)
            out = Out()
            out.representation = Rep()
            out.representation.phi_i = torch.cat([flat, flat[:, :1] * 0.25], dim=1)
            out.representation.phi_v = out.representation.phi_i + torch.cat(
                [0.1 * flat, flat[:, :1] - flat[:, 1:2]],
                dim=1,
            )
            out.representation.delta_phi_v = out.representation.phi_v - out.representation.phi_i
            out.y_inv = torch.zeros(x.shape[0], 1, device=x.device)
            return out

    canonicalizer = _fit_residual_canonicalizer(
        cfg={
            "stage3_v2": {},
            "online": {
                "stage3": {
                    "canonicalize_delta": True,
                    "canonicalizer_method": "whitened_procrustes_temporal_residual_cca_v6",
                    "drift_source": "delta_phi_v",
                }
            },
        },
        train_ds=dataset,
        batch_size=1,
        device=torch.device("cpu"),
        aligned_model=DummyAligned(),
    )

    assert isinstance(canonicalizer, TrainSetWhitenedProcrustesTemporalResidualCCAAligner)
    assert canonicalizer.fit_sample_count == 12


def test_fit_residual_canonicalizer_can_select_residualized_rrcca_v7():
    torch.manual_seed(73)
    n = 24
    latent = torch.cumsum(torch.randn(n, 2), dim=0)
    invariant_noise = torch.randn(n, 2)
    x = torch.cat([invariant_noise, latent], dim=1).reshape(n, 4, 1)
    y = torch.stack([latent[:, 0] + 0.2 * latent[:, 1]], dim=1)
    dataset = torch.utils.data.TensorDataset(x, y)

    class DummyAligned(nn.Module):
        def forward(self, x):
            class Out:
                pass

            class Rep:
                pass

            flat = x.reshape(x.shape[0], -1)
            phi_i = torch.cat([flat[:, :2], flat[:, :1] * 0.1, flat[:, 1:2] * -0.2], dim=1)
            signal = torch.cat(
                [
                    torch.zeros_like(flat[:, :1]),
                    flat[:, 2:3],
                    torch.zeros_like(flat[:, :1]),
                    -flat[:, 3:4],
                ],
                dim=1,
            )
            out = Out()
            out.representation = Rep()
            out.representation.phi_i = phi_i
            out.representation.phi_v = phi_i + signal
            out.representation.delta_phi_v = signal
            out.y_inv = torch.zeros(x.shape[0], 1, device=x.device)
            return out

    canonicalizer = _fit_residual_canonicalizer(
        cfg={
            "stage3_v2": {},
            "online": {
                "stage3": {
                    "canonicalize_delta": True,
                    "canonicalizer_method": "residualized_rrcca_v7",
                    "canonicalizer_max_rank": 2,
                    "canonicalizer_min_fit_correlation": 0.01,
                    "canonicalizer_min_calibration_correlation": 0.01,
                    "canonicalizer_min_calibration_r2": 0.0001,
                    "canonicalizer_bootstrap_samples": 8,
                    "canonicalizer_min_bootstrap_sign_stability": 0.5,
                    "drift_source": "delta_phi_v",
                }
            },
        },
        train_ds=dataset,
        batch_size=1,
        device=torch.device("cpu"),
        aligned_model=DummyAligned(),
    )

    assert isinstance(canonicalizer, TrainSetChronologicalResidualizedReducedRankCCAAligner)
    assert canonicalizer.fit_sample_count == n
    assert canonicalizer.active_rank >= 1


def test_alignment_diagnostic_rows_report_pre_and_post_canonicalizer_metrics():
    torch.manual_seed(57)
    dataset = torch.utils.data.TensorDataset(torch.randn(12, 2, 1), torch.randn(12, 1))

    class DummyAligned(nn.Module):
        def forward(self, x):
            class Out:
                pass

            class Rep:
                pass

            flat = x.reshape(x.shape[0], -1)
            out = Out()
            out.representation = Rep()
            out.representation.delta_phi_v = torch.cat([flat, flat[:, :1] - flat[:, 1:2]], dim=1)
            out.y_inv = torch.zeros(x.shape[0], 1, device=x.device)
            return out

    aligned = DummyAligned()
    canonicalizer = _fit_residual_canonicalizer(
        cfg={
            "stage3_v2": {},
            "online": {
                "stage3": {
                    "canonicalize_delta": True,
                    "canonicalizer_method": "temporal_residual_cca_v3",
                    "drift_source": "delta_phi_v",
                }
            },
        },
        train_ds=dataset,
        batch_size=1,
        device=torch.device("cpu"),
        aligned_model=aligned,
    )

    rows = _alignment_diagnostic_rows(
        dataset=dataset,
        split="train",
        batch_size=1,
        device=torch.device("cpu"),
        aligned_model=aligned,
        max_batches=None,
        residual_canonicalizer=canonicalizer,
    )

    assert [row["alignment_stage"] for row in rows] == ["pre_canonical", "post_canonical"]
    assert rows[0]["split"] == "train"
    assert "top_abs_transition_residual_corr" in rows[0]
    assert "mean_best_abs_transition_residual_corr" in rows[1]


def test_hyperlora_checkpoint_round_trip_preserves_architecture(tmp_path: Path):
    model = nn.Linear(3, 2, bias=False)
    gen = LayerAwareHyperLoRAGenerator(
        named_param_shapes={"weight": tuple(model.weight.shape)},
        drift_dim=4,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )
    path = tmp_path / "generator.pt"
    save_stage3_generator_checkpoint(
        path=path,
        generator=gen,
        metadata={"run_id": "roundtrip"},
    )

    loaded, metadata = load_stage3_generator_checkpoint(path, model=model)

    assert metadata["run_id"] == "roundtrip"
    assert isinstance(loaded, LayerAwareHyperLoRAGenerator)
    assert loaded.registered_param_names == gen.registered_param_names
    drift = torch.randn(4)
    delta_before = gen.delta_for_param("weight", model.weight, drift)
    delta_after = loaded.delta_for_param("weight", model.weight, drift)
    assert torch.allclose(delta_before, delta_after)


def test_hyperadapter_checkpoint_round_trip_preserves_bias_adapter(tmp_path: Path):
    model = nn.Linear(3, 2, bias=True)
    gen = LayerAwareHyperAdapterGenerator(
        named_param_shapes={
            "weight": tuple(model.weight.shape),
            "bias": tuple(model.bias.shape),
        },
        drift_dim=4,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )
    path = tmp_path / "generator_v8.pt"
    save_stage3_generator_checkpoint(
        path=path,
        generator=gen,
        metadata={"run_id": "roundtrip_v8"},
    )

    loaded, metadata = load_stage3_generator_checkpoint(path, model=model)

    assert metadata["run_id"] == "roundtrip_v8"
    assert isinstance(loaded, LayerAwareHyperAdapterGenerator)
    assert loaded.registered_param_names == gen.registered_param_names
    drift = torch.randn(4)
    deltas_before = {
        name: gen.delta_for_param(name, param, drift)
        for name, param in model.named_parameters()
    }
    deltas_after = {
        name: loaded.delta_for_param(name, param, drift)
        for name, param in model.named_parameters()
    }
    assert set(deltas_after) == {"weight", "bias"}
    for name in deltas_before:
        assert torch.allclose(deltas_before[name], deltas_after[name])


def test_stage3_v2_audit_cli_overrides_signal_mode_without_mutating_base_config(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_apply_dataset_override(cfg, dataset_name):
        return cfg

    def fake_load_yaml(path):
        return {
            "seed": 42,
            "data": {"source": "synthetic", "path": "synthetic"},
            "online": {"stage3": {"signal_mode": "delta_vector"}},
            "stage3_v2": {"generator": {}},
        }

    def fake_set_seed(seed):
        captured["seed"] = seed

    def stop_after_override(cfg):
        captured["signal_mode"] = cfg["online"]["stage3"]["signal_mode"]
        captured["normalize_vector"] = cfg["online"]["stage3"]["normalize_vector"]
        raise RuntimeError("stop")

    monkeypatch.setattr("train.run_stage3_v2_sensitivity_audit.load_yaml", fake_load_yaml)
    monkeypatch.setattr("train.run_stage3_v2_sensitivity_audit._apply_dataset_override", fake_apply_dataset_override)
    monkeypatch.setattr("train.run_stage3_v2_sensitivity_audit.set_seed", fake_set_seed)
    monkeypatch.setattr("train.run_stage3_v2_sensitivity_audit._load_series_and_features", stop_after_override)

    class Args:
        config = "dummy.yaml"
        dataset = None
        epochs = None
        invariant_epochs = None
        generator_epochs = None
        drift_source = None
        drift_polarity = None
        signal_mode = "level_vector"
        normalize_vector = False
        seed = 43
        run_id = "override_test"
        out_root = str(tmp_path / "logs")
        checkpoint_root = str(tmp_path / "ckpt")
        device = "cpu"
        max_train_batches = None
        max_eval_batches = None
        max_generator_batches = None
        modes = ["real"]

    try:
        run_stage3_v2_audit(Args())
    except RuntimeError as exc:
        assert str(exc) == "stop"

    assert captured["seed"] == 43
    assert captured["signal_mode"] == "level_vector"
    assert captured["normalize_vector"] is False


def test_sensitivity_gate_row_records_split_name_and_gate_metrics():
    rows = [
        {
            "mode": "real",
            "online_base_mse": 1.0,
            "online_stage3_mse": 0.8,
            "online_base_mae": 1.0,
            "online_stage3_mae": 0.8,
            "online_stage3_delta_norm_pre_clamp": 0.3,
            "online_stage3_pred_shift": 0.1,
        },
        {
            "mode": "zero",
            "online_base_mse": 1.0,
            "online_stage3_mse": 1.0,
            "online_base_mae": 1.0,
            "online_stage3_mae": 1.0,
            "online_stage3_delta_norm_pre_clamp": 0.0,
            "online_stage3_pred_shift": 0.0,
        },
    ]

    gate = _sensitivity_gate_row(
        cfg={"stage3_v2": {"claim_gate": {"min_real_mse_gain": 0.0}}},
        rows=rows,
        run_id="run",
        dataset_label="data",
        seed=42,
        split="validation",
    )

    assert gate["split"] == "validation"
    assert gate["run_id"] == "run"
    assert gate["claim_gate_pass"] is True


def test_run_sensitivity_rows_can_collect_step_metrics_with_mode_context():
    torch.manual_seed(58)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([[1.5], [2.5], [3.5]]),
    )
    model = nn.Linear(1, 1, bias=False)
    generator = LayerAwareHyperLoRAGenerator(
        named_param_shapes={"weight": tuple(model.weight.shape)},
        drift_dim=2,
        hidden_dim=8,
        rank=2,
        max_scale=0.1,
    )

    class DummyAligned(nn.Module):
        def forward(self, x):
            class Out:
                pass

            class Rep:
                pass

            drift = torch.cat([x.reshape(x.shape[0], 1), 2.0 * x.reshape(x.shape[0], 1)], dim=1)
            out = Out()
            out.representation = Rep()
            out.representation.delta_phi_v = drift
            out.representation.phi_i = torch.zeros_like(drift)
            out.representation.phi_v = drift
            return out

    cfg = {
        "stage3_v2": {
            "online_batch_size": 1,
            "counterfactual_intervention_space": "generator_input",
        },
        "online": {
            "stage3": {
                "enable": True,
                "apply_transient": True,
                "causal_online": False,
                "drift_source": "delta_phi_v",
                "signal_mode": "level_vector",
                "normalize_vector": False,
                "drift_polarity": 1.0,
                "target_modules": [],
                "max_delta": 0.1,
                "max_pred_shift": 10.0,
                "blend_alpha": 1.0,
            }
        },
    }
    step_rows: list[dict] = []

    metric_rows = _run_sensitivity_rows(
        cfg=cfg,
        eval_split="test",
        eval_ds=dataset,
        dataset_label="toy",
        run_id="trace_run",
        seed=42,
        device=torch.device("cpu"),
        aligned_model=DummyAligned(),
        model=model,
        generator=generator,
        risk_gate=None,
        generator_ckpt=Path("generator.pt"),
        train_mse=0.1,
        train_mae=0.2,
        offline_eval_mse=0.3,
        offline_eval_mae=0.4,
        modes=["real", "amplified_2x"],
        max_eval_batches=2,
        delta_standardizer=None,
        residual_canonicalizer=None,
        step_rows_out=step_rows,
    )

    assert [row["mode"] for row in metric_rows] == ["real", "amplified_2x"]
    assert "step_rows" not in metric_rows[0]
    assert len(step_rows) == 4
    assert {row["mode"] for row in step_rows} == {"real", "amplified_2x"}
    assert {row["split"] for row in step_rows} == {"test"}
    assert {row["run_id"] for row in step_rows} == {"trace_run"}
    assert all(row["counterfactual_intervention_space"] == "generator_input" for row in step_rows)
