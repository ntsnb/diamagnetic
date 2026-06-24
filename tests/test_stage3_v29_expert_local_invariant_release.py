from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.factor_mechanism_v29 import FactorMechanismV29Config, build_factor_mechanism_v29
from train.online_loop_stage3_v29 import run_online_adaptation_v29
import train.run_stage3_v29_expert_local_invariant_release_probe as v29_probe
from train.run_stage3_v29_expert_local_invariant_release_probe import (
    _collect_split_tensors_v29,
    _rename_v18_outputs_to_v29,
    collect_stage3_v29_online_gate_dynamic_feature_rows,
    run_stage3_v29_expert_local_invariant_release_probe,
    validate_v29_config_contract,
    write_stage3_v29_online_gate_dynamic_features,
)


def _tiny_cfg() -> dict:
    return {
        "data": {"lookback": 4, "horizon": 1},
        "model_v29": {"repr_dim": 8},
        "stage3_v2": {
            "protocol_version": "v29_expert_local_invariant_release_probe",
            "log_root": "logs/s3v29expert",
            "checkpoint_root": "ckpt/s3v29expert",
            "results_root": "results/s3v29expert",
        },
        "online": {
            "stage3": {
                "version": "v29_expert_local_invariant_release_probe",
                "drift_source": "pooled_servable_change",
                "signal_mode": "fluctuation_delta",
                "causal_online": True,
                "online_batch_size": 1,
            }
        },
    }


def _tiny_model():
    return build_factor_mechanism_v29(
        FactorMechanismV29Config(
            input_dim=3,
            seq_len=4,
            out_dim=1,
            d_model=16,
            n_experts=2,
            latent_dim=8,
            repr_dim=8,
            expert_hidden_dim=16,
            history_hidden_dim=12,
            release_hidden_dim=12,
            change_memory_hidden_dim=12,
            env_count=3,
            router_env_hidden_dim=8,
            invariant_release_hidden_dim=12,
            router_invariant_hidden_dim=8,
            temporal_encoder="transformer",
            nhead=2,
            num_layers=1,
            dropout=0.0,
            use_revin=False,
            use_vae=True,
            vae_eval_mode="mean",
            zero_init_variant=True,
            use_projected_residual=True,
        )
    )


def test_validate_v29_config_contract_requires_v29_roots() -> None:
    cfg = _tiny_cfg()
    validate_v29_config_contract(cfg)
    cfg["stage3_v2"]["log_root"] = "logs/s3v28expert"
    with pytest.raises(ValueError):
        validate_v29_config_contract(cfg)


def test_validate_v29_config_contract_requires_v29_drift_source() -> None:
    cfg = _tiny_cfg()
    cfg["online"]["stage3"]["drift_source"] = "delta_phi_v"
    with pytest.raises(ValueError):
        validate_v29_config_contract(cfg)


def test_collect_split_tensors_v29_uses_expert_local_release_package() -> None:
    torch.manual_seed(31)
    model = _tiny_model().eval()
    ds = TensorDataset(torch.randn(6, 4, 3), torch.randn(6, 1))
    raw, y_base, y = _collect_split_tensors_v29(
        dataset=ds,
        batch_size=2,
        device=torch.device("cpu"),
        model=model,
        max_batches=2,
    )
    assert raw.shape[0] == y_base.shape[0] == y.shape[0] == 4
    assert raw.shape[1] == 7 * 8 + 2 * 3 + 4 + 5 * 1


def test_collect_stage3_v29_online_gate_dynamic_feature_rows_are_pre_outcome() -> None:
    torch.manual_seed(37)
    model = _tiny_model().eval()
    ds = TensorDataset(torch.randn(6, 4, 3), torch.randn(6, 1))

    rows = collect_stage3_v29_online_gate_dynamic_feature_rows(
        dataset=ds,
        batch_size=2,
        device=torch.device("cpu"),
        model=model,
        max_batches=2,
        split="validation",
        target_name="raw_residual_vector",
        window_size=2,
    )

    assert len(rows) == 2
    assert rows[0]["validation_window_id"] == "0"
    assert rows[1]["validation_window_id"] == "1"
    assert rows[0]["feature_dynamic_status"] == "available_from_model_forward"
    assert rows[0]["feature_dynamic_sample_count"] == 2
    banned_tokens = ("gain", "counterfactual", "label", "partial", "teacher_real", "outcome", "best", "target_y")
    leaking_feature_columns = [
        key
        for key in rows[0]
        if key.startswith("feature_") and any(token in key for token in banned_tokens)
    ]
    assert leaking_feature_columns == []


def test_write_stage3_v29_online_gate_dynamic_features_writes_validation_csv(tmp_path) -> None:
    torch.manual_seed(43)
    model = _tiny_model().eval()
    ds = TensorDataset(torch.randn(6, 4, 3), torch.randn(6, 1))
    args = argparse.Namespace(max_eval_batches=2, validation_windows=2, targets=["raw_residual_vector"])

    rows = write_stage3_v29_online_gate_dynamic_features(
        out_dir=tmp_path,
        cfg=_tiny_cfg(),
        model=model,
        train_ds=ds,
        val_ds=ds,
        test_ds=ds,
        batch_size=2,
        device=torch.device("cpu"),
        args=args,
    )

    path = tmp_path / "stage3_v29_online_gate_dynamic_features.csv"
    assert path.exists()
    written = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == len(written) == 2
    assert written[0]["split"] == "validation"
    assert written[0]["feature_dynamic_status"] == "available_from_model_forward"
    assert written[0]["validation_window_id"] == "0"
    banned_tokens = ("gain", "counterfactual", "label", "partial", "teacher_real", "outcome", "best", "target_y")
    leaking_feature_columns = [
        key
        for key in written[0]
        if key.startswith("feature_") and any(token in key for token in banned_tokens)
    ]
    assert leaking_feature_columns == []


def test_write_stage3_v29_online_gate_dynamic_features_resolves_config_targets(tmp_path) -> None:
    torch.manual_seed(47)
    model = _tiny_model().eval()
    ds = TensorDataset(torch.randn(4, 4, 3), torch.randn(4, 1))
    cfg = _tiny_cfg()
    cfg["stage3_v2"]["v15_targets"] = {
        "primary": ["raw_residual_vector"],
        "secondary": ["residual_sign"],
    }
    args = argparse.Namespace(max_eval_batches=1, validation_windows=1, targets=None)

    rows = write_stage3_v29_online_gate_dynamic_features(
        out_dir=tmp_path,
        cfg=cfg,
        model=model,
        train_ds=ds,
        val_ds=ds,
        test_ds=ds,
        batch_size=2,
        device=torch.device("cpu"),
        args=args,
    )

    assert rows[0]["target_name"] == "raw_residual_vector"


def test_run_stage3_v29_installs_online_gate_dynamic_writer(monkeypatch, tmp_path) -> None:
    cfg = _tiny_cfg()
    captured: dict[str, object] = {}

    def fake_run_stage3_v18(args):
        captured["hook"] = getattr(v29_probe.v18_probe, "write_online_gate_dynamic_features", None)
        out_dir = tmp_path / "fake_v29_run"
        out_dir.mkdir()
        return out_dir

    monkeypatch.setattr(v29_probe, "load_yaml", lambda _path: cfg)
    monkeypatch.setattr(v29_probe.v18_probe, "run_stage3_v18_structured_env_release_probe", fake_run_stage3_v18)

    out_dir = run_stage3_v29_expert_local_invariant_release_probe(argparse.Namespace(config="ignored.yaml"))

    assert out_dir == tmp_path / "fake_v29_run"
    assert captured["hook"] is write_stage3_v29_online_gate_dynamic_features


def test_rename_v18_outputs_to_v29_records_online_gate_dynamic_csv_manifest(tmp_path) -> None:
    dynamic_path = tmp_path / "stage3_v29_online_gate_dynamic_features.csv"
    dynamic_path.write_text(
        "\n".join(
            [
                "split,target_name,feature_dynamic_status",
                "validation,raw_residual_vector,available_from_model_forward",
                "validation,raw_residual_vector,available_from_model_forward",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "run_manifest_v18.json").write_text(
        json.dumps({"protocol_version": "v18_structured_env_release_probe"}),
        encoding="utf-8",
    )

    _rename_v18_outputs_to_v29(tmp_path)

    manifest = json.loads((tmp_path / "run_manifest_v29.json").read_text(encoding="utf-8"))
    entry = manifest["stage3_v29_online_gate_dynamic_features_csv"]
    assert entry["path"] == str(dynamic_path)
    assert entry["sha256"] == hashlib.sha256(dynamic_path.read_bytes()).hexdigest()
    assert entry["row_count"] == 2
    assert entry["scope"] == "validation_model_forward_pre_outcome"
    assert entry["leakage_contract"] == "no_ground_truth_gain_counterfactual_or_oracle_label_inputs"


def test_run_online_adaptation_v29_rejects_non_causal_batch() -> None:
    with pytest.raises(Exception):
        run_online_adaptation_v29(
            base_model=None,
            generator=None,
            stream=[None, None],
            stage3_cfg={"causal_online": True, "online_batch_size": 2},
            device=torch.device("cpu"),
        )
