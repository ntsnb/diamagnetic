from __future__ import annotations

import csv

from train.run_stage3_ablation_benchmark import append_rows, make_trace_path, result_row, settings_from_row


def test_result_row_contains_metrics_tuning_and_provenance():
    settings = {
        "lr_scale": 0.0005,
        "max_delta": 0.02,
        "blend_alpha": 0.28,
        "max_pred_shift": 0.15,
    }
    result = {
        "online_base_mse": 1.10,
        "online_base_mae": 0.40,
        "online_stage2_mse": 1.05,
        "online_stage2_mae": 0.39,
        "online_stage3_mse": 0.98,
        "online_stage3_mae": 0.35,
        "online_stage3_drift_dim": 16.0,
        "online_stage3_signal_dim": 64.0,
        "online_stage3_raw_abs_mean": 0.5,
        "online_stage3_signal_abs_mean": 0.2,
        "online_stage3_signal_norm": 3.0,
        "online_stage3_scalar_ratio": 0.1,
    }
    offline = {
        "offline_train_mse": 1.30,
        "offline_train_mae": 0.55,
        "offline_val_mse": 1.25,
        "offline_test_mse": 1.23,
        "offline_test_mae": 0.45,
        "stage3_lr_decay": 0.997,
        "split_mode": "chronological_shifted",
        "train_ratio": 0.55,
        "val_ratio": 0.10,
        "online_test_ratio": 0.20,
        "shift_pool_ratio": 0.35,
        "split_info_json": '{"test_start": 100}',
    }

    row = result_row(
        timestamp="2026-05-24T00:00:00",
        run_id="rid",
        config_path="configs/best_load_online.yaml",
        config_anchor="public",
        dataset_family="public",
        dataset_name="panama",
        seed=42,
        split_name="test",
        variant_name="stage3",
        signal_mode="delta_vector",
        settings=settings,
        result=result,
        offline=offline,
        trace_path="logs/trace.csv",
    )

    required = {
        "timestamp",
        "run_id",
        "script",
        "config",
        "config_anchor",
        "dataset",
        "dataset_family",
        "seed",
        "split",
        "variant",
        "stage3_signal_mode",
        "signal_level_or_delta",
        "signal_scalar_or_vector",
        "projection_method",
        "stage3_lr_scale",
        "stage3_lr_decay",
        "stage3_max_delta",
        "stage3_blend_alpha",
        "stage3_blend_alpha_min",
        "stage3_blend_alpha_max",
        "stage3_max_pred_shift",
        "offline_test_mse",
        "offline_test_mae",
        "online_base_mse",
        "online_base_mae",
        "online_stage3_mse",
        "online_stage3_mae",
        "adapted_online_mse",
        "adapted_online_mae",
        "gain_mse_vs_base",
        "gain_mae_vs_base",
        "stage3_drift_dim",
        "stage3_signal_dim",
        "trace_path",
    }
    assert required <= set(row)
    assert row["adapted_online_mse"] == "0.980000"
    assert row["adapted_online_mae"] == "0.350000"
    assert row["gain_mse_vs_base"] == "0.120000"
    assert row["gain_mae_vs_base"] == "0.050000"
    assert row["signal_level_or_delta"] == "delta"
    assert row["signal_scalar_or_vector"] == "vector"
    assert row["projection_method"] == "train_pca_to_lrp_bottleneck"


def test_settings_round_trip_includes_max_delta_and_blend_bounds():
    row = {
        "stage3_lr_scale": "0.00050000",
        "stage3_max_delta": "0.02000000",
        "stage3_blend_alpha": "0.28000000",
        "stage3_blend_alpha_min": "0.20000000",
        "stage3_blend_alpha_max": "0.34000000",
        "stage3_max_pred_shift": "0.15000000",
    }

    settings = settings_from_row(row)

    assert settings["lr_scale"] == 0.0005
    assert settings["max_delta"] == 0.02
    assert settings["blend_alpha"] == 0.28
    assert settings["max_pred_shift"] == 0.15
    assert round(settings["blend_alpha_min_delta"], 8) == 0.08
    assert round(settings["blend_alpha_max_delta"], 8) == 0.06


def test_append_rows_preserves_existing_rows_when_new_columns_appear(tmp_path):
    path = tmp_path / "results.csv"
    append_rows(path, [{"dataset": "a", "variant": "stage3", "online_stage3_mse": "1.0"}])
    append_rows(
        path,
        [
            {
                "dataset": "b",
                "variant": "stage3",
                "online_stage3_mse": "2.0",
                "seed": "42",
                "train_epochs": "4",
            }
        ],
    )

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["dataset"] == "a"
    assert rows[0]["online_stage3_mse"] == "1.0"
    assert "seed" in rows[0]
    assert rows[0]["seed"] == ""
    assert rows[1]["seed"] == "42"
    assert rows[1]["train_epochs"] == "4"


def test_trace_path_uses_short_filename_for_windows_path_limits(tmp_path):
    path = make_trace_path(
        trace_dir=tmp_path / "logs" / "load_benchmarks" / "stage3_ablation" / "traces",
        run_id="codex_stage3_ablation_public3_seed42_e5_with_a_very_long_suffix",
        dataset_name="spain_energy_weather",
        seed=42,
        split_name="validation",
        variant_name="stage1_stage2_stage3",
        signal_mode="delta_vector",
        settings={
            "lr_scale": 0.0003,
            "max_delta": 0.015,
            "blend_alpha": 0.22,
            "max_pred_shift": 0.15,
        },
    )

    assert path.name.endswith(".csv")
    assert "spain_energy_weather" not in path.name
    assert len(path.name) <= 80
