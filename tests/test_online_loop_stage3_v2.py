from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.adaptation_v2 import IdentitySignedSeparableAdaptationGenerator, LayerAwareHyperAdapterGenerator
from models.factor_mechanism import FactorMechanismConfig, build_factor_mechanism
from models.factor_mechanism_v2 import FactorMechanismV2Config, build_factor_mechanism_v2
import train.online_loop_stage3_v2 as online_loop
from train.online_loop_stage3_v2 import run_online_adaptation_v2


def _cfg(allow_untrained: bool = False) -> dict:
    return {
        "online": {
            "stage3": {
                "enable": True,
                "apply_transient": True,
                "allow_untrained_generator": allow_untrained,
                "causal_online": False,
                "signal_mode": "level_vector",
                "normalize_vector": False,
                "target_modules": [],
                "max_delta": 0.1,
                "max_pred_shift": 10.0,
                "blend_alpha": 1.0,
            },
        },
    }


def test_v2_online_loop_refuses_missing_generator_unless_diagnostic_mode():
    model = nn.Linear(2, 1, bias=False)

    with pytest.raises(ValueError, match="trained Stage III V2 generator"):
        run_online_adaptation_v2(
            stream=[],
            model=model,
            generator=None,
            cfg=_cfg(allow_untrained=False),
            device=torch.device("cpu"),
        )


def test_v2_online_loop_zero_drift_identity_matches_base_prediction():
    torch.manual_seed(51)
    model = nn.Linear(2, 1, bias=False)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=8,
        max_dim=8,
        max_scale=0.2,
    )
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    phi_i = torch.ones(4, 3)
    phi_v = torch.ones(4, 3)

    metrics = run_online_adaptation_v2(
        stream=[(x, y, phi_i, phi_v)],
        model=model,
        generator=gen,
        cfg=_cfg(),
        device=torch.device("cpu"),
    )

    assert metrics["online_base_mse"] == metrics["online_stage3_mse"]
    assert metrics["online_base_mae"] == metrics["online_stage3_mae"]
    assert metrics["online_stage3_delta_norm_pre_clamp"] == 0.0
    assert metrics["online_stage3_use_rate"] == 1.0


def test_v2_online_loop_reports_reject_reason_when_candidate_shift_is_too_large():
    torch.manual_seed(52)
    model = nn.Linear(2, 1, bias=False)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=8,
        max_dim=8,
        max_scale=5.0,
    )
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    phi_i = torch.zeros(4, 3)
    phi_v = torch.randn(4, 3)
    cfg = _cfg()
    cfg["online"]["stage3"]["max_pred_shift"] = 0.0

    metrics = run_online_adaptation_v2(
        stream=[(x, y, phi_i, phi_v)],
        model=model,
        generator=gen,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert metrics["online_stage3_use_rate"] == 0.0
    assert metrics["online_stage3_reject_shift_count"] == 1.0


def test_v2_online_loop_passes_configured_drift_polarity(monkeypatch):
    torch.manual_seed(53)
    model = nn.Linear(2, 1, bias=False)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=8,
        max_dim=8,
        max_scale=0.2,
    )
    x = torch.randn(1, 2)
    y = torch.randn(1, 1)
    phi_i = torch.zeros(1, 3)
    phi_v = torch.tensor([[1.0, -2.0, 3.0]])
    cfg = _cfg()
    cfg["online"]["stage3"]["drift_polarity"] = -1.0
    seen_polarities = []
    original = online_loop.build_stage3_drift_vector

    def spy_build_stage3_drift_vector(*args, **kwargs):
        seen_polarities.append(kwargs.get("polarity"))
        return original(*args, **kwargs)

    monkeypatch.setattr(online_loop, "build_stage3_drift_vector", spy_build_stage3_drift_vector)

    run_online_adaptation_v2(
        stream=[(x, y, phi_i, phi_v)],
        model=model,
        generator=gen,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert seen_polarities == [-1.0]


def test_v2_online_loop_consumes_stream_incrementally_without_prefetching_all_batches():
    torch.manual_seed(54)

    class CountingLinear(nn.Linear):
        def __init__(self):
            super().__init__(2, 1, bias=False)
            self.forward_count = 0

        def forward(self, x):
            self.forward_count += 1
            return super().forward(x)

    model = CountingLinear()
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=8,
        max_dim=8,
        max_scale=0.2,
    )
    cfg = _cfg()
    x = torch.randn(1, 2)
    y = torch.randn(1, 1)
    phi_i = torch.zeros(1, 3)
    phi_v = torch.ones(1, 3)

    def online_stream():
        yield x, y, phi_i, phi_v
        assert model.forward_count > 0
        yield x, y, phi_i, phi_v

    metrics = run_online_adaptation_v2(
        stream=online_stream(),
        model=model,
        generator=gen,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert model.forward_count >= 2
    assert metrics["online_stage3_reject_missing_drift_count"] == 0.0
    assert metrics["online_batch_count"] == 2.0
    assert metrics["online_sample_count"] == 2.0


def test_v2_online_loop_risk_gate_can_reject_candidate_without_using_labels():
    torch.manual_seed(55)

    class RejectAllGate(torch.nn.Module):
        def forward(self, features):
            return torch.full((features.shape[0],), 0.1, device=features.device)

    model = nn.Linear(2, 1, bias=False)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=8,
        max_dim=8,
        max_scale=0.2,
    )
    cfg = _cfg()
    cfg["online"]["stage3"]["risk_gate_enable"] = True
    cfg["online"]["stage3"]["risk_gate_threshold"] = 0.5
    x = torch.randn(1, 2)
    y = torch.randn(1, 1)
    phi_i = torch.zeros(1, 3)
    phi_v = torch.randn(1, 3)

    metrics = run_online_adaptation_v2(
        stream=[(x, y, phi_i, phi_v)],
        model=model,
        generator=gen,
        risk_gate=RejectAllGate(),
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert metrics["online_stage3_use_rate"] == 0.0
    assert metrics["online_stage3_reject_risk_count"] == 1.0
    assert metrics["online_stage3_risk_score"] == pytest.approx(0.1)


def test_v2_online_loop_can_return_per_step_trace_rows_for_drift_diagnostics():
    torch.manual_seed(550)
    model = nn.Linear(2, 1, bias=False)
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=8,
        max_dim=8,
        max_scale=0.2,
    )
    cfg = _cfg()
    cfg["online"]["stage3"]["risk_gate_enable"] = True
    cfg["online"]["stage3"]["risk_gate_threshold"] = 0.5

    class RejectAllGate(torch.nn.Module):
        def forward(self, features):
            return torch.full((features.shape[0],), 0.1, device=features.device)

    x = torch.randn(1, 2)
    y = torch.randn(1, 1)
    phi_i = torch.zeros(1, 3)
    phi_v = torch.randn(1, 3)

    metrics = run_online_adaptation_v2(
        stream=[(x, y, phi_i, phi_v)],
        model=model,
        generator=gen,
        risk_gate=RejectAllGate(),
        cfg=cfg,
        device=torch.device("cpu"),
        return_step_rows=True,
        trace_context={"run_id": "trace_test", "split": "test", "mode": "real"},
    )

    assert len(metrics["step_rows"]) == 1
    row = metrics["step_rows"][0]
    assert row["run_id"] == "trace_test"
    assert row["split"] == "test"
    assert row["mode"] == "real"
    assert row["step_index"] == 0
    assert row["sample_count"] == 1.0
    assert row["used"] == 0.0
    assert row["reject_reason"] == "risk_gate"
    assert row["base_mse"] == pytest.approx(metrics["online_base_mse"])
    assert row["final_mse"] == pytest.approx(metrics["online_stage3_mse"])
    assert row["candidate_mse"] != ""
    assert row["would_apply_blended_mse"] != ""
    assert row["would_apply_blended_mae"] != ""
    assert row["drift_norm"] > 0.0
    assert row["delta_norm_pre_clamp"] == pytest.approx(metrics["online_stage3_delta_norm_pre_clamp"])
    assert row["risk_score"] == pytest.approx(0.1)


def test_v2_online_loop_can_apply_configured_bias_target_adapter():
    torch.manual_seed(551)
    model = nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    gen = LayerAwareHyperAdapterGenerator(
        named_param_shapes={"bias": tuple(model.bias.shape)},
        drift_dim=3,
        hidden_dim=6,
        rank=2,
        max_scale=1.0,
    )
    with torch.no_grad():
        first_linear = gen.drift_conditioner[0]
        second_linear = gen.drift_conditioner[2]
        first_linear.weight.zero_()
        second_linear.weight.zero_()
        first_linear.weight[0, 0] = 1.0
        second_linear.weight[0, 0] = 1.0
        next(iter(gen.bias_adapters.values())).basis[0, 0] = 0.25
    cfg = _cfg()
    cfg["online"]["stage3"]["target_names"] = ["bias"]
    x = torch.zeros(1, 2)
    y = torch.full((1, 1), 0.25)
    phi_i = torch.zeros(1, 3)
    phi_v = torch.tensor([[1.0, 0.0, 0.0]])

    metrics = run_online_adaptation_v2(
        stream=[(x, y, phi_i, phi_v)],
        model=model,
        generator=gen,
        cfg=cfg,
        device=torch.device("cpu"),
        return_step_rows=True,
    )

    assert metrics["online_stage3_mse"] < metrics["online_base_mse"]
    assert metrics["step_rows"][0]["delta_param_count"] == 1
    assert metrics["step_rows"][0]["used"] == 1.0


def test_v2_online_loop_rejects_legacy_stochastic_vae_model():
    model = build_factor_mechanism(
        FactorMechanismConfig(
            input_dim=3,
            seq_len=4,
            out_dim=1,
            d_model=16,
            n_experts=2,
            latent_dim=6,
            temporal_encoder="transformer",
            nhead=2,
            num_layers=1,
            dropout=0.0,
            use_vae=True,
        )
    )
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=8,
        max_dim=128,
        max_scale=0.2,
    )

    with pytest.raises(ValueError, match="deterministic VAE eval"):
        run_online_adaptation_v2(
            stream=[],
            model=model,
            generator=gen,
            cfg=_cfg(),
            device=torch.device("cpu"),
        )


def test_v2_online_loop_rejects_explicit_sample_eval_without_mc_mode():
    model = build_factor_mechanism_v2(
        FactorMechanismV2Config(
            input_dim=3,
            seq_len=4,
            out_dim=1,
            d_model=16,
            n_experts=2,
            latent_dim=6,
            temporal_encoder="transformer",
            nhead=2,
            num_layers=1,
            dropout=0.0,
            use_vae=True,
            vae_eval_mode="sample",
        )
    )
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=8,
        max_dim=128,
        max_scale=0.2,
    )

    with pytest.raises(ValueError, match="deterministic VAE eval"):
        run_online_adaptation_v2(
            stream=[],
            model=model,
            generator=gen,
            cfg=_cfg(),
            device=torch.device("cpu"),
        )


def test_v2_online_loop_accepts_deterministic_vae_eval_model():
    torch.manual_seed(56)
    model = build_factor_mechanism_v2(
        FactorMechanismV2Config(
            input_dim=3,
            seq_len=4,
            out_dim=1,
            d_model=16,
            n_experts=2,
            latent_dim=6,
            temporal_encoder="transformer",
            nhead=2,
            num_layers=1,
            dropout=0.0,
            use_vae=True,
            vae_eval_mode="mean",
        )
    )
    gen = IdentitySignedSeparableAdaptationGenerator(
        drift_dim=3,
        hidden_dim=8,
        max_dim=128,
        max_scale=0.2,
    )
    x = torch.randn(1, 4, 3)
    y = torch.randn(1, 1)
    phi_i = torch.zeros(1, 3)
    phi_v = torch.ones(1, 3)

    first = run_online_adaptation_v2(
        stream=[(x, y, phi_i, phi_v)],
        model=model,
        generator=gen,
        cfg=_cfg(),
        device=torch.device("cpu"),
    )
    second = run_online_adaptation_v2(
        stream=[(x, y, phi_i, phi_v)],
        model=model,
        generator=gen,
        cfg=_cfg(),
        device=torch.device("cpu"),
    )

    assert first["online_base_mse"] == pytest.approx(second["online_base_mse"])
    assert first["online_stage3_mse"] == pytest.approx(second["online_stage3_mse"])
    assert first["online_stage3_recon_error"] == pytest.approx(second["online_stage3_recon_error"])
