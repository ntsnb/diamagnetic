"""Structural ablations must change real computation and survive the whole lifecycle."""
import copy
from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest
import torch

from paper_impl import run as cli
from paper_impl.adaptation import CenteredBoundedGenerator, OnlineConfig, OnlineController
from paper_impl.evaluation import stage_variants
from paper_impl.foil import FoilConfig, train_foil_branches
from paper_impl.model import ForecastBranches, ModelConfig, PaperForecaster
from tools.build_full_experiment_matrix import (BASELINES, STRUCTURAL_ABLATIONS,
                                                apply_structural_ablation, generate)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def tiny_configuration(path):
    return {
        "seed": 71, "device": "cpu", "cpu_threads": 1,
        "data": {"path": str(path), "timestamp_column": "time", "target_columns": ["load"],
                 "feature_columns": ["load", "temperature"], "lookback": 4, "horizon": 2,
                 "train_fraction": .55, "validation_fraction": .2, "extra_delay": 1},
        "model": {"hidden": 4, "heads": 1, "n_experts": 2, "depth": 1, "latent_block": 2,
                  "use_vae": True, "use_revin": True},
        "foil": {"mode": "foil", "contrast": "difference", "n_environments": 2, "rounds": 1, "mil_epochs": 1,
                 "mei_epochs": 1, "batch_size": 8},
        "training": {"epochs": 1, "batch_size": 8, "specialization_weight": .05},
        "generator": {"hidden_dim": 3, "epsilon": .1, "mode": "centered_bounded"},
        "online": {"reference_windows": 2, "reference_interval": 2, "history_capacity": 4,
                   "extra_delay": 1, "channel_budget": 1},
        "replay": {"episodes": 1, "max_windows": 3}, "test_windows": 3,
        "variants": ["stages_111"],
    }


def create_data(path):
    count = 48
    pd.DataFrame({"time": pd.date_range("2024-01-01", periods=count, freq="h"),
                  "load": 2 + np.sin(np.arange(count) / 3.) + np.arange(count) / 25.,
                  "temperature": 1 + np.cos(np.arange(count) / 5.)}).to_csv(path, index=False)


@pytest.mark.parametrize("ablation", list(STRUCTURAL_ABLATIONS))
def test_each_ablation_train_evaluate_delay_and_checkpoint_roundtrip(tmp_path, ablation):
    path = tmp_path / "series.csv"
    create_data(path)
    config = apply_structural_ablation(tiny_configuration(path), ablation)
    output = tmp_path / "trained"
    summary = cli.run(config, output)
    assert summary["status"] == "completed"
    effective = read_json(output / "effective_config.json")
    forecaster = read_json(output / "forecaster_history.json")
    foil = read_json(output / "foil_history.json")
    replay = read_json(output / "replay_history.json")
    metadata = read_json(output / "data_metadata.json")
    assert effective["data"]["extra_delay"] == effective["online"]["extra_delay"]
    delay = effective["online"]["extra_delay"]
    assert np.isfinite(forecaster["best_validation_mse"])
    assert all(np.isfinite(row["normalized_mse"]) for row in summary["results"])
    assert metadata["train_end"] > metadata["offline_training_cutoff"]
    if ablation == "noSpec":
        assert effective["training"]["specialization_weight"] == 0
        assert all(row["specialization"] == 0 for row in forecaster["history"])
    if ablation == "Single":
        assert effective["model"]["n_experts"] == 1
        assert forecaster["diagnostic"]["expert_pairs"] == 0
        assert forecaster["diagnostic"]["valid_pair_mean_distance"] is None
        assert all(row["specialization"] == 0 for row in forecaster["history"])
    if ablation == "noVAE":
        assert forecaster["diagnostic"]["vae_enabled"] is False
        assert all(row["reconstruction"] == row["kl"] == 0 for row in forecaster["history"])
    if ablation == "woInvariantDisabled":
        assert foil["status"] == "skipped"
        assert replay["reason"] == "identity_generator_ablation"
        assert effective["online"]["stage_iii"] is False
    else:
        assert replay["episodes"][0]["updates"] == 3
        assert replay["episodes"][0]["max_gradient_norm"] > 0
        for event in replay["episodes"][0]["events"]:
            assert all(event["time"] >= issue + 2 + delay for issue in event["issue_times"])
    if ablation in ("ERMpair", "VariantOnly"):
        assert foil["method"] == effective["foil"]["mode"]
        assert foil["variant"]["environment_inference"] is False
        if ablation == "VariantOnly":
            assert "invariant" not in foil
    saved = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    assert saved["configuration"]["foil"]["mode"] == effective["foil"]["mode"]
    if ablation == "noVAE":
        assert not any(key.startswith(("mu.", "log_var.", "decoder.")) for key in saved["model"])
        assert any(key.startswith("gate_features.") for key in saved["model"])
    if ablation == "woInvariantDisabled":
        assert saved["branches"] == {} and saved["generator"] == {}
    if ablation == "Single":
        assert not any(key.startswith("experts.1.") for key in saved["model"])
        assert saved["model"]["router.weight"].shape[0] == 1
    resumed = tmp_path / "reevaluated"
    cli.run({}, resumed, output / "model.pt")
    for variant in effective["variants"]:
        original = np.load(output / f"{variant}_forecasts.npz")
        restored = np.load(resumed / f"{variant}_forecasts.npz")
        for field in ("issue_times", "predictions", "targets"):
            np.testing.assert_array_equal(original[field], restored[field])
        updates = read_json(output / f"{variant}_updates.json")
        assert updates
        for event in updates:
            assert all(event["time"] >= issue + 2 + delay for issue in event["issue_times"])


@pytest.mark.parametrize("use_vae,n_experts,use_revin", [(True, 1, False), (False, 2, False),
                                                       (True, 2, True), (False, 1, True)])
def test_actual_structure_prediction_lrp_and_gradients(use_vae, n_experts, use_revin):
    torch.manual_seed(17)
    model = PaperForecaster(ModelConfig(n_variables=2, lookback=4, horizon=2, hidden=4, heads=1,
                                        n_experts=n_experts, latent_block=2, use_vae=use_vae,
                                        use_revin=use_revin, target_indices=(1,)))
    x = torch.randn(3, 4, 2, requires_grad=True)
    result = model(x, stochastic=True)
    assert result.prediction.shape == (3, 2, 1)
    assert len(model.experts) == n_experts and result.gates.shape[-1] == n_experts
    if n_experts == 1:
        torch.testing.assert_close(result.gates, torch.ones_like(result.gates), rtol=0, atol=0)
    if not use_vae:
        assert not any(hasattr(model, name) for name in ("mu", "log_var", "decoder"))
        assert result.mu is result.log_var is result.reconstruction is None
        torch.testing.assert_close(model(x, stochastic=True).gates, result.gates, rtol=0, atol=0)
        torch.testing.assert_close(model(x, stochastic=True).prediction, result.prediction, rtol=0, atol=0)
    loss = (result.prediction - 1).square().mean()
    if result.reconstruction is not None:
        loss = loss + (result.reconstruction - x).square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
    assert any(g.abs().sum() > 0 for name, g in gradients.items() if name.startswith("experts."))
    if not use_vae and n_experts > 1:
        assert any(g.abs().sum() > 0 for name, g in gradients.items() if name.startswith("gate_features."))
    if use_revin:
        assert model.revin_weight.grad is not None and model.revin_bias.grad is not None
    attribution = model.attribute(x.detach(), freeze_head=True)
    assert attribution.maps.shape == (3, 1, n_experts, 2, 4)
    assert torch.isfinite(attribution.maps).all()
    assert attribution.conservation_error.max() < 1e-3


@pytest.mark.parametrize("mode,expected_branches", [("foil", 2), ("erm_pair", 2),
                                                    ("variant_only", 1), ("disabled", 0)])
def test_auxiliary_branch_modes_remove_real_modules_and_freeze_trained_parameters(mode, expected_branches):
    torch.manual_seed(29)
    config = ModelConfig(2, 4, 2, hidden=4, heads=1, n_experts=2, latent_block=2)
    branches = ForecastBranches(config, mode=mode)
    assert sum(branch is not None for branch in (branches.invariant, branches.variant)) == expected_branches
    x = torch.randn(6, 4, 2, requires_grad=True)
    y = torch.randn(6, 2, 1)
    contrast = branches(x)
    assert contrast.shape == (6, 2)
    if mode == "disabled":
        assert not list(branches.parameters())
        assert not contrast.requires_grad and torch.count_nonzero(contrast) == 0
    else:
        contrast.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branches.parameters())
        if mode == "variant_only":
            torch.testing.assert_close(contrast, branches.variant(x).flatten(1))
        else:
            assert not ({id(p) for p in branches.invariant.parameters()} & {id(p) for p in branches.variant.parameters()})
    history = train_foil_branches(branches, x.detach(), y, x.detach()[:2], y[:2],
                                  FoilConfig(mode=mode, n_environments=2, rounds=1, mil_epochs=1,
                                             mei_epochs=1, batch_size=3, seed=5))
    assert all(not p.requires_grad for p in branches.parameters())
    assert history["config"]["mode"] == mode
    with pytest.raises(ValueError, match="does not match"):
        train_foil_branches(branches, x.detach(), y, x.detach()[:2], y[:2],
                            FoilConfig(mode="erm_pair" if mode != "erm_pair" else "foil"))


def test_revin_reverses_into_the_explicit_target_channel_and_rejects_official_wrapping():
    config = ModelConfig(2, 4, 2, hidden=4, heads=1, n_experts=1, use_revin=True,
                         target_indices=(1,))
    model = PaperForecaster(config)
    with torch.no_grad():
        for parameter in model.output_head.parameters():
            parameter.zero_()
    x = torch.tensor([[[1., 10.], [2., 20.], [3., 30.], [4., 40.]]])
    prediction = model(x).prediction
    torch.testing.assert_close(prediction, torch.full_like(prediction, 25.))
    with pytest.raises(ValueError, match="external RevIN"):
        ModelConfig(2, 24, 12, hidden=8, heads=8, use_revin=True, expert_backbone="itransformer")


def test_disabled_requires_identity_generator_and_disallows_stage_iii(tmp_path):
    path = tmp_path / "series.csv"
    create_data(path)
    config = tiny_configuration(path)
    config["foil"]["mode"] = "disabled"
    with pytest.raises(ValueError, match="requires generator.mode=identity"):
        cli._normalize(config)
    config = apply_structural_ablation(config, "woInvariantDisabled")
    config["variants"] = ["stages_111"]
    with pytest.raises(ValueError, match="no Stage III contrast"):
        cli.run(config, tmp_path / "invalid")


def test_extra_delay_zero_does_not_make_future_labels_instantly_available():
    model = PaperForecaster(ModelConfig(2, 4, 2, hidden=4, heads=1, n_experts=1))
    controller = OnlineController(model, OnlineConfig(stage_i=False, stage_ii=False, stage_iii=False,
                                                       extra_delay=0))
    prediction = controller.predict(torch.randn(1, 4, 2), 10)
    assert prediction.release_time == 12
    with pytest.raises(ValueError):
        controller.release(11, {10: torch.ones(2, 1)})


def test_matrix_covers_all_stage_bits_structures_and_baselines_without_running(tmp_path):
    path = tmp_path / "series.csv"
    create_data(path)
    inventory = [{"id": "synthetic_contract_test", "input_data": {
        "path": str(path), "timestamp_column": "time", "target_columns": ["load"],
        "feature_columns": ["load", "temperature"], "expected_interval_seconds": 3600}}]
    result = generate(tmp_path / "matrix", inventory, seeds=[7], backbones=["native", "itransformer"])
    assert set(result["stage_variants"]) == {f"stages_{i:03b}" for i in range(8)}
    assert result["job_count"] == 2 * len(STRUCTURAL_ABLATIONS)
    assert result["baseline_job_count"] == 7
    assert {item["name"] for item in result["baseline_inventory"]} == {"Frozen", "GD", "Head-GD", "SOLID", "SOLID++", "OneNet", "PROCEED"}
    onenet = next(job for job in result["baseline_jobs"] if job["method"] == "OneNet")
    assert onenet["backbone"] == "itransformer"
    assert onenet["onenet_mode"] == "onlinetsf_cross_backbone"
    assert onenet["secondary_backbone"] == "fsnet_channel"
    assert onenet["secondary_adapter"] == "paper_impl.official_fsnet.OfficialFSNet"
    for job in result["jobs"]:
        config = read_json(tmp_path / "matrix" / job["config"])
        assert config["data"]["max_rows"] is None and config["test_windows"] is None
        assert config["replay"]["max_windows"] is None
        assert config["data"]["extra_delay"] == config["online"]["extra_delay"]
        if job["ablation"] == "woInvariantDisabled":
            assert config["foil"]["mode"] == "disabled"
            assert config["generator"]["mode"] == "identity" and config["variants"] == ["stages_110"]
        if job["ablation"] == "full":
            assert len(config["variants"]) == 8 and config["online"]["extra_delay"] > 0
            assert len(config["selection_comparisons"]) == 4 and len(job["evaluation_variants"]) == 12
        else:
            assert config["selection_comparisons"] == []
        assert config["expert_diagnostics"] == {"enabled": True, "distance_interval": 1}
        if job["ablation"] == "NoExtraDelay":
            assert config["online"]["extra_delay"] == 0
        if job["ablation"] == "woRevIN":
            if job["backbone"] == "native":
                assert config["model"]["use_revin"] is False
            else:
                assert config["model"]["official_options"]["itransformer_use_norm"] is False
    with pytest.raises(FileExistsError):
        generate(tmp_path / "matrix", inventory)


@pytest.mark.parametrize("contrast", ["difference", "variant", "invariant"])
def test_contrast_ablation_keeps_both_trained_branches_and_selects_exact_coordinate(contrast):
    config = ModelConfig(2, 4, 2, hidden=4, heads=1, n_experts=2)
    torch.manual_seed(72)
    branches = ForecastBranches(config, mode="foil", contrast=contrast)
    assert branches.invariant is not None and branches.variant is not None
    x = torch.randn(6, 4, 2)
    y = torch.randn(6, 2, 1)
    expected = {"difference": lambda: branches.variant(x) - branches.invariant(x),
                "variant": lambda: branches.variant(x), "invariant": lambda: branches.invariant(x)}[contrast]()
    torch.testing.assert_close(branches(x), expected.flatten(1), rtol=0, atol=0)
    history = train_foil_branches(branches, x, y, x[:2], y[:2],
                                  FoilConfig(contrast=contrast, n_environments=2, rounds=1,
                                             mil_epochs=1, mei_epochs=1, batch_size=3))
    assert "invariant" in history and "variant" in history
    assert list(branches.invariant.parameters()) and list(branches.variant.parameters())
    assert all(not parameter.requires_grad for parameter in branches.parameters())


def test_no_revin_is_only_generated_for_models_with_a_real_normalization_switch(tmp_path):
    path = tmp_path / "series.csv"
    create_data(path)
    config = tiny_configuration(path)
    config["model"]["use_revin"] = False
    config["model"]["expert_backbone"] = "autoformer"
    with pytest.raises(ValueError, match="not a defined ablation"):
        apply_structural_ablation(config, "woRevIN")
    config["model"]["expert_backbone"] = "patchtst"
    assert apply_structural_ablation(config, "woRevIN")["model"]["official_options"]["patch_revin"] is False


def controller_for(scope="sparse_experts", **changes):
    torch.manual_seed(102)
    model = PaperForecaster(ModelConfig(2, 4, 2, hidden=4, heads=1, n_experts=2, use_vae=False))
    options = dict(stage_i=False, stage_ii=True, stage_iii=False, stage_ii_scope=scope,
                   selector="gradient_only", error_threshold=0., reference_windows=1,
                   reference_interval=1, history_capacity=3)
    options.update(changes)
    x = torch.randn(1, 4, 2) + 2
    return OnlineController(model, OnlineConfig(**options), reference_inputs=x.clone()), x


@pytest.mark.parametrize("scope", ["sparse_experts", "dense_experts", "experts_router_head", "all_forecaster"])
def test_stage_ii_scope_changes_exactly_the_permitted_parameter_families(scope):
    controller, x = controller_for(scope)
    original = {name: p.detach().clone() for name, p in controller.model.named_parameters()}
    issued = controller.predict(x, 0)
    update = controller.release(2, {0: issued.prediction + 2}, current_x=x)
    assert update.experts.accepted and update.experts.gradient_norm > 0
    changed = {name for name, p in controller.model.named_parameters() if not torch.equal(original[name], p)}
    assert changed and any(name.startswith("experts.") for name in changed)
    if scope == "sparse_experts":
        assert all(name in controller.model.expert_weights() for name in changed)
        assert update.experts.selected_channels
    elif scope == "dense_experts":
        assert all(name.startswith("experts.") for name in changed)
        assert any(name.endswith("bias") for name in changed)
        assert not update.experts.selected_channels
    elif scope == "experts_router_head":
        assert all(name.startswith(("experts.", "router.", "output_head.")) for name in changed)
        assert any(name.startswith("router.") for name in changed)
        assert any(name.startswith("output_head.") for name in changed)
        assert not any(name.startswith("encoder.") for name in changed)
    else:
        assert any(name.startswith("encoder.") for name in changed)
        assert any(name.startswith("gate_features.") for name in changed)


@pytest.mark.parametrize("mode", ["ema", "current", "frozen"])
def test_reference_modes_have_different_real_state_semantics(mode):
    controller, x = controller_for(reference_mode=mode, reference_ema=.5,
                                   selector="relevance_error", error_threshold=float("inf"))
    original = {name: value.detach().clone() for name, value in controller.reference.state_dict().items()}
    with torch.no_grad():
        for parameter in controller.model.expert_weights().values():
            parameter.add_(.1)
        controller.model.encoder.embedding.weight.add_(.3)
    scoring_snapshot = {name: value.detach().clone() for name, value in controller.model.state_dict().items()}
    issued = controller.predict(x, 0)
    update = controller.release(2, {0: issued.prediction + 1}, current_x=x)
    reference = controller.reference.state_dict()
    if mode == "frozen":
        assert not update.reference_refreshed
        for name in reference:
            torch.testing.assert_close(reference[name], original[name], rtol=0, atol=0)
    elif mode == "current":
        assert not update.reference_refreshed
        for name in reference:
            torch.testing.assert_close(reference[name], scoring_snapshot[name], rtol=0, atol=0)
        assert not torch.equal(controller.reference_maps, torch.zeros_like(controller.reference_maps))
    else:
        assert update.reference_refreshed
        weights = controller.model.expert_weights()
        for name in reference:
            expected = .5 * (original[name] + controller.model.state_dict()[name]) if name in weights else original[name]
            torch.testing.assert_close(reference[name], expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(reference["encoder.embedding.weight"], original["encoder.embedding.weight"], rtol=0, atol=0)


@pytest.mark.parametrize("rollback,proposal_is_finite", [(True, True), (False, True), (False, False)])
def test_acceptance_ablation_accepts_finite_worsening_only_when_disabled_and_never_nan(monkeypatch, rollback, proposal_is_finite):
    controller, x = controller_for("dense_experts", rollback_on_worse=rollback, max_backtracks=2)
    original = {name: p.detach().clone() for name, p in controller.model.named_parameters()}
    issued = controller.predict(x, 0)
    real_loss = controller._loss
    calls = []

    def controlled_proposal_loss(batch, gains):
        if not calls:
            value = real_loss(batch, gains)
            calls.append(float(value.detach()))
            return value
        # Controlled post-update loss probes rejection without corrupting the real gradient.
        return batch.targets.new_tensor(calls[0] + 1 if proposal_is_finite else float("nan"))

    monkeypatch.setattr(controller, "_loss", controlled_proposal_loss)
    update = controller.release(2, {0: issued.prediction + 2}, current_x=x)
    if not rollback and proposal_is_finite:
        assert update.experts.accepted
        assert update.experts.loss_after > update.experts.loss_before
        assert "acceptance_test_disabled" in update.experts.reason
        assert any(not torch.equal(original[name], p) for name, p in controller.model.named_parameters())
    else:
        assert not update.experts.accepted
        assert all(torch.equal(original[name], p) for name, p in controller.model.named_parameters())
