import torch

from train.stage3_v12d_context_features import CausalResidualHistoryFeatureBuilderV12d
from train.stage3_v12d_context_features import build_stage3_v12d_context_features


def test_v12d_history_features_do_not_use_current_label() -> None:
    y_base = torch.zeros(5, 1)
    y_true = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    builder = CausalResidualHistoryFeatureBuilderV12d(
        output_shape=(1,),
        lags=[1, 2],
        include_availability_mask=False,
        standardize=False,
    ).fit(y_base_train=y_base, y_train=y_true, split="train")

    before = builder.transform(y_base=y_base, y_true=y_true).features
    changed = y_true.clone()
    changed[2, 0] = 99.0
    after = builder.transform(y_base=y_base, y_true=changed).features

    assert torch.allclose(after[2], before[2])
    assert not torch.allclose(after[3], before[3])


def test_v12d_history_features_accept_previous_tail_for_window_continuity() -> None:
    y_base_train = torch.zeros(3, 1)
    y_train = torch.tensor([[10.0], [20.0], [30.0]])
    y_base_window = torch.zeros(2, 1)
    y_window = torch.tensor([[1.0], [2.0]])
    builder = CausalResidualHistoryFeatureBuilderV12d(
        output_shape=(1,),
        lags=[1, 2],
        include_availability_mask=False,
        standardize=False,
    ).fit(y_base_train=y_base_train, y_train=y_train, split="train")

    tail = builder.residual_tail(y_base=y_base_train, y_true=y_train)
    result = builder.transform(y_base=y_base_window, y_true=y_window, previous_residual_tail=tail)

    assert result.features.shape == (2, 2)
    assert torch.allclose(result.features[0], torch.tensor([30.0, 20.0]))
    assert torch.allclose(result.features[1], torch.tensor([1.0, 30.0]))


def test_v12d_context_feature_modes_are_explicit_and_shape_safe() -> None:
    drift = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    history = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    drift_only = build_stage3_v12d_context_features(drift_signal=drift, history_features=history, mode="drift_only")
    history_only = build_stage3_v12d_context_features(
        drift_signal=drift,
        history_features=history,
        mode="history_only",
    )
    both = build_stage3_v12d_context_features(drift_signal=drift, history_features=history, mode="drift_plus_history")

    assert torch.allclose(drift_only.features, drift)
    assert torch.allclose(history_only.features, history)
    assert both.features.shape == (4, 5)
    assert both.drift_dim == 3
    assert both.history_dim == 2
    assert both.feature_dim == 5
