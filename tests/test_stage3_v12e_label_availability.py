import torch

from train.stage3_v12e_label_availability import PerHorizonLabelAvailabilityHistoryBuilderV12e


def _residual_grid(n: int = 6, horizon: int = 3) -> torch.Tensor:
    rows = []
    for step in range(n):
        rows.append([10.0 * step + float(h) for h in range(1, horizon + 1)])
    return torch.tensor(rows)


def test_v12e_masks_each_horizon_until_its_label_has_arrived() -> None:
    y_base = torch.zeros(6, 3)
    y_true = _residual_grid()
    builder = PerHorizonLabelAvailabilityHistoryBuilderV12e(
        output_shape=(3,),
        lags=[1, 2, 3],
        horizon_offsets=[1, 2, 3],
        label_delay=0,
        include_availability_mask=False,
        standardize=False,
    ).fit(y_base_train=y_base, y_train=y_true, split="train")

    result = builder.transform(y_base=y_base, y_true=y_true)

    assert result.residual_history.shape == (6, 9)
    assert torch.allclose(
        result.residual_history[3],
        torch.tensor([21.0, 0.0, 0.0, 11.0, 12.0, 0.0, 1.0, 2.0, 3.0]),
    )
    assert torch.allclose(
        result.availability_mask[3],
        torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
    )


def test_v12e_label_delay_shifts_horizon_availability() -> None:
    y_base = torch.zeros(6, 2)
    y_true = _residual_grid(n=6, horizon=2)
    builder = PerHorizonLabelAvailabilityHistoryBuilderV12e(
        output_shape=(2,),
        lags=[1, 2, 3],
        horizon_offsets=[1, 2],
        label_delay=1,
        include_availability_mask=False,
        standardize=False,
    ).fit(y_base_train=y_base, y_train=y_true, split="train")

    result = builder.transform(y_base=y_base, y_true=y_true)

    assert torch.allclose(result.residual_history[4], torch.tensor([0.0, 0.0, 21.0, 0.0, 11.0, 12.0]))
    assert torch.allclose(result.availability_mask[4], torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]))


def test_v12e_future_horizon_value_only_affects_features_after_arrival() -> None:
    y_base = torch.zeros(7, 3)
    y_true = _residual_grid(n=7, horizon=3)
    builder = PerHorizonLabelAvailabilityHistoryBuilderV12e(
        output_shape=(3,),
        lags=[1, 2, 3],
        horizon_offsets=[1, 2, 3],
        label_delay=0,
        include_availability_mask=False,
        standardize=False,
    ).fit(y_base_train=y_base, y_train=y_true, split="train")
    before = builder.transform(y_base=y_base, y_true=y_true).residual_history

    changed = y_true.clone()
    changed[2, 2] = 999.0
    after = builder.transform(y_base=y_base, y_true=changed).residual_history

    assert torch.allclose(after[3], before[3])
    assert not torch.allclose(after[5], before[5])


def test_v12e_window_tail_respects_per_horizon_availability() -> None:
    y_base_train = torch.zeros(4, 3)
    y_train = _residual_grid(n=4, horizon=3)
    y_base_window = torch.zeros(2, 3)
    y_window = torch.tensor([[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]])
    builder = PerHorizonLabelAvailabilityHistoryBuilderV12e(
        output_shape=(3,),
        lags=[1, 2, 3],
        horizon_offsets=[1, 2, 3],
        label_delay=0,
        include_availability_mask=False,
        standardize=False,
    ).fit(y_base_train=y_base_train, y_train=y_train, split="train")

    tail = builder.residual_tail(y_base=y_base_train, y_true=y_train)
    result = builder.transform(y_base=y_base_window, y_true=y_window, previous_residual_tail=tail)

    assert torch.allclose(
        result.residual_history[0],
        torch.tensor([31.0, 0.0, 0.0, 21.0, 22.0, 0.0, 11.0, 12.0, 13.0]),
    )
