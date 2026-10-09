"""用解析例、独立 float64 参考与梯度检验核对论文定义。"""

from __future__ import annotations

import itertools
import math

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from paper_impl.discrepancy import (
    absolute_discrepancy,
    normalized_discrepancy,
    pairwise_discrepancies,
    relevance_profiles,
    specialization_loss,
)


def _random(shape: tuple[int, ...], seed: int = 19) -> torch.Tensor:
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64) + 0.1


def _reference(a: torch.Tensor, b: torch.Tensor, *, normalized: bool) -> float:
    """用 Python 标量和 fsum 独立计算单个地图对。"""
    rows_a, rows_b = a.tolist(), b.tolist()
    masses_a = [math.fsum(row) for row in rows_a]
    masses_b = [math.fsum(row) for row in rows_b]
    total_a, total_b = math.fsum(masses_a), math.fsum(masses_b)
    if normalized and (total_a == 0 or total_b == 0):
        raise ValueError("undefined normalized distance")
    length = a.shape[-1]
    terms = []
    for row_a, row_b, mass_a, mass_b in zip(rows_a, rows_b, masses_a, masses_b):
        weight_a = mass_a / total_a if normalized else mass_a
        weight_b = mass_b / total_b if normalized else mass_b
        temporal = 0.0
        if mass_a > 0 and mass_b > 0 and length > 1:
            temporal = math.fsum(
                abs(math.fsum(row_a[:s]) / mass_a - math.fsum(row_b[:s]) / mass_b)
                for s in range(1, length)
            ) / (length - 1)
        terms.append(0.5 * abs(weight_a - weight_b) + min(weight_a, weight_b) * temporal)
    return math.fsum(terms)


def test_profiles_exact_zero_rows_and_dimensions():
    maps = torch.tensor([[[[1.0, 3.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]])
    profiles = relevance_profiles(maps)
    assert profiles.row_mass.shape == (1, 2, 2)
    torch.testing.assert_close(profiles.row_mass, torch.tensor([[[4.0, 0.0], [0.0, 0.0]]], dtype=torch.float64))
    torch.testing.assert_close(profiles.total_mass, torch.tensor([[4.0, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(profiles.relative_mass, torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.float64))
    torch.testing.assert_close(profiles.lag_distribution[0, 0, 0], torch.tensor([0.25, 0.75], dtype=torch.float64))
    assert torch.count_nonzero(profiles.lag_distribution[0, 1]) == 0
    assert profiles.valid.tolist() == [[True, False]]


@pytest.mark.parametrize("normalized", [False, True])
def test_float64_matches_independent_reference(normalized):
    a, b = _random((3, 5), 17), _random((3, 5), 21)
    a[1] = 0
    b[2] = 0
    function = normalized_discrepancy if normalized else absolute_discrepancy
    actual = function(a, b)
    assert actual.dtype == torch.float64
    assert actual.item() == pytest.approx(_reference(a, b, normalized=normalized), abs=1e-14)


def test_identity_and_independent_scale_invariance():
    a, b = _random((4, 3, 7)), _random((4, 3, 7), 31)
    torch.testing.assert_close(normalized_discrepancy(a, a), torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(absolute_discrepancy(a, a), torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(normalized_discrepancy(a, 13 * a), torch.zeros(4, dtype=torch.float64), atol=1e-14, rtol=0)
    expected = normalized_discrepancy(a, b)
    torch.testing.assert_close(normalized_discrepancy(a * 1e-150, b * 1e150), expected, atol=1e-14, rtol=1e-13)
    torch.testing.assert_close(absolute_discrepancy(3 * a, 3 * b), 3 * absolute_discrepancy(a, b))
    torch.testing.assert_close(absolute_discrepancy(a, 13 * a), 6 * a.sum(dim=(-2, -1)))


def test_matching_variable_identity_is_preserved():
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
    swapped = a.flip(0)
    torch.testing.assert_close(a.sum(0), swapped.sum(0))
    assert normalized_discrepancy(a, swapped).item() == 1.0
    x, y = _random((4, 3, 5)), _random((4, 3, 5), 45)
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        normalized_discrepancy(x[:, permutation], y[:, permutation]),
        normalized_discrepancy(x, y),
    )


def test_lag_displacement_uses_grid_distance():
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    adjacent = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    far = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    assert normalized_discrepancy(a, adjacent).item() == pytest.approx(1 / 3)
    assert normalized_discrepancy(a, far).item() == 1.0


@pytest.mark.parametrize("normalized", [False, True])
def test_random_symmetry_triangle_and_bounds(normalized):
    a, b, c = (_random((128, 4, 6), seed) for seed in (100, 101, 102))
    a[::3, 1] = 0
    b[::4, 2] = 0
    c[::5, 3] = 0
    if not normalized:
        c[::7] = 0
    function = normalized_discrepancy if normalized else absolute_discrepancy
    ab, ac, cb = function(a, b), function(a, c), function(c, b)
    assert bool((ab <= ac + cb + 1e-12).all())
    torch.testing.assert_close(ab, function(b, a))
    if normalized:
        assert bool(((ab >= 0) & (ab <= 1 + 1e-14)).all())
    else:
        ta, tb = a.sum((-2, -1)), b.sum((-2, -1))
        assert bool((ab >= 0.5 * (ta - tb).abs() - 1e-12).all())
        assert bool((ab <= 0.5 * (ta + tb) + 1e-12).all())


def test_absolute_zero_map_boundary():
    a = _random((3, 2, 4))
    zero = torch.zeros_like(a)
    torch.testing.assert_close(absolute_discrepancy(zero, a), 0.5 * a.sum((-2, -1)))
    torch.testing.assert_close(absolute_discrepancy(zero, zero), torch.zeros(3, dtype=torch.float64))
    with pytest.raises(ValueError, match="正总质量"):
        normalized_discrepancy(zero, a)
    with pytest.raises(ValueError, match="正总质量"):
        normalized_discrepancy(zero, zero)


def test_single_lag_is_total_variation_only():
    a = torch.tensor([[2.0], [0.0], [2.0]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[0.0], [4.0], [4.0]], dtype=torch.float64)
    assert normalized_discrepancy(a, b).item() == 0.5
    assert absolute_discrepancy(a, b).item() == 4.0
    normalized_discrepancy(a, b).backward()
    assert bool(torch.isfinite(a.grad).all())


def test_broadcasted_same_expert_reference_comparisons():
    current = _random((2, 1, 3, 4))
    references = _random((1, 5, 3, 4), 37)
    result = normalized_discrepancy(current, references)
    assert result.shape == (2, 5)
    for i, j in itertools.product(range(2), range(5)):
        assert result[i, j].item() == pytest.approx(_reference(current[i, 0], references[0, j], normalized=True))


def test_pairwise_indices_and_zero_map_validity():
    maps = _random((2, 3, 4, 2, 3))
    maps[0, 1, 2] = 0
    pairs = pairwise_discrepancies(maps)
    assert pairs.distances.shape == (2, 3, 6)
    assert pairs.indices.tolist() == [[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]]
    assert pairs.valid[0, 1].tolist() == [True, False, True, False, True, False]
    assert bool(torch.isnan(pairs.distances[~pairs.valid]).all())
    for pair, (m, n) in enumerate(pairs.indices.T.tolist()):
        torch.testing.assert_close(pairs.distances[1, 2, pair], normalized_discrepancy(maps[1, 2, m], maps[1, 2, n]))


def test_specialization_keeps_all_pairs_in_denominator():
    maps = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 0.0]]], dtype=torch.float64, requires_grad=True)
    margin = 0.6
    loss = specialization_loss(maps, margin)
    assert loss.item() == pytest.approx(2 * margin**2 / 3)
    loss.backward()
    assert bool(torch.isfinite(maps.grad).all())
    assert torch.count_nonzero(maps.grad) == 0


def test_specialization_all_zero_and_identical_maps():
    zero = torch.zeros(2, 4, 3, 5, dtype=torch.float64, requires_grad=True)
    loss = specialization_loss(zero, 0.4)
    assert loss.item() == pytest.approx(0.4**2)
    loss.backward()
    assert torch.count_nonzero(zero.grad) == 0
    assert bool(torch.isfinite(zero.grad).all())
    identical = _random((2, 1, 3, 5)).expand(2, 4, 3, 5)
    assert specialization_loss(identical, 0.4).item() == pytest.approx(0.4**2)


def test_specialization_matches_fixed_pair_reference_and_scaling():
    maps = _random((2, 3, 4, 2, 5))
    maps[1, 2, 3] = 0
    margin = 0.7
    expected = []
    for sample in maps.reshape(-1, 4, 2, 5):
        for a, b in itertools.combinations(sample, 2):
            if a.sum() == 0 or b.sum() == 0:
                expected.append(margin**2)
            else:
                expected.append(max(0.0, margin - _reference(a, b, normalized=True)) ** 2)
    actual = specialization_loss(maps, margin)
    assert actual.item() == pytest.approx(math.fsum(expected) / len(expected), abs=1e-14)
    scales = torch.tensor([1e-100, 3.0, 1e100, 0.25], dtype=torch.float64).reshape(1, 1, 4, 1, 1)
    torch.testing.assert_close(specialization_loss(maps * scales, margin), actual, atol=1e-14, rtol=1e-13)
    assert 0 <= actual.item() <= margin**2


@pytest.mark.parametrize("function", [normalized_discrepancy, absolute_discrepancy])
def test_nonzero_map_gradcheck(function):
    a = _random((2, 3), 84).requires_grad_()
    b = _random((2, 3), 98).requires_grad_()
    assert gradcheck(function, (a, b), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_specialization_gradcheck_and_second_derivatives():
    maps = _random((3, 2, 3), 72).requires_grad_()
    assert gradcheck(lambda value: specialization_loss(value, 0.9), (maps,), eps=1e-6, atol=1e-5, rtol=1e-4)
    assert gradgradcheck(lambda value: specialization_loss(value, 0.9), (maps,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_float32_input_preserves_gradients_through_float64_promotion():
    maps = _random((3, 2, 4)).float().requires_grad_()
    loss = specialization_loss(maps, 0.9)
    assert loss.dtype == torch.float64
    loss.backward()
    assert maps.grad.dtype == torch.float32
    assert bool(torch.isfinite(maps.grad).all())
    assert torch.count_nonzero(maps.grad) > 0


def test_positive_tiny_maps_are_not_thresholded():
    a, b = _random((2, 4)), _random((2, 4), 41)
    tiny_a, tiny_b = a * 1e-250, b * 1e-250
    assert bool(relevance_profiles(tiny_a).valid)
    torch.testing.assert_close(normalized_discrepancy(tiny_a, tiny_b), normalized_discrepancy(a, b), atol=1e-14, rtol=1e-13)
    tiny_float32 = (a.float() * 1e-40)
    profiles = relevance_profiles(tiny_float32)
    assert bool(profiles.valid)
    assert profiles.total_mass > 0
    torch.testing.assert_close(profiles.relative_mass.sum(), torch.tensor(1.0, dtype=torch.float64))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_rejects_invalid_relevance_values(bad):
    maps = torch.tensor([[1.0, bad]])
    with pytest.raises(ValueError):
        relevance_profiles(maps)


def test_input_shape_dtype_and_overflow_guards():
    with pytest.raises(TypeError):
        relevance_profiles([[1.0]])
    with pytest.raises(TypeError):
        relevance_profiles(torch.ones(2, 3, dtype=torch.int64))
    with pytest.raises(ValueError):
        relevance_profiles(torch.ones(3))
    with pytest.raises(ValueError):
        relevance_profiles(torch.ones(2, 0))
    with pytest.raises(ValueError, match="变量和滞后"):
        normalized_discrepancy(torch.ones(2, 3), torch.ones(3, 2))
    with pytest.raises(ValueError, match="广播"):
        normalized_discrepancy(torch.ones(2, 2, 3), torch.ones(3, 2, 3))
    with pytest.raises(ValueError, match="float64"):
        relevance_profiles(torch.full((2, 2), torch.finfo(torch.float64).max, dtype=torch.float64))
    with pytest.raises(ValueError, match="K"):
        pairwise_discrepancies(torch.ones(1, 2, 3))
    with pytest.raises(ValueError, match="样本"):
        specialization_loss(torch.empty(0, 2, 2, 3), 0.5)


@pytest.mark.parametrize("margin", [0.0, 1.0, -0.1, float("nan"), float("inf")])
def test_rejects_invalid_margin(margin):
    with pytest.raises(ValueError, match="margin"):
        specialization_loss(torch.ones(2, 2, 3), margin)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_and_backward_is_finite():
    maps = _random((2, 3, 2, 4))
    maps[0, 1] = 0
    gpu_maps = maps.cuda().requires_grad_()
    loss = specialization_loss(gpu_maps, 0.8)
    torch.testing.assert_close(loss.cpu(), specialization_loss(maps, 0.8))
    loss.backward()
    assert bool(torch.isfinite(gpu_maps.grad).all())
