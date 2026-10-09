"""论文中的质量—时序差异；使用 float64 计算并保留输入梯度。"""

from __future__ import annotations

import math
from numbers import Real
from typing import NamedTuple

import torch
from torch import Tensor


class RelevanceProfiles(NamedTuple):
    """末两维为变量和滞后；valid 表示地图总质量严格为正。"""

    row_mass: Tensor
    total_mass: Tensor
    relative_mass: Tensor
    lag_distribution: Tensor
    valid: Tensor


class PairwiseDiscrepancies(NamedTuple):
    """固定上三角专家对；无效距离为 NaN，使用时须读取 valid。"""

    distances: Tensor
    valid: Tensor
    indices: Tensor


def _checked_maps(maps: Tensor, name: str = "maps") -> Tensor:
    if not isinstance(maps, Tensor):
        raise TypeError(f"{name} 必须是 PyTorch 张量")
    if not maps.is_floating_point() or maps.layout != torch.strided:
        raise TypeError(f"{name} 必须是稠密浮点张量")
    if maps.ndim < 2 or maps.shape[-2] < 1 or maps.shape[-1] < 1:
        raise ValueError(f"{name} 的形状必须为 [..., N, L]，且 N、L 为正")
    if not bool(torch.isfinite(maps).all()):
        raise ValueError(f"{name} 必须全部有限")
    if bool((maps < 0).any()):
        raise ValueError(f"{name} 必须非负；signed relevance 应先按论文取绝对值")
    return maps.to(dtype=torch.float64)


def _profiles(maps: Tensor) -> RelevanceProfiles:
    row_mass = maps.sum(dim=-1)
    total_mass = row_mass.sum(dim=-1)
    if not bool(torch.isfinite(total_mass).all()):
        raise ValueError("质量求和超出 float64 范围")
    valid = total_mass > 0
    # 仅替换精确零分母；不加 epsilon，也不把微小正质量归零。
    safe_rows = torch.where(row_mass > 0, row_mass, torch.ones_like(row_mass))
    safe_total = torch.where(valid, total_mass, torch.ones_like(total_mass))
    return RelevanceProfiles(
        row_mass=row_mass,
        total_mass=total_mass,
        relative_mass=row_mass / safe_total.unsqueeze(-1),
        lag_distribution=maps / safe_rows.unsqueeze(-1),
        valid=valid,
    )


def relevance_profiles(maps: Tensor) -> RelevanceProfiles:
    """分解 [..., N, L] 或 [..., K, N, L] 地图；零行的条件分布用零占位。"""
    return _profiles(_checked_maps(maps))


def _paired_profiles(a: Tensor, b: Tensor) -> tuple[RelevanceProfiles, RelevanceProfiles]:
    a = _checked_maps(a, "a")
    b = _checked_maps(b, "b")
    if a.device != b.device:
        raise ValueError("a、b 必须位于同一设备")
    if a.shape[-2:] != b.shape[-2:]:
        raise ValueError("a、b 必须具有相同的变量和滞后维度")
    try:
        a, b = torch.broadcast_tensors(a, b)
    except RuntimeError as error:
        raise ValueError("a、b 的前导维度不能广播") from error
    return _profiles(a), _profiles(b)


def _lag_distance(p: Tensor, q: Tensor) -> Tensor:
    length = p.shape[-1]
    if length == 1:
        return torch.zeros_like(p[..., 0])
    cdf_difference = (p - q).cumsum(dim=-1)[..., :-1]
    return cdf_difference.abs().sum(dim=-1) / (length - 1)


def _distance(a: RelevanceProfiles, b: RelevanceProfiles, *, normalized: bool) -> Tensor:
    mass_a = a.relative_mass if normalized else a.row_mass
    mass_b = b.relative_mass if normalized else b.row_mass
    mass_term = 0.5 * (mass_a - mass_b).abs().sum(dim=-1)
    temporal_term = (
        torch.minimum(mass_a, mass_b)
        * _lag_distance(a.lag_distribution, b.lag_distribution)
    ).sum(dim=-1)
    return mass_term + temporal_term


def normalized_discrepancy(a: Tensor, b: Tensor) -> Tensor:
    """比较 [..., N, L] 的相对结构，前导维可广播；零总量无定义并报错。

    输出为 float64。无需预先归一化，可直接比较同一专家的当前与参考地图。
    """
    profiles_a, profiles_b = _paired_profiles(a, b)
    if not bool((profiles_a.valid & profiles_b.valid).all()):
        raise ValueError("归一化差异仅定义于正总质量地图；请先显式处理无效地图")
    return _distance(profiles_a, profiles_b, normalized=True)


def absolute_discrepancy(a: Tensor, b: Tensor) -> Tensor:
    """绝对 cone 距离，允许零地图；前导维可广播，输出为 float64。"""
    profiles_a, profiles_b = _paired_profiles(a, b)
    return _distance(profiles_a, profiles_b, normalized=False)


def _select_experts(profiles: RelevanceProfiles, indices: Tensor) -> RelevanceProfiles:
    return RelevanceProfiles(
        row_mass=profiles.row_mass.index_select(-2, indices),
        total_mass=profiles.total_mass.index_select(-1, indices),
        relative_mass=profiles.relative_mass.index_select(-2, indices),
        lag_distribution=profiles.lag_distribution.index_select(-3, indices),
        valid=profiles.valid.index_select(-1, indices),
    )


def pairwise_discrepancies(maps: Tensor) -> PairwiseDiscrepancies:
    """返回 [..., K, N, L] 的全部 K(K-1)/2 个专家对，顺序为 m<n。

    distances、valid 形状为 [..., P]，indices 形状为 [2, P]。
    零总量 pair 的距离无定义，以 NaN 标记，不视作零距离或最大距离。
    """
    maps = _checked_maps(maps)
    if maps.ndim < 3 or maps.shape[-3] < 2:
        raise ValueError("专家地图必须为 [..., K, N, L]，且 K 至少为 2")
    count = maps.shape[-3]
    indices = torch.triu_indices(count, count, offset=1, device=maps.device)
    profiles = _profiles(maps)
    a = _select_experts(profiles, indices[0])
    b = _select_experts(profiles, indices[1])
    valid = a.valid & b.valid
    distances = _distance(a, b, normalized=True)
    distances = torch.where(valid, distances, torch.full_like(distances, float("nan")))
    return PairwiseDiscrepancies(distances, valid, indices)


def specialization_loss(maps: Tensor, margin: float) -> Tensor:
    """固定全部专家对的 squared-hinge 均值；任一零地图 pair 罚 margin²。

    maps 为 [..., K, N, L]；对专家对及所有前导样本取均值，分母不删减。
    零地图分支保留零梯度；不人为增加恢复项或归一化阈值。
    """
    if not isinstance(margin, Real) or not math.isfinite(margin) or not 0 < margin < 1:
        raise ValueError("margin 必须是 0 与 1 之间的有限实数")
    pairs = pairwise_discrepancies(maps)
    if pairs.distances.numel() == 0:
        raise ValueError("specialization_loss 至少需要一个样本")
    # 先替换 NaN 占位再计算平方，避免未选分支把 NaN 带入梯度。
    safe_distances = torch.where(pairs.valid, pairs.distances, torch.zeros_like(pairs.distances))
    hinge = (float(margin) - safe_distances).clamp_min(0).square()
    penalties = torch.where(pairs.valid, hinge, torch.full_like(hinge, float(margin) ** 2))
    return penalties.mean()
