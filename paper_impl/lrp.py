"""可微逐算子 AttnLRP 规则；保留偏置、稳定项及常量的相关性账本。

规则依据 Achtibat et al., ICML 2024, arXiv:2402.05602 §§3.2–3.3。
这里显式重分配 relevance，不通过替换 autograd 的 backward 实现；因此
specialization loss 对参数求导时不会把 LRP 规则误当作普通前向梯度。
"""
from dataclasses import dataclass, field
from typing import Callable
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class Node:
    value: Tensor
    parents: tuple[int, ...] = ()
    rule: Callable | None = None
    name: str = "input"
    channel_key: str | None = None


@dataclass
class Propagation:
    input_relevance: Tensor
    residual: Tensor
    conservation_error: Tensor
    channels: dict[str, Tensor] = field(default_factory=dict)
    residual_by_operation: dict[str, Tensor] = field(default_factory=dict)


def sample_sum(x: Tensor) -> Tensor:
    return x.reshape(x.shape[0], -1).sum(-1)


class RelevanceTape:
    """batch轴始终为第0维；所有非输入节点均有明确的传播规则。"""

    def __init__(self, x: Tensor, epsilon: float = 1e-6):
        if epsilon <= 0:
            raise ValueError("LRP epsilon must be positive")
        self.nodes = [Node(x)]
        self.epsilon = epsilon

    def value(self, node: int) -> Tensor:
        return self.nodes[node].value

    def stable(self, z: Tensor) -> Tensor:
        return z + self.epsilon * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))

    def push(self, value, parents, rule, name, channel_key=None):
        self.nodes.append(Node(value, tuple(parents), rule, name, channel_key))
        return len(self.nodes) - 1

    def linear(self, node, weight, bias=None, key=None):
        x = self.value(node)
        if weight.ndim == 2:
            z = F.linear(x, weight, bias)
            rule = lambda r: (x * torch.matmul(r / self.stable(z), weight),)
        elif weight.ndim == 3:
            z = torch.einsum("b...i,boi->b...o", x, weight)
            if bias is not None:
                z = z + bias
            rule = lambda r: (x * torch.einsum("b...o,boi->b...i", r / self.stable(z), weight),)
        else:
            raise ValueError("weight must be [out,in] or [batch,out,in]")
        return self.push(z, (node,), rule, "linear", key)

    def reshape(self, node, shape):
        x = self.value(node)
        return self.push(x.reshape(shape), (node,), lambda r: (r.reshape(x.shape),), "reshape")

    def permute(self, node, dims):
        inverse = tuple(dims.index(i) for i in range(len(dims)))
        return self.push(self.value(node).permute(dims), (node,), lambda r: (r.permute(inverse),), "permute")

    def scale(self, node, scale):
        # 固定非零逐点乘法不分流相关性。
        return self.push(self.value(node) * scale, (node,), lambda r: (r,), "scale")

    def constant_add(self, node, constant):
        x = self.value(node)
        z = x + constant
        return self.push(z, (node,), lambda r: (x * (r / self.stable(z)),), "position_or_variable_embedding")

    def add(self, left, right):
        x, y = self.value(left), self.value(right)
        if x.shape != y.shape:
            raise ValueError("residual-add operands must have identical shape")
        z = x + y
        return self.push(z, (left, right), lambda r: (x * (r / self.stable(z)), y * (r / self.stable(z))), "residual_add")

    def gelu(self, node):
        return self.push(F.gelu(self.value(node)), (node,), lambda r: (r,), "gelu")

    def softmax(self, node):
        x = self.value(node)
        y = torch.softmax(x, dim=-1)
        return self.push(y, (node,), lambda r: (x * (r - y * r.sum(-1, keepdim=True)),), "softmax_taylor")

    def matmul(self, left, right):
        a, b = self.value(left), self.value(right)
        z = a @ b
        def redistribute(r):
            s = r / self.stable(z)
            return .5 * a * (s @ b.transpose(-2, -1)), .5 * b * (a.transpose(-2, -1) @ s)
        return self.push(z, (left, right), redistribute, "bilinear_matmul")

    def mean(self, node, dim):
        x = self.value(node)
        z = x.mean(dim=dim)
        return self.push(z, (node,), lambda r: (x / x.shape[dim] * (r / self.stable(z)).unsqueeze(dim),), "mean_pool")

    def layer_norm(self, node, layer):
        x = self.value(node)
        centered = x - x.mean(-1, keepdim=True)
        def center_rule(r):
            s = r / self.stable(centered)
            return (x * (s - s.mean(-1, keepdim=True)),)
        c = self.push(centered, (node,), center_rule, "mean_center")
        u = centered / torch.sqrt(centered.square().mean(-1, keepdim=True) + layer.eps)
        n = self.push(u, (c,), lambda r: (r,), "normalization_identity")
        y = u * layer.weight + layer.bias
        return self.push(y, (n,), lambda r: (u * layer.weight * (r / self.stable(y)),), "normalization_scale_bias")

    def propagate(self, output: int, relevance: Tensor) -> Propagation:
        if relevance.shape != self.value(output).shape:
            raise ValueError("output relevance shape mismatch")
        pending = {output: relevance}
        residual = torch.zeros_like(sample_sum(relevance))
        channels, ledger = {}, {}
        for idx in range(output, 0, -1):
            if idx not in pending:
                continue
            r = pending.pop(idx)
            node = self.nodes[idx]
            if node.channel_key is not None:
                dims = tuple(range(1, r.ndim - 1))
                mass = r.abs().sum(dims) if dims else r.abs()
                channels[node.channel_key] = channels.get(node.channel_key, 0) + mass
            values = node.rule(r)
            lost = sample_sum(r) - sum(sample_sum(v) for v in values)
            residual = residual + lost
            ledger[node.name] = ledger.get(node.name, 0) + lost
            for parent, part in zip(node.parents, values):
                pending[parent] = pending.get(parent, 0) + part
        result = pending.get(0, torch.zeros_like(self.value(0)))
        error = sample_sum(relevance) - sample_sum(result) - residual
        return Propagation(result, residual, error, channels, ledger)
