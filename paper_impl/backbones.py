"""将仓库现有八个自定义 style 编码器接为 [B, L, N] -> [B, D]。

实现复用 models/attention_blocks.py，名称中的 style 必须保留；这些
模块不是相应论文的官方模型，也没有在此补造官方机制或位置编码。
隐藏维、注意力头数、层数沿用调用方配置，前馈宽度为 4D，dropout=0。
编码器不读取 horizon/targets，由调用方的输出头负责预测形状。

长度适配：PatchTST 在右端复制最后一步至 patch_size 的整数倍，既
支持 L < patch_size，也保留尾部；复制值也参与原模块的通道均值。
FEDformer 的频点数取 min(4, floor(L/2)+1)，使短序列频域投影匹配。
池化类输入整理为变量优先的连续布局，使原模块转置后的池化输入连续，
规避当前 PyTorch/CUDA 对非连续平均池化输入的反传差异。
其余行为保持旧模块定义。仅支持构造时指定的 L、N，不隐式截断输入。
"""

from __future__ import annotations

from numbers import Integral
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models import attention_blocks as legacy


LEGACY_ENCODERS: Mapping[str, type[nn.Module]] = MappingProxyType({
    "transformer_style": legacy.TransformerTemporalEncoder,
    "itransformer_style": legacy.InvertedTransformerTemporalEncoder,
    "patchtst_style": legacy.PatchTSTTemporalEncoder,
    "informer_style": legacy.InformerTemporalEncoder,
    "autoformer_style": legacy.AutoformerTemporalEncoder,
    "fedformer_style": legacy.FEDformerTemporalEncoder,
    "pyraformer_style": legacy.PyraformerTemporalEncoder,
    "crossformer_style": legacy.CrossformerTemporalEncoder,
})

# 旧代码的直接本地依赖；父训练流程可将其 SHA256 写入运行清单。
LEGACY_DEPENDENCY_FILES = ("models/__init__.py", "models/attention_blocks.py")


def _positive_config_integer(config, name, *, default=None):
    value = getattr(config, name, default)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"config.{name} must be a positive integer")
    return int(value)


class LegacySummaryEncoder(nn.Module):
    """只做配置/长度适配；backbone 是原模块实例，输出维度为 output_dim。"""

    def __init__(self, config):
        super().__init__()
        name = getattr(config, "encoder", None)
        if not isinstance(name, str) or name not in LEGACY_ENCODERS:
            raise ValueError(f"unknown style encoder {name!r}; expected one of {tuple(LEGACY_ENCODERS)}")
        self.encoder_name = name
        self.lookback = _positive_config_integer(config, "lookback")
        self.n_variables = _positive_config_integer(config, "n_variables")
        self.output_dim = _positive_config_integer(config, "hidden")
        heads = _positive_config_integer(config, "heads")
        depth = _positive_config_integer(config, "depth")
        if self.output_dim % heads:
            raise ValueError("config.hidden must be divisible by config.heads")

        self.right_padding = 0
        self.patch_size = None
        if name == "patchtst_style":
            self.patch_size = _positive_config_integer(config, "patch_size", default=4)
            self.right_padding = (-self.lookback) % self.patch_size
        cfg = legacy.TransformerEncoderConfig(
            d_model=self.output_dim,
            nhead=heads,
            num_layers=depth,
            dim_feedforward=4 * self.output_dim,
            dropout=0.0,
            pooling="mean",
            seq_len=self.lookback + self.right_padding,
        )
        self.legacy_config = cfg
        encoder_type = LEGACY_ENCODERS[name]
        if name in ("transformer_style", "itransformer_style"):
            self.backbone = encoder_type(self.n_variables, cfg, use_custom=True)
        elif name == "patchtst_style":
            self.backbone = encoder_type(self.n_variables, cfg, patch_len=self.patch_size, stride=self.patch_size)
        elif name == "fedformer_style":
            self.backbone = encoder_type(self.n_variables, cfg, topk_freq=min(4, self.lookback // 2 + 1))
        else:
            self.backbone = encoder_type(self.n_variables, cfg)

    def forward(self, x: Tensor) -> Tensor:
        """保留输入与全部有效参数的 autograd 路径，不改变 dtype/device。"""
        if not isinstance(x, Tensor):
            raise TypeError("encoder input must be a tensor")
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1:] != (self.lookback, self.n_variables):
            raise ValueError(f"expected nonempty [B, {self.lookback}, {self.n_variables}] input")
        if not x.is_floating_point():
            raise TypeError("encoder input must be floating point")
        if self.encoder_name in ("autoformer_style", "fedformer_style", "pyraformer_style"):
            # 原模块用 x.transpose(1, 2) 池化；显式布局不改变值或梯度定义。
            x = x.transpose(1, 2).contiguous().transpose(1, 2)
        if self.right_padding:
            x = F.pad(x.transpose(1, 2), (0, self.right_padding), mode="replicate").transpose(1, 2)
        return self.backbone(x)


def build_legacy_encoder(config) -> LegacySummaryEncoder:
    """使用 config 的 encoder/n_variables/lookback/hidden/heads/depth 字段。"""
    return LegacySummaryEncoder(config)
