"""Complete author FSNet encoders with device-safe, checkpointed fast memory.

The TSEncoder, residual blocks and dilated encoder execute the pinned, untouched
author files. Only SamePadConv's device handling/persistence is ported here.
Both source variants are explicit because their memory equations differ.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F


_ROOT = Path(__file__).resolve().parents[1] / "third_party/official_online"


class PersistentSamePadConv(nn.Module):
    """Author calibration convolution; all cross-step memory is persistent.

    ``onlinetsf`` deliberately preserves the author's index-weighted memory
    retrieval. ``onenet`` uses top-k attention values, as that repository does.
    The detached EMA is read only through top-k indices/detached weights, so
    retaining its autograd graph would have no useful gradient contribution.
    """

    source_variant = "onlinetsf"

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 groups=1, gamma=.9, device=None):
        super().__init__()
        if groups != 1 or out_channels % in_channels:
            raise ValueError("author FSNet calibration requires groups=1 and divisible channels")
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.padding = self.receptive_field // 2
        self.dilation, self.kernel_size = dilation, kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.dim = self.conv.weight.numel()
        self.grad_dim, self.shape = [self.dim], [self.conv.weight.shape]
        self.in_channels, self.out_features, self.n_chunks = in_channels, out_channels, in_channels
        self.chunk_in_d, self.chunk_out_d = self.dim // in_channels, kernel_size
        self.controller = nn.Sequential(nn.Linear(self.chunk_in_d, 64), nn.SiLU())
        self.calib_w = nn.Linear(64, kernel_size)
        self.calib_b = nn.Linear(64, out_channels // in_channels)
        self.calib_f = nn.Linear(64, out_channels // in_channels)
        memory_dim = in_channels * (kernel_size + 2 * out_channels // in_channels)
        self.W = nn.Parameter(torch.empty(memory_dim, 32), requires_grad=False)
        nn.init.xavier_uniform_(self.W)
        with torch.no_grad():
            self.W.copy_(self.W / (torch.relu(torch.norm(self.W) - 1) + 1))
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        self.gamma, self.f_gamma, self.tau = gamma, .3, .75
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.register_buffer("grads", torch.zeros(self.dim))
        self.register_buffer("f_grads", torch.zeros(self.dim))
        self.register_buffer("q_ema", torch.zeros(memory_dim))
        self.register_buffer("q_initialized", torch.tensor(False))
        self.register_buffer("trigger", torch.tensor(False))
        self.register_buffer("try_trigger", torch.tensor(False))

    def ctrl_params(self):
        for layer in (self.controller, self.calib_w, self.calib_b, self.calib_f):
            yield from layer.parameters()

    @torch.no_grad()
    def store_grad(self):
        if self.conv.weight.grad is None:
            return
        grad = F.normalize(self.conv.weight.grad.detach().clone()).flatten()
        self.f_grads.mul_(self.f_gamma).add_(grad, alpha=1 - self.f_gamma)
        if self.source_variant == "onlinetsf":
            if bool(self.try_trigger) and self.cos(self.f_grads, self.grads) < -self.tau:
                self.trigger.fill_(True)
            self.grads.mul_(self.gamma).add_(grad, alpha=1 - self.gamma)
        else:
            self.grads.mul_(self.gamma).add_(grad, alpha=1 - self.gamma)
            if not self.training and self.cos(self.f_grads, self.grads) < -self.tau:
                self.trigger.fill_(True)

    def fw_chunks(self):
        rep = self.controller(self.grads.reshape(self.n_chunks, -1))
        w, b, f = self.calib_w(rep), self.calib_b(rep), self.calib_f(rep)
        q = torch.cat((w.flatten(), b.flatten(), f.flatten()))
        with torch.no_grad():
            if self.source_variant == "onlinetsf" and not bool(self.q_initialized):
                query = q.detach()
            else:
                self.q_ema.mul_(self.f_gamma).add_(q.detach(), alpha=1 - self.f_gamma)
                query = self.q_ema
            self.q_initialized.fill_(True)
            if bool(self.trigger):
                self.trigger.fill_(False)
                attention = F.softmax(query @ self.W / .5, dim=0)
                values, indices = torch.topk(attention, 2)
                weights = indices.float() if self.source_variant == "onlinetsf" else values
                recalled = self.W[:, indices] @ weights.unsqueeze(1)
                sparse = torch.zeros_like(attention)
                sparse[indices] = values
                write = recalled @ sparse.unsqueeze(0)
                mask = torch.ones_like(self.W)
                mask[:, indices] = self.tau
                self.W.copy_(mask * self.W + (1 - mask) * write)
                self.W.div_(torch.relu(torch.norm(self.W) - 1) + 1)
            else:
                recalled = None
        if recalled is not None:
            rw, rb, rf = recalled.flatten().split((w.numel(), b.numel(), f.numel()))
            w = self.tau * w + (1 - self.tau) * rw.reshape_as(w)
            b = self.tau * b + (1 - self.tau) * rb.reshape_as(b)
            f = self.tau * f + (1 - self.tau) * rf.reshape_as(f)
        return w.unsqueeze(0), b.flatten(), f.flatten().view(1, -1, 1)

    def forward(self, x):
        w, b, f = self.fw_chunks()
        return f * F.conv1d(x, self.conv.weight * w, bias=self.bias * b,
                             padding=self.padding, dilation=self.dilation)

    def representation(self, x):
        result = self.conv(x)
        return result[:, :, :-self.remove] if self.remove else result

    _forward = representation


class OnlineTSFSamePadConv(PersistentSamePadConv):
    source_variant = "onlinetsf"


class AuthorSamePadConv(PersistentSamePadConv):
    source_variant = "onenet"


def _load_source_encoder(source):
    package = f"_paper_fsnet_{source}"
    if package + ".fsnet" in sys.modules:
        return sys.modules[package + ".fsnet"].TSEncoder
    folder = _ROOT / ("OnlineTSF/layers/ts2vec" if source == "onlinetsf" else "OneNet/models/ts2vec")
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(folder)]
    sys.modules[package] = namespace
    for filename in ("fsnet_", "fsnet"):
        spec = importlib.util.spec_from_file_location(package + "." + filename, folder / (filename + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if filename == "fsnet_":
            module.SamePadConv = OnlineTSFSamePadConv if source == "onlinetsf" else AuthorSamePadConv
    return sys.modules[package + ".fsnet"].TSEncoder


@dataclass(frozen=True)
class OfficialFSNetConfig:
    lookback: int
    horizon: int
    n_variables: int
    target_indices: tuple[int, ...]
    source: str = "onlinetsf"
    branch: str = "channel"
    hidden_dims: int = 64
    output_dims: int = 320
    depth: int = 10
    gamma: float = .9
    context_length: int = 60
    revin: bool = True

    def __post_init__(self):
        object.__setattr__(self, "target_indices", tuple(self.target_indices))
        for name in ("lookback", "horizon", "n_variables", "hidden_dims", "output_dims", "depth", "context_length"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"FSNet {name} must be a positive integer")
        if self.output_dims % self.hidden_dims:
            raise ValueError("FSNet output_dims must be a multiple of hidden_dims")
        if self.source not in ("onlinetsf", "onenet") or self.branch not in ("channel", "time"):
            raise ValueError("unknown FSNet source or branch")
        if not self.target_indices or any(i < 0 or i >= self.n_variables for i in self.target_indices):
            raise ValueError("FSNet target_indices outside input variables")
        if not 0 <= self.gamma < 1:
            raise ValueError("FSNet gamma must be in [0,1)")
        if self.branch == "time" and self.revin:
            raise ValueError("author temporal FSNet branch does not apply RevIN")


class OfficialFSNet(nn.Module):
    """Full FSNet branch using the author's TSEncoder/11 residual blocks.

    Default hidden=64, output=320, depth=10 reproduce the author architecture.
    Calendar features are seven zero channels when none are provided, matching
    the OnlineTSF model's documented no-calendar fallback. Target selection is
    performed after the complete multivariate prediction and RevIN inversion.
    """

    def __init__(self, config: OfficialFSNetConfig):
        super().__init__()
        self.config = config
        encoder_class = _load_source_encoder(config.source)
        kwargs = {"input_dims": config.lookback if config.branch == "time" else config.n_variables + 7,
                  "output_dims": config.output_dims, "hidden_dims": config.hidden_dims,
                  "depth": config.depth, "gamma": config.gamma}
        if config.source == "onenet":
            kwargs["device"] = torch.device("cpu")
        self.encoder = encoder_class(**kwargs)
        self.regressor = nn.Linear(config.output_dims,
                                   config.horizon if config.branch == "time" else config.horizon * config.n_variables)
        if config.revin:
            self.affine_weight = nn.Parameter(torch.ones(config.n_variables))
            self.affine_bias = nn.Parameter(torch.zeros(config.n_variables))
        # Original OneNet performs its online update with the encoders in eval;
        # OnlineTSF explicitly switches to train before its delayed update.
        self.online_training_mode = config.source != "onenet"

    def requires_grad_(self, requires_grad=True):
        super().requires_grad_(requires_grad)
        for layer in self.memory_layers():
            layer.W.requires_grad_(False)
        return self

    def memory_layers(self):
        return (layer for layer in self.encoder.modules() if isinstance(layer, PersistentSamePadConv))

    def try_trigger_(self, enabled=True):
        for layer in self.memory_layers():
            layer.try_trigger.fill_(bool(enabled))

    def store_grad(self):
        for layer in self.memory_layers():
            layer.store_grad()

    def forward(self, x, x_mark=None):
        cfg = self.config
        if x.ndim != 3 or x.shape[1:] != (cfg.lookback, cfg.n_variables):
            raise ValueError("FSNet input must have configured [B,lookback,N] shape")
        if cfg.branch == "time":
            # The original temporal branch bypasses the all_true wrapper and
            # therefore retains its author binomial masking during training.
            prediction = self.regressor(self.encoder.forward_time(x.clone())).transpose(1, 2)
        else:
            x = x[:, -cfg.context_length:].clone()
            if cfg.revin:
                mean = x.mean(dim=1, keepdim=True).detach()
                std = (x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
                x = (x - mean) / std * self.affine_weight + self.affine_bias
            if x_mark is None:
                marks = x.new_zeros(x.shape[0], x.shape[1], 7)
            else:
                marks = x_mark[:, -x.shape[1]:]
                if marks.shape != (x.shape[0], x.shape[1], 7):
                    raise ValueError("FSNet calendar covariates must have seven channels")
            rep = self.encoder(torch.cat((x, marks), dim=-1), mask="all_true")[:, -1]
            prediction = self.regressor(rep).reshape(len(x), cfg.horizon, cfg.n_variables)
            if cfg.revin:
                prediction = (prediction - self.affine_bias) / (self.affine_weight + 1e-10)
                prediction = prediction * std + mean
        return prediction[..., list(cfg.target_indices)]

    def head_parameters(self):
        return self.regressor.parameters()

    def head_parameter_names(self):
        return tuple("regressor." + name for name, _ in self.regressor.named_parameters())

    def get_extra_state(self):
        return {"configuration": asdict(self.config), "source_variant": self.config.source,
                "calendar_policy": "seven observed calendar features or explicit zero fallback",
                "checkpoint_memory": "W, grads, f_grads, q_ema, q_initialized, trigger, try_trigger"}

    def set_extra_state(self, state):
        if state["configuration"] != asdict(self.config):
            raise ValueError("FSNet checkpoint architecture/source configuration mismatch")


def author_onenet_branches(lookback, horizon, n_variables, target_indices, **kwargs):
    """Original OneNet temporal/channel FSNet pair, including its source rules."""
    context_length = kwargs.pop("context_length", lookback)
    if context_length != lookback:
        raise ValueError("original OneNet channel branch uses the complete lookback")
    common = dict(lookback=lookback, horizon=horizon, n_variables=n_variables,
                  target_indices=tuple(target_indices), source="onenet", revin=False,
                  context_length=context_length, **kwargs)
    return (OfficialFSNet(OfficialFSNetConfig(**common, branch="time")),
            OfficialFSNet(OfficialFSNetConfig(**common, branch="channel")))
