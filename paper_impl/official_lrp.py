"""Explicit operator-wise relevance propagation through full official forecasters.

AttnLRP rules are used for softmax, bilinear products, affine operations and
LayerNorm. FFT/complex operations use a documented real-coordinate epsilon-LRP
extension. This is an explicit reverse relevance tape, not network input*gradient.
The supported ATen allowlist is closed: new operations fail instead of silently
falling back to a whole-network gradient explanation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import operator

import torch
from torch import Tensor, nn
from torch.fx import Node, GraphModule, map_arg
from torch.fx.experimental.proxy_tensor import make_fx, get_proxy_mode, get_proxy_slot

from .official_backbones import OfficialForecaster


class UnsupportedLRPOperator(RuntimeError):
    """An executed operator has no audited relevance rule."""


@dataclass
class OfficialAttribution:
    maps: Tensor                         # [B, T, N, L], mean_h(abs(R_h))
    channels: dict[str, Tensor]          # [B, T, output_channels]
    residual: Tensor                    # [B, T], signed mean_h
    conservation_error: Tensor          # [B, T], mean_h(abs(signed accounting error))
    residual_by_operation: dict[str, Tensor] = field(default_factory=dict)
    predictions: Tensor | None = None   # [B, H, T], precisely the explained output
    operators: tuple[str, ...] = ()


_ROUTING = {
    "view", "_unsafe_view", "reshape", "permute", "transpose", "t", "unsqueeze", "squeeze",
    "clone", "alias", "expand", "repeat", "unfold", "slice", "select", "index", "gather",
    "cat", "stack", "slice_scatter", "select_scatter", "index_put", "constant_pad_nd",
    "replication_pad1d", "copy", "masked_fill", "where", "topk", "max", "max_pool2d_with_indices",
    "_to_copy", "split", "split_with_sizes", "unbind", "roll", "flip",
}
_AFFINE = {"add", "sub", "addmm", "mm", "bmm", "convolution", "mean", "sum", "cumsum", "avg_pool2d", "neg"}
_IDENTITY_ACTIVATIONS = {"gelu", "relu", "elu", "tanh", "sqrt", "rsqrt"}
_CONSTANTS = {
    "new_empty", "new_zeros", "new_ones", "empty", "empty_like", "zeros", "zeros_like", "ones",
    "ones_like", "full", "full_like", "arange", "randint", "lift_fresh_copy", "scalar_tensor",
}
_OTHER = {"detach", "_conj", "_softmax", "native_layer_norm", "_native_batch_norm_legit_no_training",
          "native_batch_norm", "cudnn_batch_norm", "mul", "div", "var", "_fft_r2c", "_fft_c2r", "_fft_c2c", "abs",
          "triu", "bitwise_not", "eq", "ne", "lt", "le", "gt", "ge"}


def _kind(target) -> str:
    if target is operator.getitem:
        return "getitem"
    name = str(target)
    if not name.startswith("aten."):
        raise UnsupportedLRPOperator(f"unsupported non-ATen target {name}")
    kind = name.split(".")[1]
    if kind not in _ROUTING | _AFFINE | _IDENTITY_ACTIVATIONS | _CONSTANTS | _OTHER:
        raise UnsupportedLRPOperator(f"no explicit LRP rule for {name}")
    return kind


def _real(x: Tensor) -> Tensor:
    return torch.view_as_real(x.resolve_conj()) if x.is_complex() else x


def _stable(x: Tensor, epsilon: float) -> Tensor:
    return x + epsilon * torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def _execute(target, args, kwargs):
    """Exact functional real-coordinate implementation for unsupported complex scatters."""
    kind = _kind(target)
    if kind in ("slice_scatter", "select_scatter") and args[0].is_complex():
        base, source, *tail = args
        if tail:
            tail[0] = tail[0] % base.ndim
        adjusted = dict(kwargs)
        if "dim" in adjusted:
            adjusted["dim"] %= base.ndim
        result = target(_real(base), _real(source), *tail, **adjusted)
        return torch.view_as_complex(result.contiguous())
    if kind == "copy":
        # Functionalization emits copy(dst, src), not an in-place copy_.
        return args[1].to(dtype=args[0].dtype, device=args[0].device).expand_as(args[0]).clone()
    return target(*args, **kwargs)


@contextmanager
def _evaluation(module: nn.Module):
    modes = [(m, m.training) for m in module.modules()]
    # RevIN caches per-call statistics as ordinary attributes. Functionalization
    # wraps those tensors; restore the original attributes after tracing.
    statistics = [(m, key, hasattr(m, key), getattr(m, key, None)) for m in module.modules()
                  if type(m).__name__ == "RevIN" for key in ("mean", "stdev", "last")]
    try:
        module.eval()
        yield
    finally:
        for m, key, existed, value in statistics:
            if existed:
                setattr(m, key, value)
            elif hasattr(m, key):
                delattr(m, key)
        for m, training in modes:
            m.training = training


@contextmanager
def _sampling(x: Tensor, seed: int):
    devices = [x.device.index or 0] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if x.is_cuda:
            with torch.cuda.device(x.device):
                torch.cuda.manual_seed(seed)
        yield


def _functionalize_linear_bias_additions(graph: GraphModule) -> None:
    """Remove a Torch linear decomposition mutation on an unaliased fresh mm.

    Functionalization does not wrap closed-over parameter-only paths. A Linear
    applied to Crossformer's learned routing tokens can therefore still lower
    to mm -> view -> add_. Replacing that addition is safe only if the fresh
    matrix product and every intermediate view have a single consumer. General
    mutations (including input/buffer mutations and observable aliases) fail.
    """
    views = {torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default}
    changed = False
    for node in graph.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.add_.Tensor:
            continue
        operand = node.args[0]
        while isinstance(operand, Node) and operand.op == "call_function" and operand.target in views:
            if len(operand.users) != 1:
                raise UnsupportedLRPOperator("aliased in-place linear bias addition")
            operand = operand.args[0]
        if not (isinstance(operand, Node) and operand.op == "call_function"
                and operand.target == torch.ops.aten.mm.default and len(operand.users) == 1):
            raise UnsupportedLRPOperator("in-place addition is not an isolated fresh linear output")
        node.target = torch.ops.aten.add.Tensor
        node.meta["functionalized_linear_bias"] = True
        changed = True
    if changed:
        graph.graph.lint()
        graph.recompile()


def _trace(model: OfficialForecaster, sample: Tensor, freeze_head: bool, seed: int) -> GraphModule:
    hooks = []
    frozen = {id(p) for p in model.head_parameters()} if freeze_head else set()
    parameters = {name: p.detach() if id(p) in frozen else p for name, p in model.named_parameters()}

    def annotate(name):
        def hook(module, args, output):
            mode = get_proxy_mode()
            if mode is not None:
                raw = torch._from_functional_tensor(output) if torch._is_functional_tensor(output) else output
                slot = get_proxy_slot(raw, mode.tracer, default=None)
                if slot is None:
                    raise RuntimeError(f"cannot locate full Linear output {name} in the trace")
                slot.proxy.node.meta["official_linear"] = name + ".weight"
            return output
        return hook

    with _evaluation(model):
        try:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    hooks.append(module.register_forward_hook(annotate(name)))
            def complete_forward(value):
                return torch.func.functional_call(model, parameters, (value,))
            # Complex select_scatter lacks native autograd in Torch 2.7, so trace
            # without autograd; _execute supplies its exact differentiable form.
            with _sampling(sample, seed), torch.no_grad():
                graph = make_fx(torch.func.functionalize(complete_forward, remove="mutations"))(sample)
        finally:
            for hook in hooks:
                hook.remove()
    _functionalize_linear_bias_additions(graph)
    for node in graph.graph.nodes:
        if node.op == "call_function":
            _kind(node.target)
    return graph


class OperatorRelevanceTape:
    """A single-sample pure ATen graph plus explicit local redistribution rules."""

    def __init__(self, graph: GraphModule, sample: Tensor, epsilon=1e-6, create_graph=True):
        if sample.shape[0] != 1:
            raise ValueError("operator tape explains one sample at a time")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite")
        self.graph, self.epsilon, self.create_graph = graph, epsilon, create_graph
        self.values, self.active = {}, {}
        self.input_node = self.output_node = None
        for node in graph.graph.nodes:
            if node.op == "placeholder":
                value = sample
                self.input_node = node
                active = True
            elif node.op == "get_attr":
                value = graph
                for component in str(node.target).split("."):
                    value = getattr(value, component)
                active = False
            elif node.op == "call_function":
                kind = _kind(node.target)
                args = map_arg(node.args, self.values.__getitem__)
                kwargs = map_arg(node.kwargs, self.values.__getitem__)
                value = args[0][args[1]] if kind == "getitem" else _execute(node.target, args, kwargs)
                active = kind not in _CONSTANTS and any(self.active.get(n, False) for n in node.all_input_nodes)
                # Integer selections/masks are discrete routing state, not value
                # relevance destinations. Recompute them at every explained input.
                if isinstance(value, Tensor):
                    active = active and (value.is_floating_point() or value.is_complex())
            elif node.op == "output":
                self.output_node = node.args[0]
                continue
            else:
                raise UnsupportedLRPOperator(f"unsupported FX node kind {node.op}")
            self.values[node], self.active[node] = value, active
        if not isinstance(self.output_node, Node) or not isinstance(self.values[self.output_node], Tensor):
            raise UnsupportedLRPOperator("a single Tensor forecasting output is required")
        self.output = self.values[self.output_node]

    def _parents(self, node):
        return [n for n in node.all_input_nodes if self.active.get(n, False) and isinstance(self.values[n], Tensor)
                and (self.values[n].is_floating_point() or self.values[n].is_complex())]

    def _vjp(self, node, relevance, *, epsilon_rule=False, degree=1.0, parents=None):
        """Local VJP only: each cloned operand is independent at this operation.

        Clones retain their upstream parameter graphs. Thus differentiating the
        explicit LRP formula is ordinary autograd, without custom backward rules
        that would confuse prediction gradients with relevance propagation.
        """
        parents = self._parents(node) if parents is None else parents
        if not parents:
            return {}, relevance.sum() * 0
        local = [_real(self.values[p]).clone().requires_grad_(True) for p in parents]
        replacement = {p: torch.view_as_complex(v.contiguous()) if self.values[p].is_complex() else v
                       for p, v in zip(parents, local)}
        def operand(n):
            return replacement.get(n, self.values[n])
        out = _execute(node.target, map_arg(node.args, operand), map_arg(node.kwargs, operand))
        if isinstance(out, (tuple, list)):
            out = out[0]
        real_out = _real(out)
        original = self.values[node]
        if isinstance(original, (tuple, list)):
            original = original[0]
        real_original = _real(original)
        scaling = relevance / _stable(real_original, self.epsilon) if epsilon_rule else relevance
        gradients = torch.autograd.grad(real_out, local, grad_outputs=scaling,
                                        create_graph=self.create_graph, retain_graph=True, allow_unused=True)
        parts = {p: (v * g / degree if epsilon_rule else g) if g is not None else torch.zeros_like(v)
                 for p, v, g in zip(parents, local, gradients)}
        stabilizer = (relevance - relevance * real_original / _stable(real_original, self.epsilon)).sum() if epsilon_rule else relevance.sum() * 0
        return parts, stabilizer

    def _rule(self, node, r):
        kind = _kind(node.target)
        parents = self._parents(node)
        if not parents:
            return {}, r.sum() * 0, "constant"
        if kind in ("detach", "_conj"):
            return {parents[0]: r}, r.sum() * 0, kind + "_identity"
        if kind == "_softmax":
            x = self.values[node.args[0]]
            y = self.values[node]
            dim = node.args[1]
            # Masked -inf logits represent a routing constant. 0*inf is avoided
            # without altering the softmax or the unmasked Taylor formula.
            finite_x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
            part = finite_x * (r - y * r.sum(dim, keepdim=True))
            return {node.args[0]: part}, r.sum() * 0, "softmax_taylor"
        if kind == "native_layer_norm":
            xnode = node.args[0]
            x = self.values[xnode]
            normalized_shape = node.args[1]
            dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))
            centered = x - x.mean(dims, keepdim=True)
            args = map_arg(node.args, self.values.__getitem__)
            std = torch.sqrt(centered.square().mean(dims, keepdim=True) + args[4])
            u = centered / std
            weight = args[2] if args[2] is not None else 1
            output = self.values[node][0]
            scale = r / _stable(output, self.epsilon)
            rc = u * weight * scale
            rc_scale = rc / _stable(centered, self.epsilon)
            part = x * (rc_scale - rc_scale.mean(dims, keepdim=True))
            stab = (r - r * output / _stable(output, self.epsilon)).sum()
            stab = stab + (rc - rc * centered / _stable(centered, self.epsilon)).sum()
            return {xnode: part}, stab, "layer_norm_center_epsilon_normalization_identity"
        if kind in ("_native_batch_norm_legit_no_training", "native_batch_norm", "cudnn_batch_norm"):
            args = map_arg(node.args, self.values.__getitem__)
            if kind != "_native_batch_norm_legit_no_training" and args[5]:
                raise UnsupportedLRPOperator("training-mode BatchNorm has no single-sample relevance rule")
            xnode = node.args[0]
            x = self.values[xnode]
            weight = args[1] if args[1] is not None else torch.ones_like(args[4])
            factor = weight / torch.sqrt(args[4] + args[-1])
            factor = factor.reshape(1, -1, *([1] * (x.ndim - 2)))
            output = self.values[node][0]
            part = x * factor * r / _stable(output, self.epsilon)
            stab = (r - r * output / _stable(output, self.epsilon)).sum()
            return {xnode: part}, stab, "batch_norm_eval_affine_epsilon"
        if kind == "div":
            first = node.args[0]
            if not isinstance(first, Node) or first not in parents:
                raise UnsupportedLRPOperator("division requires a data numerator; denominator-only relevance has no audited rule")
            return {first: r.sum_to_size(_real(self.values[first]).shape)}, r.sum() * 0, "division_denominator_fixed"
        if kind == "mul" and not any(self.values[p].is_complex() for p in parents):
            parts = {p: r.sum_to_size(self.values[p].shape) / len(parents) for p in parents}
            return parts, r.sum() * 0, "elementwise_product_uniform"
        if kind in _IDENTITY_ACTIVATIONS and not self.values[parents[0]].is_complex():
            return {parents[0]: r}, r.sum() * 0, kind + "_identity"
        if kind in _ROUTING:
            parts, stabilizer = self._vjp(node, r)
            return parts, stabilizer, kind + "_routing"
        if kind in _AFFINE or kind in ("mul", "var", "_fft_r2c", "_fft_c2r", "_fft_c2c", "abs", "tanh", "_native_batch_norm_legit_no_training", "native_batch_norm"):
            degree = 1.0
            if kind in ("mm", "bmm", "mul") and sum(isinstance(arg, Node) and arg in parents for arg in node.args[:2]) > 1:
                degree = 2.0
            if kind == "var":
                degree = 2.0
            parts, stabilizer = self._vjp(node, r, epsilon_rule=True, degree=degree)
            label = "complex_real_coordinate_" + kind if any(self.values[p].is_complex() for p in parents) or kind.startswith("_fft") else kind
            return parts, stabilizer, label + "_epsilon"
        raise UnsupportedLRPOperator(f"no relevance rule for data-bearing {node.target}")

    def propagate(self, relevance: Tensor):
        if relevance.shape != _real(self.output).shape:
            raise ValueError("output relevance shape mismatch")
        pending, channels, ledger = {(self.output_node, None): relevance}, {}, {}
        for node in reversed(tuple(self.graph.graph.nodes)):
            if node.op in ("output", "placeholder"):
                continue
            value = self.values[node]
            key = (node, None) if isinstance(value, Tensor) else (node, 0)
            if key not in pending:
                continue
            r = pending.pop(key)
            if not isinstance(value, Tensor):
                extra = [k for k in pending if k[0] is node]
                if extra:
                    raise UnsupportedLRPOperator("relevance reached auxiliary normalization/index outputs")
            channel_name = node.meta.get("official_linear")
            if channel_name is not None:
                if r.ndim == 1:
                    mass = r.abs()
                else:
                    mass = r.abs().sum(tuple(range(r.ndim - 1)))
                channels[channel_name] = channels.get(channel_name, 0) + mass
            if node.op == "get_attr":
                ledger["parameter_or_buffer_constant"] = ledger.get("parameter_or_buffer_constant", 0) + r.sum()
                continue
            if node.target is operator.getitem:
                parent, index = node.args
                if not isinstance(self.values[parent], (tuple, list)) or not isinstance(index, int):
                    raise UnsupportedLRPOperator("only tuple-result FX getitem is supported")
                ref = (parent, index)
                pending[ref] = pending.get(ref, 0) + r
                continue
            parts, stabilization, label = self._rule(node, r)
            loss = r.sum() - sum((part.sum() for part in parts.values()), r.sum() * 0)
            ledger[label + ":stabilization"] = ledger.get(label + ":stabilization", 0) + stabilization
            ledger[label + ":bias_or_constant"] = ledger.get(label + ":bias_or_constant", 0) + loss - stabilization
            for parent, part in parts.items():
                ref = (parent, None)
                pending[ref] = pending.get(ref, 0) + part
        input_relevance = pending.pop((self.input_node, None), torch.zeros_like(_real(self.values[self.input_node])))
        if pending:
            raise UnsupportedLRPOperator("unconsumed relevance destinations: " + str([(n.name, slot) for n, slot in pending]))
        residual = sum(ledger.values(), relevance.sum() * 0)
        error = relevance.sum() - input_relevance.sum() - residual
        return input_relevance, channels, residual, error, ledger


def attribute_official(model: OfficialForecaster, x: Tensor, *, epsilon=1e-6, freeze_head=True,
                       sampling_seed=42, create_graph: bool | None = None) -> OfficialAttribution:
    """Explain each output step separately, then average absolute relevance.

    Evaluation mode and a preserved fixed RNG stream make sparse attention
    reproducible. Computation is per sample because official models fold batch
    and variable axes differently. No gate is used in this expert attribution.
    """
    if not isinstance(model, OfficialForecaster):
        raise TypeError("attribute_official requires a complete OfficialForecaster")
    retain = torch.is_grad_enabled() if create_graph is None else create_graph
    linear_widths = {name + ".weight": module.out_features for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    sample_maps, sample_channels, sample_residuals, sample_errors, sample_ledgers, predictions = [], [], [], [], [], []
    operators = ()
    with torch.enable_grad():
        graph = _trace(model, x[:1], freeze_head, sampling_seed)
        operators = tuple(sorted({str(node.target) for node in graph.graph.nodes if node.op == "call_function"}))
        for sample in x.split(1):
            with _sampling(sample, sampling_seed):
                tape = OperatorRelevanceTape(graph, sample, epsilon, retain)
            predictions.append(tape.output)
            target_maps, target_channels, target_residuals, target_errors, target_ledgers = [], [], [], [], []
            for target in range(model.n_targets):
                maps, residuals, errors, channels, ledgers = [], [], [], {}, {}
                for step in range(model.config.horizon):
                    mask = torch.zeros_like(tape.output)
                    mask[:, step, target] = 1
                    relevance = tape.output * mask
                    input_r, local_channels, residual, error, ledger = tape.propagate(relevance)
                    maps.append(input_r[0].transpose(0, 1).abs())
                    residuals.append(residual)
                    errors.append(error.abs())
                    for key, value in local_channels.items():
                        channels[key] = channels.get(key, 0) + value / model.config.horizon
                    for key, value in ledger.items():
                        ledgers[key] = ledgers.get(key, 0) + value / model.config.horizon
                target_maps.append(torch.stack(maps).mean(0))
                target_residuals.append(torch.stack(residuals).mean())
                target_errors.append(torch.stack(errors).mean())
                target_channels.append({key: channels.get(key, x.new_zeros(width)) for key, width in linear_widths.items()})
                target_ledgers.append(ledgers)
            sample_maps.append(torch.stack(target_maps))
            sample_channels.append({key: torch.stack([target[key] for target in target_channels]) for key in linear_widths})
            sample_residuals.append(torch.stack(target_residuals))
            sample_errors.append(torch.stack(target_errors))
            all_keys = set().union(*(set(ledger) for ledger in target_ledgers))
            sample_ledgers.append({key: torch.stack([ledger.get(key, x.new_zeros(())) for ledger in target_ledgers]) for key in all_keys})
        keys = set().union(*(set(ledger) for ledger in sample_ledgers))
        result = OfficialAttribution(
            torch.stack(sample_maps), {key: torch.stack([channel[key] for channel in sample_channels]) for key in linear_widths},
            torch.stack(sample_residuals), torch.stack(sample_errors),
            {key: torch.stack([ledger.get(key, x.new_zeros(model.n_targets)) for ledger in sample_ledgers]) for key in keys},
            torch.cat(predictions), operators,
        )
    if not retain:
        result.maps = result.maps.detach()
        result.channels = {key: value.detach() for key, value in result.channels.items()}
        result.residual = result.residual.detach()
        result.conservation_error = result.conservation_error.detach()
        result.residual_by_operation = {key: value.detach() for key, value in result.residual_by_operation.items()}
        result.predictions = result.predictions.detach()
    return result


def attribute_official_mixture(model, x: Tensor, freeze_head=True):
    """Explain full expert paths; expose only the controller's eligible channels."""
    from .model import Attribution
    result = [attribute_official(expert, x, epsilon=model.config.lrp_epsilon, freeze_head=freeze_head,
                                 sampling_seed=model.config.official_sampling_seed + i)
              for i, expert in enumerate(model.experts)]
    all_channels = {f"experts.{i}.{key}": value for i, item in enumerate(result) for key, value in item.channels.items()}
    eligible = model.expert_weights()
    missing = set(eligible) - all_channels.keys()
    if missing:
        raise RuntimeError("Missing relevance for eligible expert weights: " + ", ".join(sorted(missing)))
    channels = {key: all_channels[key] for key in eligible}
    keys = set().union(*(set(item.residual_by_operation) for item in result))
    ledger = {key: torch.stack([item.residual_by_operation.get(key, torch.zeros_like(item.residual)) for item in result], dim=2) for key in keys}
    return Attribution(torch.stack([item.maps for item in result], dim=2), channels,
                       torch.stack([item.residual for item in result], dim=2),
                       torch.stack([item.conservation_error for item in result], dim=2), ledger)
