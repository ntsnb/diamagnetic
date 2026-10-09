"""Side-effect-free parameter and persistent-buffer accounting for experiments.

Counts describe instantiated tensors, not FLOPs, latency, optimizer state or
peak working memory. Complex tensor elements count once logically and twice
as real scalars. Parameter identities, not repeated registration paths, define
unique parameters; distinct Parameter objects sharing storage remain distinct.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import random

import numpy as np
import torch
from torch import nn


def _components(prefix, value):
    if value is None:
        return []
    if isinstance(value, (nn.Module, nn.Parameter)):
        return [(prefix, value)]
    if isinstance(value, Mapping):
        result = []
        for name, component in value.items():
            if not isinstance(name, str) or not name:
                raise ValueError("component names must be nonempty strings")
            result.extend(_components(prefix + "." + name, component))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [entry for index, component in enumerate(value)
                for entry in _components(f"{prefix}.{index}", component)]
    raise TypeError("components must be modules, Parameters, mappings/sequences of them, or None")


def _named_parameters(component):
    if isinstance(component, nn.Parameter):
        return [("", component)]
    return list(component.named_parameters(remove_duplicate=True))


def _tensor_counts(tensors, *, parameter):
    unique = {id(tensor): tensor for tensor in tensors}.values()
    result = {"tensor_count": 0, "logical_count": 0, "real_scalar_count": 0, "tensor_bytes": 0}
    if parameter:
        result.update(trainable_tensor_count=0, trainable_logical_count=0, trainable_real_scalar_count=0)
    for tensor in unique:
        if isinstance(tensor, (nn.parameter.UninitializedParameter, nn.parameter.UninitializedBuffer)):
            raise ValueError("materialize lazy tensors before reporting resource counts")
        logical = tensor.numel()
        real = logical * (2 if tensor.is_complex() else 1)
        result["tensor_count"] += 1
        result["logical_count"] += logical
        result["real_scalar_count"] += real
        result["tensor_bytes"] += logical * tensor.element_size()
        if parameter and tensor.requires_grad:
            result["trainable_tensor_count"] += 1
            result["trainable_logical_count"] += logical
            result["trainable_real_scalar_count"] += real
    if parameter:
        return {"parameter_tensor_count": result["tensor_count"],
                "logical_parameter_count": result["logical_count"],
                "real_scalar_count": result["real_scalar_count"],
                "parameter_bytes": result["tensor_bytes"],
                "trainable_parameter_tensor_count": result["trainable_tensor_count"],
                "trainable_logical_parameter_count": result["trainable_logical_count"],
                "trainable_real_scalar_count": result["trainable_real_scalar_count"]}
    return {"buffer_tensor_count": result["tensor_count"], "logical_buffer_count": result["logical_count"],
            "buffer_real_scalar_count": result["real_scalar_count"], "buffer_bytes": result["tensor_bytes"]}


def framework_parameter_counts(model, branches=None, generator=None, *, include_parameter_details=False):
    """Count a model plus optional branches/generator with global identity dedup.

    Components accept an ``nn.Module``, ``nn.Parameter``, mapping or sequence.
    Top-level counts form the deduplicated union. A component's ``inclusive``
    count is useful in isolation; ``exclusive_contribution`` assigns a shared
    parameter to its first component (model, branches, then generator), so
    exclusive contributions sum exactly to the union. Buffer numbers include
    registered nonpersistent buffers as actual resident model tensors.
    """
    components = (_components("model", model) + _components("branches", branches)
                  + _components("generator", generator))
    names = [name for name, _ in components]
    if len(names) != len(set(names)):
        raise ValueError("flattened component names must be unique")
    all_parameters, all_buffers, seen, seen_buffers, records = [], [], set(), set(), {}
    for name, component in components:
        named = _named_parameters(component)
        parameters = [parameter for _, parameter in named]
        buffers = list(component.buffers()) if isinstance(component, nn.Module) else []
        exclusive = [parameter for parameter in parameters if id(parameter) not in seen]
        exclusive_buffers = [buffer for buffer in buffers if id(buffer) not in seen_buffers]
        children = {}
        if isinstance(component, nn.Module):
            for child_name, child in component.named_children():
                children[child_name] = _tensor_counts(child.parameters(), parameter=True)
        record = {"class": type(component).__name__,
                  "inclusive": {**_tensor_counts(parameters, parameter=True), **_tensor_counts(buffers, parameter=False)},
                  "exclusive_contribution": {**_tensor_counts(exclusive, parameter=True),
                                              **_tensor_counts(exclusive_buffers, parameter=False)},
                  "direct_children_inclusive": children,
                  "shared_parameter_tensor_count": len(parameters) - len(exclusive)}
        if include_parameter_details:
            record["parameters"] = [{"name": parameter_name, "shape": list(parameter.shape),
                                      "dtype": str(parameter.dtype), "device": str(parameter.device),
                                      "requires_grad": parameter.requires_grad,
                                      **_tensor_counts([parameter], parameter=True)}
                                     for parameter_name, parameter in named]
        records[name] = record
        all_parameters.extend(parameters)
        all_buffers.extend(buffers)
        seen.update(id(parameter) for parameter in parameters)
        seen_buffers.update(id(buffer) for buffer in buffers)
    return {"schema_version": 1, **_tensor_counts(all_parameters, parameter=True),
            **_tensor_counts(all_buffers, parameter=False), "components": records,
            "deduplication": "Python Parameter/Tensor identity; distinct Parameters sharing storage count separately",
            "complex_convention": "one complex element = one logical parameter = two real scalars",
            "trainable_convention": "requires_grad at time of inspection, not stage-specific update masks",
            "excluded_costs": ["optimizer state", "gradients", "activations", "temporary adaptations", "FLOPs", "latency", "peak memory"],
            "buffer_scope": "all registered buffers, including persistent=False; unregistered Python state excluded"}


@contextmanager
def isolated_cpu_initialization(seed=0):
    """Restore Python/NumPy/CPU Torch RNG and default device after CPU counting.

    No CUDA APIs are called and no CUDA RNG is initialized. CUDA seeds cannot
    be modified by using only the CPU default generator's manual_seed.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    old_device = torch.get_default_device()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.set_default_device("cpu")
            random.seed(seed)
            np.random.seed(seed % 2 ** 32)
            torch.random.default_generator.manual_seed(seed)
            yield
        finally:
            torch.set_default_device(old_device)
            random.setstate(python_state)
            np.random.set_state(numpy_state)


__all__ = ["framework_parameter_counts", "isolated_cpu_initialization"]
