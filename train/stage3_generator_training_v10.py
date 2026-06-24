from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.adaptation_v10 import ResidualBasisHyperGeneratorV10
from train.stage3_generator_training import (
    Stage3GeneratorLossConfig,
    load_stage3_generator_checkpoint as _load_stage3_generator_checkpoint_v2,
    save_stage3_generator_checkpoint,
    stage3_generator_train_step,
)


def load_stage3_generator_checkpoint(path: str | Path, *, model: nn.Module):
    """Load Stage III generator checkpoints, including the V10 generator class."""

    checkpoint = torch.load(Path(path), map_location="cpu")
    class_name = checkpoint.get("generator_class") or checkpoint.get("config", {}).get("class")
    if class_name != "ResidualBasisHyperGeneratorV10":
        return _load_stage3_generator_checkpoint_v2(path, model=model)

    config = checkpoint.get("generator_config") or checkpoint["config"]
    generator = ResidualBasisHyperGeneratorV10(
        named_param_shapes={key: tuple(value) for key, value in config["named_param_shapes"].items()},
        drift_dim=int(config["drift_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        rank=int(config["rank"]),
        output_shape=tuple(config["output_shape"]),
        max_scale=float(config["max_scale"]),
        residual_scale=float(config.get("residual_scale", 1.0)),
    )
    generator.load_state_dict(checkpoint.get("generator_state_dict") or checkpoint["state_dict"])
    return generator, dict(checkpoint.get("metadata", {}))
