"""主预测器离线训练、验证选模与完整checkpoint保存。"""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import random
import numpy as np
import torch
from torch.nn import functional as F
from .discrepancy import specialization_loss, pairwise_discrepancies


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 8
    attribution_batch_size: int = 1
    learning_rate: float = .001
    vae_weight: float = .01
    kl_weight: float = .05
    specialization_weight: float = .05
    margin: float = .3
    grad_clip: float = 5.
    seed: int = 42

    def __post_init__(self):
        if (isinstance(self.attribution_batch_size, bool)
                or not isinstance(self.attribution_batch_size, int)
                or self.attribution_batch_size < 1):
            raise ValueError("attribution_batch_size must be a positive integer")
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("invalid training configuration")
        if min(self.vae_weight, self.kl_weight, self.specialization_weight) < 0 or not 0 < self.margin < 1:
            raise ValueError("invalid loss weights or margin")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_offline(model, x, y, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    square, absolute, count = 0., 0., 0
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start + batch_size].to(device), y[start:start + batch_size].to(device)
        error = model(xb, stochastic=False).prediction - yb
        if not torch.isfinite(error).all():
            raise FloatingPointError("nonfinite offline validation prediction or target")
        square += error.square().sum().item()
        absolute += error.abs().sum().item()
        count += error.numel()
    if count == 0:
        raise ValueError("evaluation set is empty")
    if not np.isfinite(square) or not np.isfinite(absolute):
        raise FloatingPointError("nonfinite offline validation metric")
    return {"mse": square / count, "mae": absolute / count, "n_values": count}


def train_forecaster(model, train_x, train_y, validation_x, validation_y, config: TrainingConfig):
    """测试集不进入此接口；最佳参数以验证MSE选取。"""
    if len(train_x) == 0 or len(validation_x) == 0:
        raise ValueError("training and validation sets must be nonempty")
    seed_everything(config.seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best, best_state, history = float("inf"), None, []
    generator = torch.Generator().manual_seed(config.seed)
    for epoch in range(config.epochs):
        model.train()
        totals = dict(prediction=0., reconstruction=0., kl=0., specialization=0., loss=0.)
        processed = 0
        for ids in torch.randperm(len(train_x), generator=generator).split(config.batch_size):
            x, y = train_x[ids].to(device), train_y[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x, stochastic=True)
            prediction = F.mse_loss(output.prediction, y)
            reconstruction = F.mse_loss(output.reconstruction, x) if output.reconstruction is not None else prediction.new_zeros(())
            kl = (-.5 * (1 + output.log_var - output.mu.square() - output.log_var.exp()).sum(-1).mean()
                  if output.mu is not None else prediction.new_zeros(()))
            supervised = prediction + config.vae_weight * (reconstruction + config.kl_weight * kl)
            if not torch.isfinite(supervised):
                raise FloatingPointError("nonfinite offline training loss")
            # 固定专家对的逐样本均值可分块求导；先累积全部梯度，再统一裁剪和更新。
            spec = prediction.new_zeros(())
            if config.specialization_weight and model.config.n_experts > 1:
                for chunk in x.split(config.attribution_batch_size):
                    attribution = model.attribute(chunk, freeze_head=True)
                    chunk_spec = specialization_loss(attribution.maps, config.margin)
                    if not torch.isfinite(chunk_spec):
                        raise FloatingPointError("nonfinite offline specialization loss")
                    fraction = len(chunk) / len(x)
                    (config.specialization_weight * fraction * chunk_spec).backward()
                    spec = spec + fraction * chunk_spec.detach()
                    del attribution, chunk_spec
            supervised.backward()
            loss = supervised.detach() + config.specialization_weight * spec
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            for key, value in zip(totals, (prediction, reconstruction, kl, spec, loss)):
                totals[key] += float(value.detach()) * len(ids)
            processed += len(ids)
        validation = evaluate_offline(model, validation_x, validation_y, config.batch_size)
        row = {"epoch": epoch + 1, **{k: v / processed for k, v in totals.items()}, "validation": validation}
        history.append(row)
        if validation["mse"] < best:
            best = validation["mse"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        sample = validation_x[:min(config.batch_size, len(validation_x))].to(device)
        attr = model.attribute(sample)
        pairs = pairwise_discrepancies(attr.maps) if model.config.n_experts > 1 else None
        valid = pairs.distances[pairs.valid] if pairs is not None else attr.maps.new_empty(0)
        diagnostic = {"zero_map_fraction": float((attr.maps.sum((-1, -2)) == 0).float().mean()),
                      "valid_pair_mean_distance": float(valid.mean()) if valid.numel() else None,
                      "max_signed_conservation_error": float(attr.conservation_error.max()),
                      "expert_pairs": model.config.n_experts * (model.config.n_experts - 1) // 2,
                      "vae_enabled": model.config.use_vae}
    return {"history": history, "best_validation_mse": best, "diagnostic": diagnostic, "config": asdict(config)}


def save_checkpoint(path, model, branches, generator, configuration, data_metadata, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "model_config": asdict(model.config), "model": model.state_dict(),
                "branches": branches.state_dict(), "generator": generator.state_dict(),
                "configuration": configuration, "data_metadata": data_metadata, "history": history}, path)
