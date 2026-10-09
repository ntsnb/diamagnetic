"""训练前缀的延迟反馈回放；generator和持久更新有明确先后顺序。"""
from dataclasses import asdict, dataclass
import copy
import torch
from torch.nn import functional as F
from torch.func import functional_call
from .adaptation import OnlineController, OnlineConfig
from .stream import DelayedFeedbackStream


@dataclass
class ReplayConfig:
    episodes: int = 2
    learning_rate: float = .001
    max_windows: int | None = None
    grad_clip: float = 5.

    def __post_init__(self):
        if self.episodes < 1 or self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("invalid replay configuration")
        if self.max_windows is not None and self.max_windows < 1:
            raise ValueError("max_windows must be positive")


def train_generator(model, branches, generator, values, targets, train_end,
                    online_config: OnlineConfig, config: ReplayConfig):
    """只读取训练前缀。每episode恢复持久模型与所有控制器状态。"""
    if not online_config.stage_iii:
        raise ValueError("generator training requires Stage III enabled")
    if generator.mode != "centered_bounded":
        raise ValueError("identity ablation has no trainable modulation")
    if train_end > len(values) or train_end > len(targets):
        raise ValueError("training prefix exceeds observed data")
    branches.freeze()
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(generator.parameters(), lr=config.learning_rate)
    episode_reports = []
    first_issue = model.config.lookback - 1
    last_issue = train_end - model.config.horizon - online_config.extra_delay - 1
    if config.max_windows is not None:
        last_issue = min(last_issue, first_issue + config.max_windows - 1)
    if last_issue < first_issue:
        raise ValueError("training prefix too short for delayed replay")
    for episode in range(config.episodes):
        working = copy.deepcopy(model)
        controller = OnlineController(working, online_config, generator=generator, contrast_fn=branches)
        losses, gradients, contrasts, update_times = [], [], [], []

        def update_generator(batch, time):
            # detached持久值不接受generator目标的梯度，gain路径保持可微。
            state = {name: p.detach() for name, p in working.named_parameters()}
            state.update({name: b.detach() for name, b in working.named_buffers()})
            with torch.no_grad():
                contrast = branches(batch.inputs)
            generator.train()
            optimizer.zero_grad(set_to_none=True)
            gains = generator(contrast)
            prediction = functional_call(working, state, (batch.inputs,),
                                         {"stochastic": False, "modulations": gains}).prediction
            loss = F.mse_loss(prediction, batch.targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite generator loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            generator.eval()
            losses.append(float(loss.detach()))
            gradients.append(float(norm))
            contrasts.append(float(contrast.square().mean().sqrt()))
            update_times.append({"time": time, "issue_times": list(batch.issue_times)})

        stream = DelayedFeedbackStream(controller, before_update=update_generator, allow_missing=True)
        last_release = last_issue + model.config.horizon + online_config.extra_delay
        for time in range(first_issue, last_release + 1):
            x = values[time - model.config.lookback + 1:time + 1].unsqueeze(0).to(device)
            if time <= last_issue:
                stream.step(time, x, targets[time])
            else:
                stream.observe(time, targets[time], current_x=x)
        if not losses:
            raise RuntimeError("no delayed labels released during generator replay")
        episode_reports.append({"episode": episode + 1, "updates": len(losses),
                                "skipped_incomplete_label_updates": len(stream.skipped_forecasts),
                                "mean_loss": sum(losses) / len(losses), "max_gradient_norm": max(gradients),
                                "mean_contrast_rms": sum(contrasts) / len(contrasts),
                                "zero_contrast_fraction": sum(v == 0 for v in contrasts) / len(contrasts),
                                "events": update_times})
    generator.eval()
    return {"config": asdict(config), "episodes": episode_reports,
            "first_issue": first_issue, "last_issue": last_issue, "last_release": last_release,
            "training_end_exclusive": train_end}
