from __future__ import annotations

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None

from utils.paths import resolve_project_path


def get_tb_writer(log_dir: str):
    if SummaryWriter is None:
        return None
    return SummaryWriter(log_dir=str(resolve_project_path(log_dir)))
