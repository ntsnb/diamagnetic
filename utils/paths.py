from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROJECT_PATH_ANCHORS = (
    "dataset",
    "data",
    "configs",
    "logs",
    "ckpt",
    "checkpoints",
    "docs",
    "models",
    "train",
    "tests",
)


def ensure_dir(path: str) -> None:
    os.makedirs(resolve_project_path(path), exist_ok=True)


def init_log_dirs() -> None:
    ensure_dir("logs")
    ensure_dir("logs/tensorboard")
    ensure_dir("checkpoints")


def project_root() -> Path:
    return PROJECT_ROOT


def _legacy_project_suffix(path_text: str, anchors: Iterable[str]) -> Path | None:
    normalized = path_text.strip().strip("\"'").replace("\\", "/")
    lowered = normalized.lower()
    for anchor in anchors:
        token = f"/{anchor.lower()}/"
        idx = lowered.find(token)
        if idx >= 0:
            return Path(normalized[idx + 1 :])
        if lowered.startswith(f"{anchor.lower()}/"):
            return Path(normalized)
    return None


def resolve_project_path(
    path: str | os.PathLike[str],
    *,
    must_exist: bool = False,
    anchors: Iterable[str] = PROJECT_PATH_ANCHORS,
) -> Path:
    """Resolve repo-relative paths and salvage old machine-specific paths.

    Several historical configs contain absolute paths from a previous Windows
    checkout, including mojibake in the Chinese directory name. When such a path
    no longer exists, this helper keeps the evidence path suffix under known
    project anchors such as ``dataset`` or ``configs`` and resolves it in the
    current checkout.
    """

    raw = os.fspath(path)
    expanded = os.path.expandvars(os.path.expanduser(raw.strip().strip("\"'")))
    candidate = Path(expanded)

    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        suffix = _legacy_project_suffix(expanded, anchors)
        if suffix is not None:
            rescued = PROJECT_ROOT / suffix
            if rescued.exists() or not must_exist:
                return rescued
        if must_exist:
            raise FileNotFoundError(f"path does not exist: {raw}")
        return candidate

    resolved = PROJECT_ROOT / candidate
    if resolved.exists() or not must_exist:
        return resolved

    suffix = _legacy_project_suffix(expanded, anchors)
    if suffix is not None:
        rescued = PROJECT_ROOT / suffix
        if rescued.exists() or not must_exist:
            return rescued

    if must_exist:
        raise FileNotFoundError(f"path does not exist: {raw}")
    return resolved
