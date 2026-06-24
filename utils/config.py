from __future__ import annotations

import yaml

from utils.paths import resolve_project_path


def normalize_config_paths(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        return cfg

    data_cfg = cfg.get("data")
    if isinstance(data_cfg, dict) and data_cfg.get("path"):
        data_cfg["path"] = str(resolve_project_path(data_cfg["path"]))
    return cfg


def load_yaml(path: str):
    resolved = resolve_project_path(path, must_exist=True)
    with open(resolved, "r", encoding="utf-8") as f:
        return normalize_config_paths(yaml.safe_load(f))


def save_yaml(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
