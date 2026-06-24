from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import resolve_project_path  # noqa: E402


LEGACY_ABSOLUTE_MARKERS = ("C:/Users", "C:\\Users")


def _load_raw_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def audit_configs(config_root: Path) -> tuple[list[str], list[str]]:
    legacy_path_hits: list[str] = []
    unresolved_data_paths: list[str] = []

    for path in sorted(config_root.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in LEGACY_ABSOLUTE_MARKERS):
            legacy_path_hits.append(str(path.relative_to(ROOT)))

        data = _load_raw_yaml(path)
        if not isinstance(data, dict):
            continue
        data_cfg = data.get("data")
        if not isinstance(data_cfg, dict) or not data_cfg.get("path"):
            continue

        try:
            resolve_project_path(data_cfg["path"], must_exist=True)
        except FileNotFoundError:
            unresolved_data_paths.append(str(path.relative_to(ROOT)))

    return legacy_path_hits, unresolved_data_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit repo config paths without modifying files.")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    config_root = resolve_project_path(args.config_root, must_exist=True)
    legacy_path_hits, unresolved_data_paths = audit_configs(config_root)

    print(f"checked_config_root={config_root}")
    print(f"legacy_absolute_path_hits={len(legacy_path_hits)}")
    for item in legacy_path_hits:
        print(f"  legacy: {item}")
    print(f"unresolved_data_paths={len(unresolved_data_paths)}")
    for item in unresolved_data_paths:
        print(f"  missing: {item}")

    if args.strict and (legacy_path_hits or unresolved_data_paths):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
