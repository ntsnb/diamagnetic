from __future__ import annotations

from pathlib import Path

from utils.config import load_yaml
from utils.paths import project_root, resolve_project_path


def test_resolve_relative_project_path() -> None:
    resolved = resolve_project_path("dataset/weather.csv", must_exist=True)

    assert resolved == project_root() / "dataset" / "weather.csv"
    assert resolved.exists()


def test_resolve_legacy_absolute_project_suffix() -> None:
    legacy = (
        r"C:\Users\admin\Desktop\old-checkout"
        r"\IEEE-Transactions-LaTeX2e-templates-and-instructions\dataset\weather.csv"
    )

    resolved = resolve_project_path(legacy, must_exist=True)

    assert resolved == project_root() / "dataset" / "weather.csv"


def test_load_yaml_normalizes_data_path() -> None:
    cfg = load_yaml("configs/base.yaml")
    data_path = Path(cfg["data"]["path"])

    assert data_path.exists()
    assert data_path == project_root() / "dataset" / "weather.csv"
