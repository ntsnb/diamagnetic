from __future__ import annotations

import csv
import os
from typing import Dict, Optional

from utils.paths import resolve_project_path


def log_dict(prefix: str, metrics: Dict[str, float]) -> None:
    msg = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    print(f"[{prefix}] {msg}")


def log_csv(path: str, row: Dict[str, float], header: Optional[Dict[str, float]] = None) -> None:
    resolved = resolve_project_path(path)
    os.makedirs(resolved.parent, exist_ok=True)
    file_exists = resolved.exists()
    with open(resolved, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
