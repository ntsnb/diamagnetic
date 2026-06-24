from __future__ import annotations

from typing import Optional


def get_progress(total: int, desc: str):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="samples")
    except Exception:
        return None
