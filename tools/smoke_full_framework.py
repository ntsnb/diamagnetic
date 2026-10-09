"""Bounded, real-data functional checks; never a performance benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_impl.official_backbones import OFFICIAL_MODEL_NAMES
from paper_impl.run import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+", default=list(OFFICIAL_MODEL_NAMES))
    args = parser.parse_args()
    registry = json.loads((ROOT / "data/lishui_online/v2/registry.json").read_text(encoding="utf-8"))
    tasks = registry.get("tasks", registry.get("datasets"))
    if tasks is None:
        raise ValueError("registry has no task list")
    city = next(item for item in tasks if item.get("entity_id") == "lishui_city")
    data = {key: city[key] for key in ("timestamp_column", "target_columns", "feature_columns", "target_input_columns",
                                      "target_mask_columns", "expected_interval_seconds")}
    data["path"] = str(ROOT / "data/lishui_online/v2" / city["path"])
    data.update(lookback=16, horizon=2, max_rows=48, train_fraction=.55, validation_fraction=.25, extra_delay=1)
    config = {
        "seed": 20261007, "device": args.device, "cpu_threads": 1, "data": data,
        "model": {"n_experts": 2, "hidden": 16, "heads": 8, "depth": 1, "latent_block": 4},
        "foil": {"n_environments": 2, "rounds": 1, "mil_epochs": 1, "mei_epochs": 1, "batch_size": 8},
        "training": {"epochs": 1, "batch_size": 4, "specialization_weight": .02},
        "generator": {"hidden_dim": 4, "epsilon": .1},
        "online": {"extra_delay": 1, "reference_windows": 2, "reference_interval": 2,
                   "history_capacity": 8, "channel_budget": 1, "max_backtracks": 3},
        "replay": {"episodes": 1, "max_windows": 2},
        "variants": ["stages_000", "stages_111"], "test_windows": 4,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for name in args.models:
        config["model"]["expert_backbone"] = name
        config["model"]["official_options"] = {"dropout": 0., "d_ff": 32,
            "patch_len": 4, "patch_stride": 2, "moving_avg": 3,
            "crossformer_seg_len": 4, "fedformer_modes": 4}
        try:
            result = run(config, args.output / name)
            records.append({"model": name, "status": "passed", "seconds": result["seconds"]})
        except Exception as error:
            records.append({"model": name, "status": "failed", "error": repr(error)})
        print(json.dumps(records[-1]), flush=True)
        (args.output / "progress.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if any(r["status"] != "passed" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
