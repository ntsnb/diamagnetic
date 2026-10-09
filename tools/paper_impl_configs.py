"""生成真实负荷数据的有界验证配置；不自动运行或覆盖已有输出。"""
import argparse
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def generate(dataset_root: Path, output: Path, spain_source: Path | None = None):
    base = json.loads((ROOT / "configs/paper_impl/panama_smoke.json").read_text(encoding="utf8"))
    base["variants"] = ["stages_000", "stages_111"]
    base["training"]["epochs"] = 1
    base["foil"].update(rounds=1, mil_epochs=1, mei_epochs=1)
    base["replay"]["max_windows"] = 12
    base["test_windows"] = 12
    specifications = {
        "panama": ("panama.csv", "datetime", ["nat_demand"]),
        "tetouan": ("tetouan.csv", "DateTime", ["Zone 1 Power Consumption", "Zone 2  Power Consumption", "Zone 3  Power Consumption"]),
        "spain": ("spain_energy_weather.csv", "time", ["actual_load"]),
    }
    for path in sorted((dataset_root / "lishui").glob("lishui_*.csv")):
        if path.name == "lishui_summary.csv":
            continue
        specifications[path.stem] = (path.relative_to(dataset_root).as_posix(), "timestamp", ["load"])
    configs = {}
    for name, (file, timestamp, targets) in specifications.items():
        config = copy.deepcopy(base)
        config["data"].update(path=(dataset_root / file).as_posix(), timestamp_column=timestamp,
                              target_columns=targets, feature_columns=None,
                              expected_interval_seconds=600 if name == "tetouan" else 3600)
        if name == "spain" and spain_source is not None:
            config["data"]["path"] = spain_source.resolve().as_posix()
        configs[name] = config
    for encoder in ("inverted_transformer", "patch_transformer", "transformer_style", "itransformer_style",
                    "patchtst_style", "informer_style", "autoformer_style", "fedformer_style",
                    "pyraformer_style", "crossformer_style"):
        config = copy.deepcopy(base)
        config["data"]["path"] = (dataset_root / "panama.csv").as_posix()
        config["model"]["encoder"] = encoder
        configs["panama_" + encoder] = config
    # 相同的主体和训练预算，比较纯监督梯度定位与LRP+误差定位。
    config = copy.deepcopy(base)
    config["data"]["path"] = (dataset_root / "panama.csv").as_posix()
    config["online"]["selector"] = "gradient_only"
    configs["panama_gradient_only"] = config
    config = copy.deepcopy(base)
    config["data"]["path"] = (dataset_root / "panama.csv").as_posix()
    config["generator"]["share_matching_shapes"] = True
    configs["panama_shared_generator"] = config
    output.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        path = output / (name + ".json")
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf8")
    return list(configs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/paper_impl/validation_suite")
    parser.add_argument("--spain-source", type=Path, help="explicit audited clock-corrected Spain copy")
    args = parser.parse_args()
    names = generate(args.dataset_root.resolve(), args.output.resolve(), args.spain_source)
    print(json.dumps({"count": len(names), "configurations": names,
                      "scope": "bounded implementation checks, not performance comparison"}, ensure_ascii=False))
