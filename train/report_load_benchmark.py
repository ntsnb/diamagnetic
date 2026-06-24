from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


DATASET_SPECS = [
    ("panama", "dataset/load_benchmarks/panama.csv", "datetime", "nat_demand"),
    ("tetouan", "dataset/load_benchmarks/tetouan.csv", "DateTime", "Zone 1 Power Consumption"),
    ("spain_energy_weather", "dataset/load_benchmarks/spain_energy_weather.csv", "time", "actual_load"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="logs/load_benchmarks/results.csv")
    parser.add_argument("--outdir", default="logs/load_benchmarks")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.results)

    latest = (
        df.sort_values("timestamp")
        .groupby(["dataset", "variant"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    pivot = latest.pivot(index="dataset", columns="variant", values="adapted_online_mse")
    gain = latest.pivot(index="dataset", columns="variant", values="improve_vs_base")

    pivot.to_csv(os.path.join(args.outdir, "pivot_adapted_online_mse.csv"))
    gain.to_csv(os.path.join(args.outdir, "pivot_gain_vs_base.csv"))

    dataset_rows = []
    for dataset, path, ts_col, target_col in DATASET_SPECS:
        df_ds = pd.read_csv(path)
        dataset_rows.append(
            {
                "dataset": dataset,
                "rows": len(df_ds),
                "num_variables": len(df_ds.columns) - 1,
                "timestamp_col": ts_col,
                "target_col": target_col,
                "missing_values": int(df_ds.isna().sum().sum()),
            }
        )
    dataset_df = pd.DataFrame(dataset_rows)
    dataset_df.to_csv(os.path.join(args.outdir, "dataset_summary.csv"), index=False)

    variants = [c for c in ["baseline", "stage1", "stage2", "stage3", "stage1_stage2_stage3"] if c in pivot.columns]
    ax = pivot[variants].plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Online MSE")
    ax.set_title("Online Performance Across Load Forecasting Datasets")
    ax.legend(title="Variant")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "online_mse_compare.png"), dpi=200)
    plt.close()

    ax = gain[[c for c in variants if c != "baseline"]].plot(kind="bar", figsize=(10, 5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Improvement vs Baseline (MSE Reduction)")
    ax.set_title("Online Gain over Frozen Baseline")
    ax.legend(title="Variant")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "gain_vs_baseline.png"), dpi=200)
    plt.close()

    best_rows = latest.loc[latest.groupby("dataset")["adapted_online_mse"].idxmin()]
    with open(os.path.join(args.outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("# Load Benchmark Report\n\n")
        f.write("## Datasets\n")
        for dataset in latest["dataset"].drop_duplicates():
            subset = latest[latest["dataset"] == dataset].sort_values("adapted_online_mse")
            f.write(f"- `{dataset}`\n")
            for _, row in subset.iterrows():
                f.write(
                    f"  - `{row['variant']}`: online={row['adapted_online_mse']:.6f}, "
                    f"base={row['online_base_mse']:.6f}, gain={row['improve_vs_base']:.6f}, "
                    f"test={row['offline_test_mse']:.6f}, gap={row['overfit_gap']:.6f}\n"
                )
        f.write("\n## Best Variant Per Dataset\n")
        for _, row in best_rows.iterrows():
            f.write(
                f"- `{row['dataset']}`: best=`{row['variant']}`, "
                f"online={row['adapted_online_mse']:.6f}, "
                f"gain_vs_base={row['improve_vs_base']:.6f}\n"
            )

    with open(os.path.join(args.outdir, "detailed_report.md"), "w", encoding="utf-8") as f:
        f.write("# Detailed Load Benchmark Report\n\n")
        f.write("## Experimental Setup\n")
        f.write("- Config: `configs/best_online.yaml`\n")
        f.write("- Offline epochs: 2\n")
        f.write("- Invariant epochs: 1\n")
        f.write("- Variants: `baseline`, `stage1`, `stage2`, `stage3`, `stage1_stage2_stage3`\n")
        f.write("- Split protocol: chronological `55/10/35`; the test split is used as the online stream.\n")
        f.write("- Lookback / horizon: `24 / 12`\n\n")
        f.write("## Dataset Summary\n")
        for _, row in dataset_df.iterrows():
            f.write(
                f"- `{row['dataset']}`: rows={int(row['rows'])}, variables={int(row['num_variables'])}, "
                f"target=`{row['target_col']}`, missing={int(row['missing_values'])}\n"
            )
        f.write("\n## Per-Dataset Results\n")
        for dataset in dataset_df["dataset"]:
            sub = latest[latest["dataset"] == dataset].sort_values("adapted_online_mse")
            f.write(f"### {dataset}\n")
            base = float(sub[sub["variant"] == "baseline"]["adapted_online_mse"].iloc[0])
            for _, row in sub.iterrows():
                f.write(
                    f"- `{row['variant']}`: offline_test_mse={float(row['offline_test_mse']):.6f}, "
                    f"online_mse={float(row['adapted_online_mse']):.6f}, "
                    f"gain_vs_baseline={base - float(row['adapted_online_mse']):.6f}, "
                    f"overfit_gap={float(row['overfit_gap']):.6f}\n"
                )
            best = sub.iloc[0]
            f.write(
                f"- Best variant: `{best['variant']}` with online MSE "
                f"{float(best['adapted_online_mse']):.6f}.\n\n"
            )
        f.write("## Main Findings\n")
        f.write("- `Stage III` is the strongest and most consistent online component across all three newly downloaded load datasets.\n")
        f.write("- `Stage I` provides small positive gains on Panama and Tetouan, but remains weak on the Spain load-weather dataset.\n")
        f.write("- `Stage II` is still dataset-sensitive: it helps on Panama and Spain, but degrades on Tetouan in the current configuration.\n")
        f.write("- The full `Stage I + Stage II + Stage III` combination is not the best variant on these three load datasets; `Stage III` alone is currently strongest.\n")
        f.write("- Tetouan shows the largest offline train-test gap, which indicates the strongest overfitting risk among the three new datasets.\n")


if __name__ == "__main__":
    main()
