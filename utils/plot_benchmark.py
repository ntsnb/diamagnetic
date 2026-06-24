from __future__ import annotations

import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_cross_dataset_benchmark(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="dataset", columns="variant", values="adapted_online_mse")
    variants = [c for c in ["baseline", "stage1", "stage2", "stage3", "stage1_stage2_stage3"] if c in pivot.columns]
    ax = pivot[variants].plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Adapted Online MSE")
    ax.set_title("Online Adaptation Comparison Across Datasets")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_overfit_gap(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    base = df[df["variant"] == "baseline"][["dataset", "offline_train_mse", "offline_test_mse", "overfit_gap"]]
    ax = base.set_index("dataset")[["offline_train_mse", "offline_test_mse"]].plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("MSE")
    ax.set_title("Offline Train/Test Gap by Dataset (Baseline)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
