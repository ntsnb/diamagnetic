from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.loaders import build_dataloaders_from_series
from data.normalization import StandardScaler
from data.readers import load_tabular_timeseries
from models.factor_mechanism import FactorMechanismConfig, build_factor_mechanism
from models.invariant_learning import EnvInferenceConfig, MTIL, MTILConfig, MTEI
from models.simple_encoders import FlattenMLPEncoder, LinearRegressor
from train.offline_factor import evaluate_factor, train_offline_factor
from train.offline_invariant import train_offline_invariant
from train.online_loop_stage3_delta import run_online_adaptation
from utils.config import load_yaml
from utils.logging import log_csv
from utils.paths import init_log_dirs
from utils.seed import set_seed
from utils.tb_logger import get_tb_writer


def build_offline_scheduler(cfg: dict, opt: torch.optim.Optimizer):
    sched_cfg = cfg["train"].get("lr_schedule", {"type": "none"})
    sched_type = sched_cfg["type"]
    if sched_type == "none":
        return None
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg["train"]["epochs"], eta_min=sched_cfg["min_lr"]
        )
    if sched_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            opt, step_size=sched_cfg["step_size"], gamma=sched_cfg["gamma"]
        )
    if sched_type == "warmup_cosine":
        warmup_epochs = max(1, sched_cfg["warmup_epochs"])

        def lr_lambda(epoch: int):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, cfg["train"]["epochs"] - warmup_epochs)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    raise ValueError(f"unknown lr_schedule.type: {sched_type}")


def maybe_run_invariant(cfg: dict, series: torch.Tensor, train_dl, device: torch.device, variant_name: str):
    if not cfg["invariant"]["enable"]:
        print(f"[info] invariant learning disabled for {variant_name}")
        return None

    in_dim = cfg["data"]["lookback"] * series.shape[1]
    repr_dim = cfg["invariant"]["repr_dim"]
    encoder_i = FlattenMLPEncoder(in_dim, repr_dim).to(device)
    encoder_v = FlattenMLPEncoder(in_dim, repr_dim).to(device)
    reg = LinearRegressor(repr_dim, cfg["data"]["horizon"]).to(device)
    reg_v = LinearRegressor(repr_dim, cfg["data"]["horizon"]).to(device)

    mtil = MTIL(encoder_i, reg).to(device)
    mtei = MTEI(
        in_dim=repr_dim,
        out_dim=cfg["data"]["horizon"],
        cfg=EnvInferenceConfig(
            num_envs=cfg["invariant"]["num_envs"],
            smooth_radius=cfg["invariant"]["smooth_radius"],
        ),
    ).to(device)

    opt_mtil = torch.optim.Adam(mtil.parameters(), lr=cfg["train"]["lr"])
    opt_mtei = torch.optim.Adam(mtei.parameters(), lr=cfg["train"]["lr"])
    for _ in range(cfg["invariant"]["epochs"]):
        train_offline_invariant(
            dataloader=train_dl,
            mtil=mtil,
            mtei=mtei,
            cfg=MTILConfig(
                lambda_erm=cfg["invariant"]["lambda_erm"],
                lambda_var=cfg["invariant"]["lambda_var"],
            ),
            opt_mtil=opt_mtil,
            opt_mtei=opt_mtei,
            device=device,
        )
    # Train a light variant encoder on residual targets so Stage III can use phi_V - phi_I
    encoder_v.train()
    reg_v.train()
    opt_var = torch.optim.Adam(
        list(encoder_v.parameters()) + list(reg_v.parameters()),
        lr=cfg["train"]["lr"] * 0.5,
    )
    mtil.eval()
    residual_epochs = max(1, cfg["invariant"]["epochs"])
    for _ in range(residual_epochs):
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                _, y_inv = mtil(x)
                residual = y - y_inv
            phi_v = encoder_v(x)
            y_var = reg_v(phi_v)
            loss = torch.mean((y_var - residual) ** 2)
            opt_var.zero_grad(set_to_none=True)
            loss.backward()
            opt_var.step()
    encoder_i.eval()
    encoder_v.eval()
    return {
        "encoder_i": encoder_i,
        "encoder_v": encoder_v,
    }


def run_single_experiment(cfg: dict, save_ckpt: bool = False, variant_name: str = "single", return_artifacts: bool = False) -> dict:
    set_seed(cfg["seed"])
    init_log_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg["data"]["source"] == "synthetic":
        series = torch.randn(500, 8)
    else:
        feature_cols = cfg["data"]["feature_cols"] or None
        encoding = cfg["data"].get("csv_encoding") or None
        series, df = load_tabular_timeseries(
            path=cfg["data"]["path"],
            timestamp_col=cfg["data"]["timestamp_col"],
            feature_cols=feature_cols,
            encoding=encoding,
            missing_sentinels=cfg["data"].get("missing_sentinels"),
            impute_method=cfg["data"].get("impute_method", "drop"),
        )
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != cfg["data"]["timestamp_col"]]

    target_dim = cfg["data"].get("target_dim")
    target_col = cfg["data"].get("target_col")
    if target_col:
        if feature_cols is None:
            raise ValueError("target_col requires feature_cols or file-based inferred columns")
        if target_col not in feature_cols:
            raise ValueError(f"target_col '{target_col}' not found in feature columns")
        target_dim = feature_cols.index(target_col)
    if target_dim is None:
        raise ValueError("data.target_dim or data.target_col must be provided")

    if cfg["data"].get("normalize", True):
        split_idx = int(series.shape[0] * cfg["data"]["train_ratio"])
        scaler = StandardScaler.fit(series[:split_idx])
        series = scaler.transform(series)

    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds, split_info = build_dataloaders_from_series(
        series=series,
        lookback=cfg["data"]["lookback"],
        horizon=cfg["data"]["horizon"],
        target_dim=target_dim,
        batch_size=cfg["train"]["batch_size"],
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        split_mode=cfg["data"].get("split_mode", "chronological"),
        test_ratio=cfg["data"].get("online_test_ratio"),
        shift_pool_ratio=cfg["data"].get("shift_pool_ratio"),
        drop_last=True,
        return_datasets=True,
        return_split_info=True,
    )
    train_eval_dl = DataLoader(
        train_dl.dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=False
    )

    invariant_bundle = maybe_run_invariant(cfg, series, train_dl, device, variant_name)

    mcfg = FactorMechanismConfig(
        input_dim=series.shape[1],
        seq_len=cfg["data"]["lookback"],
        out_dim=cfg["data"]["horizon"],
        d_model=cfg["model"]["d_model"],
        n_experts=cfg["model"]["n_experts"],
        latent_dim=cfg["model"]["latent_dim"],
        temporal_encoder=cfg["model"]["temporal_encoder"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        use_revin=cfg["model"].get("use_revin", False),
        target_index=target_dim,
        use_vae=cfg["model"].get("use_vae", True),
    )
    model = build_factor_mechanism(mcfg).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    sched = build_offline_scheduler(cfg, opt)

    tb = get_tb_writer("logs/tensorboard")
    best_val = float("inf")
    for epoch in range(cfg["train"]["epochs"]):
        train_offline_factor(
            dataloader=train_dl,
            model=model,
            opt=opt,
            beta_vae=cfg["train"]["beta_vae"],
            lambda_disc=cfg["train"]["lambda_disc"],
            device=device,
            use_lrp=cfg["train"]["use_lrp"],
            lrp_method=cfg["train"]["lrp_method"],
            save_lrp=cfg["train"]["save_lrp"],
            pred_loss_type=cfg["train"].get("pred_loss_type", "mse"),
            pred_mae_weight=cfg["train"].get("pred_mae_weight", 0.0),
            pred_huber_delta=cfg["train"].get("pred_huber_delta", 1.0),
        )
        val_mse, val_mae = evaluate_factor(val_dl, model, device)
        print(f"[{variant_name}] [val] mse={val_mse:.4f}, mae={val_mae:.4f}")
        log_csv(
            "logs/offline_metrics.csv",
            {
                "variant": variant_name,
                "epoch": epoch,
                "val_mse": val_mse,
                "val_mae": val_mae,
            },
        )
        if tb is not None:
            tb.add_scalar(f"{variant_name}/val_mse", val_mse, epoch)
            tb.add_scalar(f"{variant_name}/val_mae", val_mae, epoch)
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), f"checkpoints/{variant_name}_best.pt")
        if sched is not None:
            sched.step()

    train_eval_mse, train_eval_mae = evaluate_factor(train_eval_dl, model, device)
    test_mse, test_mae = evaluate_factor(test_dl, model, device)
    print(f"[{variant_name}] [test] mse={test_mse:.4f}, mae={test_mae:.4f}")
    log_csv(
        "logs/test_metrics.csv",
        {
            "variant": variant_name,
            "train_mse": train_eval_mse,
            "train_mae": train_eval_mae,
            "test_mse": test_mse,
            "test_mae": test_mae,
        },
    )
    if tb is not None:
        tb.add_scalar(f"{variant_name}/train_mse", train_eval_mse, 0)
        tb.add_scalar(f"{variant_name}/train_mae", train_eval_mae, 0)
        tb.add_scalar(f"{variant_name}/test_mse", test_mse, 0)
        tb.add_scalar(f"{variant_name}/test_mae", test_mae, 0)

    if save_ckpt:
        torch.save(model.state_dict(), f"checkpoints/{variant_name}.pt")

    online_summary = {
        "online_base_mse": float("nan"),
        "online_stage2_mse": float("nan"),
        "online_stage3_mse": float("nan"),
    }
    if cfg["online_run"]["enable"]:
        online_dl = DataLoader(
            test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=True
        )
        def online_stream():
            for x, y in online_dl:
                x = x.to(device)
                y = y.to(device)
                if invariant_bundle is None:
                    yield x, y
                    continue
                with torch.no_grad():
                    phi_i = invariant_bundle["encoder_i"](x)
                    phi_v = invariant_bundle["encoder_v"](x)
                yield x, y, phi_i, phi_v

        stream = online_stream()
        online_summary = run_online_adaptation(stream, model, None, cfg, device)

    result = {
        "offline_train_mse": train_eval_mse,
        "offline_train_mae": train_eval_mae,
        "offline_val_mse": best_val,
        "offline_test_mse": test_mse,
        "offline_test_mae": test_mae,
        "split_info": split_info,
        **online_summary,
    }
    if return_artifacts:
        result.update({
            "model": model,
            "train_ds": train_ds,
            "val_ds": val_ds,
            "test_ds": test_ds,
            "invariant_bundle": invariant_bundle,
            "device": device,
            "target_dim": target_dim,
            "series_dim": int(series.shape[1]),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--no_stage2", action="store_true")
    parser.add_argument("--no_stage3", action="store_true")
    parser.add_argument("--no_lrp", action="store_true")
    parser.add_argument("--no_invariant", action="store_true")
    parser.add_argument("--save_ckpt", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.no_stage2:
        cfg["online"]["stage2"]["enable"] = False
    if args.no_stage3:
        cfg["online"]["stage3"]["enable"] = False
    if args.no_lrp:
        cfg["train"]["use_lrp"] = False
    if args.no_invariant:
        cfg["invariant"]["enable"] = False

    run_single_experiment(cfg, save_ckpt=args.save_ckpt, variant_name="single")


if __name__ == "__main__":
    main()
