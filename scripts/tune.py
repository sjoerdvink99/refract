from __future__ import annotations

import csv
import random
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer
import yaml

from refract.datasets.constants import BINARY_DATASETS, IMBALANCED_DATASETS, USE_ROC_AUC_STOPPING
from refract.datasets.real_world import load_real_world_dataset
from refract.models.routed_gnn import RoutedGNN
from refract.regimes.metrics import compute_regime_vectors, regime_vectors_to_tensor
from refract.regimes.taxonomy import fit_regime_gmm
from refract.training.config import load_hps, make_train_cfg
from refract.training.device import select_device
from refract.training.seed import set_seed
from refract.training.trainer import train_routed_model

warnings.filterwarnings("ignore")

app = typer.Typer(pretty_exceptions_enable=False)

_SEARCH_SPACE: dict[str, list] = {
    "hidden_dim":         [64, 128, 256, 512],
    "gate_hidden_dim":    [32, 64, 128],
    "dropout":            [0.2, 0.3, 0.5, 0.7],
    "lr":                 [0.0005, 0.001, 0.005, 0.01, 0.05],
    "weight_decay":       [0.0, 1e-4, 5e-4, 1e-3],
    "lambda_align":       [0.0, 0.005, 0.01, 0.05, 0.1],
    "lambda_smooth":      [0.0, 0.005, 0.01, 0.05],
    "lambda_expert_aux":  [0.0, 0.05, 0.1, 0.15, 0.2],
    "aux_floor":          [0.0, 0.1, 0.2, 0.3, 0.5],
    "T_gate_start":       [1.0, 2.0, 5.0],
    "T_gate_anneal":      [50, 100, 150, 200],
    "epochs":             [300, 500, 800],
    "patience":           [50, 75, 100],
    "drop_edge_p":        [0.0, 0.1, 0.2, 0.3],
    "lambda_entropy":     [0.0, 0.01, 0.05, 0.1],
    "lambda_z_loss":      [0.001, 0.005, 0.01, 0.02],
    "grad_clip":          [0.5, 1.0, 2.0],
    "warmup_epochs":      [0, 5, 10, 20],
}

_DATASET_OVERRIDES: dict[str, dict] = {
    "texas":          {"aux_floor": [0.0, 0.05, 0.1], "lambda_expert_aux": [0.0, 0.05, 0.1], "epochs": [500, 800, 1000], "lr": [0.001, 0.005, 0.01], "drop_edge_p": [0.0, 0.1, 0.2]},
    "cornell":        {"aux_floor": [0.0, 0.05, 0.1], "lambda_expert_aux": [0.0, 0.05, 0.1], "epochs": [500, 800, 1000], "lr": [0.001, 0.005, 0.01], "drop_edge_p": [0.0, 0.1, 0.2]},
    "wisconsin":      {"aux_floor": [0.0, 0.05, 0.1], "lambda_expert_aux": [0.0, 0.05, 0.1], "epochs": [500, 800, 1000], "lr": [0.001, 0.005, 0.01], "drop_edge_p": [0.0, 0.1, 0.2]},
    "minesweeper":    {"hidden_dim": [64, 128, 256], "dropout": [0.1, 0.2, 0.3, 0.5], "lambda_align": [0.0, 0.01, 0.05, 0.1], "epochs": [300, 500], "drop_edge_p": [0.0, 0.1, 0.2]},
    "chameleon":      {"hidden_dim": [128, 256, 512], "dropout": [0.3, 0.5, 0.7], "lambda_align": [0.0, 0.005, 0.01, 0.05], "drop_edge_p": [0.0, 0.1, 0.2, 0.3]},
    "squirrel":       {"hidden_dim": [128, 256, 512], "dropout": [0.3, 0.5, 0.7], "lambda_align": [0.0, 0.005, 0.01, 0.05], "drop_edge_p": [0.0, 0.1, 0.2, 0.3]},
    "actor":          {"hidden_dim": [64, 128, 256], "dropout": [0.3, 0.5, 0.7], "lambda_align": [0.0, 0.01, 0.05], "drop_edge_p": [0.0, 0.1, 0.2]},
    "cora":           {"hidden_dim": [128, 256, 512], "lr": [0.001, 0.005, 0.01], "lambda_align": [0.0, 0.005, 0.01], "drop_edge_p": [0.0, 0.1]},
    "citeseer":       {"hidden_dim": [128, 256, 512], "lr": [0.001, 0.005, 0.01], "lambda_align": [0.0, 0.005, 0.01], "drop_edge_p": [0.0, 0.1]},
    "questions":      {"hidden_dim": [64, 128], "lr": [0.001, 0.005, 0.01], "epochs": [200, 300], "dropout": [0.3, 0.5], "drop_edge_p": [0.0, 0.1]},
    "tolokers":       {"hidden_dim": [64, 128, 256], "dropout": [0.2, 0.3, 0.5], "epochs": [300, 500], "drop_edge_p": [0.0, 0.1, 0.2]},
    "roman-empire":   {"hidden_dim": [256, 512], "lr": [0.001, 0.005], "epochs": [500, 800], "dropout": [0.3, 0.5], "drop_edge_p": [0.0, 0.1]},
    "amazon-ratings": {"hidden_dim": [128, 256, 512], "lr": [0.001, 0.005, 0.01], "epochs": [500, 800], "dropout": [0.3, 0.5], "drop_edge_p": [0.0, 0.1]},
}


def _sample_config(dataset_name: str, rng: random.Random) -> dict:
    space = {k: list(v) for k, v in _SEARCH_SPACE.items()}
    for k, v in _DATASET_OVERRIDES.get(dataset_name, {}).items():
        space[k] = list(v)
    return {k: rng.choice(v) for k, v in space.items()}


def _val_score(dataset_name: str, split_idx: int, cfg: dict, root: str, device: str) -> float:
    set_seed(split_idx * 137 + 7)
    dataset = load_real_world_dataset(dataset_name, root=root, split_idx=split_idx)
    regime_vectors = compute_regime_vectors(dataset, dataset.train_mask)
    fit_regime_gmm(regime_vectors)
    regime_tensor = regime_vectors_to_tensor(regime_vectors)

    train_cfg = make_train_cfg(cfg, dataset_name, device)
    model = RoutedGNN(
        in_dim=dataset.num_features,
        hidden_dim=cfg["hidden_dim"],
        num_classes=dataset.num_classes,
        regime_dim=regime_tensor.shape[1],
        gate_hidden_dim=cfg["gate_hidden_dim"],
        dropout=cfg["dropout"],
    )
    result = train_routed_model(model, dataset, regime_tensor, train_cfg)
    return result.best_val_acc


@app.command()
def run(
    dataset: str = typer.Argument(..., help="Dataset name"),
    n_configs: int = typer.Option(30, help="Number of random configs to try"),
    n_val_splits: int = typer.Option(3, help="Validation splits to average per config"),
    root: str = typer.Option("data", help="Dataset root directory"),
    out_dir: str = typer.Option("configs/hparams", help="Output directory for YAML config"),
    log_dir: str = typer.Option("results/hparams", help="Directory for sweep logs and CSV"),
    seed: int = typer.Option(42, help="RNG seed for config sampling"),
    device: str = typer.Option("auto", help="Device: auto | cuda | mps | cpu"),
) -> None:
    dataset = dataset.lower()
    device_str = select_device(device)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[tune] {dataset} — {n_configs} configs × {n_val_splits} val splits — device={device_str}")

    rng = random.Random(seed)
    best_val = -1.0
    best_cfg: dict = {}
    all_results: list[dict] = []

    log_file = log_path / f"{dataset}_sweep.log"
    csv_file = log_path / f"{dataset}_all.csv"

    csv_fields = ["config_idx", "val_acc"] + [k for k in _SEARCH_SPACE.keys()]

    with open(log_file, "w") as lf:
        lf.write(f"[tune] {dataset} — {n_configs} configs × {n_val_splits} val splits — device={device_str}\n")

    with open(csv_file, "w", newline="") as cf:
        csv.DictWriter(cf, fieldnames=csv_fields).writeheader()

    for i in range(n_configs):
        cfg = _sample_config(dataset, rng)
        scores = []
        for split_idx in range(n_val_splits):
            try:
                scores.append(_val_score(dataset, split_idx, cfg, root, device_str))
            except Exception as e:
                typer.echo(f"  config {i+1} split {split_idx} failed: {e}", err=True)
        if not scores:
            continue

        mean_val = float(np.mean(scores))
        all_results.append({"val_acc": mean_val, "config": dict(cfg)})
        marker = " ← best" if mean_val > best_val else ""
        line = (
            f"  [{i+1:02d}/{n_configs}]  val={mean_val*100:.2f}%"
            f"  h={cfg['hidden_dim']}  lr={cfg['lr']}  floor={cfg['aux_floor']}"
            f"  aux={cfg['lambda_expert_aux']}  align={cfg['lambda_align']}{marker}"
        )
        typer.echo(line)

        with open(log_file, "a") as lf:
            lf.write(line + "\n")

        with open(csv_file, "a", newline="") as cf:
            csv.DictWriter(cf, fieldnames=csv_fields).writerow(
                {"config_idx": i + 1, "val_acc": mean_val, **cfg}
            )

        if mean_val > best_val:
            best_val = mean_val
            best_cfg = dict(cfg)

    all_results.sort(key=lambda x: x["val_acc"], reverse=True)

    typer.echo(f"\n[tune] Best val={best_val*100:.2f}%")

    yaml_doc = {
        "dataset": dataset,
        "best_val_acc": round(best_val, 6),
        "config": best_cfg,
        "metadata": {
            "n_configs": n_configs,
            "n_val_splits": n_val_splits,
            "seed": seed,
            "device": device_str,
            "tuned_at": datetime.now(timezone.utc).isoformat(),
            "top5": all_results[:5],
        },
    }
    yaml_out = out_path / f"{dataset}.yaml"
    with open(yaml_out, "w") as f:
        yaml.dump(yaml_doc, f, default_flow_style=False, sort_keys=False)

    typer.echo(f"[tune] Config  → {yaml_out}")
    typer.echo(f"[tune] Log     → {log_file}")
    typer.echo(f"[tune] CSV     → {csv_file}")


if __name__ == "__main__":
    app()
