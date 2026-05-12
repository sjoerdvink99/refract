from __future__ import annotations

import csv
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
import yaml
from sklearn.metrics import roc_auc_score

from refract.datasets.constants import BINARY_DATASETS, NUM_SPLITS
from refract.datasets.real_world import load_real_world_dataset
from refract.models.baselines import APPNPBaseline, GATBaseline, GCNBaseline, GraphSAGEBaseline, MLPBaseline
from refract.models.routed_gnn import RoutedGNN
from refract.regimes.metrics import compute_regime_vectors, regime_vectors_to_tensor
from refract.regimes.taxonomy import assign_regime_labels, fit_regime_gmm
from refract.training.config import load_hps, make_train_cfg, resolve_config
from refract.training.device import select_device
from refract.training.seed import set_seed
from refract.training.trainer import train_baseline_model, train_routed_model

warnings.filterwarnings("ignore")

app = typer.Typer(pretty_exceptions_enable=False)

_ABLATION_VARIANTS: list[tuple[str, dict]] = [
    ("no_align",   {"lambda_align": 0.0}),
    ("no_smooth",  {"lambda_smooth": 0.0}),
    ("no_entropy", {"lambda_entropy": 0.0}),
    ("no_aux",     {"lambda_expert_aux": 0.0, "aux_floor": 0.0}),
]

_BASELINES: list[tuple[str, type]] = [
    ("mlp", MLPBaseline),
    ("gcn", GCNBaseline),
    ("graphsage", GraphSAGEBaseline),
    ("gat", GATBaseline),
    ("appnp", APPNPBaseline),
]


@dataclass
class SplitResult:
    model: str
    split: int
    seed: int
    accuracy: float
    macro_f1: float
    roc_auc: float | None


def _roc_auc(probs: np.ndarray, true: np.ndarray) -> float | None:
    try:
        if probs.shape[1] == 2:
            return float(roc_auc_score(true, probs[:, 1]))
        return float(roc_auc_score(true, probs, multi_class="ovr", average="macro"))
    except Exception:
        return None


def _run_split(
    dataset_name: str, split_idx: int, seed_val: int, hps: dict,
    root: str, device: str, collect_gate: bool,
) -> tuple[list[SplitResult], list[dict], list[dict]]:
    set_seed(seed_val * 1000 + split_idx)
    ds = load_real_world_dataset(dataset_name, root=root, split_idx=split_idx)

    regime_vectors = compute_regime_vectors(ds, ds.train_mask)
    regime_gmm = fit_regime_gmm(regime_vectors)
    regime_tensor = regime_vectors_to_tensor(regime_vectors)

    train_cfg = make_train_cfg(hps, dataset_name, device)
    use_roc = dataset_name in BINARY_DATASETS

    routed = RoutedGNN(
        in_dim=ds.num_features,
        hidden_dim=hps["hidden_dim"],
        num_classes=ds.num_classes,
        regime_dim=regime_tensor.shape[1],
        gate_hidden_dim=hps["gate_hidden_dim"],
        dropout=hps["dropout"],
    )
    r_result = train_routed_model(routed, ds, regime_tensor, train_cfg)

    dev = torch.device(device)
    routed.eval()
    with torch.no_grad():
        out = routed(ds.x.to(dev), ds.edge_index.to(dev), regime_tensor.to(dev))

    test_mask = ds.test_mask.numpy()
    true = ds.y.numpy()[test_mask]
    probs = out["probs"].cpu().numpy()[test_mask]
    roc = _roc_auc(probs, true) if use_roc else None

    results: list[SplitResult] = [SplitResult(
        model="refract", split=split_idx, seed=seed_val,
        accuracy=r_result.test_result.accuracy,
        macro_f1=r_result.test_result.macro_f1,
        roc_auc=roc,
    )]

    history_rows: list[dict] = [
        {"dataset": dataset_name, "model": "refract", "split": split_idx, "seed": seed_val,
         "epoch": h["epoch"], "val_acc": h.get("val_acc", 0.0), "val_roc": h.get("val_roc")}
        for h in r_result.train_history
    ]

    gate_rows: list[dict] = []
    if collect_gate:
        regime_labels = assign_regime_labels(regime_vectors, regime_gmm)
        gate_np = out["gate"].cpu().numpy()
        n_sample = min(2000, ds.num_nodes)
        rng = np.random.default_rng(seed_val * 1000 + split_idx)
        for i in rng.choice(ds.num_nodes, n_sample, replace=False):
            gate_rows.append({
                "dataset": dataset_name, "split": split_idx, "seed": seed_val,
                "node_idx": int(i),
                "gate_attr": float(gate_np[i, 0]),
                "gate_topo": float(gate_np[i, 1]),
                "regime_label": str(regime_labels[i]),
            })

    for bname, cls in _BASELINES:
        bm = cls(ds.num_features, hps["hidden_dim"], ds.num_classes, hps["dropout"])
        b_result = train_baseline_model(bm, ds, train_cfg)
        bm.eval()
        with torch.no_grad():
            bout = bm(ds.x.to(dev), ds.edge_index.to(dev))
        b_probs = bout["probs"].cpu().numpy()[test_mask]
        b_roc = _roc_auc(b_probs, true) if use_roc else None
        results.append(SplitResult(
            model=bname, split=split_idx, seed=seed_val,
            accuracy=b_result.test_result.accuracy,
            macro_f1=b_result.test_result.macro_f1,
            roc_auc=b_roc,
        ))
        history_rows.extend([
            {"dataset": dataset_name, "model": bname, "split": split_idx, "seed": seed_val,
             "epoch": h["epoch"], "val_acc": h.get("val_acc", 0.0), "val_roc": h.get("val_roc")}
            for h in b_result.train_history
        ])

    return results, history_rows, gate_rows


def _run_ablation(
    dataset_name: str, split_idx: int, hps: dict, root: str, device: str,
) -> list[dict]:
    set_seed(split_idx * 77)
    ds = load_real_world_dataset(dataset_name, root=root, split_idx=split_idx)
    regime_vectors = compute_regime_vectors(ds, ds.train_mask)
    fit_regime_gmm(regime_vectors)
    regime_tensor = regime_vectors_to_tensor(regime_vectors)

    use_roc = dataset_name in BINARY_DATASETS
    test_mask = ds.test_mask.numpy()
    true = ds.y.numpy()[test_mask]
    dev = torch.device(device)
    rows: list[dict] = []

    for variant_name, overrides in [("full", {}), *_ABLATION_VARIANTS]:
        train_cfg = make_train_cfg({**hps, **overrides}, dataset_name, device)
        model = RoutedGNN(
            in_dim=ds.num_features,
            hidden_dim=hps["hidden_dim"],
            num_classes=ds.num_classes,
            regime_dim=regime_tensor.shape[1],
            gate_hidden_dim=hps["gate_hidden_dim"],
            dropout=hps["dropout"],
        )
        result = train_routed_model(model, ds, regime_tensor, train_cfg)
        model.eval()
        with torch.no_grad():
            out = model(ds.x.to(dev), ds.edge_index.to(dev), regime_tensor.to(dev))
        probs = out["probs"].cpu().numpy()[test_mask]
        roc = _roc_auc(probs, true) if use_roc else None
        rows.append({
            "dataset": dataset_name, "variant": variant_name, "split": split_idx,
            "accuracy": result.test_result.accuracy,
            "macro_f1": result.test_result.macro_f1,
            "roc_auc": roc if roc is not None else "",
        })
    return rows


def _print_table(dataset_name: str, all_results: list[SplitResult], n_seeds: int) -> None:
    use_roc = dataset_name in BINARY_DATASETS
    metric_key = "roc_auc" if use_roc else "accuracy"
    metric_label = "ROC-AUC" if use_roc else "Accuracy"
    models = ["refract"] + [b for b, _ in _BASELINES]
    n_splits = len({r.split for r in all_results if r.model == "refract"})

    typer.echo(f"\n{'='*65}")
    typer.echo(f"  {dataset_name.upper()}  —  {metric_label} ({n_splits} splits × {n_seeds} seeds)")
    typer.echo(f"{'='*65}")
    typer.echo(f"  {'Model':<14}  {metric_label:>10}  {'Macro-F1':>10}")
    typer.echo(f"  {'-'*50}")
    for model in models:
        rows = [r for r in all_results if r.model == model]
        if not rows:
            continue
        vals = [getattr(r, metric_key) for r in rows if getattr(r, metric_key) is not None]
        f1s = [r.macro_f1 for r in rows]
        tag = " ←" if model == "refract" else ""
        typer.echo(
            f"  {model:<14}  {np.mean(vals)*100:>8.2f}±{np.std(vals)*100:.2f}"
            f"  {np.mean(f1s)*100:>8.2f}±{np.std(f1s)*100:.2f}{tag}"
        )
    typer.echo(f"{'='*65}\n")


@app.command()
def run(
    dataset: str = typer.Argument(..., help="Dataset name"),
    config: str = typer.Option("", help="Path to YAML config (defaults to configs/hparams/{dataset}.yaml)"),
    n_splits: int = typer.Option(0, help="Number of splits (0 = all available)"),
    n_seeds: int = typer.Option(3, help="Number of random seeds per split"),
    ablate: bool = typer.Option(False, help="Run loss-component ablation (1 seed × 3 splits)"),
    device: str = typer.Option("auto", help="Device: auto | cuda | mps | cpu"),
    root: str = typer.Option("data", help="Dataset root directory"),
    out_dir: str = typer.Option("results/benchmarks", help="Output directory"),
) -> None:
    dataset = dataset.lower()
    device_str = select_device(device)

    try:
        config_path = resolve_config(dataset, config)
    except FileNotFoundError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    hps = load_hps(config_path)
    total = NUM_SPLITS.get(dataset, 10)
    n = total if n_splits == 0 else min(n_splits, total)
    use_roc = dataset in BINARY_DATASETS

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    typer.echo(
        f"[eval] {dataset}  {n} splits × {n_seeds} seeds  "
        f"device={device_str}  h={hps['hidden_dim']}  epochs={hps['epochs']}"
    )

    all_results: list[SplitResult] = []
    all_history: list[dict] = []
    all_gate: list[dict] = []

    for split_idx in range(n):
        for seed_val in range(n_seeds):
            t0 = time.time()
            results, history, gate = _run_split(
                dataset, split_idx, seed_val, hps, root, device_str,
                collect_gate=(seed_val == 0),
            )
            elapsed = time.time() - t0
            r = next(r for r in results if r.model == "refract")
            metric_str = f"roc={r.roc_auc:.4f}" if use_roc and r.roc_auc is not None else f"acc={r.accuracy:.4f}"
            typer.echo(f"  split={split_idx} seed={seed_val}  refract {metric_str}  f1={r.macro_f1:.4f}  [{elapsed:.0f}s]")
            all_results.extend(results)
            all_history.extend(history)
            all_gate.extend(gate)

    _print_table(dataset, all_results, n_seeds)

    csv_path = out_path / f"{dataset}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "split", "seed", "accuracy", "macro_f1", "roc_auc"])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "dataset": dataset, "model": r.model, "split": r.split, "seed": r.seed,
                "accuracy": f"{r.accuracy:.6f}", "macro_f1": f"{r.macro_f1:.6f}",
                "roc_auc": f"{r.roc_auc:.6f}" if r.roc_auc is not None else "",
            })
    typer.echo(f"[eval] Results  → {csv_path}")

    hist_path = out_path / f"{dataset}_history.parquet"
    pd.DataFrame(all_history).to_parquet(hist_path, index=False)
    typer.echo(f"[eval] History  → {hist_path}")

    if all_gate:
        gate_path = out_path / f"{dataset}_gate.parquet"
        pd.DataFrame(all_gate).to_parquet(gate_path, index=False)
        typer.echo(f"[eval] Gate     → {gate_path}")

    if ablate:
        typer.echo(f"[eval] Running ablation study (3 splits × 1 seed)…")
        abl_rows: list[dict] = []
        for split_idx in range(min(3, n)):
            abl_rows.extend(_run_ablation(dataset, split_idx, hps, root, device_str))
        abl_path = out_path / f"{dataset}_ablations.csv"
        with open(abl_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "split", "accuracy", "macro_f1", "roc_auc"])
            writer.writeheader()
            writer.writerows(abl_rows)
        typer.echo(f"[eval] Ablation → {abl_path}")


if __name__ == "__main__":
    app()
