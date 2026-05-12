from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import typer

from refract.artifacts.writer import write_run_artifacts
from refract.counterfactuals.perturbations import PerturbationMode
from refract.counterfactuals.runner import compute_all_counterfactuals
from refract.datasets.real_world import load_real_world_dataset
from refract.models.baselines import APPNPBaseline, GATBaseline, GCNBaseline, GraphSAGEBaseline, MLPBaseline
from refract.models.routed_gnn import RoutedGNN
from refract.projection.distortion import compute_distortion
from refract.projection.project import compute_projection
from refract.regimes.metrics import compute_regime_vectors, regime_vectors_to_tensor
from refract.regimes.taxonomy import assign_regime_labels, fit_regime_gmm
from refract.training.config import load_hps, make_train_cfg, resolve_config
from refract.training.device import select_device
from refract.training.seed import set_seed
from refract.training.trainer import train_baseline_model, train_routed_model

warnings.filterwarnings("ignore")

app = typer.Typer(pretty_exceptions_enable=False)

_BASELINES: list[tuple[str, type]] = [
    ("mlp", MLPBaseline), ("gcn", GCNBaseline), ("graphsage", GraphSAGEBaseline),
    ("gat", GATBaseline), ("appnp", APPNPBaseline),
]

_DIM_NAMES = ["attribute_evidence", "topology_evidence", "concordance", "log_degree", "clustering_coeff"]


@app.command()
def run(
    config: Path = typer.Option(..., help="Path to YAML config (from tune.py or configs/datasets/)"),
    dataset: str = typer.Option("", help="Dataset name (overrides config stem if set)"),
    split_idx: int = typer.Option(0, help="Dataset split index"),
    seed: int = typer.Option(0, help="Random seed"),
    device: str = typer.Option("auto", help="Device: auto | cuda | mps | cpu"),
    root: str = typer.Option("data", help="Dataset root directory"),
    save_artifacts: bool = typer.Option(False, help="Write run artifacts for the web app"),
    artifacts_dir: str = typer.Option("results/runs", help="Output directory for artifacts"),
) -> None:
    device_str = select_device(device)
    set_seed(seed)

    dataset_name = dataset.lower() if dataset else config.stem
    hps = load_hps(config)

    ds = load_real_world_dataset(dataset_name, root=root, split_idx=split_idx)
    regime_vectors = compute_regime_vectors(ds, ds.train_mask)
    regime_gmm = fit_regime_gmm(regime_vectors)
    regime_tensor = regime_vectors_to_tensor(regime_vectors)

    train_cfg = make_train_cfg(hps, dataset_name, device_str)
    routed = RoutedGNN(
        in_dim=ds.num_features,
        hidden_dim=hps["hidden_dim"],
        num_classes=ds.num_classes,
        regime_dim=regime_tensor.shape[1],
        gate_hidden_dim=hps["gate_hidden_dim"],
        dropout=hps["dropout"],
    )
    result = train_routed_model(routed, ds, regime_tensor, train_cfg)
    typer.echo(
        f"[train] {dataset_name}  split={split_idx}  "
        f"acc={result.test_result.accuracy:.4f}  f1={result.test_result.macro_f1:.4f}  "
        f"epoch={result.best_epoch}  device={device_str}"
    )

    if not save_artifacts:
        return

    dev = torch.device(device_str)
    routed.eval()
    with torch.no_grad():
        out = routed(ds.x.to(dev), ds.edge_index.to(dev), regime_tensor.to(dev))
    gate_np = out["gate"].cpu().numpy()
    probs_np = out["probs"].cpu().numpy()
    expert_hidden_np = out["expert_hidden"].cpu().numpy()

    baseline_results: dict = {}
    for bname, cls in _BASELINES:
        bm = cls(ds.num_features, hps["hidden_dim"], ds.num_classes, hps["dropout"])
        baseline_results[bname] = train_baseline_model(bm, ds, train_cfg)

    cf_records = compute_all_counterfactuals(
        routed, ds, regime_tensor,
        node_indices=ds.test_mask.numpy().nonzero()[0].tolist()[:500],
        modes=[
            PerturbationMode.MEAN_FEATURE_MASK, PerturbationMode.EGO_EDGE_MASK,
            PerturbationMode.ATTRIBUTE_EXPERT_ONLY, PerturbationMode.TOPOLOGY_EXPERT_ONLY,
        ],
        device=dev,
    )

    regime_labels = assign_regime_labels(regime_vectors, regime_gmm)
    regime_matrix = np.stack([
        regime_vectors.attribute_evidence, regime_vectors.topology_evidence,
        regime_vectors.concordance, regime_vectors.log_degree, regime_vectors.clustering_coeff,
    ], axis=1)
    projection = compute_projection(regime_matrix, _DIM_NAMES, "umap", seed)
    distortion = compute_distortion(regime_matrix, projection.xy)

    model_comparison: dict = {"routed": {
        "accuracy": result.test_result.accuracy,
        "macro_f1": result.test_result.macro_f1,
    }}
    for bname, br in baseline_results.items():
        model_comparison[bname] = {"accuracy": br.test_result.accuracy, "macro_f1": br.test_result.macro_f1}

    run_id = f"{dataset_name}_split{split_idx}_seed{seed}"
    run_dir = Path(artifacts_dir) / run_id
    write_run_artifacts(
        run_dir, run_id, ds, regime_vectors, gate_np, probs_np,
        projection, distortion, cf_records,
        {"accuracy": result.test_result.accuracy, "macro_f1": result.test_result.macro_f1,
         "model_comparison": model_comparison},
        expert_hidden=expert_hidden_np,
        gate_history=result.gate_history,
        hidden_dim=hps["hidden_dim"],
        regime_dim=regime_tensor.shape[1],
        config={},
        regime_gmm=regime_gmm,
    )
    typer.echo(f"[train] Artifacts → {run_dir}")


if __name__ == "__main__":
    app()
