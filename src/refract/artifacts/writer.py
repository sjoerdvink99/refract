import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from refract.artifacts.manifest import write_manifest
from refract.counterfactuals.runner import CounterfactualRecord
from refract.datasets.base import GraphDataset
from refract.projection.project import ProjectionResult
from refract.regimes.metrics import RegimeVectors
from refract.regimes.taxonomy import (
    RegimeGMM,
    assign_regime_labels,
    compute_soft_regime_membership,
    dominant_expert,
    fit_regime_gmm,
)
from refract.training.trainer import GateSnapshot
from schemas.run import DatasetInfo, ModelInfo, RunFiles, RunManifest


def write_nodes_table(
    path: Path,
    dataset: GraphDataset,
    regime_vectors: RegimeVectors,
    regime_labels: np.ndarray,
    gate: np.ndarray,
    probs: np.ndarray,
    split_labels: np.ndarray,
    soft_membership: np.ndarray | None = None,
) -> None:
    class_names = dataset.class_names or [str(c) for c in range(dataset.num_classes)]
    true_labels = dataset.y.numpy()
    preds = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    gate_entropy = -(gate * np.log(gate + 1e-10)).sum(axis=1)

    ei = dataset.edge_index.numpy()
    degree = np.bincount(ei[0], minlength=dataset.num_nodes).astype(np.int32)

    same_label = (true_labels[ei[0]] == true_labels[ei[1]]).astype(np.float64)
    edge_counts = np.bincount(ei[0], minlength=dataset.num_nodes).astype(np.float64)
    edge_sums = np.bincount(ei[0], weights=same_label, minlength=dataset.num_nodes)
    safe_counts = np.where(edge_counts > 0, edge_counts, 1.0)
    local_homophily = np.where(edge_counts > 0, edge_sums / safe_counts, 0.0).astype(np.float32)

    data: dict = {
        "node_id": dataset.node_ids,
        "node_index": np.arange(dataset.num_nodes),
        "true_label": [class_names[int(lbl)] for lbl in true_labels],
        "pred_label": [class_names[int(p)] for p in preds],
        "correct": (true_labels == preds),
        "confidence": confidence.astype(np.float32),
        "regime": regime_labels,
        "gate_attribute": gate[:, 0].astype(np.float32),
        "gate_topology": gate[:, 1].astype(np.float32),
        "e_attribute": regime_vectors.attribute_evidence,
        "e_topology": regime_vectors.topology_evidence,
        "concordance": regime_vectors.concordance,
        "uncertainty": regime_vectors.uncertainty,
        "log_degree": regime_vectors.log_degree,
        "clustering_coeff": regime_vectors.clustering_coeff,
        "k2_topology_evidence": regime_vectors.k2_topology_evidence,
        "hop_consistency": regime_vectors.hop_consistency,
        "degree": degree,
        "local_homophily": local_homophily,
        "gate_entropy": gate_entropy.astype(np.float32),
        "split": split_labels,
    }

    if soft_membership is not None:
        data["regime_p_attribute_dominant"] = soft_membership[:, 0]
        data["regime_p_topology_dominant"] = soft_membership[:, 1]
        data["regime_p_concordant"] = soft_membership[:, 2]
        data["regime_p_conflict"] = soft_membership[:, 3]
        data["regime_p_uncertain"] = soft_membership[:, 4]

    df = pd.DataFrame(data)
    pq.write_table(pa.Table.from_pandas(df), path)


def write_edges_table(path: Path, dataset: GraphDataset) -> None:
    ei = dataset.edge_index.numpy()
    src_ids = [dataset.node_ids[int(i)] for i in ei[0]]
    dst_ids = [dataset.node_ids[int(i)] for i in ei[1]]
    df = pd.DataFrame({"source": src_ids, "target": dst_ids})
    pq.write_table(pa.Table.from_pandas(df), path)


def write_projection_table(
    path: Path,
    dataset: GraphDataset,
    projection: ProjectionResult,
    regime_labels: np.ndarray,
    gate: np.ndarray,
    correct: np.ndarray,
    uncertainty: np.ndarray,
    distortion: np.ndarray,
    preds: np.ndarray,
    attribute_evidence: np.ndarray,
    topology_evidence: np.ndarray,
) -> None:
    dom_expert = dominant_expert(gate[:, 0], gate[:, 1])
    df = pd.DataFrame({
        "node_id": dataset.node_ids,
        "x": projection.xy[:, 0].astype(np.float32),
        "y": projection.xy[:, 1].astype(np.float32),
        "regime": regime_labels,
        "dominant_expert": dom_expert,
        "correct": correct,
        "uncertainty": uncertainty.astype(np.float32),
        "distortion": distortion.astype(np.float32),
        "gate_attribute": gate[:, 0].astype(np.float32),
        "gate_topology": gate[:, 1].astype(np.float32),
        "pred_class": preds.astype(np.int16),
        "attribute_evidence": attribute_evidence.astype(np.float32),
        "topology_evidence": topology_evidence.astype(np.float32),
    })
    pq.write_table(pa.Table.from_pandas(df), path)


def write_counterfactuals_table(
    path: Path, records: list[CounterfactualRecord], node_ids: list[str]
) -> None:
    rows = [
        {
            "node_id": node_ids[r.node_idx],
            "mode": r.mode,
            "original_confidence": r.original_confidence,
            "counterfactual_confidence": r.counterfactual_confidence,
            "damage": r.damage,
            "original_pred_label": str(r.original_pred_label),
            "counterfactual_pred_label": str(r.counterfactual_pred_label),
            "gate_attribute": r.gate_attribute,
            "gate_topology": r.gate_topology,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df), path)


def write_expert_embeddings_table(
    path: Path,
    dataset: GraphDataset,
    expert_hidden: np.ndarray,
) -> None:
    n, num_experts, hidden_dim = expert_hidden.shape
    rows: dict[str, list] = {"node_id": dataset.node_ids}
    for e_idx, name in enumerate(["attribute", "topology"][:num_experts]):
        for d in range(hidden_dim):
            rows[f"h_{name}_{d}"] = expert_hidden[:, e_idx, d].tolist()
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), path)


def write_training_history_table(
    path: Path,
    gate_history: list[GateSnapshot],
) -> None:
    df = pd.DataFrame([
        {
            "epoch": s.epoch,
            "gate_attribute_mean": s.gate_attribute_mean,
            "gate_attribute_std": s.gate_attribute_std,
            "gate_topology_mean": s.gate_topology_mean,
            "gate_topology_std": s.gate_topology_std,
        }
        for s in gate_history
    ])
    pq.write_table(pa.Table.from_pandas(df), path)


def write_metrics_json(path: Path, metrics: dict) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def write_run_artifacts(
    run_dir: Path,
    run_id: str,
    dataset: GraphDataset,
    regime_vectors: RegimeVectors,
    gate: np.ndarray,
    probs: np.ndarray,
    projection: ProjectionResult,
    distortion: np.ndarray,
    counterfactual_records: list[CounterfactualRecord],
    metrics: dict,
    expert_hidden: np.ndarray | None = None,
    gate_history: list[GateSnapshot] | None = None,
    model_name: str = "RoutedGNN",
    hidden_dim: int = 64,
    regime_dim: int = 7,
    config: dict | None = None,
    regime_gmm: RegimeGMM | None = None,
) -> RunManifest:
    run_dir.mkdir(parents=True, exist_ok=True)

    if regime_gmm is None:
        regime_gmm = fit_regime_gmm(regime_vectors)

    regime_labels = assign_regime_labels(regime_vectors, regime_gmm)
    soft_membership = compute_soft_regime_membership(regime_vectors, regime_gmm)
    true_labels = dataset.y.numpy()
    preds = probs.argmax(axis=1)
    correct = true_labels == preds

    train_mask = dataset.train_mask.numpy()
    val_mask = dataset.val_mask.numpy()
    split_labels = np.where(train_mask, "train", np.where(val_mask, "val", "test"))

    write_nodes_table(
        run_dir / "nodes.parquet",
        dataset, regime_vectors, regime_labels, gate, probs, split_labels, soft_membership,
    )
    write_edges_table(run_dir / "edges.parquet", dataset)
    write_projection_table(
        run_dir / "projection.parquet",
        dataset, projection, regime_labels, gate, correct,
        regime_vectors.uncertainty, distortion, preds,
        regime_vectors.attribute_evidence,
        regime_vectors.topology_evidence,
    )
    if counterfactual_records:
        write_counterfactuals_table(
            run_dir / "counterfactuals.parquet", counterfactual_records, dataset.node_ids
        )
    if expert_hidden is not None:
        write_expert_embeddings_table(run_dir / "expert_embeddings.parquet", dataset, expert_hidden)
    if gate_history:
        write_training_history_table(run_dir / "training_history.parquet", gate_history)

    write_metrics_json(run_dir / "metrics.json", metrics)

    manifest = RunManifest(
        run_id=run_id,
        dataset=DatasetInfo(
            name=dataset.name,
            num_nodes=dataset.num_nodes,
            num_edges=dataset.num_edges,
            num_classes=dataset.num_classes,
            feature_dim=dataset.num_features,
        ),
        model=ModelInfo(
            name=model_name,
            hidden_dim=hidden_dim,
            num_layers=2,
            experts=["attribute", "topology"],
            regime_dim=regime_dim,
        ),
        files=RunFiles(
            nodes="nodes.parquet",
            edges="edges.parquet",
            projection="projection.parquet",
            metrics="metrics.json",
            counterfactuals="counterfactuals.parquet" if counterfactual_records else None,
            expert_embeddings="expert_embeddings.parquet" if expert_hidden is not None else None,
            training_history="training_history.parquet" if gate_history else None,
        ),
        config=config,
    )
    write_manifest(run_dir / "manifest.json", manifest)
    return manifest
