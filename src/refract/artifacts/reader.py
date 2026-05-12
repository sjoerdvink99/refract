import functools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from refract.artifacts.manifest import read_manifest
from schemas.counterfactual import CounterfactualResult, NodeCounterfactuals
from schemas.graph import EgoGraph, GraphEdge, GraphNode, Subgraph
from schemas.metrics import RunMetrics
from schemas.node import (
    CounterfactualSummary,
    GateWeights,
    NeighborInfo,
    NodeDetail,
    NodeMetrics,
    RegimeVector,
)
from schemas.projection import ProjectionData, ProjectionPoint
from schemas.run import RunManifest, RunSummary


@functools.lru_cache(maxsize=64)
def _read_parquet_cached(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


@functools.lru_cache(maxsize=32)
def _build_adjacency_cached(run_dir: Path) -> dict[str, list[str]]:
    edges_path = run_dir / "edges.parquet"
    if not edges_path.exists():
        return {}
    df = _read_parquet_cached(edges_path)
    adj: dict[str, list[str]] = {}
    for row in df.itertuples():
        src, dst = str(row.source), str(row.target)
        if src not in adj:
            adj[src] = []
        if dst not in adj:
            adj[dst] = []
        adj[src].append(dst)
        adj[dst].append(src)
    return adj


def _dominant_expert(ga: float, gt: float) -> str:
    return "topology" if gt >= ga else "attribute"


class ArtifactReader:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir

    def _run_dir(self, run_id: str) -> Path:
        return self.artifacts_dir / run_id

    def list_runs(self) -> list[RunSummary]:
        if not self.artifacts_dir.exists():
            return []
        summaries = []
        for run_dir in sorted(self.artifacts_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            manifest = read_manifest(run_dir / "manifest.json")
            if manifest is None:
                continue
            summaries.append(
                RunSummary(
                    run_id=manifest.run_id,
                    dataset_name=manifest.dataset.name,
                    num_nodes=manifest.dataset.num_nodes,
                    num_classes=manifest.dataset.num_classes,
                )
            )
        return summaries

    def read_manifest(self, run_id: str) -> RunManifest | None:
        return read_manifest(self._run_dir(run_id) / "manifest.json")

    def _load_parquet(self, run_id: str, filename: str) -> pd.DataFrame | None:
        path = self._run_dir(run_id) / filename
        if not path.exists():
            return None
        return _read_parquet_cached(path)

    def _adjacency(self, run_id: str) -> dict[str, list[str]]:
        return _build_adjacency_cached(self._run_dir(run_id))

    def read_projection(self, run_id: str) -> ProjectionData | None:
        manifest = self.read_manifest(run_id)
        if manifest is None:
            return None
        df = self._load_parquet(run_id, "projection.parquet")
        if df is None:
            return None

        method = "pca"
        if manifest.config and isinstance(manifest.config.get("projection"), dict):
            method = manifest.config["projection"].get("method", "pca")

        points = [
            ProjectionPoint(
                node_id=str(row.node_id),
                x=float(row.x),
                y=float(row.y),
                regime=str(row.regime),
                dominant_expert=str(row.dominant_expert),
                correct=bool(row.correct) if row.correct is not None else None,
                uncertainty=float(row.uncertainty),
                distortion=float(row.distortion),
                gate_attribute=float(row.gate_attribute),
                gate_topology=float(row.gate_topology),
                pred_class=int(row.pred_class),
                attribute_evidence=float(getattr(row, "attribute_evidence", 0.0)),
                topology_evidence=float(getattr(row, "topology_evidence", 0.0)),
            )
            for row in df.itertuples()
        ]
        return ProjectionData(run_id=run_id, method=method, points=points)

    def read_node_detail(self, run_id: str, node_id: str) -> NodeDetail | None:
        nodes_df = self._load_parquet(run_id, "nodes.parquet")
        if nodes_df is None:
            return None

        row_df = nodes_df[nodes_df["node_id"] == node_id]
        if row_df.empty:
            return None
        r = row_df.iloc[0]

        adj = self._adjacency(run_id)
        nb_ids = set(adj.get(node_id, []))
        nb_rows = nodes_df[nodes_df["node_id"].isin(nb_ids)]

        def _str_or_none(val: object) -> str | None:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return None
            return str(val)

        neighbors = [
            NeighborInfo(
                node_id=str(nb.node_id),
                true_label=_str_or_none(nb.true_label),
                regime=str(nb.regime),
                degree=int(nb.degree),
            )
            for nb in nb_rows.itertuples()
        ]

        cf_list: list[CounterfactualSummary] = []
        cf_df = self._load_parquet(run_id, "counterfactuals.parquet")
        if cf_df is not None:
            node_cfs = cf_df[cf_df["node_id"] == node_id]
            for cf in node_cfs.itertuples():
                cf_list.append(
                    CounterfactualSummary(
                        mode=str(cf.mode),
                        original_confidence=float(cf.original_confidence),
                        counterfactual_confidence=float(cf.counterfactual_confidence),
                        damage=float(cf.damage),
                        original_pred_label=str(cf.original_pred_label),
                        counterfactual_pred_label=str(cf.counterfactual_pred_label),
                        gate=GateWeights(
                            attribute=float(cf.gate_attribute),
                            topology=float(cf.gate_topology),
                        ),
                    )
                )

        def _safe_float(val: object, default: float = 0.0) -> float:
            try:
                return float(val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        def _safe_str(val: object) -> str | None:
            if val is None:
                return None
            if isinstance(val, float) and math.isnan(val):
                return None
            return str(val)

        return NodeDetail(
            node_id=node_id,
            index=int(r.node_index),
            true_label=_safe_str(r.true_label),
            predicted_label=str(r.pred_label),
            correct=bool(r.correct),
            confidence=float(r.confidence),
            regime=str(r.regime),
            gate=GateWeights(
                attribute=float(r.gate_attribute),
                topology=float(r.gate_topology),
            ),
            regime_vector=RegimeVector(
                attribute_evidence=float(r.e_attribute),
                topology_evidence=float(r.e_topology),
                concordance=float(r.concordance),
                uncertainty=float(r.uncertainty),
                log_degree=_safe_float(getattr(r, "log_degree", 0.0)),
                clustering_coeff=_safe_float(getattr(r, "clustering_coeff", 0.0)),
                k2_topology_evidence=_safe_float(getattr(r, "k2_topology_evidence", 0.0)),
                hop_consistency=_safe_float(getattr(r, "hop_consistency", 0.5)),
            ),
            metrics=NodeMetrics(
                degree=int(r.degree),
                local_homophily=float(r.local_homophily),
                gate_entropy=float(r.gate_entropy),
            ),
            neighbors=neighbors,
            counterfactuals=cf_list,
        )

    def read_ego_graph(self, run_id: str, node_id: str, radius: int = 1) -> EgoGraph | None:
        nodes_df = self._load_parquet(run_id, "nodes.parquet")
        proj_df = self._load_parquet(run_id, "projection.parquet")

        if nodes_df is None:
            return None
        if node_id not in nodes_df["node_id"].values:
            return None

        adj = self._adjacency(run_id)

        ego_nodes: set[str] = {node_id}
        frontier: set[str] = {node_id}
        for _ in range(radius):
            next_frontier: set[str] = set()
            for n in frontier:
                next_frontier.update(adj.get(n, []))
            ego_nodes.update(next_frontier)
            frontier = next_frontier

        proj_map: dict[str, tuple[float, float]] = {}
        if proj_df is not None:
            for row in proj_df.itertuples():
                proj_map[str(row.node_id)] = (float(row.x), float(row.y))

        sub_nodes_df = nodes_df[nodes_df["node_id"].isin(ego_nodes)]

        graph_nodes = [
            GraphNode(
                node_id=str(row.node_id),
                x=proj_map.get(str(row.node_id), (0.0, 0.0))[0],
                y=proj_map.get(str(row.node_id), (0.0, 0.0))[1],
                regime=str(row.regime),
                dominant_expert=_dominant_expert(
                    float(row.gate_attribute), float(row.gate_topology)
                ),
                correct=bool(row.correct),
                degree=int(row.degree),
                gate_attribute=float(row.gate_attribute),
                gate_topology=float(row.gate_topology),
            )
            for row in sub_nodes_df.itertuples()
        ]

        ego_node_set = {n.node_id for n in graph_nodes}
        graph_edges = [
            GraphEdge(source=src, target=dst)
            for src in ego_node_set
            for dst in adj.get(src, [])
            if dst in ego_node_set
        ]

        return EgoGraph(
            center_node_id=node_id,
            nodes=graph_nodes,
            edges=graph_edges,
            radius=radius,
        )

    def read_subgraph(self, run_id: str, node_ids: list[str]) -> Subgraph:
        nodes_df = self._load_parquet(run_id, "nodes.parquet") or pd.DataFrame()
        proj_df = self._load_parquet(run_id, "projection.parquet")

        id_set = set(node_ids)
        sub_nodes_df = nodes_df[nodes_df["node_id"].isin(id_set)] if not nodes_df.empty else nodes_df

        proj_map: dict[str, tuple[float, float]] = {}
        if proj_df is not None:
            for row in proj_df.itertuples():
                proj_map[str(row.node_id)] = (float(row.x), float(row.y))

        adj = self._adjacency(run_id)

        graph_nodes = [
            GraphNode(
                node_id=str(row.node_id),
                x=proj_map.get(str(row.node_id), (0.0, 0.0))[0],
                y=proj_map.get(str(row.node_id), (0.0, 0.0))[1],
                regime=str(row.regime),
                dominant_expert=_dominant_expert(
                    float(row.gate_attribute), float(row.gate_topology)
                ),
                correct=bool(row.correct),
                degree=int(row.degree),
                gate_attribute=float(row.gate_attribute),
                gate_topology=float(row.gate_topology),
            )
            for row in sub_nodes_df.itertuples()
        ]

        graph_edges = [
            GraphEdge(source=src, target=dst)
            for src in id_set
            for dst in adj.get(src, [])
            if dst in id_set
        ]

        return Subgraph(nodes=graph_nodes, edges=graph_edges)

    def read_metrics(self, run_id: str) -> RunMetrics | None:
        path = self._run_dir(run_id) / "metrics.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return RunMetrics.model_validate(data)

    def read_counterfactuals(self, run_id: str, node_id: str) -> NodeCounterfactuals | None:
        df = self._load_parquet(run_id, "counterfactuals.parquet")
        if df is None:
            return None
        node_df = df[df["node_id"] == node_id]
        if node_df.empty:
            return NodeCounterfactuals(node_id=node_id, results=[])

        results = [
            CounterfactualResult(
                node_id=node_id,
                mode=str(row.mode),
                original_confidence=float(row.original_confidence),
                counterfactual_confidence=float(row.counterfactual_confidence),
                damage=float(row.damage),
                original_pred_label=str(row.original_pred_label),
                counterfactual_pred_label=str(row.counterfactual_pred_label),
                gate_attribute=float(row.gate_attribute),
                gate_topology=float(row.gate_topology),
            )
            for row in node_df.itertuples()
        ]
        return NodeCounterfactuals(node_id=node_id, results=results)

    def read_expert_profile(self, run_id: str) -> dict | None:
        nodes_df = self._load_parquet(run_id, "nodes.parquet")
        if nodes_df is None:
            return None

        structural_features = ["degree", "local_homophily", "gate_entropy"]
        regime_features = ["e_attribute", "e_topology", "concordance", "uncertainty",
                           "log_degree", "clustering_coeff", "k2_topology_evidence", "hop_consistency"]
        all_features = structural_features + regime_features

        nodes_df = nodes_df.copy()
        nodes_df["dominant_expert"] = nodes_df.apply(
            lambda row: _dominant_expert(
                float(row["gate_attribute"]), float(row["gate_topology"])
            ),
            axis=1,
        )

        profile: dict = {}
        for expert_name in ["attribute", "topology"]:
            mask = nodes_df["dominant_expert"] == expert_name
            sub = nodes_df[mask]
            if sub.empty:
                profile[expert_name] = {}
                continue

            expert_data: dict = {"count": int(mask.sum())}
            correlations: dict[str, float] = {}
            gate_col = f"gate_{expert_name}"
            if gate_col in nodes_df.columns:
                for feat in all_features:
                    if feat in nodes_df.columns:
                        col_data = pd.to_numeric(nodes_df[feat], errors="coerce").dropna()
                        gate_data = pd.to_numeric(nodes_df[gate_col], errors="coerce")
                        aligned = gate_data.loc[col_data.index]
                        if len(col_data) > 5:
                            corr, _ = stats.spearmanr(col_data, aligned)
                            correlations[feat] = float(corr) if not np.isnan(corr) else 0.0

            feature_stats: dict[str, dict] = {}
            for feat in all_features:
                if feat not in sub.columns:
                    continue
                vals = pd.to_numeric(sub[feat], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                pop_vals = pd.to_numeric(nodes_df[feat], errors="coerce").dropna()
                feature_stats[feat] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "pop_mean": float(pop_vals.mean()),
                    "pop_std": float(pop_vals.std()),
                }

            expert_data["feature_stats"] = feature_stats
            expert_data["gate_correlations"] = dict(
                sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
            )
            profile[expert_name] = expert_data

        return profile

    def read_training_history(self, run_id: str) -> list[dict] | None:
        df = self._load_parquet(run_id, "training_history.parquet")
        if df is None:
            return None
        return df.to_dict(orient="records")
