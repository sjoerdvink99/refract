from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class ProjectionResult:
    xy: np.ndarray
    method: str
    input_dimensions: list[str]
    metadata: dict


def build_diagnostic_matrix(
    gate_attribute: np.ndarray,
    gate_topology: np.ndarray,
    attribute_evidence: np.ndarray,
    topology_evidence: np.ndarray,
    concordance: np.ndarray,
    confidence: np.ndarray,
    correct: np.ndarray,
    feature_damage: np.ndarray | None = None,
    edge_damage: np.ndarray | None = None,
    degree: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    columns = [
        gate_attribute,
        gate_topology,
        attribute_evidence,
        topology_evidence,
        concordance,
        confidence,
        correct.astype(float),
    ]
    names = [
        "gate_attribute", "gate_topology",
        "attribute_evidence", "topology_evidence",
        "concordance", "confidence", "correct",
    ]
    if feature_damage is not None:
        columns.append(feature_damage)
        names.append("feature_damage")
    if edge_damage is not None:
        columns.append(edge_damage)
        names.append("edge_damage")
    if degree is not None:
        columns.append(np.log1p(degree.astype(float)))
        names.append("log_degree")

    return np.stack(columns, axis=1), names


def _project_pca(matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    pca = PCA(n_components=2)
    xy = pca.fit_transform(matrix)
    return xy, {"method": "pca", "explained_variance": pca.explained_variance_ratio_.tolist()}


def _project_umap(
    matrix: np.ndarray,
    seed: int = 0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> tuple[np.ndarray, dict]:
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=seed,
        )
        xy = reducer.fit_transform(matrix)
        return xy, {"method": "umap", "neighbors": n_neighbors, "min_dist": min_dist, "seed": seed}
    except ImportError:
        return _project_pca(matrix)
    except Exception:
        return _project_pca(matrix)


def compute_projection(
    matrix: np.ndarray,
    dimension_names: list[str],
    method: str = "umap",
    seed: int = 0,
) -> ProjectionResult:
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    if method == "umap":
        xy, meta = _project_umap(matrix_scaled, seed=seed)
    else:
        xy, meta = _project_pca(matrix_scaled)

    return ProjectionResult(
        xy=xy.astype(np.float32),
        method=meta["method"],
        input_dimensions=dimension_names,
        metadata=meta,
    )
