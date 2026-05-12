from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture

from refract.regimes.metrics import RegimeVectors

REGIME_LABELS = ["attribute_dominant", "topology_dominant", "concordant", "conflict", "uncertain"]


@dataclass
class RegimeGMM:
    gmm: GaussianMixture
    component_regime_map: dict[int, str]


def fit_regime_gmm(
    vectors: RegimeVectors, n_components: int = 5, random_state: int = 42
) -> RegimeGMM:
    X = np.stack(
        [vectors.attribute_evidence, vectors.topology_evidence, vectors.concordance], axis=1
    ).astype(np.float64)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=random_state,
        n_init=5,
    )
    gmm.fit(X)
    component_regime_map = _assign_components_to_regimes(gmm.means_)
    return RegimeGMM(gmm=gmm, component_regime_map=component_regime_map)


def _assign_components_to_regimes(means: np.ndarray) -> dict[int, str]:
    n = len(means)
    available = set(range(n))
    assignments: dict[int, str] = {}

    concordant_idx = max(available, key=lambda i: (means[i, 0] + means[i, 1] + means[i, 2]) / 3)
    assignments[concordant_idx] = "concordant"
    available.discard(concordant_idx)

    conflict_idx = min(available, key=lambda i: means[i, 2])
    assignments[conflict_idx] = "conflict"
    available.discard(conflict_idx)

    attr_idx = max(available, key=lambda i: means[i, 0] - means[i, 1])
    assignments[attr_idx] = "attribute_dominant"
    available.discard(attr_idx)

    topo_idx = max(available, key=lambda i: means[i, 1] - means[i, 0])
    assignments[topo_idx] = "topology_dominant"
    available.discard(topo_idx)

    for idx in available:
        assignments[idx] = "uncertain"

    return assignments


def assign_regime_labels(vectors: RegimeVectors, regime_gmm: RegimeGMM) -> np.ndarray:
    X = np.stack(
        [vectors.attribute_evidence, vectors.topology_evidence, vectors.concordance], axis=1
    ).astype(np.float64)
    component_indices = regime_gmm.gmm.predict(X)
    return np.array([regime_gmm.component_regime_map[i] for i in component_indices], dtype=str)


def compute_soft_regime_membership(vectors: RegimeVectors, regime_gmm: RegimeGMM) -> np.ndarray:
    X = np.stack(
        [vectors.attribute_evidence, vectors.topology_evidence, vectors.concordance], axis=1
    ).astype(np.float64)
    posteriors = regime_gmm.gmm.predict_proba(X)
    regime_order = ["attribute_dominant", "topology_dominant", "concordant", "conflict", "uncertain"]
    result = np.zeros((len(X), 5), dtype=np.float64)
    for comp_idx, regime_name in regime_gmm.component_regime_map.items():
        col = regime_order.index(regime_name)
        result[:, col] += posteriors[:, comp_idx]
    return result.astype(np.float32)


def dominant_expert(
    gate_attribute: np.ndarray,
    gate_topology: np.ndarray,
) -> np.ndarray:
    return np.where(gate_topology >= gate_attribute, "topology", "attribute")
