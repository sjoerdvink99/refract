import numpy as np
import torch

from refract.datasets.base import GraphDataset
from refract.regimes.metrics import (
    compute_attribute_evidence,
    compute_concordance,
    compute_regime_vectors,
    compute_structural_features,
    compute_topology_evidence,
    compute_uncertainty,
    regime_vectors_to_tensor,
)


def _make_toy_dataset(
    n: int = 20,
    feature_dim: int = 4,
    num_classes: int = 2,
    seed: int = 0,
) -> GraphDataset:
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.standard_normal((n, feature_dim)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, num_classes, size=n).astype(np.int64))
    edges = [[i, (i + 1) % n] for i in range(n)]
    edges += [[(i + 1) % n, i] for i in range(n)]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:int(n * 0.6)] = True
    return GraphDataset(
        name="toy",
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=mask,
        val_mask=~mask,
        test_mask=~mask,
        node_ids=[f"n{i}" for i in range(n)],
    )


def test_attribute_evidence_shape():
    ds = _make_toy_dataset()
    evidence, dist = compute_attribute_evidence(ds, ds.train_mask)
    assert evidence.shape == (20,)
    assert dist.shape == (20, 2)
    assert evidence.min() >= 0.0
    assert evidence.max() <= 1.0


def test_topology_evidence_shape():
    ds = _make_toy_dataset()
    evidence, dist = compute_topology_evidence(ds, ds.train_mask)
    assert evidence.shape == (20,)
    assert dist.shape == (20, 2)
    assert evidence.min() >= 0.0
    assert evidence.max() <= 1.0


def test_concordance_range():
    ds = _make_toy_dataset()
    _, attr_dist = compute_attribute_evidence(ds, ds.train_mask)
    _, topo_dist = compute_topology_evidence(ds, ds.train_mask)
    concordance = compute_concordance(attr_dist, topo_dist)
    assert concordance.shape == (20,)
    assert concordance.min() >= 0.0
    assert concordance.max() <= 1.0


def test_uncertainty_range():
    ds = _make_toy_dataset()
    _, attr_dist = compute_attribute_evidence(ds, ds.train_mask)
    _, topo_dist = compute_topology_evidence(ds, ds.train_mask)
    uncertainty = compute_uncertainty(attr_dist, topo_dist)
    assert uncertainty.shape == (20,)
    assert uncertainty.min() >= 0.0
    assert uncertainty.max() <= 1.0


def test_concordance_uncertainty_complement():
    ds = _make_toy_dataset()
    _, attr_dist = compute_attribute_evidence(ds, ds.train_mask)
    _, topo_dist = compute_topology_evidence(ds, ds.train_mask)
    concordance = compute_concordance(attr_dist, topo_dist)
    uncertainty = compute_uncertainty(attr_dist, topo_dist)
    np.testing.assert_allclose(concordance + uncertainty, np.ones(20), atol=1e-5)


def test_structural_features_shape():
    ds = _make_toy_dataset()
    log_degree, clustering = compute_structural_features(ds)
    assert log_degree.shape == (20,)
    assert clustering.shape == (20,)
    assert log_degree.min() >= 0.0
    assert log_degree.max() <= 1.0
    assert clustering.min() >= 0.0
    assert clustering.max() <= 1.0


def test_regime_vectors_9dim():
    ds = _make_toy_dataset()
    rv = compute_regime_vectors(ds, ds.train_mask)
    tensor = regime_vectors_to_tensor(rv)
    assert tensor.shape == (20, 9)


def test_no_test_label_leakage():
    ds = _make_toy_dataset()
    evidence_full, _ = compute_attribute_evidence(ds, ds.train_mask)
    assert evidence_full is not None


def test_topology_evidence_laplace_smoothing():
    ds = _make_toy_dataset()
    evidence, dist = compute_topology_evidence(ds, ds.train_mask)
    num_classes = ds.num_classes
    assert evidence.min() >= 1.0 / num_classes - 1e-5
    np.testing.assert_allclose(dist.sum(axis=1), np.ones(len(dist)), atol=1e-5)
