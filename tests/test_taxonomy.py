import numpy as np

from refract.regimes.metrics import RegimeVectors
from refract.regimes.taxonomy import assign_regime_labels, compute_soft_regime_membership, fit_regime_gmm


def _make_vectors_batch(specs):
    parts = []
    for eA, eT, conc, n in specs:
        rv = RegimeVectors(
            attribute_evidence=np.full(n, eA, dtype=np.float32),
            topology_evidence=np.full(n, eT, dtype=np.float32),
            concordance=np.full(n, conc, dtype=np.float32),
            uncertainty=np.full(n, 1 - conc, dtype=np.float32),
            log_degree=np.zeros(n, dtype=np.float32),
            clustering_coeff=np.zeros(n, dtype=np.float32),
            k2_topology_evidence=np.full(n, eT, dtype=np.float32),
            hop_consistency=np.full(n, conc, dtype=np.float32),
            attribute_distribution=np.zeros((n, 2), dtype=np.float32),
            topology_distribution=np.zeros((n, 2), dtype=np.float32),
        )
        parts.append(rv)
    return parts


def _concat_vectors(parts):
    return RegimeVectors(
        attribute_evidence=np.concatenate([p.attribute_evidence for p in parts]),
        topology_evidence=np.concatenate([p.topology_evidence for p in parts]),
        concordance=np.concatenate([p.concordance for p in parts]),
        uncertainty=np.concatenate([p.uncertainty for p in parts]),
        log_degree=np.concatenate([p.log_degree for p in parts]),
        clustering_coeff=np.concatenate([p.clustering_coeff for p in parts]),
        k2_topology_evidence=np.concatenate([p.k2_topology_evidence for p in parts]),
        hop_consistency=np.concatenate([p.hop_consistency for p in parts]),
        attribute_distribution=np.concatenate([p.attribute_distribution for p in parts]),
        topology_distribution=np.concatenate([p.topology_distribution for p in parts]),
    )


def test_gmm_fits_and_assigns():
    parts = _make_vectors_batch([
        (0.85, 0.85, 0.90, 50),
        (0.85, 0.20, 0.60, 50),
        (0.20, 0.85, 0.60, 50),
        (0.50, 0.50, 0.10, 50),
        (0.30, 0.30, 0.50, 50),
    ])
    vectors = _concat_vectors(parts)
    gmm = fit_regime_gmm(vectors)
    labels = assign_regime_labels(vectors, gmm)
    assert len(labels) == 250
    assert set(labels).issubset({"concordant", "attribute_dominant", "topology_dominant", "conflict", "uncertain"})


def test_soft_membership_sums_to_one():
    parts = _make_vectors_batch([
        (0.85, 0.85, 0.90, 30),
        (0.85, 0.20, 0.60, 30),
        (0.20, 0.85, 0.60, 30),
        (0.50, 0.50, 0.10, 30),
        (0.30, 0.30, 0.50, 30),
    ])
    vectors = _concat_vectors(parts)
    gmm = fit_regime_gmm(vectors)
    soft = compute_soft_regime_membership(vectors, gmm)
    assert soft.shape == (150, 5)
    np.testing.assert_allclose(soft.sum(axis=1), np.ones(150), atol=1e-5)


def test_label_count_correct():
    parts = _make_vectors_batch([(0.5, 0.5, 0.5, 100)])
    vectors = _concat_vectors(parts)
    gmm = fit_regime_gmm(vectors)
    labels = assign_regime_labels(vectors, gmm)
    assert len(labels) == 100
