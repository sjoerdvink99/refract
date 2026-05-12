import numpy as np

from refract.datasets.synthetic import SyntheticConfig, generate_synthetic_graph


def test_reproducibility():
    cfg = SyntheticConfig(num_nodes=200, num_patches=4, seed=42)
    g1 = generate_synthetic_graph(cfg)
    g2 = generate_synthetic_graph(cfg)
    np.testing.assert_array_equal(g1.dataset.y.numpy(), g2.dataset.y.numpy())
    np.testing.assert_array_almost_equal(g1.dataset.x.numpy(), g2.dataset.x.numpy())


def test_node_count_bounded():
    cfg = SyntheticConfig(num_nodes=300, num_patches=4, seed=0)
    g = generate_synthetic_graph(cfg)
    assert g.dataset.num_nodes <= cfg.num_nodes + cfg.num_patches


def test_non_degenerate_labels():
    cfg = SyntheticConfig(num_nodes=300, num_classes=3, num_patches=6, seed=0)
    g = generate_synthetic_graph(cfg)
    assert len(g.dataset.y.unique()) >= 2


def test_split_masks_partition():
    cfg = SyntheticConfig(num_nodes=200, num_patches=4, seed=0)
    g = generate_synthetic_graph(cfg)
    ds = g.dataset
    assert ds.train_mask.sum() + ds.val_mask.sum() + ds.test_mask.sum() == ds.num_nodes
    assert (ds.train_mask & ds.val_mask).sum() == 0
    assert (ds.train_mask & ds.test_mask).sum() == 0
    assert (ds.val_mask & ds.test_mask).sum() == 0


def test_patch_assignment_sizes():
    cfg = SyntheticConfig(num_nodes=300, num_patches=5, seed=0)
    g = generate_synthetic_graph(cfg)
    assert len(g.patch_assignments) == g.dataset.num_nodes
    assert len(g.true_regimes) == g.dataset.num_nodes


def test_feature_dimension():
    cfg = SyntheticConfig(num_nodes=100, feature_dim=16, num_patches=4, seed=0)
    g = generate_synthetic_graph(cfg)
    assert g.dataset.x.shape[1] == cfg.feature_dim


def test_edge_index_shape():
    cfg = SyntheticConfig(num_nodes=100, num_patches=4, seed=0)
    g = generate_synthetic_graph(cfg)
    assert g.dataset.edge_index.shape[0] == 2
    assert g.dataset.edge_index.max() < g.dataset.num_nodes
