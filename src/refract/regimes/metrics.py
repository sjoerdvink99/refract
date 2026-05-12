from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import normalize

from refract.datasets.base import GraphDataset


@dataclass
class RegimeVectors:
    attribute_evidence: np.ndarray
    topology_evidence: np.ndarray
    concordance: np.ndarray
    uncertainty: np.ndarray
    log_degree: np.ndarray
    clustering_coeff: np.ndarray
    k2_topology_evidence: np.ndarray
    hop_consistency: np.ndarray
    attribute_distribution: np.ndarray
    topology_distribution: np.ndarray
    feat_deviation_1hop: np.ndarray = None  # type: ignore[assignment]
    feat_deviation_2hop: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = len(self.attribute_evidence)
        scalars = [
            self.topology_evidence,
            self.concordance,
            self.uncertainty,
            self.log_degree,
            self.clustering_coeff,
            self.k2_topology_evidence,
            self.hop_consistency,
        ]
        matrices = [self.attribute_distribution, self.topology_distribution]
        if not all(len(a) == n for a in scalars):
            raise ValueError("All 1-D RegimeVectors arrays must have the same length")
        if not all(a.shape[0] == n for a in matrices):
            raise ValueError("Distribution matrices must have n rows")


def _build_adjacency(
    edge_index: np.ndarray, num_nodes: int
) -> tuple[dict[int, list[int]], np.ndarray]:
    adj: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    for e in range(edge_index.shape[1]):
        adj[int(edge_index[0, e])].append(int(edge_index[1, e]))
    degree = np.array([len(adj[i]) for i in range(num_nodes)], dtype=np.float32)
    return adj, degree


def compute_attribute_evidence(
    dataset: GraphDataset,
    train_mask: torch.Tensor,
    temperature: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    x = dataset.x.numpy()
    y = dataset.y.numpy()
    train_idx = train_mask.numpy().nonzero()[0]

    x_norm = normalize(x, norm="l2")
    num_classes = dataset.num_classes

    prototypes = np.zeros((num_classes, x.shape[1]))
    for c in range(num_classes):
        mask = y[train_idx] == c
        if mask.sum() > 0:
            prototypes[c] = x[train_idx[mask]].mean(axis=0)

    prototypes_norm = normalize(prototypes, norm="l2")
    similarities = x_norm @ prototypes_norm.T
    similarities = np.clip(similarities, -1.0, 1.0)

    logits = similarities / temperature
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    evidence = probs.max(axis=1)
    return evidence.astype(np.float32), probs.astype(np.float32)


def _compute_label_distribution(
    adj: dict[int, list[int]],
    labeled_set: set[int],
    labels: np.ndarray,
    num_nodes: int,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = 1.0 / num_classes
    evidence = np.zeros(num_nodes, dtype=np.float32)
    distributions = np.full((num_nodes, num_classes), 1.0 / num_classes, dtype=np.float32)

    for i in range(num_nodes):
        labeled_neighbors = [nb for nb in adj[i] if nb in labeled_set]
        counts = np.zeros(num_classes)
        for nb in labeled_neighbors:
            counts[labels[nb]] += 1
        total = len(labeled_neighbors) + num_classes * alpha
        dist = (counts + alpha) / total
        distributions[i] = dist.astype(np.float32)
        evidence[i] = dist.max()

    return evidence.astype(np.float32), distributions.astype(np.float32)


def compute_topology_evidence(
    dataset: GraphDataset,
    train_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    y = dataset.y.numpy()
    labeled_set = set(train_mask.numpy().nonzero()[0].tolist())
    adj, _ = _build_adjacency(dataset.edge_index.numpy(), dataset.num_nodes)
    return _compute_label_distribution(
        adj, labeled_set, y, dataset.num_nodes, dataset.num_classes
    )


def compute_k2_topology_evidence(
    dataset: GraphDataset,
    train_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    y = dataset.y.numpy()
    labeled_set = set(train_mask.numpy().nonzero()[0].tolist())
    adj, _ = _build_adjacency(dataset.edge_index.numpy(), dataset.num_nodes)

    adj2: dict[int, list[int]] = {}
    for i in range(dataset.num_nodes):
        two_hop: set[int] = set()
        for nb in adj[i]:
            two_hop.update(adj[nb])
        two_hop.discard(i)
        adj2[i] = list(two_hop)

    return _compute_label_distribution(
        adj2, labeled_set, y, dataset.num_nodes, dataset.num_classes
    )


def _jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    m = np.clip(0.5 * (p + q), 1e-10, 1.0)
    kl_p = (p * np.log(p / m)).sum(axis=1)
    kl_q = (q * np.log(q / m)).sum(axis=1)
    return np.clip(0.5 * kl_p + 0.5 * kl_q, 0.0, np.log(2))


def compute_concordance(
    attribute_distribution: np.ndarray,
    topology_distribution: np.ndarray,
) -> np.ndarray:
    jsd = _jsd(attribute_distribution, topology_distribution)
    return (1.0 - jsd / np.log(2)).astype(np.float32)


def compute_uncertainty(
    attribute_distribution: np.ndarray,
    topology_distribution: np.ndarray,
) -> np.ndarray:
    jsd = _jsd(attribute_distribution, topology_distribution)
    return (jsd / np.log(2)).astype(np.float32)


def compute_hop_consistency(
    topology_distribution: np.ndarray,
    k2_distribution: np.ndarray,
) -> np.ndarray:
    jsd = _jsd(topology_distribution, k2_distribution)
    return (1.0 - jsd / np.log(2)).astype(np.float32)


def compute_structural_features(
    dataset: GraphDataset,
) -> tuple[np.ndarray, np.ndarray]:
    ei = dataset.edge_index.numpy()
    n = dataset.num_nodes
    adj, degree = _build_adjacency(ei, n)

    max_degree = degree.max() if degree.max() > 0 else 1.0
    log_degree = np.log1p(degree) / np.log1p(max_degree)

    clustering = np.zeros(n, dtype=np.float32)
    for i in range(n):
        d = int(degree[i])
        if d < 2:
            continue
        neighbors = set(adj[i])
        triangles = sum(len(neighbors & set(adj[nb])) for nb in adj[i])
        clustering[i] = triangles / (d * (d - 1))

    return log_degree.astype(np.float32), clustering.astype(np.float32)


def compute_feat_deviation(
    dataset: GraphDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node normalized L2 distance from 1-hop and 2-hop neighbor feature mean.

    Captures how heterophilic each node is in feature space independently of labels,
    complementing the label-derived topology_evidence.  Used as gate input features
    following Node-MoE (arXiv 2406.03464).
    """
    x = dataset.x.numpy().astype(np.float32)
    ei = dataset.edge_index.numpy()
    n = dataset.num_nodes
    adj, _ = _build_adjacency(ei, n)

    node_norm = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)
    x_normed = x / node_norm

    def _mean_neighbor(adjacency: dict[int, list[int]]) -> np.ndarray:
        result = np.zeros_like(x_normed)
        for i in range(n):
            nbs = adjacency[i]
            if nbs:
                result[i] = x_normed[nbs].mean(axis=0)
            else:
                result[i] = x_normed[i]
        return result

    nbr1_mean = _mean_neighbor(adj)
    dev1 = np.linalg.norm(x_normed - nbr1_mean, axis=1).astype(np.float32)

    adj2: dict[int, list[int]] = {}
    for i in range(n):
        two_hop: set[int] = set()
        for nb in adj[i]:
            two_hop.update(adj[nb])
        two_hop.discard(i)
        adj2[i] = list(two_hop) if two_hop else adj[i]
    nbr2_mean = _mean_neighbor(adj2)
    dev2 = np.linalg.norm(x_normed - nbr2_mean, axis=1).astype(np.float32)

    max1 = dev1.max() if dev1.max() > 0 else 1.0
    max2 = dev2.max() if dev2.max() > 0 else 1.0
    return dev1 / max1, dev2 / max2


def compute_regime_vectors(
    dataset: GraphDataset,
    train_mask: torch.Tensor,
    temperature: float = 0.2,
) -> "RegimeVectors":
    attr_evidence, attr_dist = compute_attribute_evidence(dataset, train_mask, temperature)
    topo_evidence, topo_dist = compute_topology_evidence(dataset, train_mask)
    k2_evidence, k2_dist = compute_k2_topology_evidence(dataset, train_mask)

    concordance = compute_concordance(attr_dist, topo_dist)
    uncertainty = compute_uncertainty(attr_dist, topo_dist)
    hop_consistency = compute_hop_consistency(topo_dist, k2_dist)

    log_degree, clustering_coeff = compute_structural_features(dataset)
    feat_dev1, feat_dev2 = compute_feat_deviation(dataset)

    return RegimeVectors(
        attribute_evidence=attr_evidence,
        topology_evidence=topo_evidence,
        concordance=concordance,
        uncertainty=uncertainty,
        log_degree=log_degree,
        clustering_coeff=clustering_coeff,
        k2_topology_evidence=k2_evidence,
        hop_consistency=hop_consistency,
        attribute_distribution=attr_dist,
        topology_distribution=topo_dist,
        feat_deviation_1hop=feat_dev1,
        feat_deviation_2hop=feat_dev2,
    )


def regime_vectors_to_tensor(vectors: RegimeVectors) -> torch.Tensor:
    channels = [
        vectors.attribute_evidence,
        vectors.topology_evidence,
        vectors.concordance,
        vectors.log_degree,
        vectors.clustering_coeff,
        vectors.k2_topology_evidence,
        vectors.hop_consistency,
    ]
    if vectors.feat_deviation_1hop is not None:
        channels.append(vectors.feat_deviation_1hop)
    if vectors.feat_deviation_2hop is not None:
        channels.append(vectors.feat_deviation_2hop)
    stacked = np.stack(channels, axis=1)
    return torch.from_numpy(stacked).float()
