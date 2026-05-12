from dataclasses import dataclass

import numpy as np
import torch

from refract.datasets.base import GraphDataset
from refract.datasets.splits import random_split


@dataclass
class PatchConfig:
    regime: str
    size: int
    structural_homophily: float
    feature_homophily: float
    conflict_rate: float
    intra_edge_prob: float = 0.04


@dataclass
class SyntheticConfig:
    num_nodes: int = 3000
    num_classes: int = 3
    feature_dim: int = 32
    num_patches: int = 12
    inter_patch_edge_prob: float = 0.002
    label_noise: float = 0.05
    feature_noise: float = 0.4
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    seed: int = 0
    patch_configs: list[PatchConfig] | None = None


@dataclass
class SyntheticGraph:
    dataset: GraphDataset
    patch_assignments: np.ndarray
    true_regimes: np.ndarray
    patch_configs: list[PatchConfig]


def build_default_patch_configs(
    num_patches: int, num_classes: int, patch_size: int
) -> list[PatchConfig]:
    regimes = [
        "concordant",
        "attribute_dominant",
        "topology_dominant",
        "conflict",
        "uncertain",
    ]
    configs = []
    for i in range(num_patches):
        regime = regimes[i % len(regimes)]
        if regime == "concordant":
            cfg = PatchConfig(
                regime=regime,
                size=patch_size,
                structural_homophily=0.85,
                feature_homophily=0.85,
                conflict_rate=0.0,
            )
        elif regime == "attribute_dominant":
            cfg = PatchConfig(
                regime=regime,
                size=patch_size,
                structural_homophily=0.2,
                feature_homophily=0.85,
                conflict_rate=0.05,
            )
        elif regime == "topology_dominant":
            cfg = PatchConfig(
                regime=regime,
                size=patch_size,
                structural_homophily=0.85,
                feature_homophily=0.2,
                conflict_rate=0.05,
            )
        elif regime == "conflict":
            cfg = PatchConfig(
                regime=regime,
                size=patch_size,
                structural_homophily=0.7,
                feature_homophily=0.3,
                conflict_rate=0.6,
            )
        else:
            cfg = PatchConfig(
                regime=regime,
                size=patch_size,
                structural_homophily=0.4,
                feature_homophily=0.4,
                conflict_rate=0.3,
            )
        configs.append(cfg)
    return configs


def _generate_features(
    labels: np.ndarray,
    feature_dim: int,
    feature_homophily: float,
    feature_noise: float,
    rng: np.random.Generator,
    num_classes: int,
) -> np.ndarray:
    n = len(labels)
    class_centers = rng.standard_normal((num_classes, feature_dim))
    features = np.zeros((n, feature_dim))
    for i, label in enumerate(labels):
        if rng.random() < feature_homophily:
            center = class_centers[label]
        else:
            other = rng.integers(0, num_classes - 1)
            if other >= label:
                other += 1
            center = class_centers[other]
        features[i] = center + rng.standard_normal(feature_dim) * feature_noise
    return features.astype(np.float32)


def _add_patch_edges(
    src_list: list[int],
    dst_list: list[int],
    patch_nodes: np.ndarray,
    patch_labels: np.ndarray,
    structural_homophily: float,
    intra_edge_prob: float,
    rng: np.random.Generator,
) -> None:
    n = len(patch_nodes)
    for i in range(n):
        for j in range(i + 1, n):
            same_class = patch_labels[i] == patch_labels[j]
            prob = intra_edge_prob
            if same_class and rng.random() < structural_homophily:
                prob *= 3
            if rng.random() < prob:
                src_list.append(int(patch_nodes[i]))
                dst_list.append(int(patch_nodes[j]))
                src_list.append(int(patch_nodes[j]))
                dst_list.append(int(patch_nodes[i]))


def generate_synthetic_graph(cfg: SyntheticConfig) -> SyntheticGraph:
    rng = np.random.default_rng(cfg.seed)

    patch_size = cfg.num_nodes // cfg.num_patches
    patch_configs = cfg.patch_configs or build_default_patch_configs(
        cfg.num_patches, cfg.num_classes, patch_size
    )

    all_labels = []
    all_features = []
    patch_assignments = []
    true_regimes = []
    offset = 0

    for p_idx, p_cfg in enumerate(patch_configs):
        n = p_cfg.size
        labels = rng.integers(0, cfg.num_classes, size=n)

        if p_cfg.conflict_rate > 0:
            conflict_mask = rng.random(n) < p_cfg.conflict_rate
            offsets = rng.integers(1, cfg.num_classes, size=n)
            labels[conflict_mask] = (labels[conflict_mask] + offsets[conflict_mask]) % cfg.num_classes

        features = _generate_features(
            labels, cfg.feature_dim, p_cfg.feature_homophily, cfg.feature_noise, rng, cfg.num_classes
        )

        noise_mask = rng.random(n) < cfg.label_noise
        labels[noise_mask] = rng.integers(0, cfg.num_classes, size=noise_mask.sum())

        all_labels.append(labels)
        all_features.append(features)
        patch_assignments.extend([p_idx] * n)
        true_regimes.extend([p_cfg.regime] * n)
        offset += n

    remaining = cfg.num_nodes - sum(p.size for p in patch_configs)
    if remaining > 0:
        extra_labels = rng.integers(0, cfg.num_classes, size=remaining)
        extra_features = rng.standard_normal((remaining, cfg.feature_dim)).astype(np.float32)
        all_labels.append(extra_labels)
        all_features.append(extra_features)
        patch_assignments.extend([len(patch_configs) - 1] * remaining)
        true_regimes.extend(["uncertain"] * remaining)

    labels_np = np.concatenate(all_labels)
    features_np = np.concatenate(all_features, axis=0)
    patch_assignments_np = np.array(patch_assignments)
    true_regimes_np = np.array(true_regimes)

    src_list: list[int] = []
    dst_list: list[int] = []

    for p_idx, p_cfg in enumerate(patch_configs):
        patch_nodes = np.where(patch_assignments_np == p_idx)[0]
        patch_labels = labels_np[patch_nodes]
        _add_patch_edges(
            src_list, dst_list, patch_nodes, patch_labels,
            p_cfg.structural_homophily, p_cfg.intra_edge_prob, rng
        )

    num_nodes = len(labels_np)
    num_inter = int(num_nodes * cfg.num_patches * cfg.inter_patch_edge_prob)
    for _ in range(num_inter):
        i, j = rng.integers(0, num_nodes, size=2)
        if i != j:
            src_list.append(int(i))
            dst_list.append(int(j))

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    x = torch.from_numpy(features_np)
    y = torch.from_numpy(labels_np).long()

    train_mask, val_mask, test_mask = random_split(
        num_nodes, cfg.train_ratio, cfg.val_ratio, cfg.seed
    )

    node_ids = [f"n{i}" for i in range(num_nodes)]
    class_names = [f"class_{c}" for c in range(cfg.num_classes)]

    dataset = GraphDataset(
        name="synthetic_local_regime",
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        node_ids=node_ids,
        class_names=class_names,
        metadata={
            "seed": cfg.seed,
            "num_patches": cfg.num_patches,
            "feature_noise": cfg.feature_noise,
            "label_noise": cfg.label_noise,
        },
    )

    return SyntheticGraph(
        dataset=dataset,
        patch_assignments=patch_assignments_np,
        true_regimes=true_regimes_np,
        patch_configs=patch_configs,
    )
