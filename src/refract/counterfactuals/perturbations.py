from enum import Enum

import torch


class PerturbationMode(str, Enum):
    MEAN_FEATURE_MASK = "mean_feature_mask"
    ZERO_FEATURE_MASK = "zero_feature_mask"
    EGO_EDGE_MASK = "ego_edge_mask"
    ATTRIBUTE_EXPERT_ONLY = "attribute_expert_only"
    TOPOLOGY_EXPERT_ONLY = "topology_expert_only"
    REGIME_ABLATION = "regime_ablation"


def apply_feature_mask(
    x: torch.Tensor,
    node_idx: int,
    mode: PerturbationMode = PerturbationMode.MEAN_FEATURE_MASK,
) -> torch.Tensor:
    x_perturbed = x.clone()
    if mode == PerturbationMode.ZERO_FEATURE_MASK:
        x_perturbed[node_idx] = 0.0
    else:
        x_perturbed[node_idx] = x.mean(dim=0)
    return x_perturbed


def apply_ego_edge_mask(
    edge_index: torch.Tensor, node_idx: int, radius: int = 1
) -> torch.Tensor:
    src = edge_index[0]
    dst = edge_index[1]
    mask = (src != node_idx) & (dst != node_idx)
    return edge_index[:, mask]


def apply_regime_ablation(
    regime: torch.Tensor,
    node_idx: int,
    population_mean: torch.Tensor,
) -> torch.Tensor:
    r = regime.clone()
    r[node_idx] = population_mean
    return r
