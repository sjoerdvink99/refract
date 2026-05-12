from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from refract.counterfactuals.perturbations import (
    PerturbationMode,
    apply_ego_edge_mask,
    apply_feature_mask,
    apply_regime_ablation,
)
from refract.datasets.base import GraphDataset

_EXPERT_ABLATION_MAP = {
    PerturbationMode.ATTRIBUTE_EXPERT_ONLY: [1],
    PerturbationMode.TOPOLOGY_EXPERT_ONLY: [0],
}


@dataclass
class CounterfactualRecord:
    node_idx: int
    mode: str
    original_confidence: float
    counterfactual_confidence: float
    damage: float
    original_pred_label: int
    counterfactual_pred_label: int
    gate_attribute: float
    gate_topology: float
    regime_shift: bool = False


def _infer(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    regime: torch.Tensor,
    node_idx: int,
    true_label: int,
) -> tuple[float, int, np.ndarray]:
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, regime)
        probs = out["probs"][node_idx].cpu().numpy()
        pred = int(probs.argmax())
        confidence = float(probs[true_label])
        gate = out["gate"][node_idx].cpu().numpy()
    return confidence, pred, gate


def _infer_ablated_gate(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    regime: torch.Tensor,
    node_idx: int,
    true_label: int,
    ablate_experts: list[int],
) -> tuple[float, int, np.ndarray]:
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, regime)
        gate = out["gate"].clone()
        for e in ablate_experts:
            gate[:, e] = 0.0
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        expert_logits = out["expert_logits"]
        logits = (gate.unsqueeze(-1) * expert_logits).sum(dim=1)
        probs = F.softmax(logits, dim=-1)

        p = probs[node_idx].cpu().numpy()
        pred = int(p.argmax())
        confidence = float(p[true_label])
        g = gate[node_idx].cpu().numpy()
    return confidence, pred, g


def compute_node_counterfactuals(
    model: torch.nn.Module,
    dataset: GraphDataset,
    regime: torch.Tensor,
    node_idx: int,
    modes: list[PerturbationMode] | None = None,
    device: torch.device | None = None,
) -> list[CounterfactualRecord]:
    if modes is None:
        modes = [
            PerturbationMode.MEAN_FEATURE_MASK,
            PerturbationMode.EGO_EDGE_MASK,
            PerturbationMode.ATTRIBUTE_EXPERT_ONLY,
            PerturbationMode.TOPOLOGY_EXPERT_ONLY,
        ]

    dev = device or torch.device("cpu")
    x = dataset.x.to(dev)
    edge_index = dataset.edge_index.to(dev)
    regime_d = regime.to(dev)
    true_label = int(dataset.y[node_idx].item())

    orig_conf, orig_pred, orig_gate = _infer(model, x, edge_index, regime_d, node_idx, true_label)

    records: list[CounterfactualRecord] = []
    for mode in modes:
        if mode in (PerturbationMode.MEAN_FEATURE_MASK, PerturbationMode.ZERO_FEATURE_MASK):
            x_cf = apply_feature_mask(x, node_idx, mode)
            cf_conf, cf_pred, cf_gate = _infer(model, x_cf, edge_index, regime_d, node_idx, true_label)

        elif mode == PerturbationMode.EGO_EDGE_MASK:
            ei_cf = apply_ego_edge_mask(edge_index, node_idx)
            cf_conf, cf_pred, cf_gate = _infer(model, x, ei_cf, regime_d, node_idx, true_label)

        elif mode in _EXPERT_ABLATION_MAP:
            ablate = _EXPERT_ABLATION_MAP[mode]
            cf_conf, cf_pred, cf_gate = _infer_ablated_gate(
                model, x, edge_index, regime_d, node_idx, true_label, ablate
            )

        elif mode == PerturbationMode.REGIME_ABLATION:
            pop_mean = regime_d.mean(dim=0)
            r_cf = apply_regime_ablation(regime_d, node_idx, pop_mean)
            cf_conf, cf_pred, cf_gate = _infer(model, x, edge_index, r_cf, node_idx, true_label)

        else:
            continue

        records.append(CounterfactualRecord(
            node_idx=node_idx,
            mode=mode.value,
            original_confidence=orig_conf,
            counterfactual_confidence=cf_conf,
            damage=orig_conf - cf_conf,
            original_pred_label=orig_pred,
            counterfactual_pred_label=cf_pred,
            gate_attribute=float(cf_gate[0]),
            gate_topology=float(cf_gate[1]),
            regime_shift=bool(cf_gate.argmax() != orig_gate.argmax()),
        ))

    return records


def compute_all_counterfactuals(
    model: torch.nn.Module,
    dataset: GraphDataset,
    regime: torch.Tensor,
    node_indices: list[int] | None = None,
    modes: list[PerturbationMode] | None = None,
    device: torch.device | None = None,
) -> list[CounterfactualRecord]:
    indices = node_indices if node_indices is not None else list(range(dataset.num_nodes))
    return [
        record
        for idx in indices
        for record in compute_node_counterfactuals(model, dataset, regime, idx, modes, device)
    ]
