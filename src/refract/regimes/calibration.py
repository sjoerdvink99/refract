import numpy as np
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score


def gate_calibration_l1(
    gate: np.ndarray,
    attribute_evidence: np.ndarray,
    topology_evidence: np.ndarray,
) -> float:
    denom = attribute_evidence + topology_evidence + 1e-8
    expected = np.stack([attribute_evidence / denom, topology_evidence / denom], axis=1)
    l1_error = np.abs(gate - expected).sum(axis=1).mean()
    return float(1.0 - l1_error)


def gate_calibration_rank(
    gate_attribute: np.ndarray,
    gate_topology: np.ndarray,
    attribute_evidence: np.ndarray,
    topology_evidence: np.ndarray,
) -> float:
    gate_diff = gate_attribute - gate_topology
    evidence_diff = attribute_evidence - topology_evidence
    corr, _ = stats.spearmanr(gate_diff, evidence_diff)
    return float(corr) if not np.isnan(corr) else 0.0


def counterfactual_routing_fidelity(
    gate_attribute: np.ndarray,
    gate_topology: np.ndarray,
    damage_attribute: np.ndarray,
    damage_topology: np.ndarray,
) -> float:
    gate_pref = gate_attribute - gate_topology
    damage_pref = damage_attribute - damage_topology
    correct = np.sign(gate_pref) == np.sign(damage_pref)
    return float(correct.mean())


def gate_smoothness(gate: np.ndarray, edge_index: np.ndarray) -> float:
    src = edge_index[0]
    dst = edge_index[1]
    diff = np.linalg.norm(gate[src] - gate[dst], axis=1)
    return float(diff.mean())


def structural_routing_consistency(
    dominant_expert_labels: np.ndarray,
    edge_index: np.ndarray,
) -> float:
    src = edge_index[0]
    dst = edge_index[1]
    same = dominant_expert_labels[src] == dominant_expert_labels[dst]
    return float(same.mean())


def gate_regime_mutual_information(
    dominant_expert: np.ndarray,
    regime_labels: np.ndarray,
) -> float:
    return float(normalized_mutual_info_score(regime_labels, dominant_expert))


def counterfactual_fidelity_by_regime(
    gate_attribute: np.ndarray,
    gate_topology: np.ndarray,
    damage_attribute: np.ndarray,
    damage_topology: np.ndarray,
    regime_labels: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for regime in np.unique(regime_labels):
        mask = regime_labels == regime
        if mask.sum() == 0:
            continue
        result[str(regime)] = counterfactual_routing_fidelity(
            gate_attribute[mask],
            gate_topology[mask],
            damage_attribute[mask],
            damage_topology[mask],
        )
    return result
