import torch
import torch.nn.functional as F


def prediction_loss(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.nll_loss(log_probs[mask], labels[mask], weight=weight)


def gate_alignment_loss(
    gate: torch.Tensor,
    attribute_evidence: torch.Tensor,
    topology_evidence: torch.Tensor,
) -> torch.Tensor:
    denom = attribute_evidence + topology_evidence + 1e-8
    target = torch.stack([attribute_evidence / denom, topology_evidence / denom], dim=1)
    gate_norm = gate / gate.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return (target * torch.log((target + 1e-10) / (gate_norm + 1e-10))).sum(dim=-1).mean()


def regime_consistency_loss(
    gate: torch.Tensor, edge_index: torch.Tensor, regime: torch.Tensor
) -> torch.Tensor:
    src = edge_index[0]
    dst = edge_index[1]
    regime_sim = F.cosine_similarity(regime[src], regime[dst], dim=-1).clamp(0.0, 1.0)
    diff = (gate[src] - gate[dst]).pow(2).sum(dim=-1)
    return (regime_sim * diff).mean()


def expert_diversity_loss(h_attr: torch.Tensor, h_topo: torch.Tensor) -> torch.Tensor:
    h_a = F.normalize(h_attr, dim=-1)
    h_t = F.normalize(h_topo, dim=-1)
    return F.cosine_similarity(h_a, h_t, dim=-1).clamp(min=0).mean()


def load_balance_loss(gate: torch.Tensor) -> torch.Tensor:
    num_experts = gate.shape[1]
    P = gate.mean(dim=0)
    return num_experts * (P * P).sum()


def gate_entropy_loss(gate: torch.Tensor) -> torch.Tensor:
    return (gate * torch.log(gate + 1e-10)).sum(dim=-1).mean()


def gate_z_loss(gate_logits: torch.Tensor) -> torch.Tensor:
    return (torch.logsumexp(gate_logits, dim=-1) ** 2).mean()


def total_loss(
    log_probs: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    gate: torch.Tensor,
    attribute_evidence: torch.Tensor | None = None,
    topology_evidence: torch.Tensor | None = None,
    lambda_align: float = 0.0,
    lambda_smooth: float = 0.0,
    lambda_diversity: float = 0.0,
    lambda_load: float = 0.0,
    edge_index: torch.Tensor | None = None,
    regime: torch.Tensor | None = None,
    h_attr: torch.Tensor | None = None,
    h_topo: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    lambda_entropy: float = 0.0,
    gate_logits: torch.Tensor | None = None,
    lambda_z_loss: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = prediction_loss(log_probs, labels, train_mask, weight=class_weights)
    components: dict[str, float] = {"pred": loss.item()}

    if lambda_align > 0 and attribute_evidence is not None and topology_evidence is not None:
        align_l = gate_alignment_loss(gate, attribute_evidence, topology_evidence)
        loss = loss + lambda_align * align_l
        components["gate_align"] = align_l.item()

    if lambda_smooth > 0 and edge_index is not None and regime is not None:
        smooth_l = regime_consistency_loss(gate, edge_index, regime)
        loss = loss + lambda_smooth * smooth_l
        components["gate_smooth"] = smooth_l.item()

    if lambda_diversity > 0 and h_attr is not None and h_topo is not None:
        div_l = expert_diversity_loss(h_attr, h_topo)
        loss = loss + lambda_diversity * div_l
        components["expert_diversity"] = div_l.item()

    if lambda_load > 0:
        lb_l = load_balance_loss(gate)
        loss = loss + lambda_load * lb_l
        components["load_balance"] = lb_l.item()

    if lambda_entropy > 0:
        ent_l = gate_entropy_loss(gate)
        loss = loss + lambda_entropy * ent_l
        components["gate_entropy"] = ent_l.item()

    if lambda_z_loss > 0 and gate_logits is not None:
        z_l = gate_z_loss(gate_logits)
        loss = loss + lambda_z_loss * z_l
        components["z_loss"] = z_l.item()

    return loss, components
