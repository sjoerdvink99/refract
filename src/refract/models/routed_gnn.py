import torch
import torch.nn as nn
import torch.nn.functional as F

from refract.models.experts import AttributeExpert, TopologyExpert
from refract.models.gate import RoutingGate


class RoutedGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        regime_dim: int = 7,
        gate_hidden_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.attribute_expert = AttributeExpert(in_dim, hidden_dim, num_classes, dropout)
        self.topology_expert = TopologyExpert(in_dim, hidden_dim, num_classes, dropout)
        self.gate = RoutingGate(regime_dim, gate_hidden_dim, num_experts=2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        regime: torch.Tensor,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        out_attr = self.attribute_expert(x, edge_index)
        out_topo = self.topology_expert(x, edge_index)

        gate_weights, gate_logits = self.gate(regime, temperature=temperature)
        g_attr = gate_weights[:, 0:1]
        g_topo = gate_weights[:, 1:2]

        probs_attr = F.softmax(out_attr.logits, dim=-1)
        probs_topo = F.softmax(out_topo.logits, dim=-1)
        probs_raw = g_attr * probs_attr + g_topo * probs_topo
        probs = probs_raw / probs_raw.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        log_probs = torch.log(probs + 1e-10)

        return {
            "logits": log_probs,
            "probs": probs,
            "gate": gate_weights,
            "gate_logits": gate_logits,
            "expert_logits": torch.stack([out_attr.logits, out_topo.logits], dim=1),
            "expert_hidden": torch.stack([out_attr.hidden, out_topo.hidden], dim=1),
        }
