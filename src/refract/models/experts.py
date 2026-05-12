from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, SAGEConv
from torch_geometric.utils import scatter


class ExpertOutput(NamedTuple):
    logits: torch.Tensor
    hidden: torch.Tensor


class AttributeExpert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor | None = None
    ) -> ExpertOutput:
        residual = self.proj(x)
        h = self.dropout(F.relu(self.ln1(self.fc1(x))))
        h = self.dropout(F.relu(self.ln2(self.fc2(h) + residual)))
        return ExpertOutput(logits=self.head(h), hidden=h)


def _mean_neighbor(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    return scatter(x[src], dst, dim=0, dim_size=x.shape[0], reduce="mean")


class TopologyExpert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        gin_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv1 = GINConv(gin_mlp, train_eps=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.res_proj = nn.Linear(in_dim, hidden_dim)

        self.ego_lin = nn.Linear(in_dim, hidden_dim)
        self.ego_ln = nn.LayerNorm(hidden_dim)
        self.hp_lin = nn.Linear(in_dim, hidden_dim)
        self.hp_ln = nn.LayerNorm(hidden_dim)

        self.mix = nn.Linear(3 * hidden_dim, hidden_dim)
        self.mix_ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor | None = None
    ) -> ExpertOutput:
        assert edge_index is not None, "TopologyExpert requires edge_index"

        res = self.res_proj(x)
        h = self.dropout(F.relu(self.ln1(self.conv1(x, edge_index))))
        h = self.dropout(F.relu(self.ln2(self.conv2(h, edge_index) + h)))
        h = self.dropout(F.relu(self.ln3(self.conv3(h, edge_index) + h + res)))

        h_ego = F.relu(self.ego_ln(self.ego_lin(x)))

        nbr_mean = _mean_neighbor(x, edge_index)
        h_hp = F.relu(self.hp_ln(self.hp_lin(x - nbr_mean)))

        hidden = self.dropout(F.relu(self.mix_ln(self.mix(torch.cat([h, h_ego, h_hp], dim=-1)))))
        return ExpertOutput(logits=self.head(hidden), hidden=hidden)
