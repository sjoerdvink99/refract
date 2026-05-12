import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP as APPNPProp
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class MLPBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.net(x)
        return {"logits": logits, "probs": F.softmax(logits, dim=-1)}


class GCNBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)
        logits = self.head(h)
        return {"logits": logits, "probs": F.softmax(logits, dim=-1)}


class GraphSAGEBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        h = F.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, edge_index))
        h = self.dropout(h)
        logits = self.head(h)
        return {"logits": logits, "probs": F.softmax(logits, dim=-1)}


class GATBaseline(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.3, heads: int = 4
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        h = F.elu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = F.elu(self.conv2(h, edge_index))
        h = self.dropout(h)
        logits = self.head(h)
        return {"logits": logits, "probs": F.softmax(logits, dim=-1)}


class APPNPBaseline(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
        K: int = 10,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.prop = APPNPProp(K=K, alpha=alpha)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        h = F.relu(self.lin1(x))
        h = self.dropout(h)
        h = self.lin2(h)
        logits = self.prop(h, edge_index)
        return {"logits": logits, "probs": F.softmax(logits, dim=-1)}
