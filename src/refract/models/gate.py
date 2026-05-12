import torch
import torch.nn as nn


class RoutingGate(nn.Module):
    def __init__(
        self,
        regime_dim: int = 7,
        hidden_dim: int = 32,
        num_experts: int = 2,
        gate_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(regime_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, regime: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(regime) / temperature
        return torch.sigmoid(raw), raw
