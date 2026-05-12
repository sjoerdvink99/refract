from dataclasses import dataclass, field

import torch


@dataclass
class GraphDataset:
    name: str
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    node_ids: list[str]
    feature_names: list[str] | None = None
    class_names: list[str] | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.x.shape[0]
        if self.y.shape[0] != n:
            raise ValueError(f"x has {n} nodes but y has {self.y.shape[0]}")
        if len(self.node_ids) != n:
            raise ValueError(f"x has {n} nodes but node_ids has {len(self.node_ids)}")
        if self.train_mask.shape[0] != n:
            raise ValueError("train_mask size mismatch")
        if self.val_mask.shape[0] != n:
            raise ValueError("val_mask size mismatch")
        if self.test_mask.shape[0] != n:
            raise ValueError("test_mask size mismatch")
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1
