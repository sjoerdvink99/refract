import torch_geometric.transforms as T
from torch_geometric.datasets import Actor, HeterophilousGraphDataset, Planetoid, WebKB, WikipediaNetwork

from refract.datasets.base import GraphDataset
from refract.datasets.splits import random_split, stratified_split

_HETEROPHILOUS_NAMES = {
    "roman-empire": "Roman-empire",
    "amazon-ratings": "Amazon-ratings",
    "minesweeper": "Minesweeper",
    "tolokers": "Tolokers",
    "questions": "Questions",
}

_BINARY_DATASETS = frozenset({"minesweeper", "tolokers", "questions"})


def _to_graph_dataset(name: str, data, num_classes: int, split_idx: int) -> GraphDataset:
    if data.train_mask.dim() > 1:
        train_mask = data.train_mask[:, split_idx].bool()
        val_mask = data.val_mask[:, split_idx].bool()
        test_mask = data.test_mask[:, split_idx].bool()
    else:
        train_mask = data.train_mask.bool()
        val_mask = data.val_mask.bool()
        test_mask = data.test_mask.bool()

    n = data.x.shape[0]
    return GraphDataset(
        name=name,
        x=data.x,
        edge_index=data.edge_index,
        y=data.y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        node_ids=[str(i) for i in range(n)],
        feature_names=[f"f{i}" for i in range(data.x.shape[1])],
        class_names=[f"c{i}" for i in range(num_classes)],
        metadata={"source": "real_world"},
    )


def load_real_world_dataset(name: str, root: str = "data", split_idx: int = 0) -> GraphDataset:
    import numpy as np
    transform = T.NormalizeFeatures()
    name = name.lower()

    if name in ["cora", "citeseer", "pubmed"]:
        pyg_ds = Planetoid(root=root, name=name.capitalize(), transform=transform)
        data = pyg_ds[0]
        train_mask, val_mask, test_mask = random_split(data.x.shape[0], seed=split_idx)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return _to_graph_dataset(name, data, pyg_ds.num_classes, split_idx)

    if name in ["texas", "cornell", "wisconsin"]:
        pyg_ds = WebKB(root=root, name=name.capitalize(), transform=transform)
        return _to_graph_dataset(name, pyg_ds[0], pyg_ds.num_classes, split_idx)

    if name in ["chameleon", "squirrel"]:
        pyg_ds = WikipediaNetwork(root=root, name=name, geom_gcn_preprocess=True, transform=transform)
        return _to_graph_dataset(name, pyg_ds[0], pyg_ds.num_classes, split_idx)

    if name in _HETEROPHILOUS_NAMES:
        pyg_name = _HETEROPHILOUS_NAMES[name]
        pyg_ds = HeterophilousGraphDataset(root=root, name=pyg_name, transform=transform)
        data = pyg_ds[0]
        if name in _BINARY_DATASETS:
            labels = data.y.numpy()
            train_mask, val_mask, test_mask = stratified_split(labels, seed=split_idx)
            data.train_mask = train_mask
            data.val_mask = val_mask
            data.test_mask = test_mask
        return _to_graph_dataset(name, data, pyg_ds.num_classes, split_idx)

    if name == "actor":
        pyg_ds = Actor(root=root, transform=transform)
        data = pyg_ds[0]
        train_mask, val_mask, test_mask = random_split(data.x.shape[0], seed=split_idx)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return _to_graph_dataset(name, data, pyg_ds.num_classes, split_idx)

    raise ValueError(f"Dataset '{name}' not supported.")
