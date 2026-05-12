import numpy as np
import torch


def random_split(
    num_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)

    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    def _mask(idx: np.ndarray) -> torch.Tensor:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.from_numpy(idx)] = True
        return mask

    return _mask(train_idx), _mask(val_idx), _mask(test_idx)


def stratified_split(
    labels: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    num_nodes = len(labels)
    classes = np.unique(labels)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for c in classes:
        c_idx = np.where(labels == c)[0]
        c_idx = rng.permutation(c_idx)
        n = len(c_idx)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        train_idx.extend(c_idx[:n_train].tolist())
        val_idx.extend(c_idx[n_train : n_train + n_val].tolist())
        test_idx.extend(c_idx[n_train + n_val :].tolist())

    def _mask(idx: list[int]) -> torch.Tensor:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.tensor(idx, dtype=torch.long)] = True
        return mask

    return _mask(train_idx), _mask(val_idx), _mask(test_idx)
