import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_distortion(
    high_dim: np.ndarray,
    low_dim: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    nn_high = NearestNeighbors(n_neighbors=k + 1).fit(high_dim)
    nn_low = NearestNeighbors(n_neighbors=k + 1).fit(low_dim)

    _, idx_high = nn_high.kneighbors(high_dim)
    _, idx_low = nn_low.kneighbors(low_dim)

    idx_high = idx_high[:, 1:]
    idx_low = idx_low[:, 1:]

    n = high_dim.shape[0]
    intersection = np.array([
        len(np.intersect1d(idx_high[i], idx_low[i], assume_unique=True))
        for i in range(n)
    ], dtype=np.float32)
    union = 2.0 * k - intersection

    return (1.0 - intersection / (union + 1e-8)).astype(np.float32)
