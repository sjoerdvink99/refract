import networkx as nx
import torch
from sklearn.preprocessing import StandardScaler


def normalize_features(x: torch.Tensor) -> torch.Tensor:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.numpy())
    return torch.from_numpy(x_scaled).float()


def add_structural_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    G: nx.Graph = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.t().tolist())

    degree = torch.tensor([G.degree(i) for i in range(num_nodes)], dtype=torch.float)
    log_degree = torch.log1p(degree)
    clustering = torch.tensor(
        [nx.clustering(G, i) for i in range(num_nodes)], dtype=torch.float
    )
    pagerank_dict = nx.pagerank(G)
    pagerank = torch.tensor(
        [pagerank_dict[i] for i in range(num_nodes)], dtype=torch.float
    )

    struct_feats = torch.stack([degree, log_degree, clustering, pagerank], dim=1)
    return torch.cat([x, struct_feats], dim=1)
