from pydantic import BaseModel


class DatasetInfo(BaseModel):
    name: str
    num_nodes: int
    num_edges: int
    num_classes: int
    feature_dim: int


class ModelInfo(BaseModel):
    name: str
    hidden_dim: int
    num_layers: int
    experts: list[str]
    regime_dim: int = 7


class RunFiles(BaseModel):
    nodes: str
    edges: str
    projection: str
    metrics: str
    counterfactuals: str | None = None
    expert_embeddings: str | None = None
    training_history: str | None = None


class RunManifest(BaseModel):
    run_id: str
    dataset: DatasetInfo
    model: ModelInfo
    files: RunFiles
    config: dict | None = None


class RunSummary(BaseModel):
    run_id: str
    dataset_name: str
    num_nodes: int
    num_classes: int
