from pydantic import BaseModel


class GateWeights(BaseModel):
    attribute: float
    topology: float


class RegimeVector(BaseModel):
    attribute_evidence: float
    topology_evidence: float
    concordance: float
    uncertainty: float
    log_degree: float
    clustering_coeff: float
    k2_topology_evidence: float
    hop_consistency: float


class NodeMetrics(BaseModel):
    degree: int
    local_homophily: float | None
    gate_entropy: float


class CounterfactualSummary(BaseModel):
    mode: str
    original_confidence: float
    counterfactual_confidence: float
    damage: float
    original_pred_label: str
    counterfactual_pred_label: str
    gate: GateWeights


class NeighborInfo(BaseModel):
    node_id: str
    true_label: str | None
    regime: str
    degree: int


class NodeDetail(BaseModel):
    node_id: str
    index: int
    true_label: str | None
    predicted_label: str
    correct: bool | None
    confidence: float
    regime: str
    gate: GateWeights
    regime_vector: RegimeVector
    metrics: NodeMetrics
    neighbors: list[NeighborInfo]
    counterfactuals: list[CounterfactualSummary]
