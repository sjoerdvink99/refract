from pydantic import BaseModel


class CounterfactualResult(BaseModel):
    node_id: str
    mode: str
    original_confidence: float
    counterfactual_confidence: float
    damage: float
    original_pred_label: str
    counterfactual_pred_label: str
    gate_attribute: float
    gate_topology: float


class NodeCounterfactuals(BaseModel):
    node_id: str
    results: list[CounterfactualResult]
