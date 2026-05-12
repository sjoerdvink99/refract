from pydantic import BaseModel


class ProjectionPoint(BaseModel):
    node_id: str
    x: float
    y: float
    regime: str
    dominant_expert: str
    correct: bool | None
    uncertainty: float
    distortion: float
    gate_attribute: float
    gate_topology: float
    pred_class: int
    attribute_evidence: float
    topology_evidence: float


class ProjectionData(BaseModel):
    run_id: str
    method: str
    points: list[ProjectionPoint]
