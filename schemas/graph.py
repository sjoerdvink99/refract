from pydantic import BaseModel


class GraphNode(BaseModel):
    node_id: str
    x: float
    y: float
    regime: str
    dominant_expert: str
    correct: bool | None
    degree: int
    gate_attribute: float
    gate_topology: float


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float | None = None


class EgoGraph(BaseModel):
    center_node_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    radius: int


class Subgraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
