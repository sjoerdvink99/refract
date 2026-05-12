from pydantic import BaseModel


class RegimeDistribution(BaseModel):
    attribute_dominant: float
    topology_dominant: float
    concordant: float
    conflict: float
    uncertain: float


class ExpertUsage(BaseModel):
    attribute: float
    topology: float


class RoutingMetrics(BaseModel):
    gate_calibration_l1: float
    gate_calibration_rank: float
    counterfactual_routing_fidelity: float
    gate_smoothness: float
    mean_gate_entropy: float
    structural_routing_consistency: float
    routing_regime_mi: float = 0.0


class PerformanceByRegime(BaseModel):
    regime: str
    accuracy: float
    macro_f1: float
    count: int


class RunMetrics(BaseModel):
    accuracy: float
    macro_f1: float
    regime_distribution: RegimeDistribution
    expert_usage: ExpertUsage
    routing: RoutingMetrics
    by_regime: list[PerformanceByRegime]
    model_comparison: dict[str, dict[str, float]] | None = None
    per_expert_ablation: dict[str, dict[str, float]] | None = None
