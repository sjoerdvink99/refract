export type ExpertKey = "attribute" | "topology";

export type RegimeLabel =
  | "attribute_dominant"
  | "topology_dominant"
  | "concordant"
  | "conflict"
  | "uncertain";

export type ColorMode = "regime" | "expert" | "correctness";

export interface GateWeights {
  attribute: number;
  topology: number;
}

export interface RegimeVector {
  attribute_evidence: number;
  topology_evidence: number;
  concordance: number;
  uncertainty: number;
  log_degree: number;
  clustering_coeff: number;
  k2_topology_evidence: number;
  hop_consistency: number;
}

export interface NodeMetrics {
  degree: number;
  local_homophily: number | null;
  gate_entropy: number;
}

export interface CounterfactualSummary {
  mode: string;
  original_confidence: number;
  counterfactual_confidence: number;
  damage: number;
  original_pred_label: string;
  counterfactual_pred_label: string;
  gate: GateWeights;
}

export interface NeighborInfo {
  node_id: string;
  true_label: string | null;
  regime: RegimeLabel;
  degree: number;
}

export interface NodeDetail {
  node_id: string;
  index: number;
  true_label: string | null;
  predicted_label: string;
  correct: boolean | null;
  confidence: number;
  regime: RegimeLabel;
  gate: GateWeights;
  regime_vector: RegimeVector;
  metrics: NodeMetrics;
  neighbors: NeighborInfo[];
  counterfactuals: CounterfactualSummary[];
}

export interface ProjectionPoint {
  node_id: string;
  x: number;
  y: number;
  regime: RegimeLabel;
  dominant_expert: ExpertKey;
  correct: boolean | null;
  uncertainty: number;
  distortion: number;
  gate_attribute: number;
  gate_topology: number;
  pred_class: number;
  attribute_evidence: number;
  topology_evidence: number;
}

export interface ProjectionData {
  run_id: string;
  method: string;
  points: ProjectionPoint[];
}

export interface DatasetInfo {
  name: string;
  num_nodes: number;
  num_edges: number;
  num_classes: number;
  feature_dim: number;
}

export interface ModelInfo {
  name: string;
  hidden_dim: number;
  num_layers: number;
  experts: string[];
  regime_dim?: number;
}

export interface RunManifest {
  run_id: string;
  dataset: DatasetInfo;
  model: ModelInfo;
  files: Record<string, string | null>;
  config: Record<string, unknown> | null;
}

export interface RunSummary {
  run_id: string;
  dataset_name: string;
  num_nodes: number;
  num_classes: number;
}

export interface GraphNode {
  node_id: string;
  x: number;
  y: number;
  regime: RegimeLabel;
  dominant_expert: ExpertKey;
  correct: boolean | null;
  degree: number;
  gate_attribute: number;
  gate_topology: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  weight: number | null;
}

export interface EgoGraph {
  center_node_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  radius: number;
}

export interface RegimeDistribution {
  attribute_dominant: number;
  topology_dominant: number;
  concordant: number;
  conflict: number;
  uncertain: number;
}

export interface ExpertUsage {
  attribute: number;
  topology: number;
}

export interface RoutingMetrics {
  gate_calibration_l1: number;
  gate_calibration_rank: number;
  counterfactual_routing_fidelity: number;
  gate_smoothness: number;
  mean_gate_entropy: number;
  structural_routing_consistency: number;
  routing_regime_mi: number;
}

export interface PerformanceByRegime {
  regime: RegimeLabel;
  accuracy: number;
  macro_f1: number;
  count: number;
}

export interface RunMetrics {
  accuracy: number;
  macro_f1: number;
  regime_distribution: RegimeDistribution;
  expert_usage: ExpertUsage;
  routing: RoutingMetrics;
  by_regime: PerformanceByRegime[];
  model_comparison: Record<string, Record<string, number>> | null;
  per_expert_ablation: Record<string, Record<string, number>> | null;
}

export interface CounterfactualResult {
  node_id: string;
  mode: string;
  original_confidence: number;
  counterfactual_confidence: number;
  damage: number;
  original_pred_label: string;
  counterfactual_pred_label: string;
  gate_attribute: number;
  gate_topology: number;
}

export interface NodeCounterfactuals {
  node_id: string;
  results: CounterfactualResult[];
}

export interface Subgraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ExpertFeatureStats {
  mean: number;
  std: number;
  pop_mean: number;
  pop_std: number;
}

export interface ExpertProfileEntry {
  count: number;
  feature_stats: Record<string, ExpertFeatureStats>;
  gate_correlations: Record<string, number>;
}

export type ExpertProfile = Record<ExpertKey, ExpertProfileEntry>;

export interface TrainingHistoryEntry {
  epoch: number;
  gate_attribute_mean: number;
  gate_attribute_std: number;
  gate_topology_mean: number;
  gate_topology_std: number;
}

export interface SoftRegimeMembership {
  attribute_dominant: number;
  topology_dominant: number;
  concordant: number;
  conflict: number;
  uncertain: number;
}
