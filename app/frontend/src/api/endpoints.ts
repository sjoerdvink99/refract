import { get } from "./client";
import type {
  EgoGraph,
  ExpertProfile,
  NodeCounterfactuals,
  NodeDetail,
  ProjectionData,
  RunManifest,
  RunMetrics,
  RunSummary,
  Subgraph,
  TrainingHistoryEntry,
} from "./types";

export const api = {
  runs: {
    list: () => get<RunSummary[]>("/runs"),
    manifest: (runId: string) => get<RunManifest>(`/runs/${runId}/manifest`),
    projection: (runId: string) => get<ProjectionData>(`/runs/${runId}/projection`),
    metrics: (runId: string) => get<RunMetrics>(`/runs/${runId}/metrics`),
  },
  nodes: {
    detail: (runId: string, nodeId: string) =>
      get<NodeDetail>(`/runs/${runId}/nodes/${nodeId}`),
  },
  graph: {
    ego: (runId: string, nodeId: string, radius = 1) =>
      get<EgoGraph>(`/runs/${runId}/graph/ego/${nodeId}?radius=${radius}`),
    subgraph: (runId: string, nodeIds: string[]) =>
      get<Subgraph>(`/runs/${runId}/graph/subgraph?node_ids=${nodeIds.join(",")}`),
  },
  counterfactuals: {
    node: (runId: string, nodeId: string) =>
      get<NodeCounterfactuals>(`/runs/${runId}/counterfactuals/${nodeId}`),
  },
  analysis: {
    expertProfile: (runId: string) =>
      get<ExpertProfile>(`/runs/${runId}/expert-profile`),
    trainingHistory: (runId: string) =>
      get<TrainingHistoryEntry[]>(`/runs/${runId}/training/history`),
  },
};
