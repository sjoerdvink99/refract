import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/endpoints";

export function useNodeDetail(runId: string | null, nodeId: string | null) {
  return useQuery({
    queryKey: ["runs", runId, "nodes", nodeId],
    queryFn: () => api.nodes.detail(runId!, nodeId!),
    enabled: runId !== null && nodeId !== null,
    staleTime: 60_000,
  });
}

export function useEgoGraph(runId: string | null, nodeId: string | null, radius: 1 | 2 = 1) {
  return useQuery({
    queryKey: ["runs", runId, "ego", nodeId, radius],
    queryFn: () => api.graph.ego(runId!, nodeId!, radius),
    enabled: runId !== null && nodeId !== null,
    staleTime: 60_000,
  });
}

export function useSubgraph(runId: string | null, nodeIds: string[]) {
  const sortedKey = nodeIds.slice().sort().join(",");
  return useQuery({
    queryKey: ["runs", runId, "subgraph", sortedKey],
    queryFn: () => api.graph.subgraph(runId!, nodeIds),
    enabled: runId !== null && nodeIds.length > 1,
    staleTime: 60_000,
  });
}
