import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/endpoints";

export function useProjection(runId: string | null) {
  return useQuery({
    queryKey: ["runs", runId, "projection"],
    queryFn: () => api.runs.projection(runId!),
    enabled: runId !== null,
    staleTime: 60_000,
  });
}
