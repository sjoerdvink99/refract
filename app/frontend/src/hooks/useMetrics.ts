import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/endpoints";

export function useMetrics(runId: string | null) {
  return useQuery({
    queryKey: ["runs", runId, "metrics"],
    queryFn: () => api.runs.metrics(runId!),
    enabled: runId !== null,
    staleTime: 60_000,
  });
}
