import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/endpoints";

export function useTrainingHistory(runId: string | null) {
  return useQuery({
    queryKey: ["runs", runId, "training-history"],
    queryFn: () => api.analysis.trainingHistory(runId!),
    enabled: runId !== null,
    staleTime: 300_000,
  });
}
