import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/endpoints";

export function useExpertProfile(runId: string | null) {
  return useQuery({
    queryKey: ["runs", runId, "expert-profile"],
    queryFn: () => api.analysis.expertProfile(runId!),
    enabled: runId !== null,
    staleTime: 300_000,
  });
}
