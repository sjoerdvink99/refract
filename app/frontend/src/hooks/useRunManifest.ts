import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/endpoints";

export function useRuns() {
  return useQuery({
    queryKey: ["runs"],
    queryFn: api.runs.list,
    staleTime: 30_000,
  });
}

export function useRunManifest(runId: string | null) {
  return useQuery({
    queryKey: ["runs", runId, "manifest"],
    queryFn: () => api.runs.manifest(runId!),
    enabled: runId !== null,
    staleTime: 60_000,
  });
}
