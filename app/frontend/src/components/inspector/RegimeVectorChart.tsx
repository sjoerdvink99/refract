import type { RegimeVector } from "@/api/types";
import { formatPercent } from "@/utils/format";

interface RegimeVectorChartProps {
  vector: RegimeVector;
}

const DIMS: { key: keyof RegimeVector; label: string; color: string }[] = [
  { key: "attribute_evidence", label: "Attr. Evidence", color: "#3B82F6" },
  { key: "topology_evidence", label: "Topo. Evidence", color: "#14B8A6" },
  { key: "concordance", label: "Concordance", color: "#22C55E" },
  { key: "uncertainty", label: "Uncertainty", color: "#F59E0B" },
  { key: "log_degree", label: "Log Degree", color: "#8B5CF6" },
  { key: "clustering_coeff", label: "Clustering", color: "#EC4899" },
  { key: "k2_topology_evidence", label: "2-hop Topo.", color: "#0EA5E9" },
  { key: "hop_consistency", label: "Hop Consistency", color: "#6B7280" },
];

export function RegimeVectorChart({ vector }: RegimeVectorChartProps) {
  return (
    <div className="flex flex-col gap-1.5">
      {DIMS.map(({ key, label, color }) => {
        const value = vector[key] ?? 0;
        return (
          <div key={key} className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-28 text-right">{label}</span>
            <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{ width: `${value * 100}%`, backgroundColor: color }}
              />
            </div>
            <span className="text-xs font-mono text-gray-700 w-10">
              {formatPercent(value)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
