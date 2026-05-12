import { useProjection } from "@/hooks/useProjection";
import { useSelectionStore } from "@/state/selectionStore";
import { GateWeightChart } from "./GateWeightChart";
import { REGIME_COLORS } from "@/utils/colors";
import { formatFloat, formatPercent, formatRegimeLabel } from "@/utils/format";
import type { RegimeLabel } from "@/api/types";

const REGIME_ORDER: RegimeLabel[] = [
  "attribute_dominant",
  "topology_dominant",
  "concordant",
  "conflict",
  "uncertain",
];

export function SelectionInspector() {
  const { activeRunId, selectedNodeIds } = useSelectionStore();
  const { data: projection } = useProjection(activeRunId);

  if (!projection) return null;

  const selectedSet = new Set(selectedNodeIds);
  const pts = projection.points.filter((p) => selectedSet.has(p.node_id));

  if (pts.length === 0) return null;

  const regimeCounts = new Map<RegimeLabel, number>();
  for (const pt of pts) {
    regimeCounts.set(pt.regime, (regimeCounts.get(pt.regime) ?? 0) + 1);
  }

  const withLabel = pts.filter((p) => p.correct !== null);
  const accuracy =
    withLabel.length > 0
      ? withLabel.filter((p) => p.correct).length / withLabel.length
      : null;

  const avgGate = {
    attribute: pts.reduce((s, p) => s + p.gate_attribute, 0) / pts.length,
    topology: pts.reduce((s, p) => s + p.gate_topology, 0) / pts.length,
  };

  const avgUncertainty = pts.reduce((s, p) => s + p.uncertainty, 0) / pts.length;

  return (
    <div className="p-3 flex flex-col gap-4 text-sm">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="text-xs text-gray-500">
          <span className="font-mono text-gray-800">{pts.length}</span> nodes selected
        </span>
        {accuracy !== null && (
          <span className="text-xs text-gray-500">
            Accuracy:{" "}
            <span className="font-mono text-gray-800">{formatPercent(accuracy)}</span>
          </span>
        )}
        <span className="text-xs text-gray-500">
          Uncertainty:{" "}
          <span className="font-mono text-gray-800">{formatFloat(avgUncertainty)}</span>
        </span>
      </div>

      <div>
        <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
          Regime Distribution
        </span>
        <div className="mt-1.5 flex flex-col gap-1">
          {REGIME_ORDER.filter((r) => regimeCounts.has(r)).map((regime) => {
            const count = regimeCounts.get(regime)!;
            const pct = count / pts.length;
            return (
              <div key={regime} className="flex items-center gap-2">
                <span className="text-xs text-gray-500 w-28 text-right shrink-0">
                  {formatRegimeLabel(regime)}
                </span>
                <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${pct * 100}%`,
                      backgroundColor: REGIME_COLORS[regime],
                    }}
                  />
                </div>
                <span className="text-xs font-mono text-gray-700 w-10 shrink-0">
                  {formatPercent(pct)}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div>
        <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
          Avg Gate Weights
        </span>
        <div className="mt-1.5">
          <GateWeightChart gate={avgGate} />
        </div>
      </div>
    </div>
  );
}
