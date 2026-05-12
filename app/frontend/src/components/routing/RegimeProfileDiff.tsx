import { useMemo } from "react";
import type { ProjectionPoint } from "@/api/types";

interface RegimeProfileDiffProps {
  selectedIds: Set<string>;
  allPoints: ProjectionPoint[];
}

const DIMS = [
  { key: "attribute_evidence", label: "Attr. evidence" },
  { key: "topology_evidence", label: "Topo. evidence" },
  { key: "uncertainty", label: "Uncertainty" },
  { key: "distortion", label: "Distortion" },
  { key: "gate_entropy", label: "Gate entropy" },
  { key: "evidence_imbalance", label: "Evid. imbalance" },
  { key: "gate_commitment", label: "Gate commitment" },
] as const;

type DimKey = (typeof DIMS)[number]["key"];

function extractDims(pt: ProjectionPoint): Record<DimKey, number> {
  const gA = pt.gate_attribute;
  const gT = pt.gate_topology;
  const eps = 1e-9;
  return {
    attribute_evidence: pt.attribute_evidence,
    topology_evidence: pt.topology_evidence,
    uncertainty: pt.uncertainty,
    distortion: pt.distortion,
    gate_entropy: -(gA * Math.log(gA + eps) + gT * Math.log(gT + eps)),
    evidence_imbalance: Math.abs(pt.attribute_evidence - pt.topology_evidence),
    gate_commitment: Math.abs(pt.gate_attribute - pt.gate_topology),
  };
}

function mean(vals: number[]): number {
  if (vals.length === 0) return 0;
  return vals.reduce((s, v) => s + v, 0) / vals.length;
}

function std(vals: number[], m: number): number {
  if (vals.length < 2) return 1;
  const variance = vals.reduce((s, v) => s + (v - m) ** 2, 0) / vals.length;
  return Math.sqrt(variance) || 1e-9;
}

export function RegimeProfileDiff({ selectedIds, allPoints }: RegimeProfileDiffProps) {
  const selectedPts = useMemo(
    () => allPoints.filter((p) => selectedIds.has(p.node_id)),
    [allPoints, selectedIds],
  );

  const analysis = useMemo(() => {
    const allDims = allPoints.map(extractDims);
    const selDims = selectedPts.map(extractDims);

    return DIMS.map(({ key, label }) => {
      const popVals = allDims.map((d) => d[key]);
      const selVals = selDims.map((d) => d[key]);
      const popMean = mean(popVals);
      const popStd = std(popVals, popMean);
      const selMean = mean(selVals);
      const z = (selMean - popMean) / popStd;
      return { key, label, z, selMean, popMean };
    });
  }, [allPoints, selectedPts]);

  const selCorrect = selectedPts.filter((p) => p.correct === true).length;
  const selLabeled = selectedPts.filter((p) => p.correct !== null).length;
  const selAccuracy = selLabeled > 0 ? selCorrect / selLabeled : null;

  const popCorrect = allPoints.filter((p) => p.correct === true).length;
  const popLabeled = allPoints.filter((p) => p.correct !== null).length;
  const popAccuracy = popLabeled > 0 ? popCorrect / popLabeled : null;

  const maxZ = Math.max(...analysis.map((d) => Math.abs(d.z)), 1);

  return (
    <div className="flex flex-col gap-3 p-3 h-full overflow-y-auto">
      <div className="flex items-start justify-between gap-2">
        <span className="text-xs font-semibold text-gray-700">Selection vs. population</span>
        <div className="flex items-center gap-3 font-mono text-xs shrink-0">
          <span className="text-gray-400">n={selectedIds.size}</span>
          {selAccuracy !== null && (
            <span
              className="font-semibold"
              style={{
                color:
                  popAccuracy !== null && selAccuracy > popAccuracy
                    ? "#22C55E"
                    : popAccuracy !== null && selAccuracy < popAccuracy
                    ? "#EF4444"
                    : "#6b7280",
              }}
            >
              sel {Math.round(selAccuracy * 100)}%
            </span>
          )}
          {popAccuracy !== null && (
            <span className="text-gray-400">pop {Math.round(popAccuracy * 100)}%</span>
          )}
        </div>
      </div>

      <div className="flex flex-col gap-1.5">
        {analysis.map(({ key, label, z }) => {
          const pct = (Math.abs(z) / maxZ) * 100;
          const isPos = z >= 0;
          const isSignificant = Math.abs(z) >= 1;
          const color = isPos ? "#3B82F6" : "#EF4444";
          const zLabel = `${z >= 0 ? "+" : ""}${z.toFixed(2)}σ`;
          return (
            <div key={key} className="flex items-center gap-2">
              <span
                className={`text-xs w-28 shrink-0 text-right ${
                  isSignificant ? "text-gray-700 font-medium" : "text-gray-400"
                }`}
              >
                {label}
              </span>
              <div className="flex-1 flex items-center gap-0.5 h-4">
                <div className="flex-1 flex justify-end">
                  {!isPos && (
                    <div
                      className="h-3 rounded-l-sm"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: color,
                        opacity: isSignificant ? 1 : 0.4,
                      }}
                    />
                  )}
                </div>
                <div className="w-px h-4 bg-gray-300 shrink-0" />
                <div className="flex-1">
                  {isPos && (
                    <div
                      className="h-3 rounded-r-sm"
                      style={{
                        width: `${pct}%`,
                        backgroundColor: color,
                        opacity: isSignificant ? 1 : 0.4,
                      }}
                    />
                  )}
                </div>
              </div>
              <span
                className="text-xs font-mono w-12 shrink-0"
                style={{ color: isSignificant ? color : "#9ca3af" }}
              >
                {zLabel}
              </span>
            </div>
          );
        })}
      </div>

      <p className="text-xs text-gray-400 mt-auto">
        Bars show σ from population mean. Bold = |z|≥1. Positive = above average on this dimension.
      </p>
    </div>
  );
}
