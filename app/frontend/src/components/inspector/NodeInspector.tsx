import { useNodeDetail } from "@/hooks/useNodeDetails";
import { useSelectionStore } from "@/state/selectionStore";
import { LoadingState } from "@/components/shared/LoadingState";
import { regimeColor, REGIME_COLORS, EXPERT_COLORS } from "@/utils/colors";
import { formatPercent, formatRegimeLabel } from "@/utils/format";
import type { GateWeights, RegimeVector } from "@/api/types";

function EvidenceBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-xs text-gray-400 w-14 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${value * 100}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono text-gray-600 w-8 text-right">{Math.round(value * 100)}%</span>
    </div>
  );
}

function GateBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-xs text-gray-400 w-14 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${value * 100}%`, backgroundColor: color }} />
      </div>
      <span className="text-xs font-mono text-gray-600 w-8 text-right">{Math.round(value * 100)}%</span>
    </div>
  );
}

function EvidenceSection({ vector }: { vector: RegimeVector }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Evidence</span>
      <EvidenceBar label="Attribute" value={vector.attribute_evidence} color={REGIME_COLORS.attribute_dominant} />
      <EvidenceBar label="Topology" value={vector.topology_evidence} color={REGIME_COLORS.topology_dominant} />
      <EvidenceBar label="Concordance" value={vector.concordance} color={REGIME_COLORS.concordant} />
    </div>
  );
}

function GateSection({ gate }: { gate: GateWeights }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Routing Gate</span>
      <GateBar label="Attribute" value={gate.attribute} color={EXPERT_COLORS.attribute} />
      <GateBar label="Topology" value={gate.topology} color={EXPERT_COLORS.topology} />
    </div>
  );
}

function CFSection({ featureDamage, edgeDamage }: { featureDamage: number; edgeDamage: number }) {
  const max = Math.max(Math.abs(featureDamage), Math.abs(edgeDamage), 0.01);
  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">CF Impact</span>
      <div className="flex items-center gap-1.5">
        <span className="text-xs text-gray-400 w-14 shrink-0">Feature</span>
        <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <div className="h-full bg-blue-400 rounded-full" style={{ width: `${(Math.abs(featureDamage) / max) * 100}%` }} />
        </div>
        <span className="text-xs font-mono text-gray-600 w-8 text-right">{featureDamage.toFixed(2)}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-xs text-gray-400 w-14 shrink-0">Edge</span>
        <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <div className="h-full bg-teal-400 rounded-full" style={{ width: `${(Math.abs(edgeDamage) / max) * 100}%` }} />
        </div>
        <span className="text-xs font-mono text-gray-600 w-8 text-right">{edgeDamage.toFixed(2)}</span>
      </div>
    </div>
  );
}

export function NodeInspector({ className = "" }: { className?: string }) {
  const { activeRunId, selectedNodeIds } = useSelectionStore();
  const singleNodeId = selectedNodeIds.length === 1 ? selectedNodeIds[0] : null;
  const { data: node, isLoading } = useNodeDetail(activeRunId, singleNodeId);

  return (
    <div className={`bg-white border border-gray-200 rounded-lg overflow-hidden flex flex-col ${className}`}>
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-100 shrink-0">
        <span className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
          {node ? `Node — ${node.node_id}` : "Node Inspector"}
        </span>
        {node && (
          <div className="flex items-center gap-2">
            <span
              className="text-xs px-1.5 py-0.5 rounded-full text-white font-medium"
              style={{ backgroundColor: regimeColor(node.regime) }}
            >
              {formatRegimeLabel(node.regime)}
            </span>
            {node.correct !== null && (
              <span className={`text-xs px-1.5 py-0.5 rounded-full ${node.correct ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
                {node.correct ? "✓" : "✗"}
              </span>
            )}
            <span className="text-xs font-mono text-gray-500">{formatPercent(node.confidence)}</span>
          </div>
        )}
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {selectedNodeIds.length === 0 && (
          <div className="flex items-center justify-center h-full text-xs text-gray-400 py-4">
            Click a point to inspect a node
          </div>
        )}
        {selectedNodeIds.length > 1 && (
          <div className="flex items-center justify-center h-full text-xs text-gray-400 py-4">
            {selectedNodeIds.length} nodes selected
          </div>
        )}
        {isLoading && <LoadingState />}
        {node && (
          <div className="p-3 flex flex-col gap-3">
            <div className="grid grid-cols-2 gap-3">
              <EvidenceSection vector={node.regime_vector} />
              <GateSection gate={node.gate} />
            </div>
            {(() => {
              const feat = node.counterfactuals.find((c) => c.mode === "mean_feature_mask");
              const edge = node.counterfactuals.find((c) => c.mode === "ego_edge_mask");
              if (!feat && !edge) return null;
              return (
                <CFSection
                  featureDamage={feat?.damage ?? 0}
                  edgeDamage={edge?.damage ?? 0}
                />
              );
            })()}
          </div>
        )}
      </div>
    </div>
  );
}
