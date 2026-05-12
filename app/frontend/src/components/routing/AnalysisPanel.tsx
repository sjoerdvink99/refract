import { useMemo } from "react";
import { useProjection } from "@/hooks/useProjection";
import { useSelectionStore } from "@/state/selectionStore";
import { Panel } from "@/components/layout/Panel";
import { LoadingState } from "@/components/shared/LoadingState";
import { RegimeProfileDiff } from "./RegimeProfileDiff";
import { NodeRoutingStory } from "./NodeRoutingStory";

export function AnalysisPanel() {
  const { activeRunId, selectedNodeIds } = useSelectionStore();
  const { data: projection, isLoading } = useProjection(activeRunId);

  const selectedSet = useMemo(() => new Set(selectedNodeIds), [selectedNodeIds]);

  const title =
    selectedNodeIds.length === 0
      ? "Characterize"
      : selectedNodeIds.length === 1
      ? "Calibrate"
      : `Compare  (n=${selectedNodeIds.length})`;

  return (
    <Panel title={title} className="h-full">
      {isLoading && <LoadingState />}

      {!isLoading && projection && selectedNodeIds.length === 0 && (
        <div className="flex flex-col gap-3 p-4 text-xs text-gray-500">
          <p>
            <span className="font-semibold text-gray-700">Click a point</span> to calibrate —
            see the routing verdict, evidence vs. gate alignment, nearest regime neighbors, and
            whether miscalibration changed the prediction.
          </p>
          <p>
            <span className="font-semibold text-gray-700">Shift+drag to lasso</span> to compare
            — Z-score the selection against the full population across 7 structural dimensions.
          </p>
          <p className="text-gray-400">
            Both interactions work on the scatter and the regime map.
          </p>
        </div>
      )}

      {!isLoading && projection && selectedNodeIds.length === 1 && (
        <NodeRoutingStory
          nodeId={selectedNodeIds[0]}
          allPoints={projection.points}
        />
      )}

      {!isLoading && projection && selectedNodeIds.length > 1 && (
        <RegimeProfileDiff
          selectedIds={selectedSet}
          allPoints={projection.points}
        />
      )}
    </Panel>
  );
}
