import { useCallback, useEffect, useRef, useState } from "react";
import { useProjection } from "@/hooks/useProjection";
import { useSelectionStore } from "@/state/selectionStore";
import { useViewStore } from "@/state/viewStore";
import { Panel } from "@/components/layout/Panel";
import { LoadingState } from "@/components/shared/LoadingState";
import { EmptyState } from "@/components/shared/EmptyState";
import { ColorLegend } from "@/components/shared/ColorLegend";
import { RegimeCanvas } from "./RegimeCanvas";
import { REGIME_COLORS, EXPERT_COLORS } from "@/utils/colors";
import { formatRegimeLabel, formatExpertLabel } from "@/utils/format";
import { downloadCanvasPng } from "@/utils/download";
import type { ColorMode, ProjectionPoint } from "@/api/types";

const COLOR_MODE_OPTIONS: { value: ColorMode; label: string }[] = [
  { value: "regime", label: "Regime" },
  { value: "expert", label: "Expert" },
  { value: "correctness", label: "Correctness" },
];

function legendItems(colorBy: ColorMode) {
  if (colorBy === "regime") {
    return Object.entries(REGIME_COLORS).map(([k, v]) => ({
      color: v,
      label: formatRegimeLabel(k),
    }));
  }
  if (colorBy === "expert") {
    return Object.entries(EXPERT_COLORS).map(([k, v]) => ({
      color: v,
      label: formatExpertLabel(k),
    }));
  }
  if (colorBy === "correctness") {
    return [
      { color: "#22C55E", label: "Correct" },
      { color: "#EF4444", label: "Incorrect" },
    ];
  }
  return [];
}

interface TooltipState {
  x: number;
  y: number;
  pt: ProjectionPoint;
}

export function RegimeMap({ className = "" }: { className?: string }) {
  const {
    activeRunId,
    selectedNodeIds,
    hoveredNodeId,
    toggleSelectedNode,
    setSelectedNodes,
    clearSelection,
    setHoveredNode,
  } = useSelectionStore();
  const { colorBy, setColorBy } = useViewStore();
  const { data: projection, isLoading } = useProjection(activeRunId);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 540, height: 420 });
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setCanvasSize({ width: Math.floor(width), height: Math.floor(height) });
        }
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const handleExport = useCallback(() => {
    if (canvasRef.current) downloadCanvasPng(canvasRef.current, "regime_map.png");
  }, []);

  const handleNodeHover = useCallback(
    (nodeId: string | null, event?: MouseEvent) => {
      setHoveredNode(nodeId);
      if (nodeId && event && projection && canvasRef.current) {
        const pt = projection.points.find((p) => p.node_id === nodeId);
        const rect = canvasRef.current.getBoundingClientRect();
        if (pt) setTooltip({ x: event.clientX - rect.left, y: event.clientY - rect.top, pt });
      } else {
        setTooltip(null);
      }
    },
    [setHoveredNode, projection],
  );

  const actions = (
    <div className="flex gap-2 items-center">
      <select
        className="text-xs bg-white border border-gray-200 rounded px-2 py-1 text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-400"
        value={colorBy}
        onChange={(e) => setColorBy(e.target.value as ColorMode)}
      >
        {COLOR_MODE_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </select>
      <button
        className="text-xs text-gray-500 hover:text-gray-700 px-1 py-0.5"
        onClick={handleExport}
      >
        Export
      </button>
    </div>
  );

  return (
    <Panel title="Regime Map" fill actions={actions} className={className}>
      {isLoading && <LoadingState label="Computing map…" />}
      {!isLoading && !projection && (
        <EmptyState message="No run selected" hint="Select a run to begin analysis" />
      )}
      {projection && (
        <div className="flex flex-col gap-2 p-2 h-full">
          <div ref={containerRef} className="flex-1 min-h-0 w-full relative">
            <RegimeCanvas
              ref={canvasRef}
              points={projection.points}
              colorBy={colorBy}
              selectedNodeIds={selectedNodeIds}
              hoveredNodeId={hoveredNodeId}
              onNodeToggle={toggleSelectedNode}
              onNodesLasso={setSelectedNodes}
              onEmptyClick={clearSelection}
              onNodeHover={handleNodeHover}
              width={canvasSize.width}
              height={canvasSize.height}
            />
            {tooltip && (
              <div
                className="absolute pointer-events-none bg-white border border-gray-200 rounded shadow-sm px-2 py-1 text-xs text-gray-700 z-10"
                style={{ left: tooltip.x + 12, top: tooltip.y - 10 }}
              >
                <div className="font-medium">{tooltip.pt.node_id}</div>
                <div className="text-gray-500">{tooltip.pt.regime.replace(/_/g, " ")}</div>
                <div className="text-gray-400">
                  eA {Math.round(tooltip.pt.attribute_evidence * 100)}% · eT {Math.round(tooltip.pt.topology_evidence * 100)}%
                </div>
              </div>
            )}
          </div>
          <div className="px-1 shrink-0">
            <ColorLegend items={legendItems(colorBy)} />
          </div>
        </div>
      )}
    </Panel>
  );
}
