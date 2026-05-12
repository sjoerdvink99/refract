import { useCallback, useEffect, useRef, useState } from "react";
import { useProjection } from "@/hooks/useProjection";
import { useSelectionStore } from "@/state/selectionStore";
import { Panel } from "@/components/layout/Panel";
import { LoadingState } from "@/components/shared/LoadingState";
import { EmptyState } from "@/components/shared/EmptyState";
import { regimeColor } from "@/utils/colors";
import type { ProjectionPoint } from "@/api/types";

const CANVAS_W = 300;
const CANVAS_H = 300;
const PAD = 28;

function drawRegimeSpace(
  ctx: CanvasRenderingContext2D,
  points: ProjectionPoint[],
  width: number,
  height: number,
  selectedSet: Set<string>,
  hoveredId: string | null,
  dpr: number,
): void {
  const plotW = width - 2 * PAD;
  const plotH = height - 2 * PAD;

  ctx.clearRect(0, 0, width * dpr, height * dpr);
  ctx.save();
  ctx.scale(dpr, dpr);

  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  ctx.strokeRect(PAD, PAD, plotW, plotH);

  ctx.fillStyle = "#9ca3af";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Gate: Attribute →", width / 2, height - 4);
  ctx.save();
  ctx.translate(10, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Gate: Topology →", 0, 0);
  ctx.restore();

  ctx.strokeStyle = "#d1d5db";
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 0.5;
  for (let v = 0.25; v < 1; v += 0.25) {
    const gx = PAD + v * plotW;
    const gy = PAD + v * plotH;
    ctx.beginPath();
    ctx.moveTo(gx, PAD);
    ctx.lineTo(gx, PAD + plotH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(PAD, gy);
    ctx.lineTo(PAD + plotW, gy);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  const hasSelection = selectedSet.size > 0;

  for (const pt of points) {
    const cx = PAD + pt.gate_attribute * plotW;
    const cy = PAD + (1 - pt.gate_topology) * plotH;
    const isSelected = selectedSet.has(pt.node_id);
    const isHovered = pt.node_id === hoveredId;

    if (isSelected || isHovered) continue;

    ctx.beginPath();
    ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = regimeColor(pt.regime) + (hasSelection ? "30" : "99");
    ctx.fill();
  }

  for (const id of selectedSet) {
    const pt = points.find((p) => p.node_id === id);
    if (!pt || pt.node_id === hoveredId) continue;
    const cx = PAD + pt.gate_attribute * plotW;
    const cy = PAD + (1 - pt.gate_topology) * plotH;
    ctx.beginPath();
    ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = regimeColor(pt.regime);
    ctx.fill();
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  if (hoveredId) {
    const pt = points.find((p) => p.node_id === hoveredId);
    if (pt) {
      const cx = PAD + pt.gate_attribute * plotW;
      const cy = PAD + (1 - pt.gate_topology) * plotH;
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = regimeColor(pt.regime);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  ctx.restore();
}

interface TooltipState {
  x: number;
  y: number;
  pt: ProjectionPoint;
}

export function RegimeSpaceView({ className = "" }: { className?: string }) {
  const { activeRunId, selectedNodeIds, hoveredNodeId, toggleSelectedNode, setHoveredNode } =
    useSelectionStore();
  const { data: projection, isLoading } = useProjection(activeRunId);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;

  const selectedSet = new Set(selectedNodeIds);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !projection) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    drawRegimeSpace(
      ctx,
      projection.points,
      CANVAS_W,
      CANVAS_H,
      selectedSet,
      hoveredNodeId,
      dpr,
    );
  }, [projection, selectedSet, hoveredNodeId, dpr]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = CANVAS_W * dpr;
    canvas.height = CANVAS_H * dpr;
  }, [dpr]);

  useEffect(() => {
    draw();
  }, [draw]);

  const eventToPlot = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): [number, number] | null => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const scaleX = CANVAS_W / rect.width;
      const scaleY = CANVAS_H / rect.height;
      return [
        (e.clientX - rect.left) * scaleX,
        (e.clientY - rect.top) * scaleY,
      ];
    },
    [],
  );

  const hitTest = useCallback(
    (lx: number, ly: number): ProjectionPoint | null => {
      if (!projection) return null;
      const plotW = CANVAS_W - 2 * PAD;
      const plotH = CANVAS_H - 2 * PAD;
      let best: ProjectionPoint | null = null;
      let bestDist = 8;
      for (const pt of projection.points) {
        const cx = PAD + pt.gate_attribute * plotW;
        const cy = PAD + (1 - pt.gate_topology) * plotH;
        const dist = Math.hypot(lx - cx, ly - cy);
        if (dist < bestDist) {
          bestDist = dist;
          best = pt;
        }
      }
      return best;
    },
    [projection],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const pos = eventToPlot(e);
      if (!pos) return;
      const hit = hitTest(pos[0], pos[1]);
      setHoveredNode(hit?.node_id ?? null);
      if (hit) {
        const rect = canvasRef.current!.getBoundingClientRect();
        setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, pt: hit });
      } else {
        setTooltip(null);
      }
    },
    [eventToPlot, hitTest, setHoveredNode],
  );

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const pos = eventToPlot(e);
      if (!pos) return;
      const hit = hitTest(pos[0], pos[1]);
      if (hit) toggleSelectedNode(hit.node_id);
    },
    [eventToPlot, hitTest, toggleSelectedNode],
  );

  return (
    <Panel title="Gate Weight Space" className={className}>
      {isLoading && <LoadingState />}
      {!isLoading && !projection && (
        <EmptyState message="No run selected" hint="Select a run to view regime space" />
      )}
      {projection && (
        <div className="p-2 relative">
          <canvas
            ref={canvasRef}
            width={CANVAS_W}
            height={CANVAS_H}
            style={{
              display: "block",
              width: "100%",
              cursor: "crosshair",
            }}
            onMouseMove={handleMouseMove}
            onMouseLeave={() => {
              setHoveredNode(null);
              setTooltip(null);
            }}
            onClick={handleClick}
          />
          {tooltip && (
            <div
              className="absolute pointer-events-none bg-white border border-gray-200 rounded shadow-sm px-2 py-1 text-xs text-gray-700 z-10"
              style={{ left: tooltip.x + 10, top: tooltip.y - 10 }}
            >
              <div className="font-medium">{tooltip.pt.node_id}</div>
              <div className="text-gray-500">
                {tooltip.pt.regime.replace(/_/g, " ")} · {Math.round(tooltip.pt.uncertainty * 100)}% uncertainty
              </div>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
