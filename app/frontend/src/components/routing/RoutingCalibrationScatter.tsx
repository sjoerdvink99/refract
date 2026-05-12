import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useProjection } from "@/hooks/useProjection";
import { useSelectionStore } from "@/state/selectionStore";
import { Panel } from "@/components/layout/Panel";
import { LoadingState } from "@/components/shared/LoadingState";
import { EmptyState } from "@/components/shared/EmptyState";
import type { ProjectionPoint } from "@/api/types";

const PL = 46, PR = 14, PT = 14, PB = 42;

function correctnessColor(pt: ProjectionPoint): string {
  if (pt.correct === true) return "#22C55E";
  if (pt.correct === false) return "#EF4444";
  return "#9CA3AF";
}

function evidenceRadius(pt: ProjectionPoint): number {
  const de = pt.attribute_evidence - pt.topology_evidence;
  const dg = pt.gate_attribute - pt.gate_topology;
  return 1.5 + Math.min(1, Math.sqrt(de * de + dg * dg) / Math.SQRT2) * 2;
}

function toCanvas(
  de: number,
  dg: number,
  pw: number,
  ph: number,
): [number, number] {
  return [PL + ((de + 1) / 2) * pw, PT + (1 - (dg + 1) / 2) * ph];
}

function pointInPoly(px: number, py: number, poly: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

interface QuadrantStats {
  n: number;
  accuracy: number | null;
}

function computeQuadrantStats(points: ProjectionPoint[]): Record<string, QuadrantStats> {
  const quads: Record<string, { n: number; correct: number; labeled: number }> = {
    TL: { n: 0, correct: 0, labeled: 0 },
    TR: { n: 0, correct: 0, labeled: 0 },
    BL: { n: 0, correct: 0, labeled: 0 },
    BR: { n: 0, correct: 0, labeled: 0 },
  };
  for (const pt of points) {
    const de = pt.attribute_evidence - pt.topology_evidence;
    const dg = pt.gate_attribute - pt.gate_topology;
    const q = de >= 0 ? (dg >= 0 ? "TR" : "BR") : (dg >= 0 ? "TL" : "BL");
    quads[q].n++;
    if (pt.correct !== null) {
      quads[q].labeled++;
      if (pt.correct) quads[q].correct++;
    }
  }
  return Object.fromEntries(
    Object.entries(quads).map(([k, v]) => [
      k,
      { n: v.n, accuracy: v.labeled > 0 ? v.correct / v.labeled : null },
    ]),
  );
}

function drawScatter(
  ctx: CanvasRenderingContext2D,
  points: ProjectionPoint[],
  width: number,
  height: number,
  selectedIds: Set<string>,
  hoveredId: string | null,
  lassoPoly: [number, number][],
  quadStats: Record<string, QuadrantStats>,
): void {
  const pw = width - PL - PR;
  const ph = height - PT - PB;
  const hasSelection = selectedIds.size > 0;

  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = "#f9fafb";
  ctx.fillRect(PL, PT, pw, ph);

  ctx.fillStyle = "#EF444408";
  ctx.fillRect(PL, PT, pw / 2, ph / 2);
  ctx.fillRect(PL + pw / 2, PT + ph / 2, pw / 2, ph / 2);
  ctx.fillStyle = "#22C55E06";
  ctx.fillRect(PL + pw / 2, PT, pw / 2, ph / 2);
  ctx.fillRect(PL, PT + ph / 2, pw / 2, ph / 2);

  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.strokeRect(PL, PT, pw, ph);

  ctx.save();
  ctx.beginPath();
  ctx.rect(PL, PT, pw, ph);
  ctx.clip();
  ctx.beginPath();
  ctx.moveTo(PL, PT + 0.95 * ph);
  ctx.lineTo(PL + 0.95 * pw, PT);
  ctx.lineTo(PL + pw, PT + 0.05 * ph);
  ctx.lineTo(PL + 0.05 * pw, PT + ph);
  ctx.closePath();
  ctx.fillStyle = "rgba(0,0,0,0.04)";
  ctx.fill();
  ctx.strokeStyle = "#9ca3af";
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(PL, PT + ph);
  ctx.lineTo(PL + pw, PT);
  ctx.stroke();
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(PL + pw / 2, PT);
  ctx.lineTo(PL + pw / 2, PT + ph);
  ctx.moveTo(PL, PT + ph / 2);
  ctx.lineTo(PL + pw, PT + ph / 2);
  ctx.stroke();
  ctx.restore();

  ctx.font = "8px sans-serif";
  ctx.fillStyle = "#9ca3af";
  ctx.textAlign = "center";
  for (const [v, label] of [[-1, "−1"], [0, "0"], [1, "+1"]] as [number, string][]) {
    ctx.fillText(label, PL + ((v + 1) / 2) * pw, PT + ph + 12);
  }
  ctx.textAlign = "right";
  for (const [v, label] of [[-1, "−1"], [0, "0"], [1, "+1"]] as [number, string][]) {
    ctx.fillText(label, PL - 4, PT + (1 - (v + 1) / 2) * ph + 3);
  }

  ctx.fillStyle = "#6b7280";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Evidence differential  (eA − eT)", PL + pw / 2, height - 6);
  ctx.save();
  ctx.translate(12, PT + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Gate response  (gA − gT)", 0, 0);
  ctx.restore();

  const quadLabels = [
    { q: "TL", x: PL + 4, y: PT + 14, align: "left" as const, label: "attr. over-routing" },
    { q: "TR", x: PL + pw - 4, y: PT + 14, align: "right" as const, label: "correctly attr.-routed" },
    { q: "BL", x: PL + 4, y: PT + ph - 6, align: "left" as const, label: "correctly topo.-routed" },
    { q: "BR", x: PL + pw - 4, y: PT + ph - 6, align: "right" as const, label: "topo. over-routing" },
  ];
  ctx.font = "7.5px sans-serif";
  for (const { q, x, y, align, label } of quadLabels) {
    const stats = quadStats[q];
    const isMiscal = q === "TL" || q === "BR";
    ctx.fillStyle = isMiscal ? "#EF444460" : "#22C55E60";
    ctx.textAlign = align;
    ctx.fillText(label, x, y);
    if (stats) {
      const acc = stats.accuracy !== null ? `${Math.round(stats.accuracy * 100)}% acc` : "";
      ctx.fillStyle = "#9ca3af";
      ctx.fillText(`n=${stats.n}${acc ? " · " + acc : ""}`, x, y + 10);
    }
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(PL, PT, pw, ph);
  ctx.clip();

  for (const pt of points) {
    if (selectedIds.has(pt.node_id) || pt.node_id === hoveredId) continue;
    const [cx, cy] = toCanvas(
      pt.attribute_evidence - pt.topology_evidence,
      pt.gate_attribute - pt.gate_topology,
      pw,
      ph,
    );
    ctx.beginPath();
    ctx.arc(cx, cy, evidenceRadius(pt), 0, Math.PI * 2);
    const base = correctnessColor(pt);
    ctx.fillStyle = base + (hasSelection ? "26" : "b3");
    ctx.fill();
  }

  for (const id of selectedIds) {
    const pt = points.find((p) => p.node_id === id);
    if (!pt || pt.node_id === hoveredId) continue;
    const [cx, cy] = toCanvas(
      pt.attribute_evidence - pt.topology_evidence,
      pt.gate_attribute - pt.gate_topology,
      pw,
      ph,
    );
    ctx.beginPath();
    ctx.arc(cx, cy, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = correctnessColor(pt);
    ctx.fill();
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  if (hoveredId) {
    const pt = points.find((p) => p.node_id === hoveredId);
    if (pt) {
      const [cx, cy] = toCanvas(
        pt.attribute_evidence - pt.topology_evidence,
        pt.gate_attribute - pt.gate_topology,
        pw,
        ph,
      );
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = correctnessColor(pt);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  if (lassoPoly.length > 1) {
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 1.5;
    ctx.fillStyle = "rgba(55, 65, 81, 0.08)";
    ctx.beginPath();
    ctx.moveTo(lassoPoly[0][0], lassoPoly[0][1]);
    for (let i = 1; i < lassoPoly.length; i++) ctx.lineTo(lassoPoly[i][0], lassoPoly[i][1]);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  ctx.restore();
}

interface TooltipState { x: number; y: number; pt: ProjectionPoint }

export function RoutingCalibrationScatter() {
  const { activeRunId, selectedNodeIds, hoveredNodeId, setSelectedNodes, toggleSelectedNode, clearSelection, setHoveredNode } =
    useSelectionStore();
  const { data: projection, isLoading } = useProjection(activeRunId);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 320, height: 240 });
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const dpr = window.devicePixelRatio || 1;

  const isLassoRef = useRef(false);
  const lassoPtsRef = useRef<[number, number][]>([]);
  const didLassoRef = useRef(false);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") clearSelection(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [clearSelection]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        if (width > 0 && height > 0) setSize({ width: Math.floor(width), height: Math.floor(height) });
      }
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  const selectedSet = useMemo(() => new Set(selectedNodeIds), [selectedNodeIds]);

  const quadStats = useMemo(
    () => (projection ? computeQuadrantStats(projection.points) : {}),
    [projection],
  );

  const redraw = useCallback(
    (lasso: [number, number][] = []) => {
      const canvas = canvasRef.current;
      if (!canvas || !projection) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      if (canvas.width !== size.width * dpr || canvas.height !== size.height * dpr) {
        canvas.width = size.width * dpr;
        canvas.height = size.height * dpr;
      }
      ctx.save();
      ctx.scale(dpr, dpr);
      drawScatter(ctx, projection.points, size.width, size.height, selectedSet, hoveredNodeId, lasso, quadStats);
      ctx.restore();
    },
    [projection, size, selectedSet, hoveredNodeId, dpr, quadStats],
  );

  useEffect(() => { redraw(); }, [redraw]);

  function toLogical(e: React.MouseEvent<HTMLCanvasElement>): [number, number] {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return [
      (e.clientX - rect.left) * (size.width / rect.width),
      (e.clientY - rect.top) * (size.height / rect.height),
    ];
  }

  function hitTest(lx: number, ly: number): ProjectionPoint | null {
    if (!projection) return null;
    const pw = size.width - PL - PR;
    const ph = size.height - PT - PB;
    let best: ProjectionPoint | null = null;
    let bestD = 10;
    for (const pt of projection.points) {
      const [cx, cy] = toCanvas(
        pt.attribute_evidence - pt.topology_evidence,
        pt.gate_attribute - pt.gate_topology,
        pw,
        ph,
      );
      const d = Math.hypot(lx - cx, ly - cy);
      if (d < bestD) { bestD = d; best = pt; }
    }
    return best;
  }

  const onMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!e.shiftKey) return;
    e.preventDefault();
    isLassoRef.current = true;
    didLassoRef.current = false;
    const [lx, ly] = toLogical(e);
    lassoPtsRef.current = [[lx, ly]];
  }, [size]);

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const [lx, ly] = toLogical(e);
    if (isLassoRef.current) {
      didLassoRef.current = true;
      lassoPtsRef.current.push([lx, ly]);
      redraw(lassoPtsRef.current);
      return;
    }
    const pt = hitTest(lx, ly);
    setHoveredNode(pt?.node_id ?? null);
    if (pt) {
      const rect = canvasRef.current!.getBoundingClientRect();
      setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, pt });
    } else {
      setTooltip(null);
    }
  }, [redraw, setHoveredNode, projection, size]);

  const onMouseUp = useCallback(() => {
    if (!isLassoRef.current) return;
    isLassoRef.current = false;
    if (didLassoRef.current && projection && lassoPtsRef.current.length >= 3) {
      const pw = size.width - PL - PR;
      const ph = size.height - PT - PB;
      const poly = lassoPtsRef.current;
      const inside = projection.points.filter((pt) => {
        const [cx, cy] = toCanvas(
          pt.attribute_evidence - pt.topology_evidence,
          pt.gate_attribute - pt.gate_topology,
          pw,
          ph,
        );
        return pointInPoly(cx, cy, poly);
      });
      setSelectedNodes(inside.map((p) => p.node_id));
    }
    lassoPtsRef.current = [];
    redraw();
  }, [projection, size, setSelectedNodes, redraw]);

  const onClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (didLassoRef.current || e.shiftKey) return;
    const [lx, ly] = toLogical(e);
    const pt = hitTest(lx, ly);
    if (pt) toggleSelectedNode(pt.node_id);
    else clearSelection();
  }, [toggleSelectedNode, clearSelection, projection, size]);

  const onMouseLeave = useCallback(() => {
    isLassoRef.current = false;
    lassoPtsRef.current = [];
    setHoveredNode(null);
    setTooltip(null);
    redraw();
  }, [setHoveredNode, redraw]);

  return (
    <Panel
      title="Routing Calibration"
      fill
      className="h-full"
      actions={
        <span className="text-xs text-gray-400 font-mono">
          shift+drag to lasso · color = correctness
        </span>
      }
    >
      {isLoading && <LoadingState label="Loading…" />}
      {!isLoading && !projection && <EmptyState message="No run selected" hint="Select a run" />}
      {projection && (
        <div className="flex flex-col h-full p-1.5">
          <div ref={containerRef} className="flex-1 min-h-0 relative">
            <canvas
              ref={canvasRef}
              width={size.width}
              height={size.height}
              style={{ display: "block", width: "100%", height: "100%", cursor: "crosshair" }}
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              onClick={onClick}
              onMouseLeave={onMouseLeave}
            />
            {tooltip && (
              <div
                className="absolute pointer-events-none bg-white border border-gray-200 rounded shadow-sm px-2 py-1.5 text-xs text-gray-700 z-10 space-y-0.5"
                style={{ left: tooltip.x + 12, top: tooltip.y - 10 }}
              >
                <div className="font-medium">{tooltip.pt.node_id}</div>
                <div className="text-gray-500">{tooltip.pt.regime.replace(/_/g, " ")}</div>
                <div className="font-mono text-gray-400">
                  eA {(tooltip.pt.attribute_evidence * 100).toFixed(0)}%
                  &nbsp;·&nbsp;
                  eT {(tooltip.pt.topology_evidence * 100).toFixed(0)}%
                </div>
                <div className="font-mono text-gray-400">
                  gA {(tooltip.pt.gate_attribute * 100).toFixed(0)}%
                  &nbsp;·&nbsp;
                  gT {(tooltip.pt.gate_topology * 100).toFixed(0)}%
                </div>
                <div
                  className="font-medium"
                  style={{
                    color:
                      tooltip.pt.correct === true
                        ? "#22C55E"
                        : tooltip.pt.correct === false
                        ? "#EF4444"
                        : "#9CA3AF",
                  }}
                >
                  {tooltip.pt.correct === true
                    ? "correct"
                    : tooltip.pt.correct === false
                    ? "incorrect"
                    : "unknown"}
                </div>
              </div>
            )}
          </div>
          <div className="shrink-0 flex items-center gap-4 px-2 pt-1">
            {[
              { color: "#22C55E", label: "correct" },
              { color: "#EF4444", label: "incorrect" },
              { color: "#9CA3AF", label: "unknown" },
            ].map(({ color, label }) => (
              <div key={label} className="flex items-center gap-1">
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                <span className="text-xs text-gray-500">{label}</span>
              </div>
            ))}
            <span className="text-xs text-gray-400 ml-auto">
              {selectedNodeIds.length > 0 ? `${selectedNodeIds.length} selected · Esc to clear` : ""}
            </span>
          </div>
        </div>
      )}
    </Panel>
  );
}
