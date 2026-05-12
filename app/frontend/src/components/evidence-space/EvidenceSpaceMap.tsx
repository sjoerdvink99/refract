import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useProjection } from "@/hooks/useProjection";
import { useSelectionStore } from "@/state/selectionStore";
import { Panel } from "@/components/layout/Panel";
import { LoadingState } from "@/components/shared/LoadingState";
import { EmptyState } from "@/components/shared/EmptyState";
import { regimeColor, expertColor, correctnessColor } from "@/utils/colors";
import type { ColorMode, ProjectionPoint } from "@/api/types";

const PL = 38, PR = 38, PT = 38, PB = 28;
const STRIP_H = 30, STRIP_W = 30, N_BINS = 48;

function ptColor(pt: ProjectionPoint, colorBy: ColorMode): string {
  switch (colorBy) {
    case "regime": return regimeColor(pt.regime);
    case "expert": return expertColor(pt.dominant_expert);
    case "correctness": return correctnessColor(pt.correct);
  }
}

interface BinEntry { height: number; regime: string }

function computeBins(
  points: ProjectionPoint[],
  getter: (p: ProjectionPoint) => number,
): BinEntry[] {
  const bins: { count: number; rc: Record<string, number> }[] =
    Array.from({ length: N_BINS }, () => ({ count: 0, rc: {} }));
  for (const p of points) {
    const v = getter(p);
    const bi = Math.min(Math.floor(v * N_BINS), N_BINS - 1);
    bins[bi].count++;
    bins[bi].rc[p.regime] = (bins[bi].rc[p.regime] ?? 0) + 1;
  }
  const maxCount = Math.max(...bins.map(b => b.count), 1);
  return bins.map(b => ({
    height: b.count / maxCount,
    regime: Object.entries(b.rc).sort((a, b2) => b2[1] - a[1])[0]?.[0] ?? "",
  }));
}

function drawEvidence(
  ctx: CanvasRenderingContext2D,
  points: ProjectionPoint[],
  width: number,
  height: number,
  colorBy: ColorMode,
  selectedIds: Set<string>,
  hoveredId: string | null,
): void {
  const pw = width - PL - PR;
  const ph = height - PT - PB;

  ctx.clearRect(0, 0, width, height);

  const toX = (v: number) => PL + v * pw;
  const toY = (v: number) => PT + (1 - v) * ph;

  ctx.fillStyle = "#f9fafb";
  ctx.fillRect(PL, PT, pw, ph);

  const scale = Math.min(pw, ph);
  const zones = [
    { cx: 0.88, cy: 0.88, r: 0.52, color: "#22C55E" },
    { cx: 0.88, cy: 0.12, r: 0.42, color: "#3B82F6" },
    { cx: 0.12, cy: 0.88, r: 0.42, color: "#14B8A6" },
  ] as const;
  for (const z of zones) {
    const grad = ctx.createRadialGradient(toX(z.cx), toY(z.cy), 0, toX(z.cx), toY(z.cy), z.r * scale);
    grad.addColorStop(0, z.color + "16");
    grad.addColorStop(0.6, z.color + "0a");
    grad.addColorStop(1, z.color + "00");
    ctx.fillStyle = grad;
    ctx.fillRect(PL, PT, pw, ph);
  }

  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.strokeRect(PL, PT, pw, ph);

  ctx.save();
  ctx.beginPath();
  ctx.rect(PL, PT, pw, ph);
  ctx.clip();
  ctx.strokeStyle = "#d1d5db";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(toX(0), toY(0));
  ctx.lineTo(toX(1), toY(1));
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.font = "8px sans-serif";
  ctx.fillStyle = "#9ca3af";
  ctx.translate(toX(0.52) + 6, toY(0.52) - 8);
  ctx.rotate(-Math.PI / 4);
  ctx.textAlign = "left";
  ctx.fillText("eA = eT", 0, 0);
  ctx.restore();

  const watermarks = [
    { ex: 0.83, ey: 0.83, label: "Concordant", color: "#22C55E" },
    { ex: 0.86, ey: 0.10, label: "Attribute", color: "#3B82F6" },
    { ex: 0.09, ey: 0.83, label: "Topology", color: "#14B8A6" },
  ] as const;
  ctx.font = "bold 9px sans-serif";
  ctx.textAlign = "center";
  for (const w of watermarks) {
    ctx.fillStyle = w.color + "6b";
    ctx.fillText(w.label, toX(w.ex), toY(w.ey));
  }

  ctx.font = "8px sans-serif";
  ctx.fillStyle = "#9ca3af";
  ctx.textAlign = "center";
  for (const v of [0, 0.5, 1]) ctx.fillText(v.toFixed(1), toX(v), PT + ph + 14);
  ctx.textAlign = "right";
  for (const v of [0, 0.5, 1]) ctx.fillText(v.toFixed(1), PL - 4, toY(v) + 3);

  ctx.fillStyle = "#6b7280";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Attribute Evidence →", PL + pw / 2, height - 4);
  ctx.save();
  ctx.translate(10, PT + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("← Topology Evidence", 0, 0);
  ctx.restore();

  const hasSelection = selectedIds.size > 0;
  const colorGroups = new Map<string, ProjectionPoint[]>();
  for (const pt of points) {
    if (selectedIds.has(pt.node_id) || pt.node_id === hoveredId) continue;
    const c = ptColor(pt, colorBy);
    if (!colorGroups.has(c)) colorGroups.set(c, []);
    colorGroups.get(c)!.push(pt);
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(PL, PT, pw, ph);
  ctx.clip();

  ctx.globalAlpha = hasSelection ? 0.10 : 0.18;
  for (const [color, pts] of colorGroups) {
    ctx.fillStyle = hasSelection ? "#9ca3af" : color;
    ctx.beginPath();
    for (const pt of pts) {
      const x = toX(pt.attribute_evidence);
      const y = toY(pt.topology_evidence);
      ctx.moveTo(x + 2, y);
      ctx.arc(x, y, 2, 0, Math.PI * 2);
    }
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  for (const id of selectedIds) {
    const pt = points.find(p => p.node_id === id);
    if (!pt || pt.node_id === hoveredId) continue;
    const x = toX(pt.attribute_evidence);
    const y = toY(pt.topology_evidence);
    ctx.beginPath();
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = ptColor(pt, colorBy);
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  if (hoveredId) {
    const pt = points.find(p => p.node_id === hoveredId);
    if (pt) {
      const x = toX(pt.attribute_evidence);
      const y = toY(pt.topology_evidence);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = ptColor(pt, colorBy);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
  ctx.restore();

  const eaBins = computeBins(points, p => p.attribute_evidence);
  const binW = pw / N_BINS;
  const stripBottomY = PT - 2;
  for (let i = 0; i < N_BINS; i++) {
    const { height: h, regime } = eaBins[i];
    if (h === 0) continue;
    const barH = h * (STRIP_H - 4);
    ctx.fillStyle = (regime ? regimeColor(regime) : "#9ca3af") + "b3";
    ctx.fillRect(PL + i * binW, stripBottomY - barH, Math.max(binW - 0.5, 0.5), barH);
  }

  const eTBins = computeBins(points, p => p.topology_evidence);
  const binH = ph / N_BINS;
  const stripLeftX = PL + pw + 2;
  for (let i = 0; i < N_BINS; i++) {
    const { height: h, regime } = eTBins[i];
    if (h === 0) continue;
    const barW = h * (STRIP_W - 4);
    const bTop = toY((i + 1) / N_BINS);
    ctx.fillStyle = (regime ? regimeColor(regime) : "#9ca3af") + "b3";
    ctx.fillRect(stripLeftX, bTop, barW, Math.max(binH - 0.5, 0.5));
  }
}

interface TooltipState { x: number; y: number; pt: ProjectionPoint }

export function EvidenceSpaceMap({ colorBy }: { colorBy: ColorMode }) {
  const { activeRunId, selectedNodeIds, hoveredNodeId, toggleSelectedNode, setHoveredNode } =
    useSelectionStore();
  const { data: projection, isLoading } = useProjection(activeRunId);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 320, height: 240 });
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);

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

  const dpr = window.devicePixelRatio || 1;
  const selectedSet = useMemo(() => new Set(selectedNodeIds), [selectedNodeIds]);

  const redraw = useCallback(() => {
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
    drawEvidence(ctx, projection.points, size.width, size.height, colorBy, selectedSet, hoveredNodeId);
    ctx.restore();
  }, [projection, size, colorBy, selectedSet, hoveredNodeId, dpr]);

  useEffect(() => { redraw(); }, [redraw]);

  const ptFromEvent = useCallback((e: React.MouseEvent<HTMLCanvasElement>): ProjectionPoint | null => {
    if (!projection || !canvasRef.current) return null;
    const rect = canvasRef.current.getBoundingClientRect();
    const lx = e.clientX - rect.left;
    const ly = e.clientY - rect.top;
    const pw = size.width - PL - PR;
    const ph = size.height - PT - PB;
    let closest: ProjectionPoint | null = null;
    let minD = 10;
    for (const pt of projection.points) {
      const x = PL + pt.attribute_evidence * pw;
      const y = PT + (1 - pt.topology_evidence) * ph;
      const d = Math.hypot(x - lx, y - ly);
      if (d < minD) { minD = d; closest = pt; }
    }
    return closest;
  }, [projection, size]);

  return (
    <Panel title="Evidence Space" fill className="h-full">
      {isLoading && <LoadingState label="Loading…" />}
      {!isLoading && !projection && <EmptyState message="No run selected" hint="Select a run" />}
      {projection && (
        <div className="flex flex-col h-full p-2">
          <div ref={containerRef} className="flex-1 min-h-0 relative">
            <canvas
              ref={canvasRef}
              width={size.width}
              height={size.height}
              style={{ display: "block", width: "100%", height: "100%", cursor: "crosshair" }}
              onClick={(e) => {
                const pt = ptFromEvent(e);
                if (pt) toggleSelectedNode(pt.node_id);
              }}
              onMouseMove={(e) => {
                const pt = ptFromEvent(e);
                setHoveredNode(pt?.node_id ?? null);
                if (pt && canvasRef.current) {
                  const rect = canvasRef.current.getBoundingClientRect();
                  setTooltip({ x: e.clientX - rect.left, y: e.clientY - rect.top, pt });
                } else {
                  setTooltip(null);
                }
              }}
              onMouseLeave={() => { setHoveredNode(null); setTooltip(null); }}
            />
            {tooltip && (
              <div
                className="absolute pointer-events-none bg-white border border-gray-200 rounded shadow-sm px-2 py-1 text-xs text-gray-700 z-10"
                style={{ left: tooltip.x + 10, top: tooltip.y - 10 }}
              >
                <div className="font-medium">{tooltip.pt.node_id}</div>
                <div className="text-gray-500">{tooltip.pt.regime.replace(/_/g, " ")}</div>
                <div className="text-gray-400 font-mono">
                  eA {(tooltip.pt.attribute_evidence * 100).toFixed(0)}% · eT {(tooltip.pt.topology_evidence * 100).toFixed(0)}%
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}
