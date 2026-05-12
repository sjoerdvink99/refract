import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useProjection } from "@/hooks/useProjection";
import { useSelectionStore } from "@/state/selectionStore";
import { Panel } from "@/components/layout/Panel";
import { LoadingState } from "@/components/shared/LoadingState";
import { EmptyState } from "@/components/shared/EmptyState";
import { regimeColor, expertColor, correctnessColor } from "@/utils/colors";
import type { ColorMode, ProjectionPoint } from "@/api/types";

const PL = 42, PR = 38, PT = 38, PB = 30;
const STRIP_H = 30, STRIP_W = 30, N_BINS = 16;

function ptColor(pt: ProjectionPoint, colorBy: ColorMode): string {
  switch (colorBy) {
    case "regime": return regimeColor(pt.regime);
    case "expert": return expertColor(pt.dominant_expert);
    case "correctness": return correctnessColor(pt.correct);
  }
}

interface BinEntry { height: number; regime: string }

function computeBins2(
  points: ProjectionPoint[],
  getter: (p: ProjectionPoint) => number,
): BinEntry[] {
  const bins: { count: number; rc: Record<string, number> }[] =
    Array.from({ length: N_BINS }, () => ({ count: 0, rc: {} }));
  for (const p of points) {
    const v = getter(p);
    const bi = Math.min(Math.floor((v + 1) / 2 * N_BINS), N_BINS - 1);
    bins[bi].count++;
    bins[bi].rc[p.regime] = (bins[bi].rc[p.regime] ?? 0) + 1;
  }
  const maxCount = Math.max(...bins.map(b => b.count), 1);
  return bins.map(b => ({
    height: b.count / maxCount,
    regime: Object.entries(b.rc).sort((a, b2) => b2[1] - a[1])[0]?.[0] ?? "",
  }));
}

interface CalibBin { meanDg: number; stdDg: number; count: number; regime: string; idx: number }

function computeCalibBins(points: ProjectionPoint[]): CalibBin[] {
  const bins: { sumDg: number; sumDg2: number; count: number; rc: Record<string, number> }[] =
    Array.from({ length: N_BINS }, () => ({ sumDg: 0, sumDg2: 0, count: 0, rc: {} }));
  for (const p of points) {
    const de = p.attribute_evidence - p.topology_evidence;
    const dg = p.gate_attribute - p.gate_topology;
    const bi = Math.min(Math.floor((de + 1) / 2 * N_BINS), N_BINS - 1);
    bins[bi].sumDg += dg;
    bins[bi].sumDg2 += dg * dg;
    bins[bi].count++;
    bins[bi].rc[p.regime] = (bins[bi].rc[p.regime] ?? 0) + 1;
  }
  return bins.map((b, i) => {
    if (b.count === 0) return { meanDg: 0, stdDg: 0, count: 0, regime: "", idx: i };
    const meanDg = b.sumDg / b.count;
    const stdDg = Math.sqrt(Math.max(0, b.sumDg2 / b.count - meanDg * meanDg));
    const regime = Object.entries(b.rc).sort((a, b2) => b2[1] - a[1])[0]?.[0] ?? "";
    return { meanDg, stdDg, count: b.count, regime, idx: i };
  });
}

function rankArray(arr: number[]): number[] {
  const sorted = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Array<number>(arr.length);
  for (let i = 0; i < sorted.length; i++) ranks[sorted[i].i] = i + 1;
  return ranks;
}

function spearmanCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return 0;
  const rx = rankArray(x);
  const ry = rankArray(y);
  let sumD2 = 0;
  for (let i = 0; i < n; i++) {
    const d = rx[i] - ry[i];
    sumD2 += d * d;
  }
  return 1 - (6 * sumD2) / (n * (n * n - 1));
}

function drawCalibration(
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

  const toX = (v: number) => PL + (v + 1) * pw / 2;
  const toY = (v: number) => PT + (1 - v) * ph / 2;

  ctx.fillStyle = "#f9fafb";
  ctx.fillRect(PL, PT, pw, ph);

  const pixelShift = 0.15 * ph / 2;
  ctx.save();
  ctx.beginPath();
  ctx.rect(PL, PT, pw, ph);
  ctx.clip();
  ctx.beginPath();
  ctx.moveTo(PL, toY(-1) - pixelShift);
  ctx.lineTo(PL + pw, toY(1) - pixelShift);
  ctx.lineTo(PL + pw, toY(1) + pixelShift);
  ctx.lineTo(PL, toY(-1) + pixelShift);
  ctx.closePath();
  ctx.fillStyle = "#6b728020";
  ctx.fill();
  ctx.restore();

  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.strokeRect(PL, PT, pw, ph);

  ctx.strokeStyle = "#d1d5db";
  ctx.lineWidth = 0.75;
  ctx.beginPath();
  ctx.moveTo(toX(0), PT); ctx.lineTo(toX(0), PT + ph);
  ctx.moveTo(PL, toY(0)); ctx.lineTo(PL + pw, toY(0));
  ctx.stroke();

  ctx.save();
  ctx.beginPath();
  ctx.rect(PL, PT, pw, ph);
  ctx.clip();
  ctx.strokeStyle = "#6b7280";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PL, toY(-1));
  ctx.lineTo(PL + pw, toY(1));
  ctx.stroke();
  ctx.restore();

  ctx.font = "8px sans-serif";
  ctx.fillStyle = "#9ca3af";
  ctx.textAlign = "center";
  for (const v of [-1, 0, 1]) {
    ctx.fillText(v > 0 ? `+${v}` : `${v}`, toX(v), PT + ph + 12);
  }
  ctx.textAlign = "right";
  for (const v of [-1, 0, 1]) {
    ctx.fillText(v > 0 ? `+${v}` : `${v}`, PL - 4, toY(v) + 3);
  }

  ctx.fillStyle = "#6b7280";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Evidence differential  (eA − eT)", PL + pw / 2, height - 4);
  ctx.save();
  ctx.translate(11, PT + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("Routing response  (gA − gT)", 0, 0);
  ctx.restore();

  const calibBins = computeCalibBins(points);
  const binPxW = pw / N_BINS;

  ctx.save();
  ctx.beginPath();
  ctx.rect(PL, PT, pw, ph);
  ctx.clip();

  for (const bin of calibBins) {
    if (bin.count === 0) continue;
    const bx = PL + bin.idx * binPxW;
    const yTop = toY(Math.min(bin.meanDg + bin.stdDg, 1));
    const yBot = toY(Math.max(bin.meanDg - bin.stdDg, -1));
    ctx.fillStyle = (bin.regime ? regimeColor(bin.regime) : "#9ca3af") + "1f";
    ctx.fillRect(bx, yTop, binPxW, yBot - yTop);
  }

  const hasSelection = selectedIds.size > 0;
  const colorGroups = new Map<string, ProjectionPoint[]>();
  for (const pt of points) {
    if (selectedIds.has(pt.node_id) || pt.node_id === hoveredId) continue;
    const c = ptColor(pt, colorBy);
    if (!colorGroups.has(c)) colorGroups.set(c, []);
    colorGroups.get(c)!.push(pt);
  }

  ctx.globalAlpha = hasSelection ? 0.06 : 0.09;
  for (const [color, pts] of colorGroups) {
    ctx.fillStyle = hasSelection ? "#9ca3af" : color;
    ctx.beginPath();
    for (const pt of pts) {
      const x = toX(pt.attribute_evidence - pt.topology_evidence);
      const y = toY(pt.gate_attribute - pt.gate_topology);
      ctx.moveTo(x + 2, y);
      ctx.arc(x, y, 2, 0, Math.PI * 2);
    }
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  for (const id of selectedIds) {
    const pt = points.find(p => p.node_id === id);
    if (!pt || pt.node_id === hoveredId) continue;
    const x = toX(pt.attribute_evidence - pt.topology_evidence);
    const y = toY(pt.gate_attribute - pt.gate_topology);
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = ptColor(pt, colorBy);
    ctx.fill();
    ctx.strokeStyle = "#1f2937";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  if (hoveredId) {
    const pt = points.find(p => p.node_id === hoveredId);
    if (pt) {
      const x = toX(pt.attribute_evidence - pt.topology_evidence);
      const y = toY(pt.gate_attribute - pt.gate_topology);
      ctx.beginPath();
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = ptColor(pt, colorBy);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  const filledBins = calibBins.filter(b => b.count > 0);
  if (filledBins.length > 1) {
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let i = 0; i < filledBins.length; i++) {
      const b = filledBins[i];
      const x = toX((b.idx + 0.5) / N_BINS * 2 - 1);
      const y = toY(b.meanDg);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  for (const b of filledBins) {
    const x = toX((b.idx + 0.5) / N_BINS * 2 - 1);
    const y = toY(b.meanDg);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.strokeStyle = b.regime ? regimeColor(b.regime) : "#9ca3af";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  ctx.restore();

  const annotations = [
    { ax: PL + 5, ay: PT + 14, label: "attr. over-routing", color: "#EF4444", align: "left" as const },
    { ax: PL + pw - 5, ay: PT + ph - 6, label: "topo. over-routing", color: "#14B8A6", align: "right" as const },
    { ax: PL + pw - 5, ay: PT + 14, label: "calibrated attr.", color: "#22C55E", align: "right" as const },
    { ax: PL + 5, ay: PT + ph - 6, label: "calibrated topo.", color: "#22C55E", align: "left" as const },
  ] as const;
  ctx.font = "7.5px sans-serif";
  for (const { ax, ay, label, color, align } of annotations) {
    ctx.fillStyle = color + "80";
    ctx.textAlign = align;
    ctx.fillText(label, ax, ay);
  }

  const deBins = computeBins2(points, p => p.attribute_evidence - p.topology_evidence);
  const stripBinW = pw / N_BINS;
  const stripBottomY = PT - 2;
  for (let i = 0; i < N_BINS; i++) {
    const { height: h, regime } = deBins[i];
    if (h === 0) continue;
    const barH = h * (STRIP_H - 4);
    ctx.fillStyle = (regime ? regimeColor(regime) : "#9ca3af") + "b3";
    ctx.fillRect(PL + i * stripBinW, stripBottomY - barH, Math.max(stripBinW - 0.5, 0.5), barH);
  }

  const dgBins = computeBins2(points, p => p.gate_attribute - p.gate_topology);
  const stripBinH = ph / N_BINS;
  const stripLeftX = PL + pw + 2;
  for (let i = 0; i < N_BINS; i++) {
    const { height: h, regime } = dgBins[i];
    if (h === 0) continue;
    const barW = h * (STRIP_W - 4);
    const bTop = PT + (N_BINS - i - 1) * stripBinH;
    ctx.fillStyle = (regime ? regimeColor(regime) : "#9ca3af") + "b3";
    ctx.fillRect(stripLeftX, bTop, barW, Math.max(stripBinH - 0.5, 0.5));
  }
}

interface TooltipState { x: number; y: number; pt: ProjectionPoint }

export function DualSpaceMap({ colorBy }: { colorBy: ColorMode }) {
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

  const spearman = useMemo(() => {
    if (!projection || projection.points.length < 4) return null;
    const pts = projection.points;
    const x = pts.map(p => p.attribute_evidence - p.topology_evidence);
    const y = pts.map(p => p.gate_attribute - p.gate_topology);
    return spearmanCorrelation(x, y);
  }, [projection]);

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
    drawCalibration(ctx, projection.points, size.width, size.height, colorBy, selectedSet, hoveredNodeId);
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
    let minD = 12;
    for (const pt of projection.points) {
      const x = PL + (pt.attribute_evidence - pt.topology_evidence + 1) * pw / 2;
      const y = PT + (1 - (pt.gate_attribute - pt.gate_topology)) * ph / 2;
      const d = Math.hypot(x - lx, y - ly);
      if (d < minD) { minD = d; closest = pt; }
    }
    return closest;
  }, [projection, size]);

  return (
    <Panel
      title="Routing Calibration"
      fill
      className="h-full"
      actions={
        spearman !== null ? (
          <span className="text-xs font-mono text-gray-500">ρ = {spearman.toFixed(3)}</span>
        ) : undefined
      }
    >
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
                <div className="text-gray-400 font-mono">
                  gA {(tooltip.pt.gate_attribute * 100).toFixed(0)}% · gT {(tooltip.pt.gate_topology * 100).toFixed(0)}%
                </div>
                <div className="text-gray-400 font-mono">
                  Δe {((tooltip.pt.attribute_evidence - tooltip.pt.topology_evidence) >= 0 ? "+" : "")}{((tooltip.pt.attribute_evidence - tooltip.pt.topology_evidence) * 100).toFixed(0)}% · Δg {((tooltip.pt.gate_attribute - tooltip.pt.gate_topology) >= 0 ? "+" : "")}{((tooltip.pt.gate_attribute - tooltip.pt.gate_topology) * 100).toFixed(0)}%
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}
