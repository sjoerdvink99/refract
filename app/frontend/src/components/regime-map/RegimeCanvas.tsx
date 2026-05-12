import { forwardRef, useCallback, useEffect, useRef } from "react";
import type { ProjectionPoint } from "@/api/types";
import type { ColorMode } from "@/api/types";
import { correctnessColor, expertColor, regimeColor } from "@/utils/colors";
import {
  computeViewBox,
  DEFAULT_TRANSFORM,
  hitTestZoomed,
  lassoSelect,
  panBy,
  toCanvasZoomed,
  zoomAt,
  type ViewBox,
  type ZoomTransform,
} from "./projectionUtils";

interface RegimeCanvasProps {
  points: ProjectionPoint[];
  colorBy: ColorMode;
  selectedNodeIds: string[];
  hoveredNodeId: string | null;
  onNodeToggle: (nodeId: string) => void;
  onNodesLasso: (nodeIds: string[]) => void;
  onEmptyClick: () => void;
  onNodeHover: (nodeId: string | null, event?: MouseEvent) => void;
  width: number;
  height: number;
}

function pointColor(pt: ProjectionPoint, colorBy: ColorMode): string {
  switch (colorBy) {
    case "regime":
      return regimeColor(pt.regime);
    case "expert":
      return expertColor(pt.dominant_expert);
    case "correctness":
      return correctnessColor(pt.correct);
  }
}

function hexAlpha(alpha: number): string {
  return Math.round(Math.min(1, Math.max(0, alpha)) * 255)
    .toString(16)
    .padStart(2, "0");
}

function drawScene(
  ctx: CanvasRenderingContext2D,
  points: ProjectionPoint[],
  vb: ViewBox,
  width: number,
  height: number,
  colorBy: ColorMode,
  selectedNodeIds: string[],
  hoveredNodeId: string | null,
  t: ZoomTransform,
  lassoPolygon: [number, number][],
): void {
  ctx.clearRect(0, 0, width, height);

  const hasSelection = selectedNodeIds.length > 0;
  const selectedSet = new Set(selectedNodeIds);
  const isSingle = selectedNodeIds.length === 1;

  for (const pt of points) {
    if (selectedSet.has(pt.node_id) || pt.node_id === hoveredNodeId) continue;
    const [cx, cy] = toCanvasZoomed(pt.x, pt.y, vb, width, height, t);
    const alpha = hasSelection ? 0.18 : 0.75;
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = pointColor(pt, colorBy) + hexAlpha(alpha);
    ctx.fill();
    const gateCommitment = Math.abs(pt.gate_attribute - pt.gate_topology);
    if (gateCommitment < 0.4) {
      const ringOpacity = ((0.4 - gateCommitment) / 0.4) * (hasSelection ? 0.15 : 0.5);
      ctx.beginPath();
      ctx.arc(cx, cy, 5.5, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(107,114,128,${ringOpacity.toFixed(2)})`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  for (const id of selectedNodeIds) {
    const pt = points.find((p) => p.node_id === id);
    if (!pt || pt.node_id === hoveredNodeId) continue;
    const [cx, cy] = toCanvasZoomed(pt.x, pt.y, vb, width, height, t);
    const r = isSingle ? 8 : 5;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = pointColor(pt, colorBy);
    ctx.fill();
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = isSingle ? 2 : 1.5;
    ctx.stroke();
  }

  if (hoveredNodeId) {
    const pt = points.find((p) => p.node_id === hoveredNodeId);
    if (pt) {
      const [cx, cy] = toCanvasZoomed(pt.x, pt.y, vb, width, height, t);
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fillStyle = pointColor(pt, colorBy);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  if (lassoPolygon.length > 1) {
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 1.5;
    ctx.fillStyle = "rgba(55, 65, 81, 0.08)";
    ctx.beginPath();
    ctx.moveTo(lassoPolygon[0][0], lassoPolygon[0][1]);
    for (let i = 1; i < lassoPolygon.length; i++) {
      ctx.lineTo(lassoPolygon[i][0], lassoPolygon[i][1]);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

export const RegimeCanvas = forwardRef<HTMLCanvasElement, RegimeCanvasProps>(
  function RegimeCanvas(
    {
      points,
      colorBy,
      selectedNodeIds,
      hoveredNodeId,
      onNodeToggle,
      onNodesLasso,
      onEmptyClick,
      onNodeHover,
      width,
      height,
    },
    ref,
  ) {
    const internalRef = useRef<HTMLCanvasElement>(null);
    const canvasRef = (ref as React.RefObject<HTMLCanvasElement>) ?? internalRef;
    const vbRef = useRef<ViewBox>(computeViewBox(points));
    const transformRef = useRef<ZoomTransform>(DEFAULT_TRANSFORM);
    const isDraggingRef = useRef(false);
    const hasDraggedRef = useRef(false);
    const lastPosCssRef = useRef<[number, number]>([0, 0]);
    const isLassoModeRef = useRef(false);
    const lassoPointsRef = useRef<[number, number][]>([]);
    const wasLassoRef = useRef(false);

    const dpr = window.devicePixelRatio || 1;

    useEffect(() => {
      vbRef.current = computeViewBox(points);
      transformRef.current = DEFAULT_TRANSFORM;
    }, [points]);

    const redraw = useCallback(
      (lasso: [number, number][] = []) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
          canvas.width = width * dpr;
          canvas.height = height * dpr;
        }
        ctx.save();
        ctx.scale(dpr, dpr);
        drawScene(
          ctx,
          points,
          vbRef.current,
          width,
          height,
          colorBy,
          selectedNodeIds,
          hoveredNodeId,
          transformRef.current,
          lasso,
        );
        ctx.restore();
      },
      [points, width, height, colorBy, selectedNodeIds, hoveredNodeId, dpr, canvasRef],
    );

    useEffect(() => {
      redraw();
    }, [redraw]);

    const eventToLogical = useCallback(
      (clientX: number, clientY: number): [number, number] | null => {
        const canvas = canvasRef.current;
        if (!canvas) return null;
        const rect = canvas.getBoundingClientRect();
        const canvasRatio = width / height;
        const rectRatio = rect.width / rect.height;
        let renderW, renderH, ox, oy;
        if (canvasRatio > rectRatio) {
          renderW = rect.width;
          renderH = rect.width / canvasRatio;
          ox = 0;
          oy = (rect.height - renderH) / 2;
        } else {
          renderH = rect.height;
          renderW = rect.height * canvasRatio;
          ox = (rect.width - renderW) / 2;
          oy = 0;
        }
        const lx = ((clientX - rect.left - ox) / renderW) * width;
        const ly = ((clientY - rect.top - oy) / renderH) * height;
        return [lx, ly];
      },
      [width, height, canvasRef],
    );

    const cssToLogicalScale = useCallback((): number => {
      const canvas = canvasRef.current;
      if (!canvas) return 1;
      const rect = canvas.getBoundingClientRect();
      const canvasRatio = width / height;
      const rectRatio = rect.width / rect.height;
      return canvasRatio > rectRatio ? width / rect.width : height / rect.height;
    }, [width, height, canvasRef]);

    const redrawRef = useRef(redraw);
    useEffect(() => {
      redrawRef.current = redraw;
    }, [redraw]);

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const handleWheel = (e: WheelEvent) => {
        e.preventDefault();
        const pos = eventToLogical(e.clientX, e.clientY);
        if (!pos) return;
        const factor = e.deltaY < 0 ? 1.25 : 1 / 1.25;
        transformRef.current = zoomAt(transformRef.current, pos[0], pos[1], factor);
        redrawRef.current();
      };
      canvas.addEventListener("wheel", handleWheel, { passive: false });
      return () => canvas.removeEventListener("wheel", handleWheel);
    }, [eventToLogical, canvasRef]);

    const handleMouseDown = useCallback(
      (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (e.button !== 0) return;
        wasLassoRef.current = false;
        const pos = eventToLogical(e.clientX, e.clientY);
        if (!pos) return;
        if (e.shiftKey) {
          isLassoModeRef.current = true;
          lassoPointsRef.current = [pos];
        } else {
          isDraggingRef.current = true;
          hasDraggedRef.current = false;
          lastPosCssRef.current = [e.clientX, e.clientY];
          if (canvasRef.current) canvasRef.current.style.cursor = "grabbing";
        }
      },
      [canvasRef, eventToLogical],
    );

    const handleMouseMove = useCallback(
      (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (isLassoModeRef.current) {
          const pos = eventToLogical(e.clientX, e.clientY);
          if (!pos) return;
          lassoPointsRef.current = [...lassoPointsRef.current, pos];
          redraw(lassoPointsRef.current);
          return;
        }

        if (isDraggingRef.current) {
          const dx = e.clientX - lastPosCssRef.current[0];
          const dy = e.clientY - lastPosCssRef.current[1];
          if (Math.abs(dx) > 1 || Math.abs(dy) > 1) hasDraggedRef.current = true;
          lastPosCssRef.current = [e.clientX, e.clientY];
          const s = cssToLogicalScale();
          transformRef.current = panBy(transformRef.current, dx * s, dy * s);
          redraw();
          return;
        }

        const pos = eventToLogical(e.clientX, e.clientY);
        if (!pos) return;
        const [lx, ly] = pos;
        if (lx < 0 || lx > width || ly < 0 || ly > height) {
          onNodeHover(null);
          return;
        }
        const hit = hitTestZoomed(lx, ly, points, vbRef.current, width, height, transformRef.current);
        onNodeHover(hit?.node_id ?? null, e.nativeEvent);
      },
      [points, width, height, onNodeHover, eventToLogical, cssToLogicalScale, redraw],
    );

    const handleMouseUp = useCallback(
      (_e: React.MouseEvent<HTMLCanvasElement>) => {
        if (isLassoModeRef.current) {
          isLassoModeRef.current = false;
          wasLassoRef.current = true;
          const polygon = lassoPointsRef.current;
          lassoPointsRef.current = [];
          if (polygon.length >= 3) {
            const selected = lassoSelect(polygon, points, vbRef.current, width, height, transformRef.current);
            onNodesLasso(selected.map((p) => p.node_id));
          }
          redraw();
          return;
        }
        isDraggingRef.current = false;
        if (canvasRef.current) canvasRef.current.style.cursor = "crosshair";
      },
      [points, width, height, onNodesLasso, canvasRef, redraw],
    );

    const handleClick = useCallback(
      (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (wasLassoRef.current) {
          wasLassoRef.current = false;
          return;
        }
        if (hasDraggedRef.current) return;
        const pos = eventToLogical(e.clientX, e.clientY);
        if (!pos) return;
        const [lx, ly] = pos;
        if (lx < 0 || lx > width || ly < 0 || ly > height) return;
        const hit = hitTestZoomed(lx, ly, points, vbRef.current, width, height, transformRef.current);
        if (hit) {
          onNodeToggle(hit.node_id);
        } else {
          onEmptyClick();
        }
      },
      [points, width, height, onNodeToggle, onEmptyClick, eventToLogical],
    );

    return (
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        data-regime-canvas="true"
        style={{
          cursor: "crosshair",
          display: "block",
          width: "100%",
          height: "100%",
          objectFit: "contain",
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
        onMouseLeave={() => {
          isLassoModeRef.current = false;
          isDraggingRef.current = false;
          lassoPointsRef.current = [];
          if (canvasRef.current) canvasRef.current.style.cursor = "crosshair";
          redraw();
          onNodeHover(null);
        }}
      />
    );
  },
);
