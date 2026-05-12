import { useEffect, useRef } from "react";
import type { GraphEdge, GraphNode } from "@/api/types";
import type { ColorMode } from "@/api/types";
import { expertColor, regimeColor, correctnessColor } from "@/utils/colors";
import { computeViewBox, toCanvas } from "@/components/regime-map/projectionUtils";

interface SubgraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  colorBy: ColorMode;
  selectedNodeIds: string[];
  width: number;
  height: number;
}

function nodeColor(node: GraphNode, colorBy: ColorMode): string {
  switch (colorBy) {
    case "regime":
      return regimeColor(node.regime);
    case "expert":
      return expertColor(node.dominant_expert);
    case "correctness":
      return correctnessColor(node.correct);
    default:
      return regimeColor(node.regime);
  }
}

export function SubgraphCanvas({
  nodes,
  edges,
  colorBy,
  selectedNodeIds,
  width,
  height,
}: SubgraphCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dpr = window.devicePixelRatio || 1;

  useEffect(() => {
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
    ctx.clearRect(0, 0, width, height);

    const vb = computeViewBox(nodes);
    const selectedSet = new Set(selectedNodeIds);

    const positions = new Map<string, [number, number]>();
    for (const node of nodes) {
      positions.set(node.node_id, toCanvas(node.x, node.y, vb, width, height));
    }

    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    for (const edge of edges) {
      const s = positions.get(edge.source);
      const t = positions.get(edge.target);
      if (!s || !t) continue;
      ctx.beginPath();
      ctx.moveTo(s[0], s[1]);
      ctx.lineTo(t[0], t[1]);
      ctx.stroke();
    }

    for (const node of nodes) {
      const pos = positions.get(node.node_id);
      if (!pos) continue;
      const isSelected = selectedSet.has(node.node_id);
      const radius = isSelected ? 7 : 5;
      const color = nodeColor(node, colorBy);

      ctx.beginPath();
      ctx.arc(pos[0], pos[1], radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = isSelected ? "#111827" : "#fff";
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.stroke();
    }

    ctx.restore();
  }, [nodes, edges, colorBy, selectedNodeIds, width, height, dpr]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ display: "block", width: `${width}px`, height: `${height}px` }}
    />
  );
}
