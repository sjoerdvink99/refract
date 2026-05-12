import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { EgoGraph, GraphNode } from "@/api/types";
import type { ColorMode } from "@/api/types";
import { expertColor, regimeColor, correctnessColor } from "@/utils/colors";

interface EgoGraphCanvasProps {
  egoGraph: EgoGraph;
  colorBy: ColorMode;
  selectedNodeId: string;
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

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  node: GraphNode;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  source: SimNode | string;
  target: SimNode | string;
}

export function EgoGraphCanvas({
  egoGraph,
  colorBy,
  selectedNodeId,
  width,
  height,
}: EgoGraphCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dpr = window.devicePixelRatio || 1;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const simNodes: SimNode[] = egoGraph.nodes.map((n) => ({
      id: n.node_id,
      node: n,
      x: n.node_id === egoGraph.center_node_id ? width / 2 : undefined,
      y: n.node_id === egoGraph.center_node_id ? height / 2 : undefined,
      fx: n.node_id === egoGraph.center_node_id ? width / 2 : undefined,
      fy: n.node_id === egoGraph.center_node_id ? height / 2 : undefined,
    }));

    const nodeById = new Map(simNodes.map((n) => [n.id, n]));
    const simLinks: SimLink[] = egoGraph.edges
      .map((e) => ({
        source: nodeById.get(e.source) ?? e.source,
        target: nodeById.get(e.target) ?? e.target,
      }))
      .filter((l) => typeof l.source !== "string" && typeof l.target !== "string");

    const simulation = d3
      .forceSimulation<SimNode>(simNodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimLink>(simLinks)
          .id((d) => d.id)
          .distance(40)
          .strength(0.8),
      )
      .force("charge", d3.forceManyBody<SimNode>().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide<SimNode>(12))
      .stop();

    simulation.tick(300);

    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    ctx.strokeStyle = "#e5e7eb";
    ctx.lineWidth = 1;
    for (const link of simLinks) {
      const s = link.source as SimNode;
      const t = link.target as SimNode;
      if (s.x == null || s.y == null || t.x == null || t.y == null) continue;
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.stroke();
    }

    for (const sn of simNodes) {
      if (sn.x == null || sn.y == null) continue;
      const isCenter = sn.id === selectedNodeId;
      const radius = isCenter ? 9 : 6;
      const color = nodeColor(sn.node, colorBy);

      ctx.beginPath();
      ctx.arc(sn.x, sn.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      if (isCenter) {
        ctx.strokeStyle = "#111";
        ctx.lineWidth = 2;
        ctx.stroke();
      } else {
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    ctx.restore();
  }, [egoGraph, colorBy, selectedNodeId, width, height, dpr]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ display: "block", width: `${width}px`, height: `${height}px` }}
    />
  );
}
