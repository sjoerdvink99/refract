import type { ProjectionPoint } from "@/api/types";

export interface ViewBox {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export interface ZoomTransform {
  scale: number;
  tx: number;
  ty: number;
}

export const DEFAULT_TRANSFORM: ZoomTransform = { scale: 1, tx: 0, ty: 0 };

const MIN_SCALE = 0.8;
const MAX_SCALE = 24;
const GRID_RES = 40;

interface SpatialGrid {
  cells: number[][];
  cols: number;
  rows: number;
}

export function computeViewBox(points: { x: number; y: number }[], padding = 0.08): ViewBox {
  if (points.length === 0) return { xMin: -1, xMax: 1, yMin: -1, yMax: 1 };
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const p of points) {
    if (p.x < xMin) xMin = p.x;
    if (p.x > xMax) xMax = p.x;
    if (p.y < yMin) yMin = p.y;
    if (p.y > yMax) yMax = p.y;
  }
  const xPad = (xMax - xMin) * padding;
  const yPad = (yMax - yMin) * padding;
  return { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
}

function normalize(x: number, y: number, vb: ViewBox): [number, number] {
  return [
    (x - vb.xMin) / (vb.xMax - vb.xMin),
    (y - vb.yMin) / (vb.yMax - vb.yMin),
  ];
}

export function toCanvas(
  x: number,
  y: number,
  vb: ViewBox,
  width: number,
  height: number,
): [number, number] {
  const [nx, ny] = normalize(x, y, vb);
  return [nx * width, ny * height];
}

export function toCanvasZoomed(
  x: number,
  y: number,
  vb: ViewBox,
  width: number,
  height: number,
  t: ZoomTransform,
): [number, number] {
  const [nx, ny] = normalize(x, y, vb);
  return [nx * width * t.scale + t.tx, ny * height * t.scale + t.ty];
}

export function fromCanvas(
  cx: number,
  cy: number,
  vb: ViewBox,
  width: number,
  height: number,
): [number, number] {
  return [vb.xMin + (cx / width) * (vb.xMax - vb.xMin), vb.yMin + (cy / height) * (vb.yMax - vb.yMin)];
}

export function zoomAt(t: ZoomTransform, cx: number, cy: number, factor: number): ZoomTransform {
  const newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, t.scale * factor));
  const f = newScale / t.scale;
  return { scale: newScale, tx: cx - (cx - t.tx) * f, ty: cy - (cy - t.ty) * f };
}

export function panBy(t: ZoomTransform, dx: number, dy: number): ZoomTransform {
  return { scale: t.scale, tx: t.tx + dx, ty: t.ty + dy };
}

function buildGrid(points: ProjectionPoint[], vb: ViewBox): SpatialGrid {
  const cells: number[][] = Array.from({ length: GRID_RES * GRID_RES }, () => []);
  points.forEach((pt, idx) => {
    const [nx, ny] = normalize(pt.x, pt.y, vb);
    const col = Math.max(0, Math.min(GRID_RES - 1, Math.floor(nx * GRID_RES)));
    const row = Math.max(0, Math.min(GRID_RES - 1, Math.floor(ny * GRID_RES)));
    cells[row * GRID_RES + col].push(idx);
  });
  return { cells, cols: GRID_RES, rows: GRID_RES };
}

let _cachedGrid: SpatialGrid | null = null;
let _cachedGridPoints: ProjectionPoint[] | null = null;
let _cachedGridVb: ViewBox | null = null;

function getOrBuildGrid(points: ProjectionPoint[], vb: ViewBox): SpatialGrid {
  if (
    _cachedGrid !== null &&
    _cachedGridPoints === points &&
    _cachedGridVb === vb
  ) {
    return _cachedGrid;
  }
  _cachedGrid = buildGrid(points, vb);
  _cachedGridPoints = points;
  _cachedGridVb = vb;
  return _cachedGrid;
}

export function hitTestZoomed(
  px: number,
  py: number,
  points: ProjectionPoint[],
  vb: ViewBox,
  width: number,
  height: number,
  t: ZoomTransform,
  radiusPx = 8,
): ProjectionPoint | null {
  const grid = getOrBuildGrid(points, vb);

  const nx = (px - t.tx) / (t.scale * width);
  const ny = (py - t.ty) / (t.scale * height);

  const rNx = radiusPx / (t.scale * width);
  const rNy = radiusPx / (t.scale * height);

  const col0 = Math.max(0, Math.floor((nx - rNx) * GRID_RES));
  const col1 = Math.min(GRID_RES - 1, Math.ceil((nx + rNx) * GRID_RES));
  const row0 = Math.max(0, Math.floor((ny - rNy) * GRID_RES));
  const row1 = Math.min(GRID_RES - 1, Math.ceil((ny + rNy) * GRID_RES));

  let best: ProjectionPoint | null = null;
  let bestDist = Infinity;

  for (let row = row0; row <= row1; row++) {
    for (let col = col0; col <= col1; col++) {
      for (const idx of grid.cells[row * GRID_RES + col]) {
        const pt = points[idx];
        const [cx, cy] = toCanvasZoomed(pt.x, pt.y, vb, width, height, t);
        const dist = Math.hypot(cx - px, cy - py);
        if (dist < radiusPx && dist < bestDist) {
          bestDist = dist;
          best = pt;
        }
      }
    }
  }
  return best;
}

export function hitTest(
  px: number,
  py: number,
  points: ProjectionPoint[],
  vb: ViewBox,
  width: number,
  height: number,
  radius = 8,
): ProjectionPoint | null {
  return hitTestZoomed(px, py, points, vb, width, height, DEFAULT_TRANSFORM, radius);
}

function pointInPolygon(px: number, py: number, poly: [number, number][]): boolean {
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

export function lassoSelect(
  polygon: [number, number][],
  points: ProjectionPoint[],
  vb: ViewBox,
  width: number,
  height: number,
  t: ZoomTransform,
): ProjectionPoint[] {
  if (polygon.length < 3) return [];
  return points.filter((pt) => {
    const [cx, cy] = toCanvasZoomed(pt.x, pt.y, vb, width, height, t);
    return pointInPolygon(cx, cy, polygon);
  });
}
