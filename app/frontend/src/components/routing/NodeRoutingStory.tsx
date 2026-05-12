import { useMemo } from "react";
import { useNodeDetail } from "@/hooks/useNodeDetails";
import { useSelectionStore } from "@/state/selectionStore";
import { LoadingState } from "@/components/shared/LoadingState";
import { EXPERT_COLORS } from "@/utils/colors";
import { formatPercent } from "@/utils/format";
import type { CounterfactualSummary, ProjectionPoint } from "@/api/types";

interface Props {
  nodeId: string;
  allPoints: ProjectionPoint[];
}

type Verdict =
  | "correctly-attr"
  | "correctly-topo"
  | "attr-over-routing"
  | "topo-over-routing"
  | "balanced";

function getVerdict(de: number, dg: number): Verdict {
  if (Math.abs(dg) < 0.1) return "balanced";
  if (de > 0.1 && dg > 0.1) return "correctly-attr";
  if (de < -0.1 && dg < -0.1) return "correctly-topo";
  if (de < -0.1 && dg > 0.1) return "attr-over-routing";
  if (de > 0.1 && dg < -0.1) return "topo-over-routing";
  return "balanced";
}

const VERDICT_META: Record<
  Verdict,
  { label: string; color: string; bg: string; miscalibrated: boolean }
> = {
  "correctly-attr": {
    label: "Correctly attribute-routed",
    color: "#22C55E",
    bg: "#dcfce7",
    miscalibrated: false,
  },
  "correctly-topo": {
    label: "Correctly topology-routed",
    color: "#22C55E",
    bg: "#dcfce7",
    miscalibrated: false,
  },
  "attr-over-routing": {
    label: "Attribute over-routing",
    color: "#EF4444",
    bg: "#fee2e2",
    miscalibrated: true,
  },
  "topo-over-routing": {
    label: "Topology over-routing",
    color: "#EF4444",
    bg: "#fee2e2",
    miscalibrated: true,
  },
  balanced: {
    label: "Balanced routing",
    color: "#9CA3AF",
    bg: "#f3f4f6",
    miscalibrated: false,
  },
};

function knnInRegimeSpace(
  target: ProjectionPoint,
  allPoints: ProjectionPoint[],
  k = 5,
): ProjectionPoint[] {
  const scored = allPoints
    .filter((p) => p.node_id !== target.node_id)
    .map((p) => {
      const d = Math.sqrt(
        (p.attribute_evidence - target.attribute_evidence) ** 2 +
          (p.topology_evidence - target.topology_evidence) ** 2 +
          (p.uncertainty - target.uncertainty) ** 2,
      );
      return { pt: p, d };
    });
  scored.sort((a, b) => a.d - b.d);
  return scored.slice(0, k).map((s) => s.pt);
}

function neighborVerdict(pt: ProjectionPoint): Verdict {
  return getVerdict(
    pt.attribute_evidence - pt.topology_evidence,
    pt.gate_attribute - pt.gate_topology,
  );
}

function NeighborBadge({ pt }: { pt: ProjectionPoint }) {
  const v = neighborVerdict(pt);
  const miscal = VERDICT_META[v].miscalibrated;
  const correctColor =
    pt.correct === true ? "#22C55E" : pt.correct === false ? "#EF4444" : "#9CA3AF";
  return (
    <div
      className="w-6 h-6 rounded flex items-center justify-center text-white text-[9px] font-bold border-2 shrink-0"
      style={{ backgroundColor: correctColor, borderColor: miscal ? "#EF4444" : "#22C55E" }}
      title={`${pt.node_id} · ${v.replace(/-/g, " ")} · ${pt.correct === true ? "correct" : pt.correct === false ? "incorrect" : "unknown"}`}
    >
      {miscal ? "✗" : "✓"}
    </div>
  );
}

type RoutingConsequence = {
  kind: "consequential" | "both-wrong" | "robust";
  cf: CounterfactualSummary;
  cfCorrect: boolean;
  summary: string;
  v: "attr-over-routing" | "topo-over-routing";
};

export function NodeRoutingStory({ nodeId, allPoints }: Props) {
  const { activeRunId } = useSelectionStore();
  const { data: node, isLoading } = useNodeDetail(activeRunId, nodeId);

  const selectedPt = useMemo(
    () => allPoints.find((p) => p.node_id === nodeId) ?? null,
    [allPoints, nodeId],
  );

  const neighbors = useMemo(
    () => (selectedPt ? knnInRegimeSpace(selectedPt, allPoints, 5) : []),
    [selectedPt, allPoints],
  );

  const neighborsMiscal = useMemo(
    () =>
      neighbors.filter((p) => {
        const v = neighborVerdict(p);
        return VERDICT_META[v].miscalibrated;
      }).length,
    [neighbors],
  );

  const verdict = useMemo(() => {
    if (!selectedPt) return "balanced" as Verdict;
    return getVerdict(
      selectedPt.attribute_evidence - selectedPt.topology_evidence,
      selectedPt.gate_attribute - selectedPt.gate_topology,
    );
  }, [selectedPt]);

  const routingConsequence = useMemo((): RoutingConsequence | null => {
    if (!node || !selectedPt) return null;
    const v = verdict;
    if (v !== "attr-over-routing" && v !== "topo-over-routing") return null;
    const cfMode = v === "attr-over-routing" ? "topology_expert_only" : "attribute_expert_only";
    const cf = node.counterfactuals.find((c) => c.mode === cfMode);
    if (!cf) return null;
    const cfCorrect = node.true_label !== null && cf.counterfactual_pred_label === node.true_label;
    const origCorrect = node.correct;
    if (!origCorrect && cfCorrect) {
      return { kind: "consequential", cf, cfCorrect, summary: "Misrouting cost a correct prediction.", v };
    }
    if (!origCorrect && !cfCorrect) {
      return { kind: "both-wrong", cf, cfCorrect, summary: "Routing did not determine the outcome.", v };
    }
    if (origCorrect) {
      return { kind: "robust", cf, cfCorrect, summary: "Correct despite miscalibrated routing.", v };
    }
    return null;
  }, [node, selectedPt, verdict]);

  const meta = VERDICT_META[verdict];

  if (isLoading) return <LoadingState />;

  return (
    <div className="flex flex-col gap-3 p-3 h-full overflow-y-auto">
      <div
        className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg"
        style={{ backgroundColor: meta.bg }}
      >
        <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: meta.color }} />
        <span className="text-xs font-semibold" style={{ color: meta.color }}>
          {meta.label}
        </span>
        {node && (
          <span className="ml-auto text-xs font-mono text-gray-500">{node.node_id}</span>
        )}
      </div>

      {selectedPt && (
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Evidence vs. Gate
          </span>
          {(
            [
              { label: "EVIDENCE", a: selectedPt.attribute_evidence, t: selectedPt.topology_evidence },
              { label: "GATE", a: selectedPt.gate_attribute, t: selectedPt.gate_topology },
            ] as const
          ).map(({ label, a, t }) => (
            <div key={label} className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-gray-400 w-14 shrink-0 text-right">
                {label}
              </span>
              <div className="flex-1 h-3 rounded overflow-hidden flex">
                <div
                  style={{ width: `${a * 100}%`, backgroundColor: EXPERT_COLORS.attribute }}
                />
                <div
                  style={{ width: `${t * 100}%`, backgroundColor: EXPERT_COLORS.topology }}
                />
              </div>
              <span className="text-[10px] font-mono text-gray-400 w-20 shrink-0">
                {(a * 100).toFixed(0)}%A&nbsp;/&nbsp;{(t * 100).toFixed(0)}%T
              </span>
            </div>
          ))}
          <p className="text-[10px] text-gray-400">
            Orange = attribute · Indigo = topology · Row alignment = well-calibrated.
          </p>
        </div>
      )}

      {neighbors.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            5 Nearest Regime Neighbors
          </span>
          <div className="flex items-center gap-1.5">
            {neighbors.map((nb) => (
              <NeighborBadge key={nb.node_id} pt={nb} />
            ))}
          </div>
          <p className="text-xs text-gray-500">
            {neighborsMiscal === 5
              ? "All 5 regime-similar nodes are also miscalibrated — structural failure mode."
              : neighborsMiscal >= 3
              ? `${neighborsMiscal}/5 regime-similar nodes are miscalibrated — likely structural.`
              : neighborsMiscal === 0
              ? "All 5 regime-similar nodes are well-calibrated — this node is anomalous."
              : `${neighborsMiscal}/5 regime-similar nodes are miscalibrated.`}
          </p>
          <p className="text-[10px] text-gray-400">
            Badge fill = correctness · Border = calibration status.
          </p>
        </div>
      )}

      {routingConsequence && node && selectedPt && (
        <div className="flex flex-col gap-2">
          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Routing Consequence
          </span>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex flex-col gap-1.5 p-2 rounded-lg bg-gray-50">
              <span className="text-[10px] font-semibold text-gray-500 tracking-wide">ACTUAL</span>
              <div className="flex h-2.5 rounded overflow-hidden">
                <div style={{ width: `${selectedPt.gate_attribute * 100}%`, backgroundColor: EXPERT_COLORS.attribute }} />
                <div style={{ width: `${selectedPt.gate_topology * 100}%`, backgroundColor: EXPERT_COLORS.topology }} />
              </div>
              <div className="font-mono text-gray-700">"{node.predicted_label}"</div>
              <div className="flex items-center gap-1">
                <span className="font-mono text-gray-400 text-[10px]">
                  {formatPercent(routingConsequence.cf.original_confidence)}
                </span>
                <span
                  className="font-semibold"
                  style={{
                    color:
                      node.correct === true
                        ? "#22C55E"
                        : node.correct === false
                        ? "#EF4444"
                        : "#9CA3AF",
                  }}
                >
                  {node.correct === true ? "✓" : node.correct === false ? "✗" : "?"}
                </span>
              </div>
            </div>
            <div
              className="flex flex-col gap-1.5 p-2 rounded-lg"
              style={{
                backgroundColor:
                  routingConsequence.kind === "consequential"
                    ? "#fee2e2"
                    : routingConsequence.kind === "robust"
                    ? "#f0fdf4"
                    : "#f9fafb",
              }}
            >
              <span className="text-[10px] font-semibold text-gray-500 tracking-wide">
                {routingConsequence.v === "attr-over-routing" ? "TOPO-ONLY" : "ATTR-ONLY"}
              </span>
              <div className="flex h-2.5 rounded overflow-hidden">
                {routingConsequence.v === "attr-over-routing" ? (
                  <div style={{ width: "100%", backgroundColor: EXPERT_COLORS.topology }} />
                ) : (
                  <div style={{ width: "100%", backgroundColor: EXPERT_COLORS.attribute }} />
                )}
              </div>
              <div className="font-mono text-gray-700">
                "{routingConsequence.cf.counterfactual_pred_label}"
              </div>
              <div className="flex items-center gap-1">
                <span className="font-mono text-gray-400 text-[10px]">
                  {formatPercent(routingConsequence.cf.counterfactual_confidence)}
                </span>
                <span
                  className="font-semibold"
                  style={{ color: routingConsequence.cfCorrect ? "#22C55E" : "#EF4444" }}
                >
                  {routingConsequence.cfCorrect ? "✓" : "✗"}
                </span>
              </div>
            </div>
          </div>
          <p className="text-[10px] text-gray-400">{routingConsequence.summary}</p>
        </div>
      )}

      {!routingConsequence && node && node.counterfactuals.length === 0 && (
        <p className="text-xs text-gray-400 mt-auto">No counterfactual data for this node.</p>
      )}
    </div>
  );
}
