import { useMetrics } from "@/hooks/useMetrics";
import { useRunManifest } from "@/hooks/useRunManifest";
import { useSelectionStore } from "@/state/selectionStore";
import { Panel } from "@/components/layout/Panel";
import { EmptyState } from "@/components/shared/EmptyState";
import { LoadingState } from "@/components/shared/LoadingState";
import { formatFloat, formatPercent, formatRegimeLabel } from "@/utils/format";
import { REGIME_COLORS, EXPERT_COLORS } from "@/utils/colors";

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5 bg-gray-50 rounded p-2.5">
      <span className="text-xs text-gray-400">{label}</span>
      <span className="text-sm font-mono font-medium text-gray-800">{value}</span>
    </div>
  );
}

export function DatasetFingerprint({ className = "" }: { className?: string }) {
  const { activeRunId } = useSelectionStore();
  const { data: metrics, isLoading } = useMetrics(activeRunId);
  const { data: manifest } = useRunManifest(activeRunId);

  if (!activeRunId) {
    return (
      <Panel title="Dataset Fingerprint" className={className}>
        <EmptyState message="No run selected" />
      </Panel>
    );
  }

  return (
    <Panel title="Dataset Fingerprint" className={className}>
      {isLoading && <LoadingState />}
      {metrics && (
        <div className="p-3 flex flex-col gap-4">
          <div className="grid grid-cols-6 gap-2">
            {manifest && (
              <>
                <MetricCard label="Nodes" value={manifest.dataset.num_nodes.toLocaleString()} />
                <MetricCard label="Edges" value={manifest.dataset.num_edges.toLocaleString()} />
                <MetricCard label="Classes" value={String(manifest.dataset.num_classes)} />
                <MetricCard label="Feature Dim" value={String(manifest.dataset.feature_dim)} />
              </>
            )}
            <MetricCard label="Accuracy" value={formatPercent(metrics.accuracy)} />
            <MetricCard label="Macro F1" value={formatFloat(metrics.macro_f1)} />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                Regime Distribution
              </span>
              <div className="mt-2 flex flex-col gap-1">
                {Object.entries(metrics.regime_distribution).map(([regime, frac]) => (
                  <div key={regime} className="flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: REGIME_COLORS[regime as keyof typeof REGIME_COLORS] ?? "#9CA3AF" }}
                    />
                    <span className="text-xs text-gray-600 flex-1">
                      {formatRegimeLabel(regime)}
                    </span>
                    <span className="text-xs font-mono text-gray-700">
                      {formatPercent(frac)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                Expert Usage
              </span>
              <div className="mt-2 flex flex-col gap-1">
                {Object.entries(metrics.expert_usage).map(([expert, usage]) => (
                  <div key={expert} className="flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: EXPERT_COLORS[expert as keyof typeof EXPERT_COLORS] ?? "#9CA3AF" }}
                    />
                    <span className="text-xs text-gray-600 flex-1 capitalize">{expert}</span>
                    <span className="text-xs font-mono text-gray-700">
                      {formatPercent(usage)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                Routing Metrics
              </span>
              <div className="mt-2 flex flex-col gap-1">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Gate Cal. L1</span>
                  <span className="font-mono">{formatFloat(metrics.routing.gate_calibration_l1)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Gate Cal. Rank</span>
                  <span className="font-mono">{formatFloat(metrics.routing.gate_calibration_rank)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">CF Fidelity</span>
                  <span className="font-mono">{formatPercent(metrics.routing.counterfactual_routing_fidelity)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Gate Smoothness</span>
                  <span className="font-mono">{formatFloat(metrics.routing.gate_smoothness)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Gate Entropy</span>
                  <span className="font-mono">{formatFloat(metrics.routing.mean_gate_entropy)}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-500">Routing–Regime MI</span>
                  <span className="font-mono">{formatFloat(metrics.routing.routing_regime_mi)}</span>
                </div>
              </div>
            </div>
          </div>

          {metrics.per_expert_ablation && (
            <div>
              <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                Per-Expert Ablation
              </span>
              <div className="mt-2 flex flex-col gap-1">
                {Object.entries(metrics.per_expert_ablation).map(([expert, perf]) => (
                  <div key={expert} className="flex items-center justify-between text-xs">
                    <span className="text-gray-600 capitalize">{expert}</span>
                    <span className="font-mono text-gray-700">{formatPercent(perf.accuracy)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
