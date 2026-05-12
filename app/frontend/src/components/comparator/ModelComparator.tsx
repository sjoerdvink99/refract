import { useMetrics } from "@/hooks/useMetrics";
import { useSelectionStore } from "@/state/selectionStore";
import { Panel } from "@/components/layout/Panel";
import { EmptyState } from "@/components/shared/EmptyState";
import { LoadingState } from "@/components/shared/LoadingState";
import { formatPercent, formatFloat, formatRegimeLabel } from "@/utils/format";
import { REGIME_COLORS } from "@/utils/colors";

export function ModelComparator({ className = "" }: { className?: string }) {
  const { activeRunId } = useSelectionStore();
  const { data: metrics, isLoading } = useMetrics(activeRunId);

  if (!activeRunId) {
    return (
      <Panel title="Model Comparison" className={className}>
        <EmptyState message="No run selected" />
      </Panel>
    );
  }

  return (
    <Panel title="Model Comparison" className={className}>
      {isLoading && <LoadingState />}
      {metrics && (
        <div className="p-3 flex flex-col gap-4">
          {metrics.model_comparison && (
            <div>
              <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
                Global Performance
              </span>
              <table className="mt-2 w-full text-xs">
                <thead>
                  <tr className="text-gray-400 border-b border-gray-100">
                    <th className="text-left pb-1">Model</th>
                    <th className="text-right pb-1">Accuracy</th>
                    <th className="text-right pb-1">Macro F1</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(metrics.model_comparison).map(([name, m]) => (
                    <tr key={name} className="border-b border-gray-50 last:border-0">
                      <td className="py-1 font-medium capitalize">{name}</td>
                      <td className="py-1 text-right font-mono">
                        {formatPercent(m["accuracy"] ?? 0)}
                      </td>
                      <td className="py-1 text-right font-mono">
                        {formatFloat(m["macro_f1"] ?? 0)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div>
            <span className="text-xs font-semibold text-gray-600 uppercase tracking-wide">
              Accuracy by Regime
            </span>
            <div className="mt-2 flex flex-col gap-1.5">
              {metrics.by_regime.map((r) => (
                <div key={r.regime} className="flex items-center gap-2">
                  <div
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{
                      backgroundColor:
                        REGIME_COLORS[r.regime] ?? "#9CA3AF",
                    }}
                  />
                  <span className="text-xs text-gray-600 w-32">
                    {formatRegimeLabel(r.regime)}
                  </span>
                  <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-blue-500"
                      style={{ width: `${r.accuracy * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-gray-700 w-10">
                    {formatPercent(r.accuracy)}
                  </span>
                  <span className="text-xs text-gray-400">({r.count})</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </Panel>
  );
}
