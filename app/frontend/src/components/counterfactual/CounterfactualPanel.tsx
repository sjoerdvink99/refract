import { useQuery, useQueries } from "@tanstack/react-query";
import { api } from "@/api/endpoints";
import { useSelectionStore } from "@/state/selectionStore";
import { Panel } from "@/components/layout/Panel";
import { EmptyState } from "@/components/shared/EmptyState";
import { LoadingState } from "@/components/shared/LoadingState";
import { AlphaBar } from "@/components/shared/AlphaBar";
import { formatFloat, formatPercent } from "@/utils/format";
import type { CounterfactualResult } from "@/api/types";

const MAX_NODES = 20;

function DamageRow({ result }: { result: CounterfactualResult }) {
  const isFlip = result.original_pred_label !== result.counterfactual_pred_label;

  return (
    <div className="flex items-start justify-between py-2 border-b border-gray-50 last:border-0 gap-2">
      <div className="flex flex-col gap-0.5 flex-1">
        <span className="text-xs font-medium text-gray-700">
          {result.mode.replace(/_/g, " ")}
        </span>
        <span className="text-xs text-gray-400">
          {result.original_pred_label} → {result.counterfactual_pred_label}
          {isFlip && (
            <span className="ml-1 text-red-500 font-medium">flip</span>
          )}
        </span>
      </div>
      <div className="flex flex-col items-end gap-1 flex-shrink-0">
        <div className="flex flex-col items-end gap-0.5">
          <span className="text-xs text-gray-500">Damage</span>
          <span
            className={`text-xs font-mono font-medium ${
              result.damage > 0.1 ? "text-red-600" : "text-gray-700"
            }`}
          >
            {formatFloat(result.damage, 3)}
          </span>
        </div>
        <AlphaBar attribute={result.gate_attribute} topology={result.gate_topology} size={64} />
      </div>
    </div>
  );
}

interface AggregateModeRow {
  mode: string;
  avgDamage: number;
  flipRate: number;
  count: number;
}

function AggregateRow({ row }: { row: AggregateModeRow }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
      <div className="flex flex-col gap-0.5">
        <span className="text-xs font-medium text-gray-700">
          {row.mode.replace(/_/g, " ")}
        </span>
        <span className="text-xs text-gray-400">
          {row.count} nodes · flip rate:{" "}
          <span className={row.flipRate > 0.3 ? "text-red-500 font-medium" : ""}>
            {formatPercent(row.flipRate)}
          </span>
        </span>
      </div>
      <div className="flex flex-col items-end gap-0.5">
        <span className="text-xs text-gray-500">Avg Damage</span>
        <span
          className={`text-xs font-mono font-medium ${
            row.avgDamage > 0.1 ? "text-red-600" : "text-gray-700"
          }`}
        >
          {formatFloat(row.avgDamage, 3)}
        </span>
      </div>
    </div>
  );
}

function SingleNodePanel({
  activeRunId,
  nodeId,
  className,
}: {
  activeRunId: string;
  nodeId: string;
  className: string;
}) {
  const { data, isLoading } = useQuery({
    queryKey: ["runs", activeRunId, "counterfactuals", nodeId],
    queryFn: () => api.counterfactuals.node(activeRunId, nodeId),
    staleTime: 60_000,
  });

  return (
    <Panel title="Counterfactuals" className={className}>
      {isLoading && <LoadingState />}
      {!isLoading && (!data || data.results.length === 0) && (
        <EmptyState
          message="No counterfactuals computed"
          hint="Run experiment with counterfactuals enabled"
        />
      )}
      {data && data.results.length > 0 && (
        <div className="flex flex-col">
          <p className="text-xs text-gray-400 px-3 pt-2 pb-1">Each row shows how a perturbation shifts routing and prediction confidence.</p>
          <div className="p-3 flex flex-col">
            {data.results.map((r) => (
              <DamageRow key={r.mode} result={r} />
            ))}
          </div>
        </div>
      )}
    </Panel>
  );
}

function MultiNodePanel({
  activeRunId,
  nodeIds,
  className,
}: {
  activeRunId: string;
  nodeIds: string[];
  className: string;
}) {
  const capped = nodeIds.slice(0, MAX_NODES);

  const queries = useQueries({
    queries: capped.map((nodeId) => ({
      queryKey: ["runs", activeRunId, "counterfactuals", nodeId],
      queryFn: () => api.counterfactuals.node(activeRunId, nodeId),
      staleTime: 60_000,
    })),
  });

  const isLoading = queries.some((q) => q.isLoading);
  const allResults = queries.flatMap((q) => q.data?.results ?? []);

  const byMode = new Map<string, CounterfactualResult[]>();
  for (const r of allResults) {
    const list = byMode.get(r.mode) ?? [];
    list.push(r);
    byMode.set(r.mode, list);
  }

  const aggregateRows: AggregateModeRow[] = Array.from(byMode.entries()).map(
    ([mode, results]) => ({
      mode,
      avgDamage: results.reduce((s, r) => s + r.damage, 0) / results.length,
      flipRate:
        results.filter((r) => r.original_pred_label !== r.counterfactual_pred_label).length /
        results.length,
      count: results.length,
    }),
  );

  const caption =
    nodeIds.length > MAX_NODES
      ? `First ${MAX_NODES} of ${nodeIds.length} nodes`
      : `${capped.length} nodes`;

  return (
    <Panel title={`Counterfactuals — ${caption}`} className={className}>
      {isLoading && <LoadingState />}
      {!isLoading && aggregateRows.length === 0 && (
        <EmptyState
          message="No counterfactuals computed"
          hint="Run experiment with counterfactuals enabled"
        />
      )}
      {aggregateRows.length > 0 && (
        <div className="p-3 flex flex-col">
          {aggregateRows.map((row) => (
            <AggregateRow key={row.mode} row={row} />
          ))}
        </div>
      )}
    </Panel>
  );
}

export function CounterfactualPanel({ className = "" }: { className?: string }) {
  const { activeRunId, selectedNodeIds } = useSelectionStore();

  if (selectedNodeIds.length === 0 || !activeRunId) {
    return (
      <Panel title="Counterfactuals" className={className}>
        <EmptyState message="No node selected" />
      </Panel>
    );
  }

  if (selectedNodeIds.length === 1) {
    return (
      <SingleNodePanel
        activeRunId={activeRunId}
        nodeId={selectedNodeIds[0]}
        className={className}
      />
    );
  }

  return (
    <MultiNodePanel
      activeRunId={activeRunId}
      nodeIds={selectedNodeIds}
      className={className}
    />
  );
}
