import { useEgoGraph, useSubgraph } from "@/hooks/useNodeDetails";
import { useSelectionStore } from "@/state/selectionStore";
import { useViewStore } from "@/state/viewStore";
import { Panel } from "@/components/layout/Panel";
import { LoadingState } from "@/components/shared/LoadingState";
import { EmptyState } from "@/components/shared/EmptyState";
import { EgoGraphCanvas } from "./EgoGraphCanvas";
import { SubgraphCanvas } from "./SubgraphCanvas";

const CANVAS_SIZE = { width: 360, height: 280 };

export function GraphView({ className = "" }: { className?: string }) {
  const { activeRunId, selectedNodeIds } = useSelectionStore();
  const { colorBy, egoRadius } = useViewStore();

  const singleNodeId = selectedNodeIds.length === 1 ? selectedNodeIds[0] : null;
  const isMulti = selectedNodeIds.length > 1;

  const { data: ego, isLoading: egoLoading } = useEgoGraph(activeRunId, singleNodeId, egoRadius);
  const { data: subgraph, isLoading: subLoading } = useSubgraph(
    activeRunId,
    isMulti ? selectedNodeIds : [],
  );

  if (selectedNodeIds.length === 0) {
    return (
      <Panel title="Graph View" className={className}>
        <EmptyState message="No node selected" />
      </Panel>
    );
  }

  if (isMulti) {
    return (
      <Panel title={`Subgraph — ${selectedNodeIds.length} nodes`} className={className}>
        {subLoading && <LoadingState />}
        {!subLoading && !subgraph && (
          <EmptyState message="No subgraph data" hint="Subgraph unavailable" />
        )}
        {subgraph && (
          <div className="p-2">
            <SubgraphCanvas
              nodes={subgraph.nodes}
              edges={subgraph.edges}
              colorBy={colorBy}
              selectedNodeIds={selectedNodeIds}
              width={CANVAS_SIZE.width}
              height={CANVAS_SIZE.height}
            />
            <div className="mt-1 flex gap-3 text-xs text-gray-400">
              <span>{subgraph.nodes.length} nodes</span>
              <span>{subgraph.edges.length} edges</span>
            </div>
          </div>
        )}
      </Panel>
    );
  }

  return (
    <Panel
      title={`Ego Network (r=${egoRadius}) — ${singleNodeId}`}
      className={className}
    >
      {egoLoading && <LoadingState />}
      {!egoLoading && !ego && (
        <EmptyState message="No graph data" hint="Ego network unavailable" />
      )}
      {ego && (
        <div className="p-2">
          <EgoGraphCanvas
            egoGraph={ego}
            colorBy={colorBy}
            selectedNodeId={singleNodeId!}
            width={CANVAS_SIZE.width}
            height={CANVAS_SIZE.height}
          />
          <div className="mt-1 flex gap-3 text-xs text-gray-400">
            <span>{ego.nodes.length} nodes</span>
            <span>{ego.edges.length} edges</span>
          </div>
        </div>
      )}
    </Panel>
  );
}
