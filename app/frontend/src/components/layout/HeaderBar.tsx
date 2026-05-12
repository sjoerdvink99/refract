import { useRuns } from "@/hooks/useRunManifest";
import { useSelectionStore } from "@/state/selectionStore";
import { useViewStore } from "@/state/viewStore";

export function HeaderBar() {
  const { data: runs } = useRuns();
  const { activeRunId, setActiveRun, selectedNodeIds } = useSelectionStore();
  const { statsOpen, setStatsOpen } = useViewStore();

  const runOptions = (runs ?? []).map((r) => ({
    value: r.run_id,
    label: `${r.dataset_name} · ${r.num_nodes.toLocaleString()} nodes`,
  }));

  return (
    <header className="flex items-center gap-3 px-4 py-2.5 bg-gray-900 text-white border-b border-gray-800 shrink-0">
      <span className="font-semibold text-sm tracking-tight">Refract</span>
      <div className="h-4 w-px bg-gray-700" />
      {runOptions.length > 0 && (
        <select
          className="text-xs bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
          value={activeRunId ?? ""}
          onChange={(e) => setActiveRun(e.target.value || null)}
        >
          <option value="">Select run…</option>
          {runOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      )}
      <div className="flex-1" />
      {selectedNodeIds.length === 1 && (
        <span className="text-xs text-gray-400 font-mono">Node: {selectedNodeIds[0]}</span>
      )}
      {selectedNodeIds.length > 1 && (
        <span className="text-xs text-gray-400 font-mono">{selectedNodeIds.length} selected</span>
      )}
      <button
        onClick={() => setStatsOpen(!statsOpen)}
        className={`text-xs px-2.5 py-1 rounded border transition-colors ${
          statsOpen
            ? "bg-blue-600 border-blue-500 text-white"
            : "bg-gray-800 border-gray-700 text-gray-300 hover:text-white"
        }`}
      >
        Stats
      </button>
    </header>
  );
}
