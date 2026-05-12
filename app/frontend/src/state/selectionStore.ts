import { create } from "zustand";

interface SelectionState {
  selectedNodeIds: string[];
  hoveredNodeId: string | null;
  activeRunId: string | null;
  setSelectedNode: (id: string | null) => void;
  setSelectedNodes: (ids: string[]) => void;
  toggleSelectedNode: (id: string) => void;
  clearSelection: () => void;
  setHoveredNode: (id: string | null) => void;
  setActiveRun: (id: string | null) => void;
}

export const useSelectionStore = create<SelectionState>((set) => ({
  selectedNodeIds: [],
  hoveredNodeId: null,
  activeRunId: null,
  setSelectedNode: (id) => set({ selectedNodeIds: id ? [id] : [] }),
  setSelectedNodes: (ids) => set({ selectedNodeIds: ids }),
  toggleSelectedNode: (id) =>
    set((s) => ({
      selectedNodeIds: s.selectedNodeIds.includes(id)
        ? s.selectedNodeIds.filter((x) => x !== id)
        : [...s.selectedNodeIds, id],
    })),
  clearSelection: () => set({ selectedNodeIds: [] }),
  setHoveredNode: (id) => set({ hoveredNodeId: id }),
  setActiveRun: (id) => set({ activeRunId: id }),
}));
