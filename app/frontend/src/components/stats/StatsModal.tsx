import { useEffect } from "react";
import { useViewStore } from "@/state/viewStore";
import { DatasetFingerprint } from "@/components/fingerprint/DatasetFingerprint";
import { ModelComparator } from "@/components/comparator/ModelComparator";

export function StatsModal() {
  const { setStatsOpen } = useViewStore();

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") setStatsOpen(false); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [setStatsOpen]);

  return (
    <div
      className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-6"
      onClick={() => setStatsOpen(false)}
    >
      <div
        className="bg-white rounded-xl shadow-2xl w-full max-w-5xl max-h-[85vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100">
          <span className="text-sm font-semibold text-gray-800">Run Statistics</span>
          <button
            onClick={() => setStatsOpen(false)}
            className="text-gray-400 hover:text-gray-600 text-lg leading-none"
          >
            ×
          </button>
        </div>
        <div className="p-5 flex flex-col gap-6">
          <DatasetFingerprint />
          <ModelComparator />
        </div>
      </div>
    </div>
  );
}
