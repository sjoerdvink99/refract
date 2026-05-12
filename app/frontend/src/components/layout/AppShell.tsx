import { HeaderBar } from "./HeaderBar";
import { RegimeMap } from "@/components/regime-map/RegimeMap";
import { RoutingCalibrationScatter } from "@/components/routing/RoutingCalibrationScatter";
import { AnalysisPanel } from "@/components/routing/AnalysisPanel";
import { StatsModal } from "@/components/stats/StatsModal";
import { useViewStore } from "@/state/viewStore";

export function AppShell() {
  const { statsOpen } = useViewStore();

  return (
    <div className="flex flex-col h-screen bg-gray-50 overflow-hidden">
      <HeaderBar />
      <div className="flex-1 min-h-0 grid grid-cols-[58fr_42fr] gap-2 p-2">
        <RegimeMap />
        <div className="min-h-0 flex flex-col gap-2">
          <div className="flex-[5] min-h-0">
            <RoutingCalibrationScatter />
          </div>
          <div className="flex-[5] min-h-0">
            <AnalysisPanel />
          </div>
        </div>
      </div>
      {statsOpen && <StatsModal />}
    </div>
  );
}
