import type { GateWeights } from "@/api/types";
import { EXPERT_COLORS } from "@/utils/colors";
import { formatPercent, formatExpertLabel } from "@/utils/format";

interface GateWeightChartProps {
  gate: GateWeights;
}

const EXPERTS: (keyof GateWeights)[] = ["attribute", "topology"];

export function GateWeightChart({ gate }: GateWeightChartProps) {
  return (
    <div className="flex flex-col gap-1.5">
      {EXPERTS.map((expert) => {
        const value = gate[expert];
        const pct = value * 100;
        return (
          <div key={expert} className="flex items-center gap-2">
            <span className="text-xs text-gray-500 w-20 text-right">
              {formatExpertLabel(expert)}
            </span>
            <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${pct}%`,
                  backgroundColor: EXPERT_COLORS[expert],
                }}
              />
            </div>
            <span className="text-xs font-mono text-gray-700 w-10">
              {formatPercent(value)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
