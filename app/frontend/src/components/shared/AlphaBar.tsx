import { EXPERT_COLORS } from "@/utils/colors";

interface AlphaBarProps {
  attribute: number;
  topology: number;
  size?: number;
}

export function AlphaBar({ attribute, topology, size = 64 }: AlphaBarProps) {
  const sum = attribute + topology || 1;
  const aPct = (attribute / sum) * 100;
  const tPct = (topology / sum) * 100;

  return (
    <div style={{ width: size }} className="flex flex-col gap-0.5">
      <div className="flex h-3 rounded-full overflow-hidden w-full">
        <div style={{ width: `${aPct}%`, backgroundColor: EXPERT_COLORS.attribute }} />
        <div style={{ width: `${tPct}%`, backgroundColor: EXPERT_COLORS.topology }} />
      </div>
      <div className="flex justify-between text-[9px] font-mono text-gray-400">
        <span>{Math.round(aPct)}%A</span>
        <span>{Math.round(tPct)}%T</span>
      </div>
    </div>
  );
}
