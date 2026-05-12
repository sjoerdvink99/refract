import { useExpertProfile } from "@/hooks/useExpertProfile";
import { useSelectionStore } from "@/state/selectionStore";
import { Panel } from "@/components/layout/Panel";
import { LoadingState } from "@/components/shared/LoadingState";
import { EmptyState } from "@/components/shared/EmptyState";
import { EXPERT_COLORS } from "@/utils/colors";
import { formatFloat } from "@/utils/format";
import type { ExpertKey, ExpertProfileEntry } from "@/api/types";

const FEAT_LABELS: Record<string, string> = {
  degree: "Degree",
  local_homophily: "Homophily",
  gate_entropy: "Gate Entropy",
  e_attribute: "Attr. Evidence",
  e_topology: "Topo. Evidence",
  concordance: "Concordance",
  uncertainty: "Uncertainty",
  log_degree: "Log Degree",
  clustering_coeff: "Clustering",
  k2_topology_evidence: "2-hop Topo.",
  hop_consistency: "Hop Consistency",
};

function CorrBar({ value }: { value: number }) {
  const pct = Math.abs(value) * 100;
  const color = value >= 0 ? "#3B82F6" : "#EF4444";
  return (
    <div className="flex items-center gap-1.5 w-full">
      <div className="flex-1 flex items-center gap-0.5">
        <div className="flex-1 bg-gray-100 rounded-full h-1.5 overflow-hidden flex justify-end">
          {value < 0 && (
            <div
              className="h-full rounded-full"
              style={{ width: `${pct}%`, backgroundColor: color }}
            />
          )}
        </div>
        <div className="w-px h-3 bg-gray-300 flex-shrink-0" />
        <div className="flex-1 bg-gray-100 rounded-full h-1.5 overflow-hidden">
          {value >= 0 && (
            <div
              className="h-full rounded-full"
              style={{ width: `${pct}%`, backgroundColor: color }}
            />
          )}
        </div>
      </div>
      <span className="text-xs font-mono text-gray-600 w-12 text-right">
        {value >= 0 ? "+" : ""}{formatFloat(value, 2)}
      </span>
    </div>
  );
}

function FeatureBar({
  mean,
  std,
  popMean,
  popStd,
}: {
  mean: number;
  std: number;
  popMean: number;
  popStd: number;
}) {
  const scale = Math.max(popMean + 2 * popStd, mean + std, 1e-8);
  const lo = Math.max(0, (mean - std) / scale);
  const hi = Math.min(1, (mean + std) / scale);
  const meanN = Math.min(1, mean / scale);
  const popMeanN = Math.min(1, popMean / scale);
  return (
    <div className="relative h-3 bg-gray-100 rounded-full overflow-hidden">
      <div
        className="absolute h-full rounded-full bg-blue-200 opacity-60"
        style={{
          left: `${lo * 100}%`,
          width: `${(hi - lo) * 100}%`,
        }}
      />
      <div
        className="absolute w-0.5 h-full bg-blue-500"
        style={{ left: `${meanN * 100}%` }}
      />
      <div
        className="absolute w-0.5 h-full bg-gray-400 opacity-60"
        style={{ left: `${popMeanN * 100}%` }}
      />
    </div>
  );
}

function ExpertCard({
  expertKey,
  entry,
}: {
  expertKey: ExpertKey;
  entry: ExpertProfileEntry;
}) {
  const color = EXPERT_COLORS[expertKey];
  const corrs = Object.entries(entry.gate_correlations).slice(0, 6);
  const stats = Object.entries(entry.feature_stats).slice(0, 6);

  return (
    <div className="border border-gray-100 rounded-lg p-3 flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
        <span className="text-xs font-semibold text-gray-700 capitalize">{expertKey} Expert</span>
        <span className="ml-auto text-xs text-gray-400">{entry.count} nodes</span>
      </div>

      {corrs.length > 0 && (
        <div>
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Gate Spearman ρ
          </span>
          <div className="mt-1.5 flex flex-col gap-1">
            {corrs.map(([feat, rho]) => (
              <div key={feat} className="flex items-center gap-2">
                <span className="text-xs text-gray-500 w-24 flex-shrink-0">
                  {FEAT_LABELS[feat] ?? feat}
                </span>
                <div className="flex-1">
                  <CorrBar value={rho} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {stats.length > 0 && (
        <div>
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Feature Distribution
          </span>
          <div className="mt-1.5 flex flex-col gap-1.5">
            {stats.map(([feat, s]) => (
              <div key={feat} className="flex items-center gap-2">
                <span className="text-xs text-gray-500 w-24 flex-shrink-0">
                  {FEAT_LABELS[feat] ?? feat}
                </span>
                <div className="flex-1">
                  <FeatureBar mean={s.mean} std={s.std} popMean={s.pop_mean} popStd={s.pop_std} />
                </div>
                <span className="text-xs font-mono text-gray-600 w-8 text-right">
                  {formatFloat(s.mean, 2)}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-1 flex items-center gap-2 text-xs text-gray-400">
            <div className="w-3 h-0.5 bg-blue-500 rounded" /> expert mean
            <div className="w-3 h-0.5 bg-gray-400 rounded opacity-60" /> population mean
          </div>
        </div>
      )}
    </div>
  );
}

export function ExpertProfileView({ className = "" }: { className?: string }) {
  const { activeRunId } = useSelectionStore();
  const { data: profile, isLoading } = useExpertProfile(activeRunId);

  return (
    <Panel title="Expert Profiles" className={className}>
      {isLoading && <LoadingState />}
      {!isLoading && !profile && (
        <EmptyState
          message="No expert profile available"
          hint="Run experiment to generate expert profiles"
        />
      )}
      {profile && (
        <div className="p-3 flex flex-col gap-3">
          {(["attribute", "topology"] as ExpertKey[]).map((key) =>
            profile[key] ? (
              <ExpertCard key={key} expertKey={key} entry={profile[key]} />
            ) : null,
          )}
        </div>
      )}
    </Panel>
  );
}
