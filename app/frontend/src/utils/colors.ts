import type { ExpertKey, RegimeLabel } from "@/api/types";

export const REGIME_COLORS: Record<RegimeLabel, string> = {
  attribute_dominant: "#3B82F6",
  topology_dominant: "#14B8A6",
  concordant: "#22C55E",
  conflict: "#F97316",
  uncertain: "#9CA3AF",
};

export const EXPERT_COLORS: Record<ExpertKey, string> = {
  attribute: "#F97316",
  topology: "#6366F1",
};

export const CORRECTNESS_COLORS = {
  correct: "#22C55E",
  incorrect: "#EF4444",
  unknown: "#9CA3AF",
} as const;

export function regimeColor(regime: string): string {
  return REGIME_COLORS[regime as RegimeLabel] ?? "#9CA3AF";
}

export function expertColor(expert: string): string {
  return EXPERT_COLORS[expert as ExpertKey] ?? "#9CA3AF";
}

export function correctnessColor(correct: boolean | null): string {
  if (correct === null) return CORRECTNESS_COLORS.unknown;
  return correct ? CORRECTNESS_COLORS.correct : CORRECTNESS_COLORS.incorrect;
}
