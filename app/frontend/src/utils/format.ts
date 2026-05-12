export function formatPercent(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatFloat(value: number, decimals = 3): string {
  return value.toFixed(decimals);
}

export function formatRegimeLabel(regime: string): string {
  return regime.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatExpertLabel(expert: string): string {
  return expert.charAt(0).toUpperCase() + expert.slice(1);
}

export function formatNodeId(nodeId: string): string {
  return nodeId.length > 12 ? `${nodeId.slice(0, 10)}…` : nodeId;
}
