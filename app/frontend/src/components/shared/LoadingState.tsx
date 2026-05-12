export function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center h-full">
      <span className="text-sm text-gray-400 animate-pulse">{label}</span>
    </div>
  );
}
