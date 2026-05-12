import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
  fill?: boolean;
}

export function Panel({ title, children, actions, className = "", fill = false }: PanelProps) {
  return (
    <div className={`flex flex-col bg-white border border-gray-200 rounded-lg overflow-hidden ${className}`}>
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-100 flex-shrink-0">
        <span className="text-xs font-semibold text-gray-700 uppercase tracking-wide">
          {title}
        </span>
        {actions && <div className="flex items-center gap-1">{actions}</div>}
      </div>
      <div className={fill ? "flex-1 min-h-0" : ""}>{children}</div>
    </div>
  );
}
