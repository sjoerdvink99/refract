interface LegendItem {
  color: string;
  label: string;
}

interface ColorLegendProps {
  items: LegendItem[];
  title?: string;
}

export function ColorLegend({ items, title }: ColorLegendProps) {
  return (
    <div className="flex flex-col gap-1.5">
      {title && <span className="text-xs font-medium text-gray-500">{title}</span>}
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {items.map((item) => (
          <div key={item.label} className="flex items-center gap-1.5">
            <div
              className="w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ backgroundColor: item.color }}
            />
            <span className="text-xs text-gray-600">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
