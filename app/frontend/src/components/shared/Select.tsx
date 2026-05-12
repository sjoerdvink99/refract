import type { SelectHTMLAttributes } from "react";

interface Option {
  value: string;
  label: string;
}

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  options: Option[];
  label?: string;
}

export function Select({ options, label, className = "", ...props }: SelectProps) {
  return (
    <label className="flex flex-col gap-1">
      {label && <span className="text-xs text-gray-500 font-medium">{label}</span>}
      <select
        className={`text-sm border border-gray-200 rounded px-2.5 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 ${className}`}
        {...props}
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  );
}
