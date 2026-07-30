import React from "react";

// A row of buttons where exactly one is selected. Used for the backend picker
// and for the forecast/explain tabs.
export default function Switcher({ label, options, active, onChange }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-16 shrink-0 text-xs uppercase tracking-wide text-gray-400">{label}</span>
      <div className="flex flex-1 gap-1 rounded-lg border border-gray-800 bg-gray-900/60 p-1 text-sm">
        {options.map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => onChange(option.id)}
            className={`flex-1 rounded-md px-3 py-1.5 transition ${
              active === option.id
                ? "bg-emerald-600 text-white"
                : "text-gray-300 hover:text-white"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
