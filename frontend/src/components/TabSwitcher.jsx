import React from "react";

export default function TabSwitcher({ tabs, active, onChange }) {
  return (
    <div className="flex gap-1 rounded-lg border border-gray-800 bg-gray-900/60 p-1 text-sm">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          type="button"
          onClick={() => onChange(tab.id)}
          className={`flex-1 rounded-md px-3 py-1.5 transition ${
            active === tab.id
              ? "bg-emerald-600 text-white"
              : "text-gray-300 hover:text-white"
          }`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
