import React from "react";

export default function Header({ healthy }) {
  return (
    <header className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-semibold text-white tracking-tight">StockVision</h1>
        <ServiceStatus healthy={healthy} />
      </div>
      <p className="text-sm text-gray-400">
        Time-series forecasting across multiple architectures, served from a unified API.
      </p>
    </header>
  );
}

function ServiceStatus({ healthy }) {
  if (healthy === null) {
    return <span className="text-xs text-gray-500">checking…</span>;
  }
  const colour = healthy ? "bg-emerald-500" : "bg-red-500";
  const label = healthy ? "API online" : "API unreachable";
  return (
    <span className="flex items-center gap-2 text-xs text-gray-300">
      <span className={`inline-block h-2 w-2 rounded-full ${colour}`} aria-hidden />
      {label}
    </span>
  );
}
