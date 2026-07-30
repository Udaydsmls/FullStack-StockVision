import React from "react";

export default function Header({ online }) {
  return (
    <header className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-semibold tracking-tight text-white">StockVision</h1>
        <BackendStatus online={online} />
      </div>
      <p className="text-sm text-gray-400">
        Nine forecasting architectures, served by three interchangeable backends.
      </p>
    </header>
  );
}

function BackendStatus({ online }) {
  if (online === null) return <span className="text-xs text-gray-500">checking…</span>;
  return (
    <span className="flex items-center gap-2 text-xs text-gray-300">
      <span
        className={`inline-block h-2 w-2 rounded-full ${online ? "bg-emerald-500" : "bg-red-500"}`}
        aria-hidden
      />
      {online ? "backend online" : "backend unreachable"}
    </span>
  );
}
