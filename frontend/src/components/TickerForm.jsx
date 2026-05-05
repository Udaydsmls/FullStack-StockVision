import React from "react";

export default function TickerForm({
  ticker,
  onTickerChange,
  model,
  onModelChange,
  models,
  onSubmit,
  loading,
}) {
  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit();
  };

  return (
    <form
      className="grid gap-3 sm:grid-cols-[1fr_180px_140px] items-stretch"
      onSubmit={handleSubmit}
    >
      <input
        type="text"
        value={ticker}
        onChange={(e) => onTickerChange(e.target.value.toUpperCase().trim())}
        placeholder="Ticker (e.g. AAPL, MSFT, TSLA)"
        autoComplete="off"
        spellCheck="false"
        aria-label="Ticker symbol"
        className="rounded-lg bg-gray-900 border border-gray-700 px-4 py-2 text-white uppercase placeholder:normal-case placeholder:text-gray-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
      />
      <select
        value={model}
        onChange={(e) => onModelChange(e.target.value)}
        aria-label="Model architecture"
        className="rounded-lg bg-gray-900 border border-gray-700 px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
      >
        {models.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>
      <button
        type="submit"
        disabled={loading || !ticker}
        className="rounded-lg bg-emerald-600 px-4 py-2 font-medium text-white transition hover:bg-emerald-500 disabled:opacity-50 disabled:hover:bg-emerald-600"
      >
        {loading ? "Loading…" : "Predict"}
      </button>
    </form>
  );
}
