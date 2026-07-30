import React from "react";

import { formatCurrency, formatPercent } from "../utils/format";

export default function PredictionPanel({ data }) {
  if (!data) return null;

  const change = data.prediction - data.last_close;
  const percent = data.last_close === 0 ? 0 : change / data.last_close;

  return (
    <section className="grid gap-4 sm:grid-cols-3">
      <Card label="Last close" value={formatCurrency(data.last_close)} />
      <Card label="Next-step prediction" value={formatCurrency(data.prediction)} highlight />
      <Card
        label="Implied move"
        value={`${formatCurrency(change)} (${formatPercent(percent)})`}
        colour={change >= 0 ? "text-emerald-400" : "text-red-400"}
      />
    </section>
  );
}

function Card({ label, value, highlight = false, colour = "text-white" }) {
  return (
    <div
      className={`rounded-xl border border-gray-800 bg-gray-900/60 p-4 ${
        highlight ? "ring-1 ring-emerald-500/40" : ""
      }`}
    >
      <div className="text-xs uppercase tracking-wide text-gray-400">{label}</div>
      <div className={`mt-1 text-2xl font-semibold ${colour}`}>{value}</div>
    </div>
  );
}
