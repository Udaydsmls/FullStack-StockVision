import React from "react";

import { formatCurrency, formatPercent, priceDelta } from "../utils/format";

export default function PredictionPanel({ data }) {
  if (!data) return null;
  const { absolute, relative } = priceDelta(data.prediction, data.last_close);
  const positive = (absolute ?? 0) >= 0;

  return (
    <section className="grid gap-4 sm:grid-cols-3">
      <Card label="Last close" value={formatCurrency(data.last_close)} />
      <Card label="Next-step prediction" value={formatCurrency(data.prediction)} highlight />
      <Card
        label="Implied move"
        value={`${formatCurrency(absolute)} (${formatPercent(relative)})`}
        tone={positive ? "positive" : "negative"}
      />
    </section>
  );
}

function Card({ label, value, highlight = false, tone = "neutral" }) {
  const tones = {
    neutral: "text-white",
    positive: "text-emerald-400",
    negative: "text-red-400",
  };
  return (
    <div
      className={`rounded-xl border border-gray-800 bg-gray-900/60 p-4 ${
        highlight ? "ring-1 ring-emerald-500/40" : ""
      }`}
    >
      <div className="text-xs uppercase tracking-wide text-gray-400">{label}</div>
      <div className={`mt-1 text-2xl font-semibold ${tones[tone]}`}>{value}</div>
    </div>
  );
}
