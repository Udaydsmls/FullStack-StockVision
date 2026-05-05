export function formatCurrency(value, currency = "USD") {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    maximumFractionDigits: 2,
  }).format(value);
}

export function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
}

export function priceDelta(prediction, lastClose) {
  if (!Number.isFinite(prediction) || !Number.isFinite(lastClose) || lastClose === 0) {
    return { absolute: null, relative: null };
  }
  const absolute = prediction - lastClose;
  const relative = absolute / lastClose;
  return { absolute, relative };
}
