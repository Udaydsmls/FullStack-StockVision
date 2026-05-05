import React, { useMemo } from "react";
import { Line } from "react-chartjs-2";
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

export default function PriceChart({ history, historyDates, prediction }) {
  const chart = useMemo(() => buildChart(history, historyDates, prediction), [
    history,
    historyDates,
    prediction,
  ]);
  if (!history || history.length === 0) return null;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
      <Line data={chart.data} options={chart.options} />
    </div>
  );
}

function buildChart(history, historyDates, prediction) {
  const labels = (historyDates && historyDates.length === history.length
    ? historyDates
    : history.map((_, i) => `t-${history.length - i - 1}`)
  ).slice();

  const predictionSeries = prediction === null || prediction === undefined
    ? null
    : new Array(history.length - 1).fill(null).concat(prediction);

  if (prediction !== null && prediction !== undefined) {
    labels.push("forecast");
    predictionSeries.push(prediction);
  }

  const datasets = [
    {
      label: "Close",
      data: prediction !== null && prediction !== undefined ? [...history, null] : history,
      borderColor: "rgb(96, 165, 250)",
      backgroundColor: "rgba(96, 165, 250, 0.15)",
      fill: true,
      tension: 0.25,
      pointRadius: 0,
    },
  ];

  if (predictionSeries) {
    datasets.push({
      label: "Forecast",
      data: predictionSeries,
      borderColor: "rgb(52, 211, 153)",
      backgroundColor: "rgb(52, 211, 153)",
      borderDash: [4, 4],
      pointRadius: 4,
      pointBackgroundColor: "rgb(52, 211, 153)",
      tension: 0,
    });
  }

  const all = history.concat(prediction !== null && prediction !== undefined ? [prediction] : []);
  const min = Math.min(...all);
  const max = Math.max(...all);
  const padding = (max - min) * 0.1 || 1;

  return {
    data: { labels, datasets },
    options: {
      responsive: true,
      animation: false,
      plugins: {
        legend: { labels: { color: "#e5e7eb" } },
        tooltip: { mode: "index", intersect: false },
      },
      interaction: { mode: "nearest", axis: "x", intersect: false },
      scales: {
        x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: {
          ticks: { color: "#9ca3af" },
          grid: { color: "rgba(255,255,255,0.05)" },
          min: min - padding,
          max: max + padding,
        },
      },
    },
  };
}
