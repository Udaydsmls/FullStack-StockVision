import React from "react";
import { Line } from "react-chartjs-2";
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

export default function PriceChart({ history, historyDates, prediction }) {
  if (!history || history.length === 0) return null;

  const hasForecast = typeof prediction === "number";
  const labels = historyDates?.length === history.length
    ? [...historyDates]
    : history.map((_, index) => `t-${history.length - 1 - index}`);
  if (hasForecast) labels.push("next");

  const datasets = [
    {
      label: "Close",
      data: hasForecast ? [...history, null] : history,
      borderColor: "rgb(96, 165, 250)",
      backgroundColor: "rgba(96, 165, 250, 0.15)",
      fill: true,
      tension: 0.25,
      pointRadius: 0,
    },
  ];

  if (hasForecast) {
    // A two-point dashed line joining the last real close to the forecast.
    datasets.push({
      label: "Forecast",
      data: [...new Array(history.length - 1).fill(null), history[history.length - 1], prediction],
      borderColor: "rgb(52, 211, 153)",
      backgroundColor: "rgb(52, 211, 153)",
      borderDash: [4, 4],
      pointRadius: 4,
    });
  }

  // Leave 10% headroom above and below so the line is not glued to the frame.
  const values = hasForecast ? [...history, prediction] : history;
  const lowest = Math.min(...values);
  const highest = Math.max(...values);
  const padding = (highest - lowest) * 0.1 || 1;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
      <Line
        data={{ labels, datasets }}
        options={{
          responsive: true,
          animation: false,
          interaction: { mode: "nearest", axis: "x", intersect: false },
          plugins: { legend: { labels: { color: "#e5e7eb" } } },
          scales: {
            x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.05)" } },
            y: {
              ticks: { color: "#9ca3af" },
              grid: { color: "rgba(255,255,255,0.05)" },
              min: lowest - padding,
              max: highest + padding,
            },
          },
        }}
      />
    </div>
  );
}
