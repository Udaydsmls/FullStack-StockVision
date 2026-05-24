import React, { useMemo } from "react";
import { Bar } from "react-chartjs-2";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

export default function ExplainChart({ data }) {
  const chart = useMemo(() => buildChart(data), [data]);
  if (!data) return null;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
      <div className="mb-2 flex items-center justify-between text-sm text-gray-300">
        <span>
          Prediction: <span className="text-white">{data.prediction?.toFixed(2)}</span>
        </span>
        <span>
          Base value: <span className="text-white">{data.base_value?.toFixed(2)}</span>
        </span>
      </div>
      <Bar data={chart.data} options={chart.options} />
    </div>
  );
}

function buildChart(data) {
  const entries = Object.entries(data?.shap_values || {}).sort((a, b) => b[1] - a[1]);
  return {
    data: {
      labels: entries.map(([name]) => name),
      datasets: [
        {
          label: "|SHAP|",
          data: entries.map(([, value]) => value),
          backgroundColor: "rgba(52, 211, 153, 0.7)",
          borderColor: "rgb(52, 211, 153)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      animation: false,
      plugins: { legend: { labels: { color: "#e5e7eb" } } },
      scales: {
        x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.05)" } },
        y: { ticks: { color: "#e5e7eb" }, grid: { color: "rgba(255,255,255,0.05)" } },
      },
    },
  };
}
