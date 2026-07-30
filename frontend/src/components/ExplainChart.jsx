import React from "react";
import { Bar } from "react-chartjs-2";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

export default function ExplainChart({ data }) {
  if (!data) return null;

  const features = Object.entries(data.shap_values || {}).sort((a, b) => b[1] - a[1]);

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
      <Bar
        data={{
          labels: features.map(([name]) => name),
          datasets: [
            {
              label: "|SHAP|",
              data: features.map(([, value]) => value),
              backgroundColor: "rgba(52, 211, 153, 0.7)",
              borderColor: "rgb(52, 211, 153)",
              borderWidth: 1,
            },
          ],
        }}
        options={{
          indexAxis: "y", // horizontal bars, so the feature names stay readable
          responsive: true,
          animation: false,
          plugins: { legend: { labels: { color: "#e5e7eb" } } },
          scales: {
            x: { ticks: { color: "#9ca3af" }, grid: { color: "rgba(255,255,255,0.05)" } },
            y: { ticks: { color: "#e5e7eb" }, grid: { color: "rgba(255,255,255,0.05)" } },
          },
        }}
      />
    </div>
  );
}
