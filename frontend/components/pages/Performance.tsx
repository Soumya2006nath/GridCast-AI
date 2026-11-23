"use client";
import MetricsChart from "../charts/MetricsChart";

interface ModelMetrics { MAE: number; RMSE: number; MAPE: number | null }
interface Props {
  metrics: Record<string, ModelMetrics> | null;
  bestModel: string | null;
}

export default function Performance({ metrics, bestModel }: Props) {
  if (!metrics || Object.keys(metrics).length === 0) return (
    <div className="flex items-center justify-center h-64">
      <p style={{ color: "var(--muted)" }}>Train at least one model in Forecasting first.</p>
    </div>
  );

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold" style={{ color: "var(--cyan)" }}>📈 Model Performance</h1>

      {bestModel && <p className="text-sm text-green-400">⭐ Best model (lowest MAE): <strong>{bestModel}</strong></p>}

      {/* Comparison table */}
      <div className="rounded-xl p-5 border" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
        <table className="w-full text-sm">
          <thead>
            <tr style={{ color: "var(--muted)" }}>
              {["Model", "MAE", "RMSE", "MAPE (%)"].map(h => (
                <th key={h} className="text-left pb-3 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(metrics).map(([m, v]) => (
              <tr key={m} className="border-t" style={{ borderColor: "var(--border)" }}>
                <td className="py-2.5 font-semibold" style={{ color: m === bestModel ? "var(--cyan)" : "var(--text)" }}>
                  {m === bestModel ? "⭐ " : ""}{m}
                </td>
                <td className="py-2.5 font-mono">{v.MAE.toFixed(4)}</td>
                <td className="py-2.5 font-mono">{v.RMSE.toFixed(4)}</td>
                <td className="py-2.5 font-mono">{v.MAPE != null ? v.MAPE.toFixed(2) : "N/A"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Bar chart */}
      <div className="rounded-xl p-5 border" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
        <p className="text-sm font-semibold mb-3" style={{ color: "var(--cyan)" }}>Visual Comparison</p>
        <MetricsChart metrics={metrics} />
      </div>
    </div>
  );
}
