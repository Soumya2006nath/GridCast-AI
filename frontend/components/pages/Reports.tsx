"use client";
import { getReportUrl } from "../../lib/api";

interface ModelMetrics { MAE: number; RMSE: number; MAPE: number | null }
interface Props {
  metrics: Record<string, ModelMetrics> | null;
  bestModel: string | null;
  stats: { start_date: string; end_date: string; n_records: number; total_kwh: number } | null;
}

export default function Reports({ metrics, bestModel, stats }: Props) {
  const hasData = metrics && Object.keys(metrics).length > 0;

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold" style={{ color: "var(--cyan)" }}>📄 Reports</h1>

      {!hasData ? (
        <p style={{ color: "var(--muted)" }}>Train at least one model before generating a report.</p>
      ) : (
        <div className="space-y-4">
          <div className="rounded-xl p-5 border" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
            <p className="text-sm mb-4" style={{ color: "var(--muted)" }}>
              The PDF report contains dataset info, model performance, and the 7-day forecast for the best model (<strong style={{ color: "var(--cyan)" }}>{bestModel}</strong>).
            </p>
            <a href={getReportUrl()} target="_blank" rel="noopener noreferrer"
              className="inline-block px-6 py-2.5 rounded-lg font-semibold text-sm transition-all hover:opacity-90"
              style={{ background: "linear-gradient(135deg,#1e90ff,#00d4ff)", color: "#0a1628" }}>
              ⬇ Download PDF Report
            </a>
          </div>

          {/* Summary preview */}
          {stats && (
            <div className="rounded-xl p-5 border" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
              <p className="text-sm font-semibold mb-3" style={{ color: "var(--cyan)" }}>Report Preview</p>
              <div className="grid grid-cols-2 gap-3 text-sm mb-4">
                {[
                  ["Dataset Start", stats.start_date],
                  ["Dataset End", stats.end_date],
                  ["Records", stats.n_records.toLocaleString()],
                  ["Total kWh", stats.total_kwh.toFixed(2)],
                ].map(([k, v]) => (
                  <div key={k}>
                    <span style={{ color: "var(--muted)" }}>{k}: </span>
                    <span style={{ color: "var(--text)" }}>{v}</span>
                  </div>
                ))}
              </div>

              <table className="w-full text-sm">
                <thead>
                  <tr style={{ color: "var(--muted)" }}>
                    {["Model", "MAE", "RMSE", "MAPE (%)"].map(h => <th key={h} className="text-left pb-2 font-medium">{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(metrics!).map(([m, v]) => (
                    <tr key={m} className="border-t" style={{ borderColor: "var(--border)" }}>
                      <td className="py-1.5" style={{ color: m === bestModel ? "var(--cyan)" : "var(--text)" }}>{m}</td>
                      <td className="py-1.5 font-mono">{v.MAE.toFixed(4)}</td>
                      <td className="py-1.5 font-mono">{v.RMSE.toFixed(4)}</td>
                      <td className="py-1.5 font-mono">{v.MAPE != null ? v.MAPE.toFixed(2) : "N/A"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
