"use client";
import KpiCard from "../KpiCard";
import HistoricalChart from "../charts/HistoricalChart";

interface Stats {
  total_kwh: number; mean_kwh: number; max_kwh: number; min_kwh: number;
  n_records: number; start_date: string; end_date: string;
}

interface Props {
  stats: Stats | null;
  historical: { timestamp: string; consumption: number }[];
  metrics: Record<string, { MAE: number; RMSE: number; MAPE: number | null }> | null;
  bestModel: string | null;
}

export default function Overview({ stats, historical, metrics, bestModel }: Props) {
  if (!stats) return (
    <div className="flex items-center justify-center h-64">
      <p style={{ color: "var(--muted)" }}>Load a dataset from the sidebar to begin.</p>
    </div>
  );

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold" style={{ color: "var(--cyan)" }}>⚡ Overview Dashboard</h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard label="Total Consumption" value={stats.total_kwh.toLocaleString("en-IN", { maximumFractionDigits: 0 })} unit="kWh" color="var(--cyan)" />
        <KpiCard label="Avg per Record" value={stats.mean_kwh.toFixed(4)} unit="kWh" color="var(--blue)" />
        <KpiCard label="Peak Record" value={stats.max_kwh.toFixed(4)} unit="kWh" color="var(--orange)" />
        <KpiCard label="Total Records" value={stats.n_records.toLocaleString()} color="var(--green)" />
      </div>

      <p className="text-sm" style={{ color: "var(--muted)" }}>
        Dataset: <span style={{ color: "var(--text)" }}>{stats.start_date}</span> → <span style={{ color: "var(--text)" }}>{stats.end_date}</span>
        &nbsp;|&nbsp; Last 30 days held out for testing
      </p>

      <div className="rounded-xl p-5 border" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
        <p className="text-sm font-semibold mb-3" style={{ color: "var(--cyan)" }}>Historical Consumption</p>
        {historical.length > 0 ? <HistoricalChart data={historical} /> : <p style={{ color: "var(--muted)" }}>Loading chart…</p>}
      </div>

      {metrics && (
        <div className="rounded-xl p-5 border" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
          <p className="text-sm font-semibold mb-3" style={{ color: "var(--cyan)" }}>Latest Model Evaluation</p>
          {bestModel && <p className="text-sm mb-3 text-green-400">⭐ Best model (lowest MAE): <strong>{bestModel}</strong></p>}
          <table className="w-full text-sm">
            <thead>
              <tr style={{ color: "var(--muted)" }}>
                {["Model", "MAE", "RMSE", "MAPE (%)"].map(h => <th key={h} className="text-left pb-2 font-medium">{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {Object.entries(metrics).map(([m, v]) => (
                <tr key={m} className="border-t" style={{ borderColor: "var(--border)" }}>
                  <td className="py-2 font-semibold" style={{ color: m === bestModel ? "var(--cyan)" : "var(--text)" }}>{m === bestModel ? "⭐ " : ""}{m}</td>
                  <td className="py-2">{v.MAE.toFixed(4)}</td>
                  <td className="py-2">{v.RMSE.toFixed(4)}</td>
                  <td className="py-2">{v.MAPE != null ? v.MAPE.toFixed(2) : "N/A"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
