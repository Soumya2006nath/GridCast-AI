"use client";
import { useState } from "react";
import KpiCard from "../KpiCard";
import ForecastChart from "../charts/ForecastChart";
import { runForecast } from "../../lib/api";

interface ForecastRow { Date: string; Predicted_kWh: number }
interface Summary { avg_daily_kwh: number; peak_daily_kwh: number; total_7d_kwh: number }

interface Props { historicalAvg: number; onMetricsUpdate: () => void }

export default function Forecasting({ historicalAvg, onMetricsUpdate }: Props) {
  const [model, setModel] = useState("ARIMA");
  const [lookback, setLookback] = useState(48);
  const [epochs, setEpochs] = useState(30);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forecast, setForecast] = useState<ForecastRow[] | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);

  const run = async () => {
    setLoading(true); setError(null);
    try {
      const res = await runForecast(model, lookback, epochs);
      setForecast(res.data.forecast);
      setSummary(res.data.summary);
      onMetricsUpdate();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setError(msg);
    } finally { setLoading(false); }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold" style={{ color: "var(--cyan)" }}>🔮 7-Day Forecasting</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Controls */}
        <div className="rounded-xl p-5 border space-y-4" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
          <div>
            <label className="text-xs uppercase tracking-widest block mb-2" style={{ color: "var(--muted)" }}>Model</label>
            <select value={model} onChange={e => setModel(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border text-sm"
              style={{ background: "#0a1628", borderColor: "var(--border)", color: "var(--text)" }}>
              {["ARIMA", "SARIMA", "LSTM"].map(m => <option key={m}>{m}</option>)}
            </select>
          </div>

          {model === "LSTM" && (
            <>
              <div>
                <label className="text-xs uppercase tracking-widest block mb-2" style={{ color: "var(--muted)" }}>Lookback Window: {lookback}h</label>
                <input type="range" min={24} max={168} step={12} value={lookback} onChange={e => setLookback(+e.target.value)} className="w-full accent-blue-500" />
              </div>
              <div>
                <label className="text-xs uppercase tracking-widest block mb-2" style={{ color: "var(--muted)" }}>Epochs: {epochs}</label>
                <input type="range" min={5} max={50} step={5} value={epochs} onChange={e => setEpochs(+e.target.value)} className="w-full accent-blue-500" />
              </div>
            </>
          )}

          <button onClick={run} disabled={loading}
            className="w-full py-2.5 rounded-lg font-semibold text-sm transition-all disabled:opacity-50"
            style={{ background: "linear-gradient(135deg,#1e90ff,#00d4ff)", color: "#0a1628" }}>
            {loading ? `Training ${model}…` : `▶ Generate ${model} Forecast`}
          </button>

          {error && <p className="text-xs text-red-400">{error}</p>}
        </div>

        {/* Results */}
        <div className="md:col-span-2 space-y-4">
          {summary && (
            <div className="grid grid-cols-3 gap-3">
              <KpiCard label="7-Day Total" value={summary.total_7d_kwh.toFixed(2)} unit="kWh" color="var(--cyan)" />
              <KpiCard label="Avg Daily" value={summary.avg_daily_kwh.toFixed(2)} unit="kWh" color="var(--blue)" />
              <KpiCard label="Peak Day" value={summary.peak_daily_kwh.toFixed(2)} unit="kWh" color="var(--orange)" />
            </div>
          )}

          {forecast && (
            <div className="rounded-xl p-5 border" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
              <ForecastChart data={forecast} historicalAvg={historicalAvg} model={model} />
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr style={{ color: "var(--muted)" }}>
                      <th className="text-left pb-2">Date</th>
                      <th className="text-right pb-2">Predicted kWh</th>
                    </tr>
                  </thead>
                  <tbody>
                    {forecast.map(row => (
                      <tr key={row.Date} className="border-t" style={{ borderColor: "var(--border)" }}>
                        <td className="py-1.5">{row.Date}</td>
                        <td className="py-1.5 text-right font-mono" style={{ color: "var(--cyan)" }}>{row.Predicted_kWh.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {!forecast && !loading && (
            <div className="rounded-xl p-10 border flex items-center justify-center" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
              <p style={{ color: "var(--muted)" }}>Select a model and click Generate.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
