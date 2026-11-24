"use client";
import { useState, useEffect, useRef } from "react";
import KpiCard from "../KpiCard";
import ForecastChart from "../charts/ForecastChart";
import { runForecast } from "../../lib/api";
import ClientOnly from "../ClientOnly";
import axios from "axios";

interface ForecastRow { Date: string; Predicted_kWh: number }
interface Summary { avg_daily_kwh: number; peak_daily_kwh: number; total_7d_kwh: number }

interface Props { historicalAvg: number; onMetricsUpdate: () => void }

export default function Forecasting({ historicalAvg, onMetricsUpdate }: Props) {
  const [model, setModel] = useState("ARIMA");
  const [lookback, setLookback] = useState(48);
  const [epochs, setEpochs] = useState(30);
  const [useCached, setUseCached] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [forecast, setForecast] = useState<ForecastRow[] | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);

  const [modelSaved, setModelSaved] = useState(false);
  const [pretraining, setPretraining] = useState(false);
  const [pretrainMsg, setPretrainMsg] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    axios.get("http://localhost:8000/api/lstm/status")
      .then(r => setModelSaved(r.data.model_saved))
      .catch(() => {});
  }, []);

  const startPretrain = async () => {
    setPretraining(true);
    setPretrainMsg("Training LSTM in background… takes a few minutes.");
    try {
      await axios.post("http://localhost:8000/api/lstm/pretrain", { lookback, epochs: 50 });
    } catch {
      setPretrainMsg("Failed to start training.");
      setPretraining(false);
      return;
    }
    pollRef.current = setInterval(async () => {
      try {
        const r = await axios.get("http://localhost:8000/api/lstm/status");
        if (r.data.error) {
          setPretrainMsg(`Training error: ${r.data.error}`);
          setPretraining(false);
          clearInterval(pollRef.current!);
        } else if (r.data.done && r.data.model_saved) {
          setModelSaved(true);
          setPretraining(false);
          setPretrainMsg("✅ Model saved! LSTM loads instantly from now on.");
          clearInterval(pollRef.current!);
        }
      } catch {}
    }, 3000);
  };

  const run = async () => {
    setLoading(true); setError(null);
    try {
      const res = await runForecast(model, lookback, epochs, model === "LSTM" ? useCached : true);
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

              <div className="rounded-lg p-3 border" style={{ borderColor: modelSaved ? "#00e676" : "var(--border)", background: "#0a1628" }}>
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-semibold" style={{ color: modelSaved ? "var(--green)" : "var(--muted)" }}>
                    {modelSaved ? "⚡ Demo Mode" : "Demo Mode"}
                  </span>
                  <button onClick={() => setUseCached(v => !v)}
                    className="w-10 h-5 rounded-full transition-all relative"
                    style={{ background: useCached ? "var(--green)" : "var(--border)" }}>
                    <span className="absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all"
                      style={{ left: useCached ? "calc(100% - 1.1rem)" : "0.1rem" }} />
                  </button>
                </div>
                <p className="text-xs" style={{ color: "var(--muted)" }}>
                  {useCached && modelSaved
                    ? "Loads saved model — instant."
                    : useCached && !modelSaved
                    ? "No saved model yet. Pre-train below."
                    : "Will retrain from scratch."}
                </p>
                {!modelSaved && (
                  <button onClick={startPretrain} disabled={pretraining}
                    className="mt-2 w-full py-1.5 rounded-lg text-xs font-semibold disabled:opacity-50"
                    style={{ background: "linear-gradient(135deg,#1e90ff,#00d4ff)", color: "#0a1628" }}>
                    {pretraining ? "Training… (check back in ~3 min)" : "Pre-train & Save Model"}
                  </button>
                )}
                {pretrainMsg && (
                  <p className="text-xs mt-1" style={{ color: modelSaved ? "var(--green)" : "var(--cyan)" }}>{pretrainMsg}</p>
                )}
              </div>
            </>
          )}

          <button onClick={run} disabled={loading}
            className="w-full py-2.5 rounded-lg font-semibold text-sm transition-all disabled:opacity-50"
            style={{ background: "linear-gradient(135deg,#1e90ff,#00d4ff)", color: "#0a1628" }}>
            {loading
              ? `${model === "LSTM" && useCached && modelSaved ? "Loading" : "Training"} ${model}…`
              : `▶ Generate ${model} Forecast`}
          </button>

          {error && <p className="text-xs text-red-400">{error}</p>}
        </div>

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
              <ClientOnly><ForecastChart data={forecast} historicalAvg={historicalAvg} model={model} /></ClientOnly>
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
