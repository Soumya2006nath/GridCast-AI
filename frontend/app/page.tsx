"use client";
import { useEffect, useState, useCallback } from "react";
import Sidebar from "../components/Sidebar";
import Overview from "../components/pages/Overview";
import Analysis from "../components/pages/Analysis";
import Forecasting from "../components/pages/Forecasting";
import Performance from "../components/pages/Performance";
import Reports from "../components/pages/Reports";
import {
  loadSample, uploadDataset, getStats, getHistorical,
  getDaily, getWeekly, getMonthly, getPreview, getMetrics, getHealth,
} from "../lib/api";

export default function Home() {
  const [page, setPage] = useState("overview");
  const [dataLoaded, setDataLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<{ msg: string; type: "ok" | "err" } | null>(null);

  // Data state
  const [stats, setStats] = useState(null);
  const [historical, setHistorical] = useState([]);
  const [daily, setDaily] = useState([]);
  const [weekly, setWeekly] = useState([]);
  const [monthly, setMonthly] = useState([]);
  const [preview, setPreview] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [bestModel, setBestModel] = useState<string | null>(null);
  const [gpuInfo, setGpuInfo] = useState<string[]>([]);

  // Health check on mount
  useEffect(() => {
    getHealth().then(r => setGpuInfo(r.data.gpus ?? [])).catch(() => {});
  }, []);

  const notify = (msg: string, type: "ok" | "err" = "ok") => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 4000);
  };

  const fetchAllData = useCallback(async () => {
    try {
      const [s, h, d, w, m, p] = await Promise.all([
        getStats(), getHistorical(2000), getDaily(), getWeekly(), getMonthly(), getPreview(100),
      ]);
      setStats(s.data);
      setHistorical(h.data);
      setDaily(d.data);
      setWeekly(w.data);
      setMonthly(m.data);
      setPreview(p.data);
      setDataLoaded(true);
    } catch (e) {
      notify("Failed to fetch data from backend.", "err");
    }
  }, []);

  const fetchMetrics = useCallback(async () => {
    try {
      const r = await getMetrics();
      setMetrics(r.data.metrics);
      setBestModel(r.data.best_model);
    } catch {}
  }, []);

  const handleLoadSample = async () => {
    setLoading(true);
    try {
      await loadSample();
      await fetchAllData();
      notify("Sample dataset loaded ✓");
    } catch { notify("Failed to load sample dataset.", "err"); }
    finally { setLoading(false); }
  };

  const handleUpload = async (file: File) => {
    setLoading(true);
    try {
      await uploadDataset(file);
      await fetchAllData();
      notify("Dataset uploaded ✓");
    } catch { notify("Upload failed. Check your CSV format.", "err"); }
    finally { setLoading(false); }
  };

  // Historical average (daily mean) for forecast chart reference line
  const historicalAvg = daily.length > 0
    ? (daily as { consumption: number }[]).reduce((s, r) => s + r.consumption, 0) / daily.length
    : 0;

  return (
    <div className="flex min-h-screen">
      <Sidebar
        active={page}
        onChange={setPage}
        dataLoaded={dataLoaded}
        onLoadSample={handleLoadSample}
        onUpload={handleUpload}
        gpuInfo={gpuInfo}
      />

      <main className="flex-1 p-8 overflow-y-auto">
        {loading && (
          <div className="fixed inset-0 flex items-center justify-center z-50" style={{ background: "rgba(10,22,40,0.8)" }}>
            <div className="rounded-xl px-8 py-6 border text-center" style={{ background: "var(--card)", borderColor: "var(--border)" }}>
              <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-3" />
              <p style={{ color: "var(--muted)" }}>Loading…</p>
            </div>
          </div>
        )}

        {toast && (
          <div className={`fixed top-4 right-4 z-50 px-4 py-3 rounded-lg text-sm font-medium shadow-lg ${toast.type === "ok" ? "text-green-400" : "text-red-400"}`}
            style={{ background: "var(--card)", border: `1px solid ${toast.type === "ok" ? "#00e676" : "#f44336"}` }}>
            {toast.msg}
          </div>
        )}

        {page === "overview" && <Overview stats={stats} historical={historical} metrics={metrics} bestModel={bestModel} />}
        {page === "analysis" && (
          dataLoaded
            ? <Analysis daily={daily} weekly={weekly} monthly={monthly} historical={historical} preview={preview} />
            : <p style={{ color: "var(--muted)" }}>Load a dataset first.</p>
        )}
        {page === "forecast" && (
          dataLoaded
            ? <Forecasting historicalAvg={historicalAvg} onMetricsUpdate={fetchMetrics} />
            : <p style={{ color: "var(--muted)" }}>Load a dataset first.</p>
        )}
        {page === "performance" && <Performance metrics={metrics} bestModel={bestModel} />}
        {page === "reports" && <Reports metrics={metrics} bestModel={bestModel} stats={stats} />}
      </main>
    </div>
  );
}
