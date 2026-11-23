"use client";
import { useState } from "react";
import { LayoutDashboard, BarChart2, TrendingUp, Activity, FileText, Zap } from "lucide-react";

const NAV = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "analysis", label: "Data Analysis", icon: BarChart2 },
  { id: "forecast", label: "Forecasting", icon: TrendingUp },
  { id: "performance", label: "Model Performance", icon: Activity },
  { id: "reports", label: "Reports", icon: FileText },
];

interface Props {
  active: string;
  onChange: (id: string) => void;
  dataLoaded: boolean;
  onLoadSample: () => void;
  onUpload: (f: File) => void;
  gpuInfo: string[];
}

export default function Sidebar({ active, onChange, dataLoaded, onLoadSample, onUpload, gpuInfo }: Props) {
  return (
    <aside className="w-64 min-h-screen flex flex-col border-r" style={{ background: "#0d1b2a", borderColor: "var(--border)" }}>
      {/* Logo */}
      <div className="p-6 border-b" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-2">
          <Zap className="text-cyan-400" size={22} />
          <span className="text-lg font-bold" style={{ color: "var(--cyan)" }}>GridCast AI</span>
        </div>
        <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>Smart Grid Forecasting · Group 14</p>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-4 space-y-1">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => onChange(id)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all ${
              active === id
                ? "text-white font-semibold"
                : "hover:bg-white/5"
            }`}
            style={active === id ? { background: "rgba(30,144,255,0.18)", color: "var(--cyan)" } : { color: "var(--muted)" }}
          >
            <Icon size={16} />
            {label}
          </button>
        ))}
      </nav>

      {/* Dataset loader */}
      <div className="p-4 border-t space-y-2" style={{ borderColor: "var(--border)" }}>
        <p className="text-xs font-semibold uppercase tracking-widest" style={{ color: "var(--muted)" }}>Load Dataset</p>
        <button
          onClick={onLoadSample}
          className="w-full py-2 rounded-lg text-sm font-semibold transition-all hover:opacity-90"
          style={{ background: "linear-gradient(135deg,#1e90ff,#00d4ff)", color: "#0a1628" }}
        >
          Use Sample Dataset
        </button>
        <label className="w-full block">
          <span className="w-full flex items-center justify-center py-2 rounded-lg text-sm cursor-pointer border transition-all hover:bg-white/5"
            style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
            Upload CSV
          </span>
          <input type="file" accept=".csv" className="hidden" onChange={e => e.target.files?.[0] && onUpload(e.target.files[0])} />
        </label>
        {dataLoaded && <p className="text-xs text-green-400">✓ Dataset loaded</p>}
      </div>

      {/* GPU badge */}
      <div className="p-4 border-t" style={{ borderColor: "var(--border)" }}>
        <p className="text-xs font-semibold uppercase tracking-widest mb-1" style={{ color: "var(--muted)" }}>Hardware</p>
        {gpuInfo.length > 0
          ? <p className="text-xs text-green-400">⚡ GPU: {gpuInfo[0].split("/").pop()}</p>
          : <p className="text-xs" style={{ color: "var(--muted)" }}>CPU only</p>}
        <div className="mt-3 space-y-0.5">
          {["Lawrence Sarkar", "Soumyadeep Nath", "Sayon De", "Pravin Agarwal"].map(m => (
            <p key={m} className="text-xs" style={{ color: "var(--muted)" }}>· {m}</p>
          ))}
        </div>
      </div>
    </aside>
  );
}
